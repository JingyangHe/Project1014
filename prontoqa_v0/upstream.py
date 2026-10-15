"""Adapter for the pinned upstream generator. Never edits upstream files."""
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from .logic import Fact, Literal, Rejected, Rule, World, require_decidable


@contextmanager
def upstream_context(path, seed=None):
    """Fix upstream's bad_patterns.txt relative read and isolate global RNGs.

    This adapter is intentionally single-threaded: upstream uses process-global
    cwd/random state. All CLI invocations are separate processes.
    """
    cwd, old_path = Path.cwd(), list(sys.path)
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        os.chdir(path)
        sys.path.insert(0, str(path))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = old_path
        random.setstate(py_state)
        np.random.set_state(np_state)


def source_hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.iterdir()) if p.is_file() and p.name != ".git"}


class Adapter:
    def __init__(self, root, config):
        self.path = (Path(root) / config["upstream_path"]).resolve()
        git = ["git", "-c", f"safe.directory={Path(root).as_posix()}",
               "-c", f"safe.directory={self.path.as_posix()}", "-C", str(self.path)]
        self.commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        if self.commit != config["upstream_commit"]:
            raise ValueError(f"upstream commit mismatch: {self.commit}")
        dirty = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty.strip():
            raise ValueError(f"upstream tracked files are modified: {dirty}")
        self.hashes = source_hashes(self.path)
        with upstream_context(self.path):
            spec = importlib.util.spec_from_file_location("_prontoqa_v0_upstream", self.path / "run_experiment.py")
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.fol = sys.modules["fol"]

    def atom(self, formula, term_type, variable=None):
        f = self.fol
        negative = type(formula) is f.FOLNot
        atom = formula.operand if negative else formula
        if type(atom) is not f.FOLFuncApplication or len(atom.args) != 1:
            raise Rejected("unsupported_atomic_form", type(atom).__name__)
        term = atom.args[0]
        if type(term) is not term_type:
            raise Rejected("unsupported_term", type(term).__name__)
        if variable is not None and term.variable != variable:
            raise Rejected("unbound_variable")
        return Literal(atom.function, negative), term

    def convert(self, formulas, target):
        f, m = self.fol, self.module
        rules, facts, entities, cats, props = [], [], set(), set(), set()

        def inventory(literal):
            (cats if m.morphology.is_noun(literal.predicate) else props).add(literal.predicate)

        for formula in formulas:
            if type(formula) is f.FOLForAll:
                body = formula.operand
                if type(body) is not f.FOLIfThen:
                    raise Rejected("unsupported_quantified_form", type(body).__name__)
                antecedent, _ = self.atom(body.antecedent, f.FOLVariable, formula.variable)
                consequent, _ = self.atom(body.consequent, f.FOLVariable, formula.variable)
                if antecedent.negative or not m.morphology.is_noun(antecedent.predicate):
                    raise Rejected("unsupported_antecedent", str(antecedent))
                inventory(antecedent)
                inventory(consequent)
                rules.append(Rule(f"r{len(rules)+1:03d}", antecedent.predicate, consequent))
            else:
                literal, term = self.atom(formula, f.FOLConstant)
                inventory(literal)
                entities.add(term.constant)
                facts.append(Fact(f"f{len(facts)+1:03d}", literal))
        conclusion, term = self.atom(target, f.FOLConstant)
        inventory(conclusion)
        entities.add(term.constant)
        if len(entities) != 1:
            raise Rejected("multiple_entities")
        return World(next(iter(entities)), tuple(sorted(cats)), tuple(sorted(props)),
                     tuple(facts), tuple(rules), Literal(conclusion.predicate)).check()

    def sample(self, hops, seed, config):
        m, f = self.module, self.fol
        captured = io.StringIO()
        with upstream_context(self.path, seed), redirect_stdout(captured):
            concepts = random.sample(config["concepts"], config["source_concepts_per_attempt"])
            m.config.max_child_count = 1
            m.config.generate_negation = True
            m.config.generate_properties = True
            output = m.generate_question(hops + 1, concepts, formula_ordering="random",
                                         ontology="fictional", add_distractor=True,
                                         deduction_rule="ModusPonens", use_dfs=False, proof_width=2)
        question, query, formulas, cot, answer, proof = output
        if formulas is None:
            raise Rejected("upstream_no_exact_proof", captured.getvalue().strip())
        world = self.convert(formulas, proof[-1].conclusion)
        result = require_decidable(world, hops)
        query_formula = m.parse_sentence(query[len("True or false: "):-1], m.morphology, False)
        query_literal, _ = self.atom(query_formula, f.FOLConstant)
        expected = result["answer"] == "True"
        if query_literal.negative:
            expected = not expected
        if str(expected) != answer or query_literal.predicate != world.query.predicate:
            raise Rejected("upstream_label_mismatch")
        # Validate the actual upstream proof AST, not just our regenerated proof.
        for step in proof:
            if step.step_type.name == "AXIOM":
                if step.premises or step.conclusion not in formulas:
                    raise Rejected("invalid_upstream_axiom")
            elif step.step_type.name == "UNIVERSAL_INSTANTIATION":
                if len(step.premises) != 2 or any(p not in proof[:proof.index(step)] for p in step.premises):
                    raise Rejected("invalid_upstream_proof_reference")
                rule, premise = step.premises
                if type(rule.conclusion) is not f.FOLForAll or type(rule.conclusion.operand) is not f.FOLIfThen:
                    raise Rejected("unsupported_upstream_proof_rule")
                body = rule.conclusion.operand
                conclusion_atom = step.conclusion.operand if type(step.conclusion) is f.FOLNot else step.conclusion
                entity_term = conclusion_atom.args[0]
                if (f.substitute(body.antecedent, f.FOLVariable(rule.conclusion.variable), entity_term) != premise.conclusion
                        or f.substitute(body.consequent, f.FOLVariable(rule.conclusion.variable), entity_term) != step.conclusion):
                    raise Rejected("invalid_upstream_instantiation")
            else:
                raise Rejected("unsupported_upstream_proof_step", step.step_type.name)
        if sum(s.step_type.name == "UNIVERSAL_INSTANTIATION" for s in proof) != hops:
            raise Rejected("upstream_proof_hops_mismatch")
        conclusion, _ = self.atom(proof[-1].conclusion, f.FOLConstant)
        if (conclusion.negative and result["answer"] != "False") or (not conclusion.negative and result["answer"] != "True"):
            raise Rejected("upstream_proof_goal_mismatch")
        positions = {id(step): i for i, step in enumerate(proof)}
        provenance = {"candidate_seed": seed, "upstream_requested_steps": hops + 1,
                      "upstream_question": question, "upstream_query": query, "upstream_answer": answer,
                      "formulas_tptp": [f.fol_to_tptp(x) for x in formulas],
                      "proof": [{"type": s.step_type.name, "conclusion_tptp": f.fol_to_tptp(s.conclusion),
                                 "premises": [positions[id(p)] for p in s.premises]} for s in proof],
                      "source_world": world.to_dict(), "source_proof_verified": True,
                      "generator_messages": captured.getvalue().splitlines()}
        return world, provenance
