"""Replay stored upstream proofs and exact IR transformations after export."""
from collections import defaultdict
from dataclasses import replace
from .logic import Fact, Literal, Rule, World, require_decidable, verify_proof
from .transform import edits, flip, identify_pair
from .upstream import Adapter, source_hashes


def audit_provenance(root, config, cores, provenance):
    adapter = Adapter(root, config)
    f, m = adapter.fol, adapter.module
    families = defaultdict(list)
    for r in cores:
        families[r["base_id"]].append(r)
    assert len(provenance) == len(families)
    assert {p["base_id"] for p in provenance} == set(families)
    for origin in provenance:
        source = World.from_dict(origin["source_world"])
        verify_proof(source, origin["source_shortest_proof"], origin["hops"])
        # A trailing sentinel handles upstream's parser lookahead at end-of-input.
        parse = lambda text: f.do_parse_fol_from_tptp(text + " ")
        formulas = [parse(text) for text in origin["formulas_tptp"]]
        statements = [parse(step["conclusion_tptp"]) for step in origin["proof"]]
        assert adapter.convert(formulas, statements[-1]) == source
        rule_steps = 0
        for i, step in enumerate(origin["proof"]):
            refs = step["premises"]
            assert all(0 <= j < i for j in refs)
            if step["type"] == "AXIOM":
                assert not refs and statements[i] in formulas
            else:
                assert step["type"] == "UNIVERSAL_INSTANTIATION" and len(refs) == 2
                rule, premise = [statements[j] for j in refs]
                assert type(rule) is f.FOLForAll and type(rule.operand) is f.FOLIfThen
                literal, term = adapter.atom(statements[i], f.FOLConstant)
                assert term.constant == source.entity
                assert f.substitute(rule.operand.antecedent, f.FOLVariable(rule.variable), term) == premise
                assert f.substitute(rule.operand.consequent, f.FOLVariable(rule.variable), term) == statements[i]
                rule_steps += 1
        assert rule_steps == origin["hops"]
        proof_goal, _ = adapter.atom(statements[-1], f.FOLConstant)
        result = require_decidable(source, origin["hops"])
        assert proof_goal == (source.query if result["answer"] == "True" else source.query.opposite())
        original_query = m.parse_sentence(origin["upstream_query"][len("True or false: "):-1], m.morphology, False)
        query_literal, _ = adapter.atom(original_query, f.FOLConstant)
        assert query_literal.predicate == source.query.predicate
        assert str((result["answer"] == "True") != query_literal.negative) == origin["upstream_answer"]
        text_formulas = [m.parse_sentence(text, m.morphology, False)
                         for text in origin["upstream_question"][:-1].split(". ")]
        assert text_formulas == formulas
        metadata = origin["adaptation"]
        mapping, refs = metadata["predicate_mapping"], metadata["rule_reference_mapping"]
        assert set(mapping) == set(source.categories) | set(source.properties)
        assert len(set(mapping.values())) == len(mapping)
        assert set(refs) == {r.id for r in source.rules} and len(set(refs.values())) == len(refs)
        lit = lambda x: Literal(mapping[x.predicate], x.negative)
        renamed = replace(source, categories=tuple(sorted(mapping[c] for c in source.categories)),
                          properties=tuple(sorted(mapping[p] for p in source.properties)),
                          facts=tuple(Fact(fact.id, lit(fact.literal)) for fact in source.facts),
                          rules=tuple(sorted((Rule(refs[r.id], mapping[r.antecedent], lit(r.consequent)) for r in source.rules), key=lambda r: r.id)),
                          query=lit(source.query))
        key, irrelevant = identify_pair(renamed, origin["hops"])
        assert (key, irrelevant) == (metadata["key_rule_id"], metadata["irrelevant_rule_id"])
        normalized = renamed if result["answer"] == "True" else flip(renamed, [key, irrelevant])
        assert metadata["source_to_canonical_true_edits"] == edits(renamed, normalized)
        for record in families[origin["base_id"]]:
            assert record["pool"] == origin["pool"] and record["hops"] == origin["hops"]
            expected = normalized if record["answer"] == "True" else flip(normalized, [key, irrelevant])
            assert World.from_dict(record["logic"]) == replace(expected, entity=record["name"])
            transforms = record["transformations"]
            assert transforms[0] == {"type": "predicate_bijection", "mapping": mapping}
            assert transforms[1] == {"type": "rule_permutation", "mapping": refs}
            assert transforms[2] == {"type": "source_to_canonical_true", "edited_rules": edits(renamed, normalized)}
            assert transforms[-1] == {"type": "entity_rename", "before": source.entity, "after": record["name"]}
    assert source_hashes(adapter.path) == adapter.hashes
    return {"status": "passed", "source_worlds": len(provenance),
            "upstream_ast_proofs": len(provenance), "exact_transform_replays": len(cores),
            "upstream_commit": adapter.commit, "upstream_unchanged": True}
