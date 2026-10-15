from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import numpy as np

from prontoqa_v0.logic import (Fact, Literal, Rejected, Rule, World, solve,
                               require_decidable, shortest_proof, verify_proof)
from prontoqa_v0.render import build_model_input, parse_text
from prontoqa_v0.transform import (core_records, diagnostic_records, fingerprint,
                                   flip, prepare_family)
from prontoqa_v0.validation import validate_dataset, validate_family, validate_record
from prontoqa_v0.dataset import build, dumps, GenerationFailure, load_config
from prontoqa_v0.upstream import Adapter, upstream_context
from prontoqa_v0.sampling import sample_bias_pair
from prontoqa_v0.audit import audit_provenance

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/prontoqa_v0.json")


def known_world(hops=2):
    categories = tuple(CONFIG["concepts"][:hops] + ["tumpus"])
    rules = [Rule(f"r{i+1:03d}", categories[i], Literal(categories[i+1])) for i in range(hops-1)]
    rules += [Rule(f"r{hops:03d}", categories[hops-1], Literal("green")),
              Rule(f"r{hops+1:03d}", "tumpus", Literal("green", True))]
    return World("Fae", tuple(sorted(categories)), ("green",),
                 (Fact("f001", Literal(categories[0])),), tuple(rules), Literal("green"))


def family(base="test", pool="training", hops=2, seed=1):
    source = known_world(hops)
    prepared = prepare_family(source, hops, random.Random(seed), CONFIG)
    return core_records(base, pool, hops, source, prepared)


class LogicTests(unittest.TestCase):
    def test_true_false_unknown_and_contradiction(self):
        world = known_world()
        self.assertEqual(solve(world)["answer"], "True")
        self.assertEqual(solve(flip(world, ["r002", "r003"]))["answer"], "False")
        self.assertEqual(solve(replace(world, facts=()))["answer"], "unknown")
        contradictory = replace(world, facts=world.facts + (Fact("f002", Literal("green", True)),))
        self.assertEqual(solve(contradictory)["answer"], "contradiction")
        with self.assertRaises(Rejected):
            require_decidable(contradictory)

    def test_zero_hops_shortcuts_cycles_and_order(self):
        world = known_world(5)
        self.assertEqual(solve(world)["hops"], 5)
        shorter = replace(world, rules=world.rules + (Rule("shortcut", CONFIG["concepts"][0], Literal("green")),))
        self.assertEqual(solve(shorter)["hops"], 1)
        self.assertEqual(solve(replace(shorter, rules=tuple(reversed(shorter.rules))))["hops"], 1)
        with self.assertRaises(Rejected):
            require_decidable(shorter, 5)
        zero = replace(world, facts=world.facts + (Fact("f002", Literal("green")),))
        self.assertEqual(solve(zero)["hops"], 0)
        cycle = replace(world, rules=world.rules + (Rule("cycle", CONFIG["concepts"][1], Literal(CONFIG["concepts"][0])),))
        self.assertEqual(solve(cycle)["hops"], 5)

    def test_proof_references_and_corruption(self):
        world = known_world(3)
        proof = shortest_proof(world)
        self.assertTrue(verify_proof(world, proof, 3))
        wrong = deepcopy(proof)
        wrong[-1]["rule_id"] = "r004"
        with self.assertRaises(AssertionError):
            verify_proof(world, wrong, 3)


class TransformationTests(unittest.TestCase):
    def test_vowel_category_articles_round_trip(self):
        text = "If someone is an impus, then they are green. Each wumpus is an impus. Nora is a wumpus."
        world = parse_text(text, "True or false: Nora is green.")
        self.assertEqual(solve(world)["hops"], 2)
        from prontoqa_v0.render import complement, rule_text
        self.assertEqual(complement(Literal("impus"), ("impus",)), "an impus")
        self.assertEqual(rule_text(world, world.rules[0], 1), "If someone is an impus, then they are green.")

    def test_names_pair_counts_interventions_round_trip(self):
        for hops in range(1, 6):
            records = family(hops=hops, seed=hops)
            checks = validate_family(records, diagnostic_records(records))
            self.assertEqual(checks["proof"], 12)
            for r in records:
                self.assertEqual(parse_text(r["question"], r["query"]), World.from_dict(r["logic"]))
            false = next(r for r in records if r["variant"] == "A_false")
            self.assertEqual(len(false["transformations"][3]["edited_rules"]), 2)

    def test_corrupt_text_and_diagnostic_are_rejected(self):
        records = family()
        corrupted = deepcopy(records[0])
        corrupted["query"] = corrupted["query"].replace(" is ", " is not ")
        with self.assertRaises(Rejected):
            validate_record(corrupted)
        diag = diagnostic_records(records)
        diag[0]["transformations"][-1]["edited_rules"] = []
        with self.assertRaises(AssertionError):
            validate_family(records, diag)

    def test_duplicate_ignores_names_order_ids_and_query(self):
        world = known_world()
        renamed = replace(world, entity="Nora", rules=tuple(replace(r, id=f"other{i}") for i, r in enumerate(reversed(world.rules))))
        self.assertEqual(fingerprint(world), fingerprint(renamed))
        self.assertNotEqual(fingerprint(world), fingerprint(flip(world, ["r002"])))

    def test_cross_pool_leak_detected(self):
        smoke = family("smoke", "smoke", 1)
        calibration = deepcopy(smoke)
        for r in calibration:
            r["id"] = r["id"].replace("smoke", "calibration")
            r["base_id"], r["pool"] = "calibration", "calibration"
        cfg = deepcopy(CONFIG)
        cfg["hops"], cfg["pools"], cfg["diagnostic_families_per_hops"] = [1], {"smoke": 1, "calibration": 1}, 1
        with self.assertRaisesRegex(AssertionError, "world reused"):
            validate_dataset(smoke + calibration, diagnostic_records(calibration), cfg, ["smoke", "calibration"])

    def test_model_input_excludes_all_metadata(self):
        record = family()[0]
        expected = build_model_input(record["question"], record["query"])
        poisoned = deepcopy(record)
        for key in record:
            if key not in ("question", "query"):
                poisoned[key] = "PRIVATE_EVALUATION_SENTINEL"
        self.assertEqual(expected, build_model_input(poisoned["question"], poisoned["query"]))
        self.assertNotIn("PRIVATE_EVALUATION_SENTINEL", expected)
        with self.assertRaises(TypeError):
            build_model_input(record, record["query"])


class SamplingTests(unittest.TestCase):
    def setUp(self):
        # Independent handcrafted training fixtures, never copied from a real pool.
        self.records = [r for h in range(1, 6) for i in range(20)
                        for r in family(f"training-h{h}-{i}", "training", h, 1000*h+i)]

    def test_exact_ratios_same_families_same_labels_both_directions(self):
        for direction in ("A", "B"):
            pair = sample_bias_pair(self.records, seed=123, true_name_group=direction)
            self.assertEqual([(r["base_id"], r["answer"]) for r in pair[0]], [(r["base_id"], r["answer"]) for r in pair[1]])
            self.assertEqual(len({r["base_id"] for r in pair[0]}), 100)
            for rows, ratio in zip(pair, (0.5, 0.9)):
                strata = defaultdict(list)
                for r in rows:
                    preferred = direction if r["answer"] == "True" else ("B" if direction == "A" else "A")
                    strata[r["hops"], r["answer"]].append(r["name_group"] == preferred)
                for matches in strata.values():
                    self.assertEqual(sum(matches) / len(matches), ratio)
                    self.assertIn(False, matches)
            # For the same family/label, rule statements and query predicate stay identical.
            for a, b in zip(*pair):
                self.assertEqual(a["logic"]["rules"], b["logic"]["rules"])
                self.assertEqual(a["logic"]["query"]["predicate"], b["logic"]["query"]["predicate"])
                self.assertEqual(a["render_plan"], b["render_plan"])

    def test_forbid_evaluation_pool_and_inexact_ratios(self):
        rows = deepcopy(self.records)
        rows[0]["pool"] = "calibration"
        with self.assertRaises(ValueError):
            sample_bias_pair(rows)
        with self.assertRaises(ValueError):
            sample_bias_pair(self.records[:8])


class UpstreamIntegrationTests(unittest.TestCase):
    def test_unsupported_forms_rejected_explicitly(self):
        adapter = Adapter(ROOT, CONFIG)
        f = adapter.fol
        atom = lambda p, term: f.FOLFuncApplication(p, [term])
        x, entity = f.FOLVariable(1), f.FOLConstant("Fae")
        fact, target = atom("wumpus", entity), atom("green", entity)
        for bad in (f.FOLForAll(1, f.FOLIfThen(f.FOLAnd([atom("wumpus", x), atom("yumpus", x)]), atom("green", x))),
                    f.FOLNot(f.FOLExists(1, atom("wumpus", x))),
                    f.FOLForAll(1, f.FOLIfThen(atom("wumpus", x), f.FOLOr([atom("green", x), atom("blue", x)])))):
            with self.assertRaises(Rejected):
                adapter.convert([fact, bad], target)
        with self.assertRaisesRegex(Rejected, "multiple_entities"):
            adapter.convert([fact, atom("wumpus", f.FOLConstant("Rex"))], target)

    def test_cwd_bad_patterns_and_rng_are_restored(self):
        path, state, np_state = Path.cwd(), random.getstate(), np.random.get_state()
        with upstream_context(ROOT / CONFIG["upstream_path"], 42):
            self.assertTrue(Path("bad_patterns.txt").exists())
            random.random()
            np.random.rand()
        self.assertEqual(Path.cwd(), path)
        self.assertEqual(random.getstate(), state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertEqual(np.random.get_state()[2:], np_state[2:])

    def test_tiny_generation_reproduction_and_source_audit(self):
        cfg = deepcopy(CONFIG)
        cfg["pools"] = {"smoke": 2, "calibration": 2, "heldout": 2}
        cfg["diagnostic_families_per_hops"] = 1
        a, b = build(ROOT, cfg), build(ROOT, cfg)
        self.assertEqual(dumps(a), dumps(b))
        self.assertEqual(a[4]["totals"], {"families": 30, "core_records": 120, "diagnostic_records": 40})
        self.assertEqual(audit_provenance(ROOT, cfg, a[0], a[2])["status"], "passed")

    def test_attempt_budget_fails_explicitly(self):
        cfg = deepcopy(CONFIG)
        cfg["max_attempts_per_pool_hops"] = 3
        with patch.object(Adapter, "sample", side_effect=Rejected("forced_unsupported")) as sample:
            with self.assertRaises(GenerationFailure) as caught:
                build(ROOT, cfg, "smoke")
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(caught.exception.detail["attempts"]["smoke/h1"]["rejected"], {"forced_unsupported": 3})


if __name__ == "__main__":
    unittest.main()
