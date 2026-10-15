from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from test_prontoqa_v0 import family, ROOT
from prontoqa_v0.dataset import read_jsonl
from prontoqa_v0.evaluation import annotate_records, evaluation_bundle, inspection_records, validate_round_trip
from prontoqa_v0.logic import Literal, World, require_decidable, verify_proof
from prontoqa_v0.render import INSTRUCTION, build_model_input, build_prompt, canonical_input_json, compute_example_id
from prontoqa_v0.transform import diagnostic_records
from prontoqa_v0.validation import validate_family


class InputIdentityTests(unittest.TestCase):
    def test_exact_canonical_json_sha256(self):
        question, query = "Nora is a jompus.", "True or false: Nora is metallic."
        expected = '{"query":"True or false: Nora is metallic.","question":"Nora is a jompus."}'
        self.assertEqual(canonical_input_json(question, query), expected)
        self.assertEqual(compute_example_id(question, query), hashlib.sha256(expected.encode("utf-8")).hexdigest())

    def test_same_input_different_provenance(self):
        first = family()[0]
        second = deepcopy(first)
        second.update(id="another-source", base_id="another-family", variant="other_origin", provenance_id="other")
        unique, mapping, _, report = evaluation_bundle([first, second])
        self.assertEqual(first["example_id"], second["example_id"])
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(report["families"], 2)

    def test_name_rule_wording_order_and_whitespace_change_id(self):
        record = family()[0]
        question, query = record["question"], record["query"]
        original = compute_example_id(question, query)
        altered = [
            (question.replace(record["name"], "Othername"), query.replace(record["name"], "Othername")),
            ("A new rule. " + question, query),
            (question + " ", query),
            (" ".join(reversed(question[:-1].split(". "))) + ".", query),
            (question, query + " "),
        ]
        for q, target in altered:
            self.assertNotEqual(compute_example_id(q, target), original)
        self.assertNotEqual(compute_example_id("Every wumpus is green.", query),
                            compute_example_id("Each wumpus is green.", query))
        self.assertNotEqual(compute_example_id("é", query), compute_example_id("e\u0301", query))

    def test_metadata_whitelist_sentinel_and_original_format(self):
        record = family()[0]
        expected = f"{INSTRUCTION}\n\n{record['question']}\n{record['query']}"
        self.assertEqual(build_prompt(record), expected)
        self.assertEqual(build_model_input(record["question"], record["query"]), expected)
        poison = {key: "BACKEND_TEST_SENTINEL" for key in (
            "base_id", "example_id", "id", "variant", "name_group", "hops", "answer", "proof",
            "comparison", "transformations", "key_rule_id", "irrelevant_rule_id", "pool", "name", "logic")}
        poisoned = {**record, **poison}
        self.assertEqual(build_prompt(poisoned), expected)
        self.assertNotIn("BACKEND_TEST_SENTINEL", build_prompt(poisoned))
        self.assertEqual(compute_example_id(poisoned["question"], poisoned["query"]), record["example_id"])
        self.assertIn(record["name"], build_prompt(record))
        for backend_key in ("base_id", "example_id", "variant", "name_group", "hops", "r001", "f001"):
            self.assertNotIn(backend_key, expected)

    def test_conflicting_gold_answers_fail(self):
        first = family()[0]
        second = {**first, "id": "another-source", "answer": "False"}
        with self.assertRaisesRegex(ValueError, "conflicting gold"):
            evaluation_bundle([first, second])

    def test_reproducible_across_process_hash_seeds(self):
        code = "from prontoqa_v0.render import compute_example_id; print(compute_example_id('Nora is a jompus.','True or false: Nora is metallic.'))"
        values = [subprocess.check_output([sys.executable, "-c", code], cwd=ROOT,
                                         env={**os.environ, "PYTHONHASHSEED": value}, text=True).strip()
                  for value in ("1", "98989")]
        self.assertEqual(values[0], values[1])
        self.assertEqual(len(values[0]), 64)


class ComparisonExportTests(unittest.TestCase):
    def test_all_records_and_comparisons_restore(self):
        cores = family()
        records = cores + diagnostic_records(cores)
        unique, mapping, comparisons, report = evaluation_bundle(records)
        self.assertEqual((len(records), len(unique), len(mapping)), (12, 8, 12))
        self.assertEqual(report["comparisons"], {"name": 2, "key_flip": 4, "irrelevant_flip": 4, "core_truth_swap": 2})
        self.assertTrue(validate_round_trip(records, unique, mapping, comparisons))
        broken = deepcopy(mapping)
        broken[0]["comparisons"] = []
        with self.assertRaises(AssertionError):
            validate_round_trip(records, unique, broken, comparisons)
        with self.assertRaises(AssertionError):
            validate_round_trip(records, unique, mapping[:-1], comparisons)
        with self.assertRaises(AssertionError):
            validate_round_trip(records, unique, mapping, comparisons[:-1])

    def test_counts_follow_actual_records(self):
        cores = family()
        for records, expected in ((cores[:1], (1, 1)), (cores, (4, 4)),
                                  (cores + diagnostic_records(cores[:1]), (6, 6))):
            unique, mapping, _, report = evaluation_bundle(records)
            self.assertEqual((report["original_records"], report["unique_inputs"]), expected)
            self.assertEqual(len(mapping), len(records))

    def test_core_truth_swap_is_two_edits_not_key_only(self):
        cores = family()
        rows = cores + diagnostic_records(cores)
        _, _, comparisons, _ = evaluation_bundle(rows)
        for comparison in comparisons:
            if comparison["modification_type"] == "core_truth_swap":
                self.assertEqual(len(comparison["edited_rules"]), 2)
            elif comparison["modification_type"] in ("key_flip", "irrelevant_flip"):
                self.assertEqual(len(comparison["edited_rules"]), 1)
        forged = deepcopy(rows)
        forged[-1]["comparison"]["modification_type"] = "key_flip"
        with self.assertRaises(AssertionError):
            evaluation_bundle(forged)

    def test_metadata_upgrade_preserves_original_payloads(self):
        cores = family()
        diags = diagnostic_records(cores)
        old_cores = [{k: v for k, v in r.items() if k != "example_id"} for r in cores]
        old_diags = [{k: v for k, v in r.items() if k not in ("example_id", "comparison")} for r in diags]
        new_cores, new_diags = annotate_records(old_cores, old_diags)
        self.assertEqual(new_cores, cores)
        self.assertEqual(new_diags, diags)
        self.assertNotIn("example_id", old_cores[0])
        self.assertNotIn("comparison", old_diags[0])

    def test_sampling_selection_and_frequency_unchanged(self):
        from prontoqa_v0.sampling import sample_bias_pair
        records = [r for i in range(20) for r in family(f"train-{i}", "training", 2, i)]
        upgraded, _ = annotate_records(records, [])
        before, after = sample_bias_pair(records, 91), sample_bias_pair(upgraded, 91)
        self.assertEqual([[r["id"] for r in rows] for rows in before],
                         [[r["id"] for r in rows] for rows in after])

    def test_real_smoke_case_twelve_records_eight_inputs(self):
        cores, _ = annotate_records(read_jsonl(ROOT / "tests/fixtures/v0-smoke-h5-0008.jsonl"), [])
        case = inspection_records(cores)
        self.assertEqual(len(case), 12)
        validate_family(case[:4], case[4:])
        unique, mapping, _, report = evaluation_bundle(case)
        self.assertEqual(len(unique), 8)
        self.assertEqual(report["families"], 1)
        by_variant = {r["variant"]: r for r in case}
        for group in ("A", "B"):
            self.assertEqual(by_variant[group+"_true__key_flip"]["example_id"],
                             by_variant[group+"_false__irrelevant_flip"]["example_id"])
            self.assertEqual(by_variant[group+"_true__irrelevant_flip"]["example_id"],
                             by_variant[group+"_false__key_flip"]["example_id"])
        self.assertTrue({r["example_id"] for r in case if r["name_group"] == "A"}.isdisjoint(
                        {r["example_id"] for r in case if r["name_group"] == "B"}))
        for r in case:
            world = World.from_dict(r["logic"])
            result = require_decidable(world, 5)
            self.assertEqual(result["answer"], r["answer"])
            antecedent = next(rule.antecedent for rule in world.rules if rule.id == "r006")
            self.assertEqual(antecedent, "timpus")
            self.assertNotIn(Literal(antecedent), result["distances"])
            self.assertTrue(verify_proof(world, r["proof"], 5))
        self.assertEqual(len(mapping), 12)


if __name__ == "__main__":
    unittest.main()
