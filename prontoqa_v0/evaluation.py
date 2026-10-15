"""Exact input IDs and lossless record/comparison links for evaluation only."""
from collections import Counter, defaultdict
from dataclasses import replace
from .logic import World
from .render import build_prompt, compute_example_id
from .transform import comparison_link, diagnostic_comparison, diagnostic_records, edits, flip

INSPECTION_BASE_ID = "v0-smoke-h5-0008"


def annotate_records(cores, diagnostics):
    """Upgrade metadata without changing or dropping any original record payload."""
    cores = [{**r, "example_id": compute_example_id(r["question"], r["query"])} for r in cores]
    index = {r["id"]: r for r in cores}
    upgraded = []
    for old in diagnostics:
        record = {**old, "example_id": compute_example_id(old["question"], old["query"])}
        record["comparison"] = diagnostic_comparison(index[record["parent_id"]], record)
        upgraded.append(record)
    return cores, upgraded


def comparisons_for(records):
    cores, comparisons = defaultdict(dict), []
    for record in records:
        if record["kind"] == "core":
            variants = cores[record["base_id"]]
            if record["variant"] in variants:
                raise ValueError("duplicate core variant")
            variants[record["variant"]] = record
    for _, variants in sorted(cores.items()):
        # Count only the actual pairs present; do not assume 12 records / 8 inputs.
        for label in ("true", "false"):
            if f"A_{label}" in variants and f"B_{label}" in variants:
                comparisons.append(comparison_link(variants[f"A_{label}"], variants[f"B_{label}"], "name", [], "same"))
        for group in ("A", "B"):
            if f"{group}_true" in variants and f"{group}_false" in variants:
                source, result = variants[f"{group}_true"], variants[f"{group}_false"]
                comparisons.append(comparison_link(source, result, "core_truth_swap",
                                   edits(World.from_dict(source["logic"]), World.from_dict(result["logic"])), "flip"))
    comparisons.extend(r["comparison"] for r in records if r["kind"] == "diagnostic")
    return sorted(comparisons, key=lambda c: c["comparison_id"])


def validate_comparison(comparison, index):
    source, result = index[comparison["source_record_id"]], index[comparison["result_record_id"]]
    assert source["base_id"] == result["base_id"] == comparison["base_id"]
    assert source["pool"] == result["pool"] == comparison["pool"]
    assert source["hops"] == result["hops"] == comparison["hops"]
    assert source["example_id"] == comparison["source_example_id"]
    assert result["example_id"] == comparison["result_example_id"]
    before, after = World.from_dict(source["logic"]), World.from_dict(result["logic"])
    delta = edits(before, after)
    assert comparison["edited_rules"] == delta
    kind = comparison["modification_type"]
    if kind == "name":
        assert source["kind"] == result["kind"] == "core"
        assert source["name_group"] == "A" and result["name_group"] == "B"
        assert before.entity != after.entity and after == replace(before, entity=after.entity)
        expected = "same"
    elif kind == "core_truth_swap":
        assert source["kind"] == result["kind"] == "core"
        ids = [source["key_rule_id"], source["irrelevant_rule_id"]]
        assert len(delta) == 2 and {d["rule_id"] for d in delta} == set(ids)
        assert after == flip(before, ids)
        expected = "flip"
    else:
        assert kind in ("key_flip", "irrelevant_flip")
        assert source["kind"] == "core" and result["kind"] == "diagnostic"
        assert result["parent_id"] == source["id"]
        rule_id = source["key_rule_id" if kind == "key_flip" else "irrelevant_rule_id"]
        assert len(delta) == 1 and delta[0]["rule_id"] == rule_id
        assert after == flip(before, [rule_id])
        assert result["comparison"] == diagnostic_comparison(source, result) == comparison
        transform = result["transformations"][-1]
        assert transform["type"] == kind and transform["parent_id"] == source["id"]
        assert transform["edited_rules"] == delta
        expected = "flip" if kind == "key_flip" else "same"
    assert source["render_plan"] == result["render_plan"]
    assert comparison["expected_answer_relation"] == expected
    assert (source["answer"] == result["answer"]) == (expected == "same")


def evaluation_bundle(records):
    """Return unique {example_id,prompt}, one mapping per record, comparisons, report.

    Original records remain authoritative for labels/proofs/frequencies. No labels
    or other backend fields enter the unique-input catalog's prompt strings.
    """
    index, unique, text_by_id, labels, family_pools = {}, {}, {}, {}, defaultdict(set)
    for record in records:
        if record["id"] in index:
            raise ValueError("duplicate record ID")
        index[record["id"]] = record
        example_id = compute_example_id(record["question"], record["query"])
        if record["example_id"] != example_id:
            raise ValueError("example_id does not match canonical question/query SHA-256")
        text = (record["question"], record["query"])
        if example_id in text_by_id and text_by_id[example_id] != text:
            raise ValueError("SHA-256 collision between different inputs")
        if example_id in labels and labels[example_id] != record["answer"]:
            raise ValueError("same example_id has conflicting gold answers")
        if record["answer"] not in ("True", "False"):
            raise ValueError("invalid gold answer")
        text_by_id[example_id], labels[example_id] = text, record["answer"]
        unique[example_id] = {"example_id": example_id, "prompt": build_prompt(record)}
        family_pools[record["base_id"]].add(record["pool"])
    if any(len(pools) != 1 for pools in family_pools.values()):
        raise ValueError("base_id leaked across pools")
    comparisons = comparisons_for(records)
    assert len({c["comparison_id"] for c in comparisons}) == len(comparisons)
    memberships = defaultdict(list)
    for comparison in comparisons:
        validate_comparison(comparison, index)
        for role in ("source", "result"):
            memberships[comparison[role + "_record_id"]].append(
                {"comparison_id": comparison["comparison_id"], "role": role})
    mapping = [{"record_id": r["id"], "base_id": r["base_id"], "example_id": r["example_id"],
                "comparisons": memberships[r["id"]]} for r in sorted(records, key=lambda r: r["id"])]
    validate_round_trip(records, list(unique.values()), mapping, comparisons)
    counts = Counter(c["modification_type"] for c in comparisons)
    pool_stats = {}
    for pool in sorted({r["pool"] for r in records}):
        rows = [r for r in records if r["pool"] == pool]
        pool_stats[pool] = {"original_records": len(rows), "unique_inputs": len({r["example_id"] for r in rows}),
                            "families": len({r["base_id"] for r in rows})}
    report = {"status": "passed", "original_records": len(records), "unique_inputs": len(unique),
              "families": len(family_pools), "deduplicated_requests": len(records) - len(unique),
              "comparisons": {key: counts[key] for key in ("name", "key_flip", "irrelevant_flip", "core_truth_swap")},
              "by_pool": pool_stats,
              "checks": {key: "passed" for key in ("canonical_sha256_ids", "one_gold_answer_per_input",
                  "all_original_records_mapped", "all_comparisons_restorable", "exact_rule_edits",
                  "expected_answer_relations", "prompt_whitelist", "base_id_pool_isolation")}}
    return sorted(unique.values(), key=lambda row: row["example_id"]), mapping, comparisons, report


def validate_round_trip(records, unique, mapping, comparisons):
    """Join exported IDs back to every original row and every comparison endpoint."""
    index = {r["id"]: r for r in records}
    catalog = {r["example_id"]: r for r in unique}
    assert len(index) == len(records) == len(mapping)
    assert len(catalog) == len(unique)
    assert len({r["record_id"] for r in mapping}) == len(mapping)
    assert {r["record_id"] for r in mapping} == set(index)
    assert set(catalog) == {r["example_id"] for r in records}
    relationships = {c["comparison_id"]: c for c in comparisons}
    assert len(relationships) == len(comparisons)
    assert comparisons == comparisons_for(records), "exported comparisons were lost or changed"
    expected_memberships = defaultdict(list)
    for comparison in comparisons:
        for role in ("source", "result"):
            expected_memberships[comparison[role + "_record_id"]].append(
                {"comparison_id": comparison["comparison_id"], "role": role})
    for row in mapping:
        original = index[row["record_id"]]
        assert row["base_id"] == original["base_id"]
        assert row["example_id"] == original["example_id"]
        assert row["comparisons"] == expected_memberships[row["record_id"]]
        exported = catalog[row["example_id"]]
        assert set(exported) == {"example_id", "prompt"}
        assert exported["prompt"] == build_prompt(original)
        assert original["example_id"] == compute_example_id(original["question"], original["query"])
    for comparison in comparisons:
        validate_comparison(comparison, index)
    return True


def inspection_records(cores):
    """A separate smoke inspection case; it does not change main-pool weights."""
    family = [r for r in cores if r["base_id"] == INSPECTION_BASE_ID]
    return family + diagnostic_records(family) if family else []
