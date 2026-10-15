"""Independent record/family/leakage checks and machine-readable distributions."""
from collections import Counter, defaultdict
from dataclasses import replace
import re
from .logic import Literal, World, require_decidable, solve, verify_proof
from .render import build_model_input, build_prompt, compute_example_id, fact_text, parse_text, render, INSTRUCTION
from .evaluation import evaluation_bundle
from .transform import canonical_world, diagnostic_records, edits, fingerprint, flip, identify_pair


def validate_record(record):
    world = World.from_dict(record["logic"])
    result = require_decidable(world, record["hops"])
    assert record["answer"] == result["answer"]
    assert record["name"] == world.entity
    assert record["world_hash"] == fingerprint(world)
    parsed = parse_text(record["question"], record["query"])
    assert parsed == world, "text/IR round-trip mismatch"
    assert record["example_id"] == compute_example_id(record["question"], record["query"])
    assert set(record["render_plan"]["order"]) == {r.id for r in world.rules} | {f.id for f in world.facts}
    assert len(record["render_plan"]["order"]) == len(world.rules) + len(world.facts)
    q, query, proof = render(world, record["render_plan"], record["proof"])
    assert q == record["question"] and query == record["query"] and proof == record["proof"]
    verify_proof(world, record["proof"], record["hops"])
    for step in record["proof"]:
        lit = Literal(step["conclusion"]["predicate"], step["conclusion"]["negative"])
        assert step["text"] == fact_text(world, lit)
    assert record["query"] == f"True or false: {world.entity} is {world.query.predicate}."
    key, irrelevant = record["key_rule_id"], record["irrelevant_rule_id"]
    rules = {r.id: r for r in world.rules}
    assert Literal(rules[irrelevant].antecedent) not in result["distances"]
    assert Literal(rules[key].antecedent) in result["distances"]
    assert rules[key].consequent.predicate == world.query.predicate == rules[irrelevant].consequent.predicate
    assert build_model_input(record["question"], record["query"]) == f"{INSTRUCTION}\n\n{q}\n{query}"
    assert build_prompt(record) == build_model_input(q, query)
    return world


def validate_family(cores, diagnostics=()):
    assert len(cores) == 4
    variants = {r["variant"]: r for r in cores}
    assert set(variants) == {"A_true", "B_true", "A_false", "B_false"}
    base = variants["A_true"]
    worlds = {k: validate_record(r) for k, r in variants.items()}
    assert len({r["base_id"] for r in cores}) == len({r["pool"] for r in cores}) == len({r["hops"] for r in cores}) == 1
    true = worlds["A_true"]
    key, irrelevant = identify_pair(true, base["hops"])
    assert (key, irrelevant) == (base["key_rule_id"], base["irrelevant_rule_id"])
    false = worlds["A_false"]
    delta = edits(true, false)
    assert len(delta) == 2 and {e["rule_id"] for e in delta} == {key, irrelevant}
    assert false == flip(true, [key, irrelevant])
    assert canonical_world(worlds["B_true"]) == canonical_world(true)
    assert canonical_world(worlds["B_false"]) == canonical_world(false)
    assert worlds["B_true"] == replace(true, entity=worlds["B_true"].entity)
    assert worlds["B_false"] == replace(false, entity=worlds["B_false"].entity)
    assert worlds["A_false"].entity == true.entity and worlds["B_false"].entity == worlds["B_true"].entity
    assert len({r["name"] for r in cores}) == 2
    assert len({len(r["logic"]["rules"]) for r in cores}) == 1
    assert len({len(re.findall(r"\bnot\b", r["question"] + " " + r["query"])) for r in cores}) == 1
    for variant, record in variants.items():
        assert record["name_group"] == variant[0]
        assert record["answer"] == ("True" if variant.endswith("true") else "False")
        assert record["render_plan"] == base["render_plan"]
        swap = [t for t in record["transformations"] if t["type"] == "core_truth_swap"]
        assert len(swap) == 1 and swap[0]["edited_rules"] == ([] if record["answer"] == "True" else delta)
        # Removing the key eliminates entailment; each single-rule intervention
        # is solved for every family, including families not exported diagnostically.
        w = worlds[variant]
        assert solve(replace(w, rules=tuple(r for r in w.rules if r.id != key)))["answer"] == "unknown"
        assert require_decidable(flip(w, [key]), base["hops"])["answer"] != record["answer"]
        assert require_decidable(flip(w, [irrelevant]), base["hops"])["answer"] == record["answer"]
    if diagnostics:
        assert len(diagnostics) == 8
        expected = {r["id"]: r for r in diagnostic_records(cores)}
        assert {r["id"] for r in diagnostics} == set(expected)
        for record in diagnostics:
            validate_record(record)
            assert record == expected[record["id"]]
            assert len(record["transformations"][-1]["edited_rules"]) == 1
    return {"name_invariance": 4, "core_two_rule_swap": 2,
            "key_only_flip": 4, "irrelevant_only_flip": 4,
            "shortest_hops": 4, "round_trip": 4 + len(diagnostics),
            "proof": 4 + len(diagnostics), "core_equal_rule_and_negation_count": 1}


def distribution(records):
    def histogram(values):
        c = Counter(values)
        return {str(k): c[k] for k in sorted(c)}
    def numeric(values):
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values),
                "histogram": histogram(values)}
    return {"families": len({r["base_id"] for r in records}), "records": len(records),
            "labels": histogram([r["answer"] for r in records]),
            "name_groups": histogram([r["name_group"] for r in records]),
            "names": histogram([r["name"] for r in records]),
            "label_x_name_group": histogram([r["answer"] + "/" + r["name_group"] for r in records]),
            "rule_count": numeric([len(r["logic"]["rules"]) for r in records]),
            "text_characters": numeric([len(r["question"] + " " + r["query"]) for r in records]),
            "text_words": numeric([len((r["question"] + " " + r["query"]).split()) for r in records]),
            "negation_words": numeric([len(re.findall(r"\bnot\b", r["question"] + " " + r["query"])) for r in records]),
            "key_rule_position": histogram([next(i+1 for i, rule in enumerate(r["logic"]["rules"]) if rule["id"] == r["key_rule_id"]) for r in records]),
            "key_antecedent": histogram([next(rule["antecedent"] for rule in r["logic"]["rules"] if rule["id"] == r["key_rule_id"]) for r in records]),
            "render_styles": histogram([s for r in records for s in r["render_plan"]["styles"].values()])}


def validate_dataset(cores, diagnostics, config, expected_pools):
    assert len({r["id"] for r in cores + diagnostics}) == len(cores) + len(diagnostics)
    groups, diags = defaultdict(list), defaultdict(list)
    for r in cores:
        assert r["kind"] == "core" and r["pool"] in expected_pools
        assert r["name"] in config["name_groups"][r["name_group"]]
        groups[r["base_id"]].append(r)
    for r in diagnostics:
        assert r["kind"] == "diagnostic" and r["pool"] == "calibration"
        diags[r["base_id"]].append(r)
    assert set(diags) <= set(groups)
    checks = Counter()
    for base_id, records in groups.items():
        checks.update(validate_family(records, diags[base_id]))
    owners, repetitions = {}, 0
    for record in cores + diagnostics:
        signature = fingerprint(World.from_dict(record["logic"]))
        owner = (record["base_id"], record["pool"])
        if signature in owners:
            assert owners[signature] == owner, "world reused across families/pools"
            repetitions += 1
        else:
            owners[signature] = owner
    stats = {}
    for pool in expected_pools:
        stats[pool] = {}
        for hops in config["hops"]:
            records = [r for r in cores if r["pool"] == pool and r["hops"] == hops]
            assert len({r["base_id"] for r in records}) == config["pools"][pool]
            assert len(records) == config["pools"][pool] * 4
            stats[pool][str(hops)] = distribution(records)
    diagnostic_stats = {}
    if "calibration" in expected_pools:
        for hops in config["hops"]:
            records = [r for r in diagnostics if r["hops"] == hops]
            assert len({r["base_id"] for r in records}) == config["diagnostic_families_per_hops"]
            assert len(records) == config["diagnostic_families_per_hops"] * 8
            diagnostic_stats[str(hops)] = distribution(records)
    else:
        assert not diagnostics
    return {"status": "passed", "core": stats, "diagnostics": diagnostic_stats,
            "evaluation": evaluation_bundle(cores + diagnostics)[3],
            "checks": dict(checks), "deduplication": {"cross_family_world_collisions": 0,
            "cross_pool_world_collisions": 0, "unique_worlds": len(owners),
            "expected_within_family_repeated_world_records": repetitions,
            "canonicalization": "entity-free, unordered facts/rules, no query, predicate spellings retained"},
            "totals": {"families": len(groups), "core_records": len(cores), "diagnostic_records": len(diagnostics)}}
