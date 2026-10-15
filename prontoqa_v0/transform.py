"""IR-only name/predicate permutations and causal polarity interventions."""
from dataclasses import replace
import hashlib
import json
from .logic import Literal, Fact, Rule, World, Rejected, require_decidable, shortest_proof, solve
from .render import compute_example_id, render


def canonical_world(world):
    """World fingerprint ignores entity, reference IDs, sentence/rule/fact order.

    It deliberately excludes query too, so querying a shared world differently
    cannot defeat cross-pool leakage checks. Predicate spellings are retained.
    """
    return {"facts": sorted(set((f.literal.predicate, f.literal.negative) for f in world.facts)),
            "rules": sorted(set((r.antecedent, r.consequent.predicate, r.consequent.negative) for r in world.rules))}


def fingerprint(world):
    return hashlib.sha256(json.dumps(canonical_world(world), sort_keys=True).encode()).hexdigest()


def edits(before, after):
    old = {r.id: r for r in before.rules}
    return [{"rule_id": r.id, "before": old[r.id].to_dict(), "after": r.to_dict()}
            for r in after.rules if old[r.id] != r]


def flip(world, rule_ids):
    ids = set(rule_ids)
    assert ids <= {r.id for r in world.rules}
    return replace(world, rules=tuple(replace(r, consequent=r.consequent.opposite()) if r.id in ids else r
                                      for r in world.rules))


def identify_pair(world, hops):
    result = require_decidable(world, hops)
    target_rules = [r for r in world.rules if r.consequent.predicate == world.query.predicate]
    reachable = [r for r in target_rules if Literal(r.antecedent) in result["distances"]]
    unreachable = [r for r in target_rules if Literal(r.antecedent) not in result["distances"]]
    if len(target_rules) != 2 or len(reachable) != 1 or len(unreachable) != 1:
        raise Rejected("non_unique_target_pair")
    key, irrelevant = reachable[0], unreachable[0]
    if key.consequent.negative == irrelevant.consequent.negative:
        raise Rejected("target_pair_not_opposite")
    without_key = replace(world, rules=tuple(r for r in world.rules if r.id != key.id))
    if solve(without_key)["answer"] != "unknown":
        raise Rejected("key_not_necessary")
    return key.id, irrelevant.id


def prepare_family(source, hops, rng, config):
    key, irrelevant = identify_pair(source, hops)
    category_mapping = dict(zip(source.categories, rng.sample(config["concepts"], len(source.categories))))
    property_mapping = dict(zip(source.properties, rng.sample(config["properties"], len(source.properties))))
    mapping = category_mapping | property_mapping
    lit = lambda x: Literal(mapping[x.predicate], x.negative)
    rules = [Rule(r.id, mapping[r.antecedent], lit(r.consequent)) for r in source.rules]
    rng.shuffle(rules)
    refs = {r.id: f"r{i+1:03d}" for i, r in enumerate(rules)}
    world = replace(source, categories=tuple(sorted(category_mapping.values())),
                    properties=tuple(sorted(property_mapping.values())),
                    facts=tuple(Fact(f.id, lit(f.literal)) for f in source.facts),
                    rules=tuple(replace(r, id=refs[r.id]) for r in rules), query=lit(source.query))
    key, irrelevant = refs[key], refs[irrelevant]
    normalized = world if require_decidable(world)["answer"] == "True" else flip(world, [key, irrelevant])
    identify_pair(normalized, hops)
    false_world = flip(normalized, [key, irrelevant])
    assert require_decidable(false_world, hops)["answer"] == "False"
    plan = {"styles": {r.id: rng.choice(config["render_styles"]) for r in normalized.rules},
            "order": [r.id for r in normalized.rules]}
    # Insert facts while preserving the shuffled rule order (also used for IDs).
    for fact in normalized.facts:
        plan["order"].insert(rng.randrange(len(plan["order"])+1), fact.id)
    names = {group: rng.choice(config["name_groups"][group]) for group in ("A", "B")}
    metadata = {"predicate_mapping": mapping, "rule_reference_mapping": refs,
                "source_to_canonical_true_edits": edits(world, normalized),
                "key_rule_id": key, "irrelevant_rule_id": irrelevant}
    return normalized, false_world, names, plan, metadata


def make_record(base_id, pool, hops, variant, group, world, plan, metadata, transforms, kind="core", parent_id=None):
    result = require_decidable(world, hops)
    question, query, proof = render(world, plan, shortest_proof(world))
    return {"id": f"{base_id}:{variant}", "base_id": base_id, "pool": pool, "hops": hops,
            "example_id": compute_example_id(question, query),
            "kind": kind, "variant": variant, "parent_id": parent_id,
            "name": world.entity, "name_group": group, "question": question, "query": query,
            "answer": result["answer"], "logic": world.to_dict(), "proof": proof,
            "key_rule_id": metadata["key_rule_id"], "irrelevant_rule_id": metadata["irrelevant_rule_id"],
            "render_plan": plan, "transformations": transforms, "world_hash": fingerprint(world),
            "provenance_id": base_id}


def core_records(base_id, pool, hops, source, prepared):
    true_world, false_world, names, plan, metadata = prepared
    records = []
    for answer, world in (("True", true_world), ("False", false_world)):
        for group in ("A", "B"):
            renamed = replace(world, entity=names[group])
            transforms = [{"type": "predicate_bijection", "mapping": metadata["predicate_mapping"]},
                          {"type": "rule_permutation", "mapping": metadata["rule_reference_mapping"]},
                          {"type": "source_to_canonical_true", "edited_rules": metadata["source_to_canonical_true_edits"]},
                          {"type": "core_truth_swap", "reference": "canonical_true",
                           "edited_rules": edits(true_world, world)},
                          {"type": "entity_rename", "before": source.entity, "after": names[group]}]
            records.append(make_record(base_id, pool, hops, f"{group}_{answer.lower()}", group,
                                       renamed, plan, metadata, transforms))
    return records


def comparison_link(source, result, modification_type, edited_rules, expected_answer_relation):
    return {"comparison_id": f"{source['id']}->{result['id']}:{modification_type}",
            "base_id": source["base_id"], "pool": source["pool"], "hops": source["hops"],
            "source_record_id": source["id"], "source_example_id": source["example_id"],
            "result_record_id": result["id"], "result_example_id": result["example_id"],
            "modification_type": modification_type, "edited_rules": edited_rules,
            "expected_answer_relation": expected_answer_relation}


def diagnostic_comparison(core, record):
    before, after = World.from_dict(core["logic"]), World.from_dict(record["logic"])
    modification_type = record["transformations"][-1]["type"]
    if modification_type not in ("key_flip", "irrelevant_flip"):
        raise ValueError("unsupported diagnostic modification")
    return comparison_link(core, record, modification_type, edits(before, after),
                           "flip" if modification_type == "key_flip" else "same")


def diagnostic_records(cores):
    records = []
    from .logic import World
    for core in cores:
        world = World.from_dict(core["logic"])
        for intervention, key in (("key_flip", "key_rule_id"), ("irrelevant_flip", "irrelevant_rule_id")):
            modified = flip(world, [core[key]])
            result = require_decidable(modified, core["hops"])
            assert (result["answer"] != core["answer"]) == (intervention == "key_flip")
            transform = {"type": intervention, "parent_id": core["id"], "edited_rules": edits(world, modified)}
            record = make_record(core["base_id"], core["pool"], core["hops"],
                                       core["variant"] + "__" + intervention, core["name_group"],
                                       modified, core["render_plan"], core,
                                       core["transformations"] + [transform], "diagnostic", core["id"])
            record["comparison"] = diagnostic_comparison(core, record)
            records.append(record)
    return records
