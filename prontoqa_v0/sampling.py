"""Paired bias sampling for a separately generated training pool only."""
from collections import defaultdict
from fractions import Fraction
import random


def sample_bias_pair(records, seed=0, correlations=(0.5, 0.9), true_name_group="A"):
    """Return two lists with identical families/labels, changing only name group.

    Exact proportions are required in EVERY (hops, label) stratum. Incompatible
    stratum sizes raise instead of rounding. At least one reverse pairing is
    retained in each stratum. Calibration, heldout and smoke inputs are refused.
    No training export is performed by the dataset CLI.
    """
    if len(correlations) != 2 or true_name_group not in ("A", "B"):
        raise ValueError("two correlations and A/B direction are required")
    ratios = [Fraction(str(p)) for p in correlations]
    if any(p not in (Fraction(1, 2), Fraction(9, 10)) for p in ratios):
        raise ValueError("v0 supports correlations 0.5 and 0.9")
    families = defaultdict(dict)
    for record in records:
        if record["pool"] != "training" or record["kind"] != "core":
            raise ValueError("sampling requires core records from an independent training pool")
        key = (record["name_group"], record["answer"])
        if key in families[record["base_id"]]:
            raise ValueError("duplicate core variant")
        families[record["base_id"]][key] = record
    by_hops = defaultdict(list)
    required = {(g, label) for g in ("A", "B") for label in ("True", "False")}
    for base_id, variants in sorted(families.items()):
        if set(variants) != required or len({r["hops"] for r in variants.values()}) != 1:
            raise ValueError("each family needs all four core variants at the same hops")
        by_hops[next(iter(variants.values()))["hops"]].append(base_id)
    if not by_hops:
        raise ValueError("empty training pool")
    rng = random.Random(seed)
    strata = defaultdict(list)
    for hops, ids in sorted(by_hops.items()):
        rng.shuffle(ids)
        if len(ids) % 2:
            raise ValueError("balanced label selection requires an even family count")
        for i, base_id in enumerate(ids):
            strata[hops, "True" if i % 2 == 0 else "False"].append(base_id)
    assignments = [dict(), dict()]
    for (hops, label), ids in sorted(strata.items()):
        rng.shuffle(ids)
        preferred = true_name_group if label == "True" else ("B" if true_name_group == "A" else "A")
        reverse = "B" if preferred == "A" else "A"
        for output, ratio in zip(assignments, ratios):
            count = len(ids) * ratio
            if count.denominator != 1 or not 0 < count < len(ids):
                raise ValueError(f"cannot realize exact {ratio} with reverse pairs in stratum {(hops, label)} of size {len(ids)}")
            for i, base_id in enumerate(ids):
                group = preferred if i < int(count) else reverse
                output[base_id] = families[base_id][group, label]
    return tuple([assignment[base_id] for base_id in sorted(families)] for assignment in assignments)
