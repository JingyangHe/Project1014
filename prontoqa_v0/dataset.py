"""Bounded generation, staged smoke gate, deterministic UTF-8 JSONL export."""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import scipy
from .logic import Rejected, World, shortest_proof
from .transform import core_records, diagnostic_records, fingerprint, prepare_family
from .upstream import Adapter, source_hashes
from .validation import validate_dataset, validate_family
from .evaluation import evaluation_bundle, inspection_records, validate_round_trip
from .render import INSTRUCTION
from .execution import dependency_versions, project_revision


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(seed, *parts):
    return int.from_bytes(hashlib.sha256(dumps([seed, *parts]).encode()).digest()[:4], "big")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["hops"] != [1, 2, 3, 4, 5] or set(config["pools"]) != {"smoke", "calibration", "heldout"}:
        raise ValueError("v0 requires hops 1..5 and smoke/calibration/heldout")
    groups = config["name_groups"]
    if set(groups) != {"A", "B"} or any(len(set(v)) != 4 for v in groups.values()) or set(groups["A"]) & set(groups["B"]):
        raise ValueError("two disjoint groups of four distinct names required")
    if set(config["concepts"]) & set(config["properties"]):
        raise ValueError("category/property vocabularies overlap")
    if any(n <= 0 for n in config["pools"].values()) or config["max_attempts_per_pool_hops"] <= 0:
        raise ValueError("counts/attempt budget must be positive")
    if not 0 < config["diagnostic_families_per_hops"] <= config["pools"]["calibration"]:
        raise ValueError("diagnostic selection exceeds calibration pool")
    return config


class GenerationFailure(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(dumps(detail))


def build(root, config, stage="full", progress=None):
    """No export occurs until all requested families pass; no requirement relaxes."""
    adapter = Adapter(root, config)
    cores, provenance, rejections, attempts = [], [], [], {}
    owners = {}  # Reserve core AND all possible intervention worlds before splitting.
    pools = ["smoke"] if stage == "smoke" else ["smoke", "calibration", "heldout"]
    smoke_gate = None
    for pool in pools:
        for hops in config["hops"]:
            accepted, rejection_counts = 0, Counter()
            for attempt in range(config["max_attempts_per_pool_hops"]):
                candidate_seed = seed_for(config["seed"], pool, hops, attempt)
                try:
                    source, origin = adapter.sample(hops, candidate_seed, config)
                    rng = random.Random(seed_for(candidate_seed, "transforms"))
                    prepared = prepare_family(source, hops, rng, config)
                    base_id = f"v0-{pool}-h{hops}-{accepted:04d}"
                    family = core_records(base_id, pool, hops, source, prepared)
                    all_diags = diagnostic_records(family)
                    validate_family(family, all_diags)
                    signatures = {fingerprint(World.from_dict(r["logic"])) for r in family + all_diags}
                    if signatures & owners.keys():
                        raise Rejected("duplicate_world", "entity/order-free collision with an earlier family")
                except Rejected as exc:
                    rejection_counts[exc.reason] += 1
                    rejections.append({"pool": pool, "hops": hops, "attempt": attempt,
                                       "candidate_seed": candidate_seed, "reason": exc.reason, "detail": str(exc)[:1000]})
                    continue
                for signature in signatures:
                    owners[signature] = (base_id, pool)
                cores.extend(family)
                origin.update({"base_id": base_id, "pool": pool, "hops": hops,
                               "source_shortest_proof": shortest_proof(source), "adaptation": prepared[-1]})
                provenance.append(origin)
                accepted += 1
                if accepted == config["pools"][pool]:
                    break
            attempts[f"{pool}/h{hops}"] = {"attempted": attempt + 1, "accepted": accepted,
                                          "rejected": dict(sorted(rejection_counts.items()))}
            if accepted != config["pools"][pool]:
                raise GenerationFailure({"status": "failed", "reason": "maximum_attempts_exhausted",
                                         "pool": pool, "hops": hops, "required": config["pools"][pool],
                                         "accepted": accepted, "attempts": attempts,
                                         "seed": config["seed"], "config": config})
            if progress:
                progress(f"{pool} hops={hops}: {accepted} families; {attempt+1} attempts")
        if pool == "smoke":
            smoke_gate = validate_dataset(cores, [], config, ["smoke"])
            if progress:
                progress("Smoke gate passed; continuing only after automatic validation.")
    diagnostics = []
    if "calibration" in pools:
        family_index = defaultdict(list)
        for record in cores:
            if record["pool"] == "calibration":
                family_index[record["base_id"]].append(record)
        for hops in config["hops"]:
            ids = sorted(base_id for base_id, family in family_index.items() if family[0]["hops"] == hops)
            selected = random.Random(seed_for(config["seed"], "diagnostic_selection", hops)).sample(
                ids, config["diagnostic_families_per_hops"])
            for base_id in sorted(selected):
                diagnostics.extend(diagnostic_records(family_index[base_id]))
    report = validate_dataset(cores, diagnostics, config, pools)
    assert source_hashes(adapter.path) == adapter.hashes, "upstream files changed during generation"
    report.update({"stage": stage, "seed": config["seed"], "config": config,
                   "attempts": attempts, "rejections": dict(Counter(r["reason"] for r in rejections)),
                   "upstream_commit": adapter.commit, "upstream_file_hashes": adapter.hashes,
                   "upstream_unchanged": True, "upstream_proofs_verified": len(provenance),
                   "dependencies": dependency_versions(),
                   "diagnostic_selection": "fixed seeded sampling of 50 calibration families per hops (count from config)",
                   "heldout_policy": "automatic construction and integrity validation only; no model or difficulty selection",
                   "known_limitations": [
                       "Single entity; unary categories and signed properties; positive antecedents; forward rules only.",
                       "Controlled English has three rule templates; no unrestricted natural-language parser.",
                       "Hops measure minimum rule applications, not empirical model difficulty; no model evaluation was run.",
                       "Name groups are configured proxies, not demographic annotations or fairness claims.",
                       "Dedup removes entity/order/query differences but retains predicate spelling; isomorphic renamed ontologies may remain.",
                       "Upstream fictional ontologies have a tree/chain bias; no conjunction, disjunction, quantified exclusions or contraposition.",
                       "The irrelevant antecedent is unreachable in the single entity's closure; universal rules are not claims of existence.",
                       "50%/90% sampling needs a separate training pool with compatible exact stratum sizes; none is exported here."]})
    return cores, diagnostics, provenance, rejections, report, smoke_gate


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(dumps(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def example_markdown(cores):
    lines = ["# PrOntoQA v0：五个完整核心配对题目族", "",
             "仅从 smoke 中按较短文本选取，便于人工阅读。每族四个版本均完整列出；未展示 heldout。", ""]
    for hops in range(1, 6):
        rows = [r for r in cores if r["pool"] == "smoke" and r["hops"] == hops]
        chosen = min(rows, key=lambda r: (len(r["question"]), r["base_id"]))["base_id"]
        family = [r for r in rows if r["base_id"] == chosen]
        lines += [f"## {hops} hops — {chosen}", ""]
        for record in family:
            lines += [f"### {record['variant']} · {record['name']} · {record['answer']}", "",
                      record["question"], "", record["query"], "", "证明：", ""]
            for step in record["proof"]:
                ref = step.get("rule_id", step.get("fact_id"))
                lines.append(f"- {step['id']} [{ref}]: {step['text']}")
            lines += ["", f"关键规则 `{record['key_rule_id']}`；无关规则 `{record['irrelevant_rule_id']}`。", ""]
    return "\n".join(lines)


def export(result, output, config_path, allow_existing=False):
    cores, diagnostics, provenance, rejections, report, smoke_gate = result
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not allow_existing and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing data: {output}")
    filenames = []
    for pool in ("smoke", "calibration", "heldout"):
        filename = pool + ".jsonl"
        write_jsonl(output / filename, [r for r in cores if r["pool"] == pool])
        filenames.append(filename)
    for filename, rows in (("diagnostics.jsonl", diagnostics), ("provenance.jsonl", provenance), ("rejections.jsonl", rejections)):
        write_jsonl(output / filename, rows)
        filenames.append(filename)
    eval_files, evaluation_report = export_evaluation(output, cores, diagnostics)
    filenames.extend(eval_files)
    report["evaluation"] = evaluation_report
    (output / "examples.md").write_text(example_markdown(cores), encoding="utf-8", newline="\n")
    write_json(output / "smoke_gate_report.json", smoke_gate)
    report["config_sha256"] = sha256_file(config_path)
    report["implementation_hashes"] = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    report["data_file_hashes"] = {name: sha256_file(output / name) for name in filenames}
    report["evaluation_manifest_sha256"] = sha256_file(output / "evaluation_manifest.json")
    report.pop("runtime", None)
    report["dependencies"] = dependency_versions()
    report["project"] = project_revision(Path(__file__).resolve().parents[1])
    manifest = {"schema_version": "prontoqa-v0-data-manifest-v1", "seed": report["seed"],
                "config": report["config"], "config_sha256": report["config_sha256"],
                "upstream_commit": report["upstream_commit"], "upstream_file_hashes": report["upstream_file_hashes"],
                "project": report["project"], "dependencies": report["dependencies"],
                "counts": report["totals"], "evaluation": report["evaluation"],
                "implementation_hashes": report["implementation_hashes"],
                "data_file_hashes": report["data_file_hashes"],
                "evaluation_manifest_sha256": report["evaluation_manifest_sha256"]}
    write_json(output / "manifest.json", manifest)
    report["manifest_sha256"] = sha256_file(output / "manifest.json")
    write_json(output / "validation_report.json", report)
    write_json(output / "config.snapshot.json", report["config"])
    return report


def validate_manifest(output, saved):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(output / "manifest.json") == saved["manifest_sha256"]
    for field in ("seed", "config", "config_sha256", "upstream_commit", "upstream_file_hashes",
                  "project", "dependencies", "implementation_hashes", "data_file_hashes", "evaluation_manifest_sha256"):
        assert manifest[field] == saved[field], f"manifest mismatch: {field}"
    assert manifest["counts"] == saved["totals"] and manifest["evaluation"] == saved["evaluation"]
    return {"status": "passed", "sha256": saved["manifest_sha256"]}


def export_evaluation(output, cores, diagnostics):
    """Only deduplicate the evaluation request catalog; all original rows survive."""
    filenames, inspection = [], inspection_records(cores)
    report = None
    bundles = [("evaluation", cores + diagnostics)]
    if inspection:
        validate_family([r for r in inspection if r["kind"] == "core"],
                        [r for r in inspection if r["kind"] == "diagnostic"])
        bundles.append(("inspection", inspection))
        write_jsonl(output / "inspection_records.jsonl", inspection)
        filenames.append("inspection_records.jsonl")
    summaries = {}
    for prefix, records in bundles:
        unique, mapping, comparisons, summary = evaluation_bundle(records)
        for suffix, rows in (("inputs", unique), ("record_map", mapping), ("comparisons", comparisons)):
            name = f"{prefix}_{suffix}.jsonl"
            write_jsonl(output / name, rows)
            filenames.append(name)
        # Validate actual serialized files, including the complete comparison join.
        validate_round_trip(records, read_jsonl(output / f"{prefix}_inputs.jsonl"),
                            read_jsonl(output / f"{prefix}_record_map.jsonl"),
                            read_jsonl(output / f"{prefix}_comparisons.jsonl"))
        summaries[prefix] = summary
        if prefix == "evaluation":
            report = summary
    if inspection:
        case = summaries["inspection"]
        assert case["original_records"] == 12 and case["unique_inputs"] == 8
        assert all(r["hops"] == 5 and r["irrelevant_rule_id"] == "r006" for r in inspection)
        summaries["inspection"]["base_id"] = inspection[0]["base_id"]
    manifest = {"status": "passed", "schema_version": "prontoqa-v0-evaluation-v1",
                "example_id_definition": "sha256(UTF8(json.dumps({question,query},ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)))",
                "text_identity": "exact strings; names, whitespace, sentence order and rule wording retained",
                "instruction": INSTRUCTION,
                "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
                "model_payload_field": "prompt",
                "response_cache_key_fields": ["model_id", "checkpoint_id", "instruction_sha256", "example_id"],
                "response_reuse_policy": "Never reuse responses across models or checkpoints; only exported prompt is sent to a model.",
                "model_calls": 0, "training_sampling_changed": False,
                "inspection_scope": "separate smoke inspection only; excluded from main pool counts and sampling",
                "summaries": summaries, "file_hashes": {name: sha256_file(output / name) for name in filenames}}
    write_json(output / "evaluation_manifest.json", manifest)
    return filenames, report


def validate_evaluation_files(output, cores, diagnostics):
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["instruction"] == INSTRUCTION
    assert manifest["instruction_sha256"] == hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
    for filename, digest in manifest["file_hashes"].items():
        assert sha256_file(output / filename) == digest, f"evaluation hash mismatch: {filename}"
    bundles = [("evaluation", cores + diagnostics)]
    inspection = inspection_records(cores)
    if inspection:
        assert read_jsonl(output / "inspection_records.jsonl") == inspection
        validate_family([r for r in inspection if r["kind"] == "core"],
                        [r for r in inspection if r["kind"] == "diagnostic"])
        bundles.append(("inspection", inspection))
    summaries = {}
    for prefix, records in bundles:
        expected = evaluation_bundle(records)
        actual = [read_jsonl(output / f"{prefix}_{suffix}.jsonl") for suffix in ("inputs", "record_map", "comparisons")]
        assert all(a == b for a, b in zip(actual, expected[:3]))
        validate_round_trip(records, *actual)
        summary = dict(expected[3])
        if prefix == "inspection":
            summary["base_id"] = inspection[0]["base_id"]
            assert summary["original_records"] == 12 and summary["unique_inputs"] == 8
        assert summary == manifest["summaries"][prefix]
        summaries[prefix] = summary
    return {"status": "passed", "summaries": summaries, "manifest_sha256": sha256_file(output / "evaluation_manifest.json")}
