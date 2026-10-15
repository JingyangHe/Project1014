"""CPU data CLI. Use explicit run directories; see scripts/submit_prontoqa.sh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from .dataset import (build, export, GenerationFailure, load_config, read_jsonl,
                      sha256_file, write_json, dumps, validate_evaluation_files, validate_manifest)
from .evaluation import annotate_records
from .execution import execution_record, reserve_directory
from .validation import validate_dataset


def execute(args, root, config_path, config, output, reports):
    if args.command == "selftest":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
        def names(suite):
            return [name for item in suite for name in (names(item) if isinstance(item, unittest.TestSuite) else [item.id()])]
        test_names = names(suite)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        write_json(reports / "selftest_report.json", {"status": "passed" if result.wasSuccessful() else "failed",
                    "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                    "test_names": test_names,
                    "test_file_hashes": {p.name: sha256_file(p) for p in sorted((root / "tests").glob("test_*.py"))}})
        return 0 if result.wasSuccessful() else 1
    if args.command == "generate":
        try:
            result = build(root, config, args.stage, progress=lambda message: print(message, flush=True))
            report = export(result, output, config_path)
        except GenerationFailure as exc:
            write_json(output / "generation_failure.json", exc.detail)
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(report["totals"], ensure_ascii=False))
        return 0
    saved = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert saved["config"] == config and saved["config_sha256"] == sha256_file(config_path)
    for filename, expected in saved["data_file_hashes"].items():
        assert sha256_file(output / filename) == expected, f"file hash mismatch: {filename}"
    cores = sum([read_jsonl(output / (pool + ".jsonl")) for pool in ("smoke", "calibration", "heldout")], [])
    diagnostics = read_jsonl(output / "diagnostics.jsonl")
    pools = ["smoke"] if saved["stage"] == "smoke" else ["smoke", "calibration", "heldout"]
    if args.command == "export-eval":
        original = {r["id"]: r for r in cores + diagnostics}
        def payload_hash(rows):
            payloads = [{k: v for k, v in r.items() if k not in ("example_id", "comparison")}
                        for r in sorted(rows, key=lambda r: r["id"])]
            return hashlib.sha256(dumps(payloads).encode("utf-8")).hexdigest()
        before_hash = payload_hash(cores + diagnostics)
        cores, diagnostics = annotate_records(cores, diagnostics)
        assert {r["id"] for r in cores + diagnostics} == set(original)
        assert payload_hash(cores + diagnostics) == before_hash, "original payload changed"
        checked = validate_dataset(cores, diagnostics, config, pools)
        from .audit import audit_provenance
        provenance = read_jsonl(output / "provenance.jsonl")
        checked["provenance_audit"] = audit_provenance(root, config, cores, provenance)
        report = {**saved, **checked, "metadata_upgrade": {
            "status": "passed", "original_records_preserved": len(original),
            "original_payload_sha256_before": before_hash, "original_payload_sha256_after": before_hash,
            "only_added_fields": ["example_id", "comparison (diagnostic records)"],
            "seed_unchanged": saved["seed"], "config_sha256_unchanged": saved["config_sha256"],
            "training_sampling_changed": False}}
        smoke_gate = validate_dataset([r for r in cores if r["pool"] == "smoke"], [], config, ["smoke"])
        export((cores, diagnostics, provenance, read_jsonl(output / "rejections.jsonl"), report, smoke_gate),
               output, config_path, allow_existing=True)
        write_json(reports / "metadata_upgrade_report.json", report["metadata_upgrade"])
        print(json.dumps(report["evaluation"], ensure_ascii=False))
        return 0
    checked = validate_dataset(cores, diagnostics, config, pools)
    # Also audit all source proofs and transformations from stored provenance.
    from .audit import audit_provenance
    checked["provenance_audit"] = audit_provenance(root, config, cores, read_jsonl(output / "provenance.jsonl"))
    checked["evaluation_exports"] = validate_evaluation_files(output, cores, diagnostics)
    checked["manifest"] = validate_manifest(output, saved)
    assert checked["evaluation_exports"]["manifest_sha256"] == saved["evaluation_manifest_sha256"]
    if args.command == "validate":
        checked["data_file_hashes"] = saved["data_file_hashes"]
        write_json(reports / "independent_validation_report.json", checked)
        print(json.dumps({"status": "passed", **checked["totals"]}, ensure_ascii=False))
        return 0
    result = build(root, config, saved["stage"], progress=lambda message: print(message, flush=True))
    repeated = export(result, args.scratch, config_path)
    matches = {name: expected == repeated["data_file_hashes"][name] for name, expected in saved["data_file_hashes"].items()}
    assert all(matches.values()), f"reproduction mismatch: {matches}"
    assert repeated["evaluation_manifest_sha256"] == saved["evaluation_manifest_sha256"]
    write_json(reports / "reproducibility_report.json", {"status": "passed", "seed": config["seed"],
               "config_sha256": saved["config_sha256"], "file_matches": matches,
               "data_file_hashes": saved["data_file_hashes"], "independent_validation_passed": True,
               "evaluation_manifest_matches": True})
    print("Byte-for-byte JSONL reproduction passed.")
    return 0


def main():
    if sys.flags.optimize:
        raise SystemExit("Validation must run without -O / PYTHONOPTIMIZE.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "export-eval", "validate", "reproduce", "selftest"])
    parser.add_argument("--config", default="configs/prontoqa_v0.json")
    parser.add_argument("--output", required=True, help="explicit dataset directory (new for generate/selftest)")
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--scratch", help="new reproduction directory; required for reproduce")
    parser.add_argument("--report-dir", help="new directory for validate/reproduce/export-eval reports")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = load_config(config_path)
    output = (root / args.output).resolve()
    if args.command in ("generate", "selftest"):
        if args.report_dir:
            parser.error("generate/selftest use --output for reports; --report-dir is for existing datasets")
        reports = reserve_directory(output)
    else:
        if not output.is_dir():
            parser.error(f"dataset directory does not exist: {output}")
        reports = reserve_directory(root / args.report_dir) if args.report_dir else Path(
            tempfile.mkdtemp(prefix=f"{output.name}-{args.command}-", dir=output.parent))
    if args.command == "reproduce":
        if not args.scratch:
            parser.error("reproduce requires --scratch pointing to a new run directory")
        args.scratch = (root / args.scratch).resolve()
        if args.scratch == output:
            parser.error("reproduction scratch must differ from reference")
        reserve_directory(args.scratch)
    print(f"Dataset/output: {output}\nReports: {reports}", flush=True)
    with execution_record(reports, args.command):
        code = execute(args, root, config_path, config, output, reports)
        if code:
            raise SystemExit(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
