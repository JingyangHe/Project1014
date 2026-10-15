from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_prontoqa_v0 import CONFIG, ROOT
from prontoqa_v0.dataset import build, export, sha256_file, validate_manifest, write_json
from prontoqa_v0.execution import dependency_versions, execution_record, reserve_directory


class ExecutionTests(unittest.TestCase):
    def test_existing_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            reserve_directory(path)
            sentinel = path / "sentinel.txt"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                reserve_directory(path)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                export(([], [], [], [], {}, {}), path, ROOT / "configs/prontoqa_v0.json")
            self.assertEqual(sentinel.read_text(), "keep")

    def test_cli_refuses_existing_output_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-m", "prontoqa_v0", "generate", "--output", tmp],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failure_execution_metadata_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "test failure"):
                with execution_record(tmp, "test"):
                    raise ValueError("test failure")
            record = json.loads((Path(tmp) / "execution.json").read_text())
            self.assertEqual((record["status"], record["exit_code"]), ("failed", 1))
            self.assertIn("hostname", record)
            self.assertIn("elapsed_seconds", record)

    def test_manifest_has_versions_commit_counts_and_hashes_without_host(self):
        cfg = deepcopy(CONFIG)
        cfg["pools"] = {"smoke": 1, "calibration": 1, "heldout": 1}
        cfg["diagnostic_families_per_hops"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config_path = path / "config.json"
            write_json(config_path, cfg)
            report = export(build(ROOT, cfg), path / "data", config_path)
            manifest_path = path / "data/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["seed"], CONFIG["seed"])
            self.assertEqual(manifest["dependencies"], dependency_versions())
            self.assertRegex(manifest["project"]["commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(manifest["project"]["dirty"], bool)
            self.assertEqual(manifest["counts"]["families"], 15)
            for name, digest in manifest["data_file_hashes"].items():
                self.assertEqual(sha256_file(path / "data" / name), digest)
            self.assertEqual(validate_manifest(path / "data", report)["status"], "passed")
            for key in ("hostname", "started_utc", "elapsed_seconds", "executable", "platform"):
                self.assertNotIn(key, manifest_path.read_text())


class DeploymentStaticTests(unittest.TestCase):
    def test_shell_files_are_lf_without_bom_and_have_cpu_only_resources(self):
        scripts = list((ROOT / "scripts").glob("*.sh")) + list((ROOT / "scripts").glob("*.sbatch"))
        scripts.append(ROOT / "configs/hipergator.env")
        scripts.append(ROOT / "tests/test_submit.sh")
        for path in scripts:
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), path)
            self.assertNotIn(b"\r", raw, path)
        text = (ROOT / "scripts/prontoqa_cpu.sbatch").read_text()
        directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
        for value in ("--account=fsu-compsci-dept", "--qos=fsu-compsci-dept", "--ntasks=1",
                      "--cpus-per-task=4", "--mem=16G", "--time=04:00:00"):
            self.assertIn("#SBATCH " + value, directives)
        self.assertFalse(any("--partition" in line or "--gres" in line or "--gpus" in line for line in directives))
        self.assertIn("SLURM_CPUS_PER_TASK", text)
        self.assertIn("OPENBLAS_NUM_THREADS=$threads", text)

    def test_environment_checker_accepts_existing_versions(self):
        result = subprocess.run([sys.executable, "scripts/check_environment.py", "--expected-prefix", sys.prefix],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
