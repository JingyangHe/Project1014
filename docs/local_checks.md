# Local checks and HiPerGator validation boundary

Checked locally on 2026-10-15; no HiPerGator connection or remote job submission was made.
These are local development results, not evidence of successful Linux/Slurm execution.

## Actually executed locally

- Existing interpreter: Python 3.12.15, NumPy 2.2.6, SciPy 1.15.3.
- 33 Python tests passed. This includes true/false/unknown/contradiction, shortest paths and shortcuts,
  proof references, paired transformations, base_id/example_id, duplicate-input comparison links,
  pool leakage, prompt whitelists, exact sampling ratios, bounded generation, environment checking,
  output overwrite refusal and data manifest structure.
- The same 33 tests passed in a separate fresh local checkout with the pinned upstream and no generated dataset.
  The pinned upstream emitted its existing `invalid escape sequence` SyntaxWarning on first import; no upstream files were changed.
- The fixed `v0-smoke-h5-0008` fixture has 12 derived records / 8 exact inputs; all proofs remain 5 steps.
- Default scientific config, smoke stage: 100 families / 400 core records / 0 main diagnostic records.
- Independent validation passed, including all 100 upstream proof ASTs and 400 transformation replays.
- A second smoke generation matched all 13 exported JSONL files byte for byte, and matched the evaluation manifest.
- Tiny full-pool generation is also exercised by Python tests: calibration/heldout membership and diagnostic links remain checked.
- Generation into an existing directory is refused before data construction.

CLI forms used, with an explicit fresh local scratch root:

```text
python -m prontoqa_v0 selftest --output <scratch>/selftest
python -m prontoqa_v0 generate --stage smoke --config configs/prontoqa_v0.json --output <scratch>/data
python -m prontoqa_v0 validate --config configs/prontoqa_v0.json --output <scratch>/data --report-dir <scratch>/validation
python -m prontoqa_v0 reproduce --config configs/prontoqa_v0.json --output <scratch>/data --scratch <scratch>/reproduction --report-dir <scratch>/repro-checks
```

Raw local artifacts are generated under ignored `work/`, not committed.

## Static and offline shell checks

- Bash 5.2.15 (`x86_64-pc-msys`) parsed both submission/Slurm scripts, `configs/hipergator.env` and `tests/test_submit.sh` with `bash -n`.
- `bash tests/test_submit.sh` passed against a local stub `sbatch`; it checked smoke/full/validate argument forwarding,
  that stdout/stderr directories exist before submission, quoting, config snapshots and existing-output refusal.
  It did not execute a worker or submit a real job.
- Shell/Slurm files checked as LF with no UTF-8 BOM. Slurm directives request 1 task, 4 CPUs, 16G, 4 hours,
  account/qos `fsu-compsci-dept`, with no partition or GPU directive.

## Pending on HiPerGator

- Availability of the `conda` module and activation of the supplied full environment prefix.
- Account/qos authorization and the site's default CPU partition selection.
- `/blue` read/write permissions, quota, and actual Slurm scheduling and worker execution.
- Server-side test suite, environment report, smoke generation/validation/reproduction, and then full generation.

Run the README smoke workflow first and inspect Slurm exit status plus all passed reports before full generation.
