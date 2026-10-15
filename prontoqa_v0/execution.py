"""Execution metadata and output reservation, separate from stable data files."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import numpy as np
import scipy


def dependency_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def project_revision(root):
    root = Path(root).resolve()
    git = ["git", "-c", f"safe.directory={root.as_posix()}",
           "-c", f"safe.directory={(root / 'third_party/prontoqa').as_posix()}", "-C", str(root)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=normal",
                                           "--ignore-submodules=untracked"], text=True)
    return {"commit": commit, "dirty": bool(status.strip())}


def reserve_directory(path):
    """Atomic reservation: even an existing empty directory is refused."""
    path = Path(path).resolve()
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing run directory: {path}") from exc
    return path


@contextmanager
def execution_record(directory, command):
    """Host, timestamps, Slurm allocation and executable never enter data manifests."""
    directory = Path(directory)
    started, start_clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    record = {"command": command, "started_utc": started, "hostname": socket.gethostname(),
              "executable": sys.executable, "platform": platform.platform(), "argv": sys.argv[1:],
              "slurm": {key: os.environ.get(key) for key in (
                  "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")},
              "threads": {key: os.environ.get(key) for key in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    try:
        yield
        record["status"], record["exit_code"] = "passed", 0
    except BaseException as exc:
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        record.update(status="failed" if code else "passed", exit_code=code, error=str(exc))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = time.perf_counter() - start_clock
        (directory / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8", newline="\n")
