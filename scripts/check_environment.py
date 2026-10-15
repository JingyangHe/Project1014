#!/usr/bin/env python3
"""Inspect the existing environment; never install or modify packages."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy
import scipy

EXPECTED = {"python": "3.12.15", "numpy": "2.2.6", "scipy": "1.15.3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    valid = versions == EXPECTED
    if args.expected_prefix:
        valid = valid and Path(sys.prefix).resolve() == args.expected_prefix.resolve()
    report = {"status": "passed" if valid else "failed", "versions": versions, "expected": EXPECTED,
              "executable": sys.executable, "prefix": sys.prefix}
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
