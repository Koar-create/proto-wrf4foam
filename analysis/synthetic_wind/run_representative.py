#!/usr/bin/env python3
"""
End-to-end pipeline for RANS-driven synthetic second-scale wind fields.

Usage (from repo root):
  python analysis/synthetic_wind/run_representative.py --stage all
  python analysis/synthetic_wind/run_representative.py --stage extract --max-cases 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RANS synthetic wind pipeline")
    parser.add_argument(
        "--stage",
        choices=["extract", "classify", "synthesize", "validate", "all"],
        default="all",
    )
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit extraction count (for testing)")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_args = ["--case", "all"]
    if args.max_cases:
        extract_args += ["--max-cases", str(args.max_cases)]
    if args.force_extract:
        extract_args.append("--force")

    stages = (
        ["extract", "classify", "synthesize", "validate"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "extract":
            _run("extract_fields.py", *extract_args)
        elif stage == "classify":
            _run("classify_states.py")
        elif stage == "synthesize":
            _run("synthesize.py", "--case", "representative", "--seed", str(args.seed))
        elif stage == "validate":
            _run("validate.py", "--case", "representative")


if __name__ == "__main__":
    main()
