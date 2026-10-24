#!/usr/bin/env python3
"""Lightweight post-run checks for URANS pimpleFoam cases."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    umax = []
    for line in text.splitlines():
        if "max(mag(U))" in line or "min/max(mag(U))" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    nums = parts[-1].strip().split()
                    umax.append(float(nums[-1]))
                except ValueError:
                    pass
    times = [float(m.group(1)) for m in re.finditer(r"^Time = ([0-9.eE+-]+)", text, re.M)]
    return {
        "Umax_series_tail": umax[-20:],
        "Umax_global": max(umax) if umax else None,
        "last_time": times[-1] if times else None,
        "has_nan": bool(
            re.search(
                r"\bnan\b|FOAM FATAL.*nan|Abort triggered|--> FOAM FATAL",
                text,
                re.I,
            )
        )
        and "Floating point exception trapping enabled" not in text[:5000],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--log", default="log.pimpleFoam")
    args = ap.parse_args()
    case = args.case_dir.resolve()
    meta_path = case / "urans_boundary_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    u_thr = float(meta.get("watchdog_Umax_suggest", 40)) / 2.5 * 2.0
    log = case / args.log
    if not log.is_file():
        for alt in sorted(case.glob("log.pimpleFoam*")):
            log = alt
            break
    if not log.is_file():
        logs_dir = case / "logs"
        if logs_dir.is_dir():
            for alt in sorted(logs_dir.glob("log.pimpleFoam*")):
                log = alt
                break
    if not log.is_file():
        print("no log found", file=sys.stderr)
        return 2
    rep = parse_log(log)
    ok = True
    if rep["has_nan"]:
        print("FAIL: nan/FPE/abort in log")
        ok = False
    if rep["Umax_global"] is not None and rep["Umax_global"] > u_thr:
        print(f"FAIL: max|U|={rep['Umax_global']:.3g} > threshold {u_thr:.3g}")
        ok = False
    print(json.dumps(rep, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
