#!/usr/bin/env python3
"""Classify 120 hourly states by stability and LLJ; pick representatives."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    OUTPUT_ROOT,
    STATS_SUBDIR,
    dump_json,
    enhanced_hdf5_path,
    list_case_ids,
)
from analysis.synthetic_wind.grid import load_enhanced_hdf5  # noqa: E402
from analysis.synthetic_wind.statistics import (  # noqa: E402
    classify_stability,
    summarize_case_stats,
)


def classify_all() -> pd.DataFrame:
    rows = []
    for cid in list_case_ids():
        h5 = enhanced_hdf5_path(cid)
        if not h5.is_file():
            rows.append({"case_id": cid, "stability": "missing", "llj": False, "has_enhanced": False})
            continue
        data = load_enhanced_hdf5(cid)
        meta = data.get("meta", {})
        stability = meta.get("stability")
        llj = meta.get("llj")
        if stability is None:
            summary = summarize_case_stats(
                data["k"], data["epsilon"], data["U"], data["coords_z"]
            )
            stability = classify_stability(summary)
            llj = summary["llj"]
        rows.append({
            "case_id": cid,
            "stability": stability,
            "llj": bool(llj),
            "has_enhanced": True,
        })
    return pd.DataFrame(rows)


def pick_representatives(df: pd.DataFrame, per_class: int = 2) -> list[str]:
    """Pick cases covering stability × LLJ combinations (enhanced only)."""
    df_ok = df[df["has_enhanced"] == True].copy()  # noqa: E712
    selected = []
    groups = defaultdict(list)
    for _, row in df_ok.iterrows():
        key = (row["stability"], row["llj"])
        groups[key].append(row["case_id"])
    for key in sorted(groups.keys()):
        selected.extend(groups[key][:per_class])
    # ensure at least one daytime and one nighttime if possible
    extras = []
    for cid in df["case_id"]:
        if "_1200_" in cid or "_1400_" in cid:
            extras.append(cid)
        if "_0200_" in cid or "_2200_" in cid:
            extras.append(cid)
    for e in extras:
        if e not in selected:
            selected.append(e)
    return list(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    df = classify_all()
    out_dir = OUTPUT_ROOT / STATS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "state_classification.csv"
    df.to_csv(csv_path, index=False)

    reps = pick_representatives(df, per_class=args.per_class)
    rep_path = out_dir / "representative_cases.json"
    dump_json(rep_path, {"representative_cases": reps, "n_total": len(df)})
    print(f"Wrote {csv_path} ({len(df)} cases)")
    print(f"Representative cases ({len(reps)}): {reps[:5]}...")


if __name__ == "__main__":
    main()
