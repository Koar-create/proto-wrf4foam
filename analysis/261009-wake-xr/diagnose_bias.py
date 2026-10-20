"""Compare near-ground X_R with the Fackrell length and with the two cases.

  python3 analysis/261009-wake-xr/diagnose_bias.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MET = REPO / "data" / "261009" / "processed" / "wake_metrics.csv"
OUT = REPO / "data" / "261009" / "processed" / "bias_summary.csv"

# Yu & Thé (2017), standard k-epsilon LW divided by the CEDVAL length.
# These ratios are model/experiment, not model/Fackrell. See the methods note.
KE_RATIO_LOW = 2.1 / 1.45
KE_RATIO_HIGH = 2.6 / 1.42
CASES = (
    "20250901_1000_two_boundaries_as_outlet",
    "20250908_0000_two_boundaries_as_outlet",
)


def _stats(s: pd.Series) -> dict:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n": 0, "median": np.nan, "p25": np.nan, "p75": np.nan}
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def main() -> None:
    df = pd.read_csv(MET)
    rows = []
    for z_tag in ("z3m", "z0.10H", "z0.25H", "z0.50H"):
        sub = df[df["z_tag"] == z_tag]
        for case, g in sub.groupby("case_id"):
            counts = g["status"].value_counts().to_dict()
            for status, n in counts.items():
                rows.append(
                    {
                        "table": "status",
                        "z_tag": z_tag,
                        "case_id": case,
                        "group": status,
                        "n": int(n),
                        "median": np.nan,
                        "p25": np.nan,
                        "p75": np.nan,
                    }
                )
            ok = g[g["status"] == "reattached"]
            for label, part in (
                ("all", ok),
                ("I<3", ok[ok["I"] < 3]),
                ("I>=3", ok[ok["I"] >= 3]),
            ):
                for col in ("r_fackrell", "XR_over_H", "LR_fackrell_over_H"):
                    st = _stats(part[col])
                    rows.append(
                        {
                            "table": col,
                            "z_tag": z_tag,
                            "case_id": case,
                            "group": label,
                            **st,
                        }
                    )
                # 3D cavity versus Fackrell. The cavity columns repeat on every z row.
                if z_tag == "z3m":
                    ratio = part["cav_length_m"] / part["LR_fackrell_m"]
                    st = _stats(ratio)
                    rows.append(
                        {
                            "table": "cav_over_fackrell",
                            "z_tag": z_tag,
                            "case_id": case,
                            "group": label,
                            **st,
                        }
                    )
                    st = _stats(part["cav_length_m"] / part["H_m"])
                    rows.append(
                        {
                            "table": "cav_over_H",
                            "z_tag": z_tag,
                            "case_id": case,
                            "group": label,
                            **st,
                        }
                    )
                    st = _stats(part["p_fit"].where(part["p_r2"] >= 0.5))
                    rows.append(
                        {
                            "table": "p_fit_r2ge0.5",
                            "z_tag": z_tag,
                            "case_id": case,
                            "group": label,
                            **st,
                        }
                    )

    # Same building, two inflows. Only where both cases reattach.
    wide = []
    for z_tag in ("z3m", "z0.25H"):
        a = df[(df["case_id"] == CASES[0]) & (df["z_tag"] == z_tag)]
        b = df[(df["case_id"] == CASES[1]) & (df["z_tag"] == z_tag)]
        if a.empty or b.empty:
            continue
        m = a.merge(b, on="bid", suffixes=("_a", "_b"))
        both = m[(m["status_a"] == "reattached") & (m["status_b"] == "reattached")]
        if both.empty:
            continue
        delta = (both["XR_over_H_a"] - both["XR_over_H_b"]).abs()
        st = _stats(delta)
        rows.append(
            {
                "table": "abs_dXR_over_H",
                "z_tag": z_tag,
                "case_id": "both",
                "group": "both_reattached",
                **st,
            }
        )
        speed = both[["bid", "U_approach_ms_a", "U_approach_ms_b", "XR_over_H_a", "XR_over_H_b", "I_a"]]
        wide.append(speed.assign(z_tag=z_tag))
        corr = float(both["XR_over_H_a"].corr(both["XR_over_H_b"]))
        rows.append(
            {
                "table": "corr_XR_over_H",
                "z_tag": z_tag,
                "case_id": "both",
                "group": "both_reattached",
                "n": int(len(both)),
                "median": corr,
                "p25": np.nan,
                "p75": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    show = out[out["z_tag"].isin(["z3m", "z0.25H"]) & out["table"].isin(
        ["status", "r_fackrell", "XR_over_H", "cav_over_fackrell", "abs_dXR_over_H", "corr_XR_over_H", "p_fit_r2ge0.5"]
    )]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print(show.to_string(index=False))
    print(f"k-epsilon / experiment band from Yu and The 2017: {KE_RATIO_LOW:.3f} to {KE_RATIO_HIGH:.3f}")
    if wide:
        w = wide[0]
        print("approach speed case A vs B (z3m pairs):")
        print(w[["bid", "U_approach_ms_a", "U_approach_ms_b", "XR_over_H_a", "XR_over_H_b"]].to_string(index=False))


if __name__ == "__main__":
    main()
