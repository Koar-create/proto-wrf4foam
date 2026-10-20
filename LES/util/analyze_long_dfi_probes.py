#!/usr/bin/env python3
"""Merge probesLidar restarts for long DFI and write verification table."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_probes_les as a  # noqa: E402

Z_M = a.Z_M
LES = Path(__file__).resolve().parents[1]


def merge_probe_files(paths: list[Path]):
    chunks = []
    for p in paths:
        if not p.is_file():
            continue
        t, u = a.read_probes_U(p)
        chunks.append((t, u))
    if not chunks:
        raise FileNotFoundError("no probe files")
    chunks.sort(key=lambda x: x[0][0])
    out_t, out_u = [], []
    last = -np.inf
    for t, u in chunks:
        sel = t > last + 1e-6
        out_t.append(t[sel])
        out_u.append(u[sel])
        last = float(t[-1])
    return np.concatenate(out_t), np.concatenate(out_u)


def platform_check(rt: np.ndarray, rti: np.ndarray, t0: float, t1: float) -> dict:
    sel = (rt >= t0) & (rt <= t1)
    tt, ti = rt[sel], rti[sel]
    if ti.size < 3:
        return {"ok": False, "reason": "too few samples"}
    rise = (ti[-1] - ti[0]) / max(ti[0], 1e-6)
    mono_up = bool(np.all(np.diff(ti) >= -1e-6) and rise > 0.30)
    return {
        "ok": not mono_up and rise <= 0.30,
        "t0": t0,
        "t1": t1,
        "ti_start": float(ti[0]),
        "ti_end": float(ti[-1]),
        "relative_rise": float(rise),
        "monotone_increasing": mono_up,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, default=LES / "experiments/20250901_1000_long_DFI")
    ap.add_argument(
        "--out",
        type=Path,
        default=LES / "docs/verification/probes_1000_long_DFI_2000.txt",
    )
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()
    case = args.case
    if args.fig is None:
        args.fig = case / "figs_long" / "GAW103_z100_rolling_TIh.png"

    base = case / "postProcessing/probesLidar"
    paths = sorted(base.glob("*/U"), key=lambda p: float(p.parent.name))
    t, U = merge_probe_files(paths)
    dt = float(np.median(np.diff(t)))
    lines = [
        f"# case={case.name}",
        f"probes n={len(t)} t=[{t[0]:.3f},{t[-1]:.3f}] dt_med={dt:.4f}",
        "# No mean-inlet control to 2000 s — station TI cannot be attributed to DFI alone.",
        "",
    ]
    windows = [
        ("w100_400", 100.0, 400.0),
        ("w400_700", 400.0, 700.0),
        ("w700_1000", 700.0, 1000.0),
        ("w1000_1400", 1000.0, 1400.0),
        ("w1400_2000", 1400.0, 2000.0),
    ]
    for wname, t0, t1 in windows:
        for det in (False, True):
            tag = f"{wname}_detrend" if det else wname
            st = a.ti_stats(U, t, t0, t1, detrend=det)
            if st is None:
                continue
            ws_mean, ti, ti_h, k = st
            lines.append(f"\n=== {tag} t={t0:.1f}:{t1:.1f} ===")
            lines.append("    z   Ws_mean    TI      TI_h      k")
            for i, z in enumerate(Z_M):
                lines.append(
                    f"  {z:3d}  {ws_mean[i]:8.3f}  {ti[i]:.4f}  {ti_h[i]:.4f}  {k[i]:.4f}"
                )
            if wname == "w1400_2000" and not det:
                j = 2
                ws = np.hypot(U[:, j, 0], U[:, j, 1])
                sel = (t >= t0) & (t <= t1)
                tt = t[sel]
                tau = a.autocorr_time_scale(ws[sel], dt)
                n_indep = (t1 - t0) / max(tau, dt)
                lines.append(
                    f"  GAW103 z=100: tau_int~{tau:.1f} s, n_indep~{n_indep:.1f} (window {t1-t0:.0f} s)"
                )

    j = 2  # GAW103 z=100 m
    rt, rti = a.rolling_ti_z100(U, t, j, 50.0)
    plat = platform_check(rt, rti, 1400.0, 2000.0)
    lines.append("\n=== GAW103 z=100 m rolling TI_h win=50 s (platform check 1400–2000) ===")
    lines.append(f"  platform_ok={plat.get('ok')} rise={plat.get('relative_rise', float('nan')):.3f} "
                 f"TI_h {plat.get('ti_start', float('nan')):.4f} -> {plat.get('ti_end', float('nan')):.4f}")
    for i in range(0, len(rt), max(len(rt) // 30, 1)):
        lines.append(f"  t={rt[i]:7.1f}  TI_h={rti[i]:.4f}")
    if len(rt):
        lines.append(f"  t_end={rt[-1]:.1f} TI_h={rti[-1]:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")

    os.environ.setdefault("MPLCONFIGDIR", str(case / ".mplconfig"))
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rt, rti, "k-", lw=0.8)
    ax.axvspan(1400, 2000, color="green", alpha=0.1, label="platform window")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("TI_h (50 s window)")
    ax.set_title("GAW103 z=100 m — long DFI")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(args.fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
