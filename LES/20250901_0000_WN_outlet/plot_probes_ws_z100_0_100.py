#!/usr/bin/env python3
"""Mean-inlet LES z=100 probe Ws for t=0–100, same layout as the DFSEM figure."""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet")
OUT = CASE / "figs_les_100s"
PROBE = CASE / "postProcessing" / "probesLidar" / "0" / "U"
os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))

STATIONS = {
    "GAW103": 0 + 2,  # z=100 is the 3rd height at each site
    "GAW104": 11 + 2,
    "GAW111": 22 + 2,
}


def read_probes_U(path: Path):
    txt = path.read_text()
    rows = []
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        t_s, rest = line.split(None, 1)
        vecs = re.findall(r"\(([^()]*)\)", rest)
        arr = np.array([list(map(float, v.split())) for v in vecs], dtype=float)
        rows.append((float(t_s), arr))
    t = np.array([r[0] for r in rows])
    U = np.stack([r[1] for r in rows])
    return t, U


def main():
    if not PROBE.is_file():
        raise SystemExit(f"missing {PROBE}")
    OUT.mkdir(parents=True, exist_ok=True)
    t, U = read_probes_U(PROBE)
    sel = (t >= 0.0) & (t <= 100.0 + 1e-9)
    t, U = t[sel], U[sel]
    print(
        f"probes n={len(t)} t=[{t[0]:.2f},{t[-1]:.2f}] "
        f"dt_med={np.median(np.diff(t)) if len(t) > 1 else np.nan:.3f}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    for ax, (name, j) in zip(axes, STATIONS.items()):
        u = U[:, j]
        ws = np.hypot(u[:, 0], u[:, 1])
        ax.plot(t, ws, lw=0.7, color="C0")
        ax.set_ylabel(f"{name}\nWs z=100 (m/s)")
        ax.grid(alpha=0.3)
        print(
            f"  {name}  mean={ws.mean():.4f}  min={ws.min():.4f}  "
            f"max={ws.max():.4f}  std={ws.std():.4f}"
        )
    axes[-1].set_xlabel("t (s)")
    axes[-1].set_xlim(0.0, 100.0)
    fig.suptitle("LES probes, horizontal wind speed at z=100 m (t=0–100)")
    fig.tight_layout()
    out = OUT / "probes_ws_z100.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
