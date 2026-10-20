#!/usr/bin/env python3
"""Probe TI + Q percentiles. Output stays in this LES case."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

CASE = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling/LES/20250901_0000_WN_outlet")
OUT = CASE / "figs_les_300s"
os.environ.setdefault("MPLCONFIGDIR", str(CASE / ".mplconfig"))

STATIONS = {
    "GAW103": (0, 11, (975, -320)),
    "GAW104": (11, 22, (450, 350)),
    "GAW111": (22, 33, (75, 30)),
}
Z_M = np.array([60, 80, 100, 120, 150, 200, 232, 300, 400, 500, 600])


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
    U = np.stack([r[1] for r in rows])  # (nt, nprobe, 3)
    return t, U


def read_of_internal_scalar(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    key = b"internalField"
    i = raw.find(key)
    if i < 0:
        raise ValueError(f"no internalField in {path}")
    chunk = raw[i:]
    m = re.search(br"nonuniform\s+List<scalar>\s+(\d+)\s*\(", chunk)
    if not m:
        raise ValueError(f"cannot parse List<scalar> in {path}")
    n = int(m.group(1))
    start = i + m.end()
    # skip one newline if present
    if start < len(raw) and raw[start : start + 1] in (b"\n", b" "):
        start += 1
    nbytes = n * 8
    buf = raw[start : start + nbytes]
    if len(buf) != nbytes:
        raise ValueError(f"{path}: expected {nbytes} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype="<f8")


def ti_block(U, t0, t1, t):
    sel = (t >= t0) & (t <= t1)
    u = U[sel]
    um = u.mean(axis=0)
    up = u - um
    k = 0.5 * (up**2).sum(axis=2).mean(axis=0)
    ws = np.hypot(u[..., 0], u[..., 1])
    ws_mean = ws.mean(axis=0)
    mag = np.linalg.norm(um, axis=1)
    ti = np.sqrt(2.0 / 3.0 * k) / np.maximum(mag, 1e-6)
    ti_h = ws.std(axis=0) / np.maximum(ws_mean, 1e-6)
    return ws_mean, mag, ti, ti_h, k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t, U = read_probes_U(CASE / "postProcessing/probesLidar/100/U")
    lines = []
    lines.append(f"probes n={len(t)} t=[{t[0]:.1f},{t[-1]:.1f}] dt_med={np.median(np.diff(t)):.3f}")
    for window, t0, t1 in (("full", t[0], t[-1]), ("last100s", max(t[-1] - 100, t[0]), t[-1])):
        ws_mean, mag, ti, ti_h, k = ti_block(U, t0, t1, t)
        lines.append(f"\n=== {window} t={t0:.1f}:{t1:.1f} ===")
        for name, (a, b, xy) in STATIONS.items():
            lines.append(f"  {name} xy={xy}")
            lines.append("    z   Ws_mean  |U|_mean    TI     TI_h      k")
            for i, z in enumerate(Z_M):
                j = a + i
                lines.append(
                    f"    {z:3.0f}  {ws_mean[j]:7.3f}  {mag[j]:7.3f}  {ti[j]:6.3f}  {ti_h[j]:6.3f}  {k[j]:.4e}"
                )
    # Q percentiles at 105 and 300
    lines.append("\n=== Q internalField ===")
    for tt in ("105", "300"):
        q = read_of_internal_scalar(CASE / tt / "Q")
        qpos = q[q > 0]
        pcts = (50, 90, 95, 99, 99.5, 99.9, 99.99)
        qp = np.percentile(qpos, pcts) if qpos.size else np.zeros(len(pcts))
        lines.append(
            f"  t={tt} n={q.size} min={q.min():.4e} max={q.max():.4e} "
            f"frac_pos={qpos.size / q.size:.4f}"
        )
        lines.append(
            "    Q>0 pct "
            + " ".join(f"p{p:g}={v:.4e}" for p, v in zip(pcts, qp))
        )
        # fraction of cells above candidate isos
        for iso in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
            frac = float((q > iso).mean())
            lines.append(f"    Q>{iso:g}: frac={frac:.6f} n={int((q > iso).sum())}")
    report = "\n".join(lines) + "\n"
    (OUT / "inspect_les_300.txt").write_text(report)
    print(report)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    z100_idx = {name: a + 2 for name, (a, b, xy) in STATIONS.items()}  # z=100 is 3rd
    for ax, (name, j) in zip(axes, z100_idx.items()):
        u = U[:, j]
        ws = np.hypot(u[:, 0], u[:, 1])
        ax.plot(t, ws, lw=0.7, color="C0")
        ax.set_ylabel(f"{name}\nWs z=100 (m/s)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("LES probes, horizontal wind speed at z=100 m (t=100–300)")
    fig.tight_layout()
    fig.savefig(OUT / "probes_ws_z100.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
