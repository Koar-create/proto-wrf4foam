#!/usr/bin/env python3
"""Probe TI tables, autocorrelation, detrended TI, rolling TI — all cases."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

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
    U = np.stack([r[1] for r in rows])
    return t, U


def autocorr_time_scale(x: np.ndarray, dt: float) -> float:
    x = x - x.mean()
    if x.size < 4 or np.var(x) < 1e-20:
        return float("nan")
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx))[:n]
    ac /= ac[0] if ac[0] > 0 else 1.0
    thresh = np.exp(-1.0)
    idx = np.where(ac < thresh)[0]
    if idx.size == 0:
        return (n - 1) * dt
    return float(idx[0] * dt)


def detrend_linear(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = np.vstack([t, np.ones_like(t)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return y - (A @ coef)


def ti_stats(U: np.ndarray, t: np.ndarray, t0: float, t1: float, detrend: bool = False):
    sel = (t >= t0) & (t <= t1)
    u = U[sel]
    tt = t[sel]
    if u.shape[0] < 3:
        return None
    um = u.mean(axis=0)
    up = u - um
    if detrend:
        for j in range(up.shape[1]):
            for c in range(3):
                up[:, j, c] = detrend_linear(up[:, j, c], tt)
    k = 0.5 * (up**2).sum(axis=2).mean(axis=0)
    ws = np.hypot(u[..., 0], u[..., 1])
    ws_mean = ws.mean(axis=0)
    mag = np.linalg.norm(um, axis=1)
    ti = np.sqrt(2.0 / 3.0 * k) / np.maximum(mag, 1e-6)
    ti_h = ws.std(axis=0) / np.maximum(ws_mean, 1e-6)
    return ws_mean, ti, ti_h, k


def rolling_ti_z100(U: np.ndarray, t: np.ndarray, j: int, win: float):
    ws = np.hypot(U[:, j, 0], U[:, j, 1])
    dt = float(np.median(np.diff(t)))
    nwin = max(int(round(win / dt)), 5)
    out_t, out_ti = [], []
    for i in range(nwin, len(t)):
        seg = ws[i - nwin : i]
        m = seg.mean()
        out_t.append(t[i])
        out_ti.append(seg.std() / max(m, 1e-6))
    return np.array(out_t), np.array(out_ti)


def resolve_probe_file(case: Path, probe_time: str) -> Path:
    base = case / "postProcessing/probesLidar"
    direct = base / probe_time / "U"
    if direct.is_file():
        return direct
    # OpenFOAM may keep a single cumulative file under 0/ or 5/.
    candidates = sorted(base.glob("*/U"), key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 10**9)
    if not candidates:
        raise FileNotFoundError(f"no probes under {base}")
    return candidates[-1]


def analyze_case(case: Path, probe_time: str, windows: list[tuple[str, float, float]], out: Path):
    probe_file = resolve_probe_file(case, probe_time)
    t, U = read_probes_U(probe_file)
    dt = float(np.median(np.diff(t)))
    lines = [
        f"# case={case.name}",
        f"probes n={len(t)} t=[{t[0]:.3f},{t[-1]:.3f}] dt_med={dt:.4f}",
    ]
    for wname, t0, t1 in windows:
        for det in (False, True):
            tag = f"{wname}_detrend" if det else wname
            st = ti_stats(U, t, t0, t1, detrend=det)
            if st is None:
                continue
            ws_mean, ti, ti_h, k = st
            lines.append(f"\n=== {tag} t={t0:.1f}:{t1:.1f} ===")
            lines.append("    z   Ws_mean    TI      TI_h      k")
            for i, z in enumerate(Z_M):
                lines.append(
                    f"    {z:3.0f}  {ws_mean[i]:7.3f}  {ti[i]:6.4f}  {ti_h[i]:6.4f}  {k[i]:.4e}"
                )
            lines.append("  station z=100 TI / TI_h / integral_tau (s) / n_indep~T/tau:")
            for name, (a, b, xy) in STATIONS.items():
                j = a + 2
                tau = autocorr_time_scale(ws := np.hypot(U[(t >= t0) & (t <= t1), j, 0], U[(t >= t0) & (t <= t1), j, 1]), dt)
                T = t1 - t0
                nindep = T / tau if tau > 0 and np.isfinite(tau) else float("nan")
                lines.append(
                    f"    {name} xy={xy}: TI={ti[j]:.4f} TI_h={ti_h[j]:.4f} tau={tau:.2f} n_indep~{nindep:.2f}"
                )
    # rolling TI at z=100 for GAW103
    j = STATIONS["GAW103"][0] + 2
    rt, rti = rolling_ti_z100(U, t, j, win=50.0)
    lines.append(f"\n=== rolling TI_h GAW103 z=100 win=50s (n={len(rt)}) ===")
    if rti.size:
        lines.append(f"  min={rti.min():.4f} max={rti.max():.4f} last={rti[-1]:.4f}")
        for frac in (0.25, 0.5, 0.75, 1.0):
            i = min(int(frac * len(rti)) - 1, len(rti) - 1)
            lines.append(f"  at t={rt[i]:.1f}s TI_h={rti[i]:.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--probe-time", default="100")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window", nargs=3, action="append", metavar=("NAME", "T0", "T1"))
    args = p.parse_args()
    windows = [(w[0], float(w[1]), float(w[2])) for w in args.window] if args.window else [
        ("full", 0.0, 1e9),
    ]
    # clamp full window to data
    probe_file = resolve_probe_file(args.case, args.probe_time)
    t, _ = read_probes_U(probe_file)
    fixed = []
    for name, t0, t1 in windows:
        if t1 > 1e8:
            t1 = t[-1]
        fixed.append((name, max(t0, t[0]), min(t1, t[-1])))
    analyze_case(args.case, args.probe_time, fixed, args.out)


if __name__ == "__main__":
    main()
