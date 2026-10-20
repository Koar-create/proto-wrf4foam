#!/usr/bin/env python3
"""Near-inlet temporal sigma_u on z=100 slices: DFI long run vs mean-inlet baseline."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

LES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LES / "20250901_0000_WN_outlet_DFSEM"))
from plot_dfsem_z100 import read_vtp_points_U  # noqa: E402

SLICE_TIMES = list(range(150, 1050, 50))
LINES = {
    "south_y-4950": ("y", -4950.0, 80.0),
    "south_y-4500": ("y", -4500.0, 80.0),
    "east_x4950": ("x", 4950.0, 80.0),
    "east_x4500": ("x", 4500.0, 80.0),
}


def load_series(case: Path, times: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    speeds = []
    ref_pts = None
    for t in times:
        p = case / "postProcessing" / "surfacesU" / str(t) / "z100.vtp"
        if not p.is_file():
            raise FileNotFoundError(p)
        pts, u = read_vtp_points_U(p, field="U")
        ws = np.linalg.norm(u, axis=1)
        if ref_pts is None:
            ref_pts = pts
        speeds.append(ws)
    return ref_pts, speeds


def line_mask(pts: np.ndarray, axis: str, val: float, tol: float) -> np.ndarray:
    j = 0 if axis == "x" else 1
    return np.abs(pts[:, j] - val) < tol


def read_of_scalar_list(path: Path) -> np.ndarray:
    txt = path.read_text()
    m = re.search(r"\n(\d+)\n\(\n", txt)
    if not m:
        raise ValueError(f"cannot parse scalar list in {path}")
    n = int(m.group(1))
    inner = txt.split(m.group(0), 1)[1].rsplit(")", 1)[0]
    vals = np.fromstring(inner.replace("\n", " "), sep=" ")
    if vals.size != n:
        raise ValueError(f"{path}: expected {n} values, got {vals.size}")
    return vals


def read_of_vector_list(path: Path) -> np.ndarray:
    txt = path.read_text()
    m = re.search(r"\n(\d+)\n\(\n", txt)
    if not m:
        raise ValueError(f"cannot parse vector list in {path}")
    n = int(m.group(1))
    triples = re.findall(r"\(([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\)", txt)
    arr = np.array(triples, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"{path}: expected {n} points, got {arr.shape[0]}")
    return arr


def k_target_z100(bd_root: Path, patch: str, z: float = 100.0) -> float:
    pts = read_of_vector_list(bd_root / patch / "points")
    k = read_of_scalar_list(bd_root / patch / "0" / "k")
    layer = np.abs(pts[:, 2] - z) < 5
    return float(np.sqrt(2.0 / 3.0 * np.mean(k[layer])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfi", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--constant", type=Path, required=True, help="DFSEM constant with boundaryData")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--times",
        type=int,
        nargs="+",
        default=None,
        help="slice times (default 150–1000 step 50)",
    )
    args = ap.parse_args()
    times = args.times if args.times is not None else SLICE_TIMES
    bd = args.constant / "boundaryData"
    pts_d, spd_d = load_series(args.dfi, times)
    _, spd_b = load_series(args.baseline, [25, 50, 75, 100])
    stack_d = np.stack(spd_d, axis=0)
    stack_b = np.stack(spd_b, axis=0)
    sigma_d = stack_d.std(axis=0)
    sigma_b = stack_b.std(axis=0)

    lines_out = [
        "# DFI long run vs mean-inlet baseline on z=100 m slices",
        f"# DFI times={times}; baseline times=[25,50,75,100]",
        "",
    ]
    for name, (axis, val, tol) in LINES.items():
        m = line_mask(pts_d, axis, val, tol)
        if m.sum() < 10:
            lines_out.append(f"## {name}: too few points ({m.sum()})")
            continue
        patch = "south" if "south" in name else "east"
        ut = k_target_z100(bd, patch)
        sd = sigma_d[m]
        sb = sigma_b[m]
        along = pts_d[m, 0] if axis == "y" else pts_d[m, 1]
        order = np.argsort(along)
        lines_out.append(f"## {name} n={m.sum()} u_target_sqrt(2k/3)={ut:.3f} m/s")
        lines_out.append("  along_m  sigma_u_DFI  sigma_u_baseline")
        for a, d, b in zip(along[order], sd[order], sb[order]):
            lines_out.append(f"  {a:10.1f}  {d:12.4f}  {b:12.4f}")
        lines_out.append(
            f"  summary: median_sigma_DFI={np.median(sd):.4f} baseline={np.median(sb):.4f} "
            f"max_DFI={sd.max():.4f} ratio_to_target={np.median(sd)/max(ut,1e-6):.2f}"
        )
        lines_out.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines_out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
