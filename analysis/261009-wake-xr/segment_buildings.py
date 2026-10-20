"""Segment the case STL into buildings and select tall, relatively isolated ones.

The two pilot cases share one binary STL (verified by MD5 before this script
is trusted). Buildings are connected components of triangles that share
vertices. Canton Tower is the component with the largest roof height and is
excluded: its geometry in this STL is known to be wrong.

Selection (plan):
  H >= 100 m
  and H >= 2 * (area-weighted mean height of other buildings whose centroid
  lies inside a radius of 2H)
  and not Canton Tower.

W and L are not computed here. They depend on the wind direction of each
case. This script stores the ground footprint and the z >= 0.5H footprint so
the frame step can project them.

Run from the repo root:
  python3 analysis/261009-wake-xr/segment_buildings.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STL = (
    REPO
    / "steady_experiments_finer_ABL"
    / "20250901_1000_two_boundaries_as_outlet"
    / "constant"
    / "triSurface"
    / "buildings.stl"
)
OUT_DATA = REPO / "data" / "261009" / "processed"
OUT_FIG = REPO / "results" / "wake_xr" / "261009"

H_MIN = 100.0
NEIGH_RADIUS_FACTOR = 2.0
HEIGHT_FACTOR = 2.0
HIGH_FRAC = 0.5
GROUND_Z = 1.0


def load_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(raw[84 : 84 + n * 50], dtype=dt)["v"].astype(np.float64)
    if tris.shape[0] != n:
        raise RuntimeError(f"STL triangle count mismatch: header {n}, got {tris.shape[0]}")
    return tris


def connected_components(tris: np.ndarray) -> np.ndarray:
    """Return a component id per triangle. Vertices match on exact float32 bits."""
    flat = np.ascontiguousarray(tris.astype(np.float32).reshape(-1, 3))
    keys = flat.view(np.dtype((np.void, 12))).ravel()
    _, inv, _ = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1, 3)
    n_vert = int(inv.max()) + 1
    parent = np.arange(n_vert, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in inv:
        union(int(a), int(b))
        union(int(b), int(c))

    roots = np.empty(n_vert, dtype=np.int32)
    for i in range(n_vert):
        roots[i] = find(i)
    # One root per triangle, from its first vertex.
    tri_root = roots[inv[:, 0]]
    _, comp = np.unique(tri_root, return_inverse=True)
    return comp.astype(np.int32)


def polygon_area_and_centroid(xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Convex hull area, polygon centroid, and hull vertices (closed not included)."""
    xy = np.unique(np.round(xy, 3), axis=0)
    if len(xy) < 3:
        if len(xy) == 0:
            return 0.0, np.array([np.nan, np.nan]), xy
        return 0.0, xy.mean(axis=0), xy
    try:
        hull = ConvexHull(xy)
    except Exception:
        return 0.0, xy.mean(axis=0), xy
    hv = xy[hull.vertices]
    x, y = hv[:, 0], hv[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    signed = 0.5 * float(cross.sum())
    if abs(signed) < 1e-6:
        return 0.0, hv.mean(axis=0), hv
    cx = float(((x + x2) * cross).sum() / (6.0 * signed))
    cy = float(((y + y2) * cross).sum() / (6.0 * signed))
    return abs(signed), np.array([cx, cy]), hv


def summarise(tris: np.ndarray, comp: np.ndarray) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    rows = []
    hulls = []
    high_hulls = []
    order = np.argsort(comp, kind="mergesort")
    comp_s = comp[order]
    breaks = np.flatnonzero(np.diff(comp_s)) + 1
    spans = np.split(order, breaks)
    ids = comp_s[np.r_[0, breaks]]
    for bid, idx in zip(ids, spans):
        verts = tris[idx].reshape(-1, 3)
        z = verts[:, 2]
        h = float(z.max())
        ground = verts[z <= GROUND_Z, :2]
        if len(ground) < 3:
            ground = verts[:, :2]
        area, cen, hull = polygon_area_and_centroid(ground)
        high = verts[z >= HIGH_FRAC * h, :2] if h > 0 else verts[:, :2]
        _, _, high_hull = polygon_area_and_centroid(high if len(high) else verts[:, :2])
        rows.append(
            {
                "bid": int(bid),
                "cx": float(cen[0]),
                "cy": float(cen[1]),
                "H": h,
                "area_m2": area,
                "n_facets": int(len(idx)),
                "z_min": float(z.min()),
            }
        )
        hulls.append(hull.astype(np.float32))
        high_hulls.append(high_hull.astype(np.float32))
    df = pd.DataFrame(rows).sort_values("H", ascending=False).reset_index(drop=True)
    hull_by_bid = {int(b): h for b, h in zip(ids, hulls)}
    high_by_bid = {int(b): h for b, h in zip(ids, high_hulls)}
    return df, hull_by_bid, high_by_bid


def select_tall(df: pd.DataFrame) -> pd.DataFrame:
    tower_bid = int(df.iloc[0]["bid"])  # already sorted by H desc
    xy = df[["cx", "cy"]].to_numpy()
    tree = cKDTree(xy)
    h_surr = np.full(len(df), np.nan)
    n_neigh = np.zeros(len(df), dtype=int)
    areas = df["area_m2"].to_numpy()
    heights = df["H"].to_numpy()
    for i, h in enumerate(heights):
        if h < H_MIN:
            continue
        idx = tree.query_ball_point(xy[i], NEIGH_RADIUS_FACTOR * h)
        idx = [j for j in idx if j != i and areas[j] > 1.0 and heights[j] > 1.0]
        n_neigh[i] = len(idx)
        if idx:
            h_surr[i] = np.average(heights[idx], weights=areas[idx])
    out = df.copy()
    out["n_neigh_2H"] = n_neigh
    out["H_surround"] = h_surr
    with np.errstate(divide="ignore", invalid="ignore"):
        out["I"] = out["H"] / out["H_surround"]
    out.loc[out["n_neigh_2H"] == 0, "I"] = np.inf
    out["is_canton_tower"] = out["bid"] == tower_bid
    isolated_enough = (out["n_neigh_2H"] == 0) | (out["H"] >= HEIGHT_FACTOR * out["H_surround"])
    out["selected"] = (out["H"] >= H_MIN) & isolated_enough & ~out["is_canton_tower"]
    return out


def save_hulls(path: Path, bids: np.ndarray, hull_by_bid: dict, high_by_bid: dict) -> None:
    hulls = [hull_by_bid[int(b)] for b in bids]
    highs = [high_by_bid[int(b)] for b in bids]
    lengths = np.array([len(h) for h in hulls], dtype=np.int32)
    high_lengths = np.array([len(h) for h in highs], dtype=np.int32)
    cat = np.vstack(hulls) if hulls else np.zeros((0, 2), np.float32)
    high_cat = np.vstack(highs) if highs else np.zeros((0, 2), np.float32)
    np.savez_compressed(
        path,
        bid=np.asarray(bids, dtype=np.int32),
        hull_len=lengths,
        hull_xy=cat.astype(np.float32),
        high_len=high_lengths,
        high_xy=high_cat.astype(np.float32),
    )


def qc_figure(df: pd.DataFrame, hull_by_bid: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    other = df[(df["H"] >= 15) & ~df["selected"] & ~df["is_canton_tower"]]
    ax.scatter(other["cx"], other["cy"], s=4, c="#b0b0b0", linewidths=0, label="other buildings (H ≥ 15 m)", zorder=1)
    sel = df[df["selected"]]
    for _, r in sel.iterrows():
        hull = hull_by_bid[int(r.bid)]
        if len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="#1f4e79", lw=0.6, zorder=2)
    sc = ax.scatter(
        sel["cx"],
        sel["cy"],
        c=np.clip(sel["I"].replace(np.inf, sel["I"].replace(np.inf, np.nan).max() * 1.05), 0, None),
        s=36,
        cmap="viridis",
        zorder=3,
        label="selected tall buildings",
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("isolation I = H / H_surround")
    tower = df[df["is_canton_tower"]]
    if len(tower):
        ax.scatter(tower["cx"], tower["cy"], marker="x", c="#b00020", s=80, zorder=4, label="Canton Tower (excluded)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Tall-building selection  n = {len(sel)}")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stl", type=Path, default=DEFAULT_STL)
    args = p.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.stl}")
    tris = load_stl(args.stl)
    print(f"triangles {len(tris)}")
    comp = connected_components(tris)
    print(f"components {comp.max() + 1}")
    df, hull_by_bid, high_by_bid = summarise(tris, comp)
    df = select_tall(df)

    counts = df["n_facets"].to_numpy()
    print(f"largest component facets {counts.max()}  Hmax {df['H'].max():.2f}")
    print("tallest 12:")
    print(df.head(12)[["bid", "cx", "cy", "H", "area_m2", "n_facets", "is_canton_tower"]].to_string(index=False))
    sel = df[df["selected"]].copy()
    print(f"selected {len(sel)}")
    print(sel[["bid", "cx", "cy", "H", "H_surround", "I", "n_neigh_2H", "area_m2"]].to_string(index=False))

    # Full catalogue is useful for downstream-building checks; hulls only for selected + tower.
    df.drop(columns=[]).to_csv(OUT_DATA / "buildings_components.csv", index=False)
    keep_cols = [
        "bid", "cx", "cy", "H", "area_m2", "n_facets", "n_neigh_2H", "H_surround", "I",
    ]
    sel[keep_cols].to_csv(OUT_DATA / "tall_buildings.csv", index=False)
    save_bids = np.r_[sel["bid"].to_numpy(), df.loc[df["is_canton_tower"], "bid"].to_numpy()]
    save_hulls(OUT_DATA / "tall_footprints.npz", save_bids, hull_by_bid, high_by_bid)
    qc_figure(df, hull_by_bid, OUT_FIG / "tall_buildings_qc.png")
    print(f"wrote {OUT_DATA / 'tall_buildings.csv'}")
    print(f"wrote {OUT_FIG / 'tall_buildings_qc.png'}")


if __name__ == "__main__":
    main()
