"""Height alignment (LiDAR AGL ↔ CFD z) and SI table S1 roof geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paths import CACHE, OUT, PROBE_Z, STATIONS


def _df_to_md(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


LIDAR_GATES_HEIGHT = None  # instrument-relative Height, 52 m + 13 m


def z_agl(height: np.ndarray, station: str) -> np.ndarray:
    return np.asarray(height, float) + STATIONS[station]["h0"]


def lidar_gates_in_cfd_range(station: str, zmin: float = 60.0, zmax: float = 600.0):
    """Unique LiDAR Height values whose AGL sits in [zmin, zmax]."""
    h = np.arange(52.0, 2000.0, 13.0)
    z = h + STATIONS[station]["h0"]
    sel = (z >= zmin) & (z <= zmax)
    return h[sel], z[sel]


def interp_column(z_src, val_src, z_tgt, axis=0):
    """Linear interpolation of a vertical profile onto target AGL heights."""
    z_src = np.asarray(z_src, float)
    val_src = np.asarray(val_src, float)
    z_tgt = np.asarray(z_tgt, float)
    out_shape = list(val_src.shape)
    out_shape[axis] = z_tgt.size
    out = np.empty(out_shape, dtype=float)
    # move profile axis to last
    v = np.moveaxis(val_src, axis, -1)
    flat = v.reshape(-1, v.shape[-1])
    acc = np.empty((flat.shape[0], z_tgt.size), dtype=float)
    for i, row in enumerate(flat):
        acc[i] = np.interp(z_tgt, z_src, row, left=np.nan, right=np.nan)
    out = np.moveaxis(acc.reshape(*v.shape[:-1], z_tgt.size), -1, axis)
    return out


def window_profile_stats(t, U, t0, t1):
    """Per-probe window mean U and component/horizontal sigma. U: (nt, 33, 3)."""
    sel = (t >= t0) & (t <= t1)
    u = U[sel]
    um = u.mean(axis=0)
    up = u - um
    sig = up.std(axis=0, ddof=1)
    ws = np.hypot(u[..., 0], u[..., 1])
    return dict(
        U_mean=um,
        sigma=sig,
        ws_mean_scalar=ws.mean(axis=0),
        ws_mean_vector=np.hypot(um[:, 0], um[:, 1]),
        sigma_ws=ws.std(axis=0, ddof=1),
        sigma_u=sig[:, 0],
        sigma_v=sig[:, 1],
        sigma_w=sig[:, 2],
        n=int(sel.sum()),
    )


def station_column(U_or_stat, name: str):
    s = STATIONS[name]
    if U_or_stat.ndim == 1:
        return U_or_stat[s["i0"]:s["i1"]]
    return U_or_stat[..., s["i0"]:s["i1"], :]


def write_s1(roof, path: Path | None = None) -> pd.DataFrame:
    rows = []
    for name, meta in STATIONS.items():
        x, y = meta["xy"]
        h_pt = roof.query_roof_height(x, y)
        h30 = roof.max_in_radius(x, y, 30.0, step=5.0)
        h60 = roof.max_in_radius(x, y, 60.0, step=5.0)
        z_gates_h, z_gates = lidar_gates_in_cfd_range(name)
        n_below_roof = int(np.sum(z_gates < h_pt)) if h_pt > 0 else 0
        rows.append(dict(
            station=name,
            x=x, y=y,
            h0_instrument_m=meta["h0"],
            roof_at_xy_m=h_pt,
            roof_max_r30_m=h30,
            roof_max_r60_m=h60,
            lowest_lidar_zAGL_m=float(z_gates[0]),
            n_gates_60_600=int(z_gates.size),
            n_gates_below_roof_at_xy=n_below_roof,
            note=(
                "roof query = 0 (instrument not on a building footprint in the STL)"
                if h_pt <= 0 else
                "non-zero roof at the CFD (x,y); lowest gates may sit in the roughness sublayer"
            ),
        ))
    df = pd.DataFrame(rows)
    path = path or (OUT / "S1_roof_heights.csv")
    df.to_csv(path, index=False)
    md = path.with_suffix(".md")
    lines = [
        "# S1 站点屋面高度与仪器高度几何核对",
        "",
        "CFD 地面取 z = 0。LiDAR 的 `Height` 为仪器以上高度；AGL = Height + h0。",
        "屋面高度由 `buildings.stl` 经 `BuildingRoofIndex.query_roof_height` 查询。",
        "对比范围限制为 z_AGL ∈ [60, 600] m。GAW111 最低门约为 83 m AGL，该站 60 m 与 80 m 探针在观测中没有对应门。",
        "",
        _df_to_md(df),
        "",
        "GAW111 在 CFD (x, y) 处的 STL 屋面高度为 18.5 m，低于仪器高度 30.9 m，说明该点要么未落在真实楼顶足迹上，要么 STL 低估了屋面。",
        "三个站点在 [60, 600] m 对比窗内均没有“门中心低于该点屋面”的情况；GAW103 在 r = 60 m 内存在 122 m 建筑，GAW104 在 r = 30 m 内存在 92.5 m 建筑，解释虚拟 DBS 误差时需考虑邻近建筑。",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return df


def save_alignment_meta():
    rows = []
    for name in STATIONS:
        h, z = lidar_gates_in_cfd_range(name)
        for hh, zz in zip(h, z):
            rows.append(dict(station=name, Height=hh, z_AGL=zz,
                             has_les_probe=bool(np.min(np.abs(np.asarray(PROBE_Z) - zz)) < 1.0)))
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "lidar_gate_alignment.csv", index=False)
    return df
