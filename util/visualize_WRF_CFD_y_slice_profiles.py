"""
WRF + CFD 垂直廓线对比（y 切片 zonal mean，WS / WD 双 panel）

在固定 y 位置（默认 y = -4995 m）对 x–z 二维场沿 x 做 zonal mean，
绘制 WRF 边界 + init_1000_run 各时刻 CFD 廓线 + 同时间戳 outlet 算例 t=5000。

Usage
-----
    # 20250903 00:00（19 时刻 + t5000 → 21 条）
    python visualize_WRF_CFD_y_slice_profiles.py \\
        steady_experiments_finer_ABL/20250903_0000_two_boundaries_as_outlet-init_1000_run

    # 20250903 15:00（init_1000_run 无 y-4995m 中间时刻，仅 WRF + outlet t5000）
    python visualize_WRF_CFD_y_slice_profiles.py \\
        steady_experiments_finer_ABL/20250903_1500_two_boundaries_as_outlet-init_1000_run
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import visualize_WRF_CFD_xz_at_time as vat
import visualize_WRF_CFD_xz_two_panel as v2p

RESULTS_PROFILE_DIR = os.path.join("results", "wrf_openfoam", "y_slice_profile")
Y_SLICE_TAG = "y-4995m"
CFD_CSV_GLOB = f"{Y_SLICE_TAG}_t*.csv"

# 赤橙黄绿青蓝紫… 色阶（亮黄改为暗金黄，白底可读）
CFD_BASE_COLORS = [
    "#E31A1C",  # 红
    "#FF7F00",  # 橙
    "#D4A017",  # 金黄
    "#33A02C",  # 绿
    "#00CED1",  # 青
    "#1F78B4",  # 蓝
    "#6A3D9A",  # 紫
    "#A65628",  # 棕
    "#999999",  # 灰
    "#4DAF4A",  # 浅绿补充
]
COLOR_T5000 = "#FF1493"  # DeepPink
WD_MIN_WS = 0.15         # m/s，统一阈值（WRF 近地层 WS 可低至 ~0.15）
WD_MIN_WS_LOW = WD_MIN_WS
WD_MIN_WS_HIGH = WD_MIN_WS
WD_HEIGHT_SPLIT = 99999.0  # 仅当 low≠high 时生效


@dataclass(frozen=True)
class ProfileSeries:
    label: str
    height: np.ndarray
    ws: np.ndarray
    wd: np.ndarray
    kind: str  # "wrf" | "cfd"


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _repo_root() -> str:
    return v2p._repo_root()


def _ws_wd_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ws = np.sqrt(u**2 + v**2)
    wd = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return ws, wd


def wd_min_ws_at_height(height: float,
                        min_ws_low: float = WD_MIN_WS_LOW,
                        min_ws_high: float = WD_MIN_WS_HIGH,
                        height_split: float = WD_HEIGHT_SPLIT) -> float:
    """近地面/高空可设不同阈值；默认二者相同（统一阈值）。"""
    if min_ws_low == min_ws_high:
        return min_ws_low
    return min_ws_low if height <= height_split else min_ws_high


def wd_profile_segments(height: np.ndarray,
                        wd: np.ndarray,
                        ws: np.ndarray,
                        min_ws_low: float = WD_MIN_WS_LOW,
                        min_ws_high: float = WD_MIN_WS_HIGH,
                        height_split: float = WD_HEIGHT_SPLIT) -> list[tuple[np.ndarray, np.ndarray]]:
    """将风向廓线拆成可绘制的连续段。

    1. WS < min_ws 时不绘制（静风区风向无意义）。
    2. 相邻层 |ΔWD| > 180° 处断开，避免 0/360° 边界产生横穿图面的伪折线。
    """
    h_arr = np.asarray(height, dtype=float)
    wd_arr = np.asarray(wd, dtype=float)
    ws_arr = np.asarray(ws, dtype=float)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    seg_h: list[float] = []
    seg_w: list[float] = []

    for i in range(len(h_arr)):
        if not (np.isfinite(h_arr[i]) and np.isfinite(wd_arr[i]) and np.isfinite(ws_arr[i])):
            if seg_h:
                segments.append((np.array(seg_h), np.array(seg_w)))
                seg_h, seg_w = [], []
            continue
        min_ws = wd_min_ws_at_height(h_arr[i], min_ws_low, min_ws_high, height_split)
        if ws_arr[i] < min_ws:
            if seg_h:
                segments.append((np.array(seg_h), np.array(seg_w)))
                seg_h, seg_w = [], []
            continue
        if seg_w and abs(wd_arr[i] - seg_w[-1]) > 180.0:
            segments.append((np.array(seg_h), np.array(seg_w)))
            seg_h, seg_w = [], []
        seg_h.append(h_arr[i])
        seg_w.append(wd_arr[i])

    if seg_h:
        segments.append((np.array(seg_h), np.array(seg_w)))
    return segments


def _zonal_mean_profile(height_2d: np.ndarray,
                        u_2d: np.ndarray,
                        v_2d: np.ndarray,
                        max_height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿 x（axis=1）对 x–z 场做 zonal mean，返回 (height, ws, wd)。"""
    mask = height_2d <= max_height
    u = np.where(mask, u_2d, np.nan)
    v = np.where(mask, v_2d, np.nan)
    h = np.where(mask, height_2d, np.nan)

    row_valid = np.any(np.isfinite(u) & np.isfinite(v), axis=1)
    u_mean = np.full(u.shape[0], np.nan)
    v_mean = np.full(v.shape[0], np.nan)
    h_mean = np.full(h.shape[0], np.nan)
    if np.any(row_valid):
        with np.errstate(invalid="ignore"):
            u_mean[row_valid] = np.nanmean(u[row_valid], axis=1)
            v_mean[row_valid] = np.nanmean(v[row_valid], axis=1)
            h_mean[row_valid] = np.nanmean(h[row_valid], axis=1)

    valid = (
        row_valid
        & np.isfinite(h_mean)
        & np.isfinite(u_mean)
        & np.isfinite(v_mean)
        & (h_mean <= max_height)
    )
    h_mean = h_mean[valid]
    u_mean = u_mean[valid]
    v_mean = v_mean[valid]

    order = np.argsort(h_mean)
    h_mean = h_mean[order]
    u_mean = u_mean[order]
    v_mean = v_mean[order]
    ws, wd = _ws_wd_from_uv(u_mean, v_mean)
    return h_mean, ws, wd


def extract_wrf_xz_with_v(nc_path: str,
                          target_lat=v2p.TARGET_LAT,
                          target_lon=v2p.TARGET_LON,
                          lat_tol=v2p.LAT_TOL,
                          lon_tol=v2p.LON_TOL,
                          max_height=v2p.MAX_HEIGHT):
    """与 ``extract_wrf_xz`` 相同，但额外返回 V（south_north）分量。"""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"WRF file not found: {nc_path}")

    import xarray as xr

    ds = xr.open_dataset(nc_path)

    def get_val(name):
        v = ds[name]
        return v.values[0] if "Time" in v.dims else v.values

    lats = get_val("XLAT")
    if lats.ndim == 3:
        lats = lats[0]
    lat_1d = np.mean(lats, axis=1)
    sn_idx = int(np.argmin(np.abs(lat_1d - target_lat)))

    lons = get_val("XLONG")
    if lons.ndim == 3:
        lons = lons[0]
    lon_1d = lons[sn_idx, :]
    we_mask = (target_lon - lon_tol <= lon_1d) & (lon_1d <= target_lon + lon_tol)
    we_idx = np.where(we_mask)[0]
    we_slice = slice(None) if len(we_idx) < 2 else slice(int(we_idx[0]), int(we_idx[-1]) + 1)

    PH_sn = get_val("PH")[:, sn_idx, :]
    PHB_sn = get_val("PHB")[:, sn_idx, :]
    U_sn = get_val("U")[:, sn_idx, :]
    V_sn = get_val("V")[:, sn_idx, :]
    V_sn1 = get_val("V")[:, sn_idx + 1, :]

    ds.close()

    H_sn = v2p._destagger_np((PH_sn + PHB_sn) / 9.81, axis=0)
    U_dest = v2p._destagger_np(U_sn, axis=1)
    V_dest = 0.5 * (V_sn + V_sn1)

    H_xz = H_sn[:, we_slice]
    U_xz = U_dest[:, we_slice]
    V_xz = V_dest[:, we_slice]
    return H_xz, U_xz, V_xz


def load_wrf_zonal_profile(nc_path: str, max_height: float = v2p.MAX_HEIGHT, **kwargs) -> ProfileSeries:
    height_2d, u_2d, v_2d = extract_wrf_xz_with_v(nc_path, max_height=max_height, **kwargs)
    h, ws, wd = _zonal_mean_profile(height_2d, u_2d, v_2d, max_height)
    return ProfileSeries(label="WRF", height=h, ws=ws, wd=wd, kind="wrf")


def load_cfd_zonal_profile(csv_path: str, max_height: float = v2p.MAX_HEIGHT) -> ProfileSeries:
    """读取 y 切片 CSV，沿 x 对 U:0/U:1 做 zonal mean。"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CFD CSV not found: {csv_path}")

    chunks = []
    req = ["Coords:0", "Coords:2", "U:0", "U:1"]
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        if any(c not in chunk.columns for c in req):
            raise KeyError(f"Missing columns in {csv_path}. Expected: {req}")
        sub = chunk[req].astype(float)
        sub = sub[sub["Coords:2"] <= max_height]
        chunks.append(sub)

    df = pd.concat(chunks, ignore_index=True)
    agg = (
        df.groupby("Coords:2", sort=True)
        .agg({"U:0": "mean", "U:1": "mean"})
        .reset_index()
        .rename(columns={"Coords:2": "height", "U:0": "u", "U:1": "v"})
    )
    h = agg["height"].to_numpy()
    u = agg["u"].to_numpy()
    v = agg["v"].to_numpy()
    ws, wd = _ws_wd_from_uv(u, v)

    _, time_str = vat.parse_time_from_csv(csv_path)
    return ProfileSeries(label=f"CFD t={time_str}", height=h, ws=ws, wd=wd, kind="cfd")


def infer_outlet_t5000_csv(cfd_dir: str) -> str | None:
    """从算例目录名推断同时间戳 ``*_two_boundaries_as_outlet`` 的 t5000 CSV。"""
    cfd_dir = os.path.abspath(cfd_dir)
    case = os.path.basename(cfd_dir.rstrip(os.sep))
    batch = os.path.dirname(cfd_dir)
    m = re.match(
        r"(\d{8}_\d{4})_(two_boundaries_as_outlet)(?:-init_1000_run)?$",
        case,
    )
    if not m:
        return None
    ts, tag = m.groups()
    csv_path = os.path.join(
        batch, f"{ts}_{tag}", "postProcessing", "y-4995m_t5000.csv",
    )
    return csv_path if os.path.isfile(csv_path) else None


def default_extra_cfd_csvs(cfd_dir: str) -> list[str]:
    """默认追加同时间戳 outlet 算例的 t=5000（若文件存在）。"""
    path = infer_outlet_t5000_csv(cfd_dir)
    return [path] if path else []


def resolve_cfd_csv_paths(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        path = p if os.path.isabs(p) else os.path.join(_repo_root(), p)
        out.append(os.path.abspath(path))
    return out


def merge_cfd_csvs(primary: list[str], extra: list[str]) -> list[str]:
    merged = list(primary)
    seen = {os.path.abspath(p) for p in merged}
    for p in extra:
        ap = os.path.abspath(p)
        if ap not in seen:
            merged.append(ap)
            seen.add(ap)
    return sorted(merged, key=_cfd_csv_sort_key)


def discover_cfd_csvs(post_dir: str, required: bool = False) -> list[str]:
    pattern = os.path.join(post_dir, CFD_CSV_GLOB)
    paths = sorted(glob.glob(pattern), key=_cfd_csv_sort_key)
    if not paths and required:
        raise FileNotFoundError(f"No CFD CSV matched: {pattern}")
    if not paths:
        warnings.warn(f"No CFD CSV matched: {pattern}")
    return paths


def _cfd_csv_sort_key(path: str) -> float:
    _, time_str = vat.parse_time_from_csv(path)
    return float(time_str)


def cfd_dir_from_post_dir(post_dir: str) -> str:
    post_dir = os.path.abspath(post_dir)
    if os.path.basename(post_dir) == "postProcessing":
        return os.path.dirname(post_dir)
    return post_dir


def default_output_path(cfd_dir: str) -> str:
    case = os.path.basename(cfd_dir.rstrip(os.sep))
    batch = os.path.basename(os.path.dirname(cfd_dir.rstrip(os.sep))) or "misc"
    return os.path.join(
        _repo_root(),
        RESULTS_PROFILE_DIR,
        batch,
        f"ws_wd_profile_{Y_SLICE_TAG}_{case}.png",
    )


def plot_profiles(profiles: Iterable[ProfileSeries],
                  output_path: str,
                  y_tag: str = Y_SLICE_TAG,
                  case_label: str = "",
                  max_height: float = v2p.MAX_HEIGHT,
                  wd_min_ws_low: float = WD_MIN_WS_LOW,
                  wd_min_ws_high: float = WD_MIN_WS_HIGH,
                  wd_height_split: float = WD_HEIGHT_SPLIT) -> None:
    configure_matplotlib_style()
    profiles = list(profiles)
    n = len(profiles)

    fig, (ax_ws, ax_wd) = plt.subplots(
        1, 2, figsize=(11.5, 6.2), constrained_layout=True,
    )

    cfd_counter = 0

    for prof in profiles:
        if prof.kind == "wrf":
            color = "#1a1a1a"
            ls = "--"
            lw = 2.2
            alpha = 1.0
        elif "t=5000" in prof.label:
            color = COLOR_T5000
            ls = "-"
            lw = 2.5
            alpha = 1.0
        else:
            c_idx = cfd_counter // 2
            ls_idx = cfd_counter % 2
            color = CFD_BASE_COLORS[c_idx % len(CFD_BASE_COLORS)]
            ls = "-" if ls_idx == 0 else "--"
            lw = 1.5
            alpha = 0.85
            cfd_counter += 1

        ax_ws.plot(prof.ws, prof.height, color=color, ls=ls, lw=lw, alpha=alpha, label=prof.label)

        wd_segments = wd_profile_segments(
            prof.height, prof.wd, prof.ws,
            min_ws_low=wd_min_ws_low,
            min_ws_high=wd_min_ws_high,
            height_split=wd_height_split,
        )
        for seg_i, (seg_h, seg_w) in enumerate(wd_segments):
            ax_wd.plot(
                seg_w, seg_h,
                color=color, ls=ls, lw=lw, alpha=alpha,
                label=prof.label if seg_i == 0 else "_nolegend_",
            )

    ax_ws.set_xlim(left=0)
    ax_ws.set_ylim(0, max_height)
    ax_ws.set_xlabel("WS (m s$^{-1}$)")
    ax_ws.set_ylabel("Height (m)")
    ax_ws.set_title("Wind Speed")

    ax_wd.set_xlim(0, 360)
    ax_wd.set_xticks([0, 90, 180, 270, 360])
    ax_wd.set_xticklabels(["N", "E", "S", "W", "N"])
    ax_wd.set_ylim(0, max_height)
    ax_wd.set_xlabel("WD (°)")
    ax_wd.set_title("Wind Direction")
    ax_wd.set_yticklabels([])

    title_case = case_label or "WRF + CFD"
    fig.suptitle(
        f"Zonal-Mean WS/WD Profiles at {y_tag}\n{title_case}  ({n} profiles)",
        fontsize=13,
        fontweight="bold",
    )

    handles, labels = ax_ws.get_legend_handles_labels()
    # 将图例移至整个图表正下方，缩短与 WS 的距离，使用多列排版使整体更紧凑
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
        frameon=True,
        ncol=5,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nDONE: Figure saved -> {output_path}  (300 DPI, {n} profiles)\n")


def build_parser() -> argparse.ArgumentParser:
    default_case = os.path.join(
        _repo_root(),
        "steady_experiments_finer_ABL",
        "20250903_0000_two_boundaries_as_outlet-init_1000_run",
    )
    p = argparse.ArgumentParser(
        description="Plot zonal-mean WS/WD profiles at y slice (1 WRF + N CFD times)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "cfd_dir",
        nargs="?",
        default=default_case,
        help="CFD case directory (postProcessing/y-4995m_t*.csv inside)",
    )
    p.add_argument(
        "--post-dir",
        default=None,
        help="Override postProcessing directory (default: <cfd_dir>/postProcessing)",
    )
    p.add_argument(
        "--extra-cfd",
        action="append",
        default=None,
        help="追加 CFD CSV（可多次指定）；默认含同时间戳 outlet 算例 y-4995m_t5000.csv",
    )
    p.add_argument("--no-extra-cfd", action="store_true",
                   help="不加载默认 extra CFD（t=5000）")
    p.add_argument("--output", default=None, help="Output PNG path")
    p.add_argument("--wrf-nc", default=None, help="Override WRF NetCDF path")
    p.add_argument("--no-wrf", action="store_true", help="Skip WRF profile")
    p.add_argument("--max-height", type=float, default=v2p.MAX_HEIGHT)
    p.add_argument(
        "--wd-min-ws", type=float, default=WD_MIN_WS,
        help="绘制风向的最小 WS (m/s) (default: 0.15)",
    )
    p.add_argument(
        "--wd-min-ws-low", type=float, default=None,
        help="近地面 WS 阈值；默认与 --wd-min-ws 相同",
    )
    p.add_argument(
        "--wd-min-ws-high", type=float, default=None,
        help="高空 WS 阈值；默认与 --wd-min-ws 相同；设不同于 low 时启用高度分界",
    )
    p.add_argument(
        "--wd-height-split", type=float, default=600.0,
        help="近地面/高空 WS 阈值分界高度 (m)；仅 low≠high 时生效 (default: 600)",
    )
    p.add_argument("--lat", type=float, default=v2p.TARGET_LAT)
    p.add_argument("--lon", type=float, default=v2p.TARGET_LON)
    p.add_argument("--lat-tol", type=float, default=v2p.LAT_TOL)
    p.add_argument("--lon-tol", type=float, default=v2p.LON_TOL)
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfd_dir = os.path.abspath(args.cfd_dir)
    post_dir = os.path.abspath(args.post_dir or os.path.join(cfd_dir, "postProcessing"))
    case_label = os.path.basename(cfd_dir.rstrip(os.sep))
    output_path = args.output or default_output_path(cfd_dir)

    cfd_csvs = discover_cfd_csvs(post_dir)
    if args.no_extra_cfd:
        extra_csvs: list[str] = resolve_cfd_csv_paths(args.extra_cfd or [])
    else:
        extra_csvs = resolve_cfd_csv_paths(
            (args.extra_cfd or []) + default_extra_cfd_csvs(cfd_dir)
        )
    cfd_csvs = merge_cfd_csvs(cfd_csvs, extra_csvs)
    if not cfd_csvs:
        raise FileNotFoundError(
            "No CFD profile CSV found. Expected y-4995m_t*.csv under postProcessing "
            "and/or a matching outlet y-4995m_t5000.csv. "
            "Use --extra-cfd to add paths explicitly."
        )
    wrf_nc_path, _, wrf_time = v2p.infer_paths(cfd_dir)
    if args.wrf_nc:
        wrf_nc_path = args.wrf_nc
    if not os.path.isabs(wrf_nc_path):
        wrf_nc_path = os.path.join(_repo_root(), wrf_nc_path)

    print("=" * 64)
    print("  WRF + CFD y-slice zonal-mean profiles")
    print("=" * 64)
    print(f"  CFD case     : {cfd_dir}")
    print(f"  postProcessing: {post_dir}")
    print(f"  CFD CSVs     : {len(cfd_csvs)} files")
    print(f"  WRF nc file  : {wrf_nc_path}")
    print(f"  Output       : {output_path}")
    print("=" * 64)

    profiles: list[ProfileSeries] = []

    if not args.no_wrf:
        if not os.path.exists(wrf_nc_path):
            warnings.warn(f"WRF file not found: {wrf_nc_path}; skipping WRF profile.")
        else:
            print(f"\n[WRF] Loading zonal profile ... ({wrf_time})")
            profiles.append(
                load_wrf_zonal_profile(
                    wrf_nc_path,
                    max_height=args.max_height,
                    target_lat=args.lat,
                    target_lon=args.lon,
                    lat_tol=args.lat_tol,
                    lon_tol=args.lon_tol,
                )
            )

    for i, csv_path in enumerate(cfd_csvs, 1):
        print(f"[CFD {i:02d}/{len(cfd_csvs)}] {os.path.basename(csv_path)}")
        profiles.append(load_cfd_zonal_profile(csv_path, max_height=args.max_height))

    if not profiles:
        raise RuntimeError("No profiles loaded.")

    wd_min_ws_low = args.wd_min_ws_low if args.wd_min_ws_low is not None else args.wd_min_ws
    wd_min_ws_high = args.wd_min_ws_high if args.wd_min_ws_high is not None else args.wd_min_ws

    print("\nRendering figure ...")
    plot_profiles(
        profiles,
        output_path=output_path,
        case_label=case_label,
        max_height=args.max_height,
        wd_min_ws_low=wd_min_ws_low,
        wd_min_ws_high=wd_min_ws_high,
        wd_height_split=args.wd_height_split,
    )


if __name__ == "__main__":
    main()
