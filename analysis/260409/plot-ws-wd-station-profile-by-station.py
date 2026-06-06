"""按单站绘制 WS/WD 垂直廓线（4 个 UTC 整点横排/图）。"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 路径与阈值 ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/260409/processed/merged_lidar_simulation_final.csv"
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_station_profile/260409/by_station"
TZ_TAG = "utc"  # 数据筛选与 CSV datetime 列一致（UTC）

WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0

METRIC_START = "2025-09-01 00:00:00"
METRIC_END = "2025-09-05 23:00:00"
METRIC_DATETIMES = pd.date_range(METRIC_START, METRIC_END, freq="h")
TIME_LABELS = {
    dt.strftime("%Y-%m-%d %H:%M:%S"): f"{dt.day:02d}_{dt.strftime('%H00')} UTC"
    for dt in METRIC_DATETIMES
}

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"


def metric_utc_days() -> list[int]:
    return sorted({dt.day for dt in METRIC_DATETIMES})


def _has_any_hour(day: int, hours: list[int]) -> bool:
    keys = set(TIME_LABELS)
    return any(f"2025-09-{day:02d} {h:02d}:00:00" in keys for h in hours)


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
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["wd_obs"] = np.degrees(np.arctan2(-df["u_obs"], -df["v_obs"])) % 360
    df["wd_wrf"] = np.degrees(np.arctan2(-df["u_wrf"], -df["v_wrf"])) % 360
    df["wd_cfd"] = np.degrees(np.arctan2(-df["u_cfd"], -df["v_cfd"])) % 360
    df["time_label"] = df["datetime"].astype(str).map(TIME_LABELS)
    return df


def quality_control(
    df: pd.DataFrame,
    ws_max_obs: float = WS_MAX_OBS,
    ws_max_cfd: float = WS_MAX_CFD,
) -> pd.DataFrame:
    out = df.copy()
    obs_ok = (out["ws_obs"] <= ws_max_obs) | out["ws_obs"].isna()
    cfd_ok = out["ws_cfd"] <= ws_max_cfd
    out["qc_ok"] = obs_ok & cfd_ok
    return out


def aggregate_profile(sub: pd.DataFrame, height_bins: np.ndarray) -> pd.DataFrame:
    sub = sub.copy()
    sub["H_bin"] = pd.cut(sub["Height"], bins=height_bins)
    for prefix in ("obs", "wrf", "cfd"):
        sub[f"u_{prefix}_dir"] = -np.sin(np.radians(sub[f"wd_{prefix}"]))
        sub[f"v_{prefix}_dir"] = -np.cos(np.radians(sub[f"wd_{prefix}"]))

    agg = sub.groupby("H_bin", observed=True).agg({
        "ws_obs": "mean",
        "ws_wrf": "mean",
        "ws_cfd": "mean",
        "u_obs_dir": "mean",
        "u_wrf_dir": "mean",
        "u_cfd_dir": "mean",
        "v_obs_dir": "mean",
        "v_wrf_dir": "mean",
        "v_cfd_dir": "mean",
        "Height": "mean",
    }).dropna(subset=["Height"]).reset_index()

    for prefix in ("obs", "wrf", "cfd"):
        agg[f"wd_{prefix}"] = (
            np.degrees(np.arctan2(-agg[f"u_{prefix}_dir"], -agg[f"v_{prefix}_dir"])) + 360
        ) % 360

    agg["ws_obs_std"] = agg["H_bin"].map(
        sub.groupby("H_bin", observed=True)["ws_obs"].std().fillna(0)
    ).to_numpy()
    return agg


def plot_one_station_window(
    df: pd.DataFrame,
    site: str,
    day: int,
    h_start: int,
    out_dir: Path,
) -> Path | None:
    hours = [h_start + i for i in range(4)]
    if not _has_any_hour(day, hours):
        return None

    height_bins = np.arange(0, 2150, 50)
    fig, axes = plt.subplots(
        1,
        8,
        figsize=(18, 3.2),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05},
    )

    plotted = False
    for i_h, hour in enumerate(hours):
        t_raw = f"2025-09-{day:02d} {hour:02d}:00:00"
        time_label = TIME_LABELS.get(t_raw, t_raw)
        ax_ws = axes[2 * i_h]
        ax_wd = axes[2 * i_h + 1]
        sub = df[(df["obtid"] == site) & (df["time_label"] == time_label) & df["qc_ok"]]

        if sub.empty:
            ax_ws.axis("off")
            ax_wd.axis("off")
            continue

        plotted = True
        agg = aggregate_profile(sub, height_bins)

        ax_ws.errorbar(
            agg["ws_obs"],
            agg["Height"],
            xerr=agg["ws_obs_std"],
            fmt="o",
            ms=2.5,
            color=COLOR_OBS,
            alpha=0.85,
            elinewidth=0.6,
            capsize=1.5,
            label="LiDAR",
        )
        ax_ws.plot(agg["ws_wrf"], agg["Height"], color=COLOR_WRF, lw=1.5, ls="--", label="WRF")
        ax_ws.plot(agg["ws_cfd"], agg["Height"], color=COLOR_CFD, lw=1.8, ls="-", label="OpenFOAM")
        ax_ws.set_ylim(0, 2100)
        ax_ws.set_xlim(left=0)
        ax_ws.set_title(time_label, fontsize=9, fontweight="bold")
        ax_ws.set_xlabel("WS (m s$^{-1}$)", fontsize=9)
        if i_h == 0:
            ax_ws.set_ylabel("Height (m)", fontsize=9)
        else:
            ax_ws.set_yticklabels([])

        ax_wd.scatter(
            sub["wd_obs"],
            sub["Height"],
            s=1,
            color=COLOR_OBS,
            alpha=0.15,
            edgecolors="none",
        )
        ax_wd.plot(agg["wd_obs"], agg["Height"], "o", ms=2.5, color=COLOR_OBS, alpha=0.9, label="LiDAR")
        ax_wd.plot(agg["wd_wrf"], agg["Height"], color=COLOR_WRF, lw=1.5, ls="--", label="WRF")
        ax_wd.plot(agg["wd_cfd"], agg["Height"], color=COLOR_CFD, lw=1.8, ls="-", label="OpenFOAM")
        ax_wd.set_ylim(0, 2100)
        ax_wd.set_xlim(0, 360)
        ax_wd.set_xticks([0, 90, 180, 270, 360])
        ax_wd.set_xticklabels(["N", "E", "S", "W", "N"], fontsize=8)
        ax_wd.set_yticklabels([])
        ax_wd.set_xlabel("WD (°)", fontsize=9)
        if i_h == len(hours) - 1:
            ax_wd.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1.0))

    if not plotted:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Wind Speed & Direction Profiles - Site {site} on 2025-09-{day:02d} "
        f"({h_start:02d}00-{h_start + 3:02d}00 UTC)",
        fontsize=13,
        fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"fig1_ws_wd_09{day:02d}_{site}_{h_start:02d}00-{h_start + 3:02d}00.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_ws_wd_station_profiles_by_station(
    df: pd.DataFrame,
    out_dir: Path,
    sites: list[str] | None = None,
    days: list[int] | None = None,
    window_starts: list[int] | None = None,
) -> list[Path]:
    selected_sites = sites or sorted(df["obtid"].dropna().unique())
    selected_days = days or metric_utc_days()
    selected_windows = window_starts or list(range(0, 24, 4))

    outputs = []
    for site in selected_sites:
        for day in selected_days:
            for h_start in selected_windows:
                save_path = plot_one_station_window(df, site, day, h_start, out_dir)
                if save_path is not None:
                    outputs.append(save_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-station WS/WD vertical profiles in 4-hour UTC windows.",
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="Merged LiDAR/WRF/OpenFOAM CSV.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--stations", nargs="+", default=None, help="Station IDs, e.g. GAW103 GAW111.")
    parser.add_argument("--days", nargs="+", type=int, default=None, help="UTC day numbers in September 2025.")
    parser.add_argument(
        "--window-starts",
        nargs="+",
        type=int,
        default=None,
        help="UTC window start hours, e.g. 0 4 8 12 16 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_style()
    df = quality_control(load_and_preprocess(args.csv))
    outputs = plot_ws_wd_station_profiles_by_station(
        df,
        args.out_dir,
        sites=args.stations,
        days=args.days,
        window_starts=args.window_starts,
    )
    print(f"Saved {len(outputs)} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
