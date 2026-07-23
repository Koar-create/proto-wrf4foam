"""按单站绘制 WS/WD 垂直廓线（4 个 UTC 整点横排/图）。"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import campaign_config as cfg

warnings.filterwarnings("ignore")

# ─── 路径与阈值 ───────────────────────────────────────────────────────────────
REPO_ROOT = cfg.REPO_ROOT
DATA_PATH = cfg.DATA_PATH
OUTPUT_DIR = REPO_ROOT / "results/ws_wd_station_profile" / cfg.RESULTS_TAG / "by_station"
TZ_TAG = "utc"  # 数据筛选与 CSV datetime 列一致（UTC）

WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0
DEFAULT_ZMAX = 1000

METRIC_START = cfg.METRIC_START
METRIC_END = cfg.METRIC_END
METRIC_DATETIMES = cfg.METRIC_DATETIMES
TIME_LABELS = cfg.TIME_LABELS

COLOR_OBS = "#1a1a2e"
COLOR_WRF = "#e07b39"
COLOR_CFD = "#2196a5"


def metric_utc_days() -> list[int]:
    return cfg.metric_utc_days()


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


def resolve_out_dir(zmax: int, out_dir: Path | None = None) -> Path:
    """默认 zmax=1000 写到 by_station；其它 zmax 写到 by_station/z{zmax}。"""
    base = out_dir or OUTPUT_DIR
    if zmax == DEFAULT_ZMAX:
        return base
    return base / f"z{int(zmax)}"


def plot_one_station_day(
    df: pd.DataFrame,
    site: str,
    day: int,
    out_dir: Path,
    hours: list[int] | None = None,
    zmax: int = DEFAULT_ZMAX,
) -> Path | None:
    hours = list(hours or cfg.SYNOPTIC_UTC_HOURS)
    if not _has_any_hour(day, hours):
        return None

    height_bins = np.arange(0, zmax + 50, 50)
    fig, axes = plt.subplots(
        1,
        2 * len(hours),
        figsize=(4.5 * len(hours), 3.2),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05},
    )
    if len(hours) == 1:
        axes = np.array([axes])

    plotted = False
    showed_height_ticks = False
    for i_h, hour in enumerate(hours):
        t_raw = f"2025-09-{day:02d} {hour:02d}:00:00"
        time_label = TIME_LABELS.get(t_raw, t_raw)
        ax_ws = axes[2 * i_h]
        ax_wd = axes[2 * i_h + 1]
        sub = df[
            (df["obtid"] == site)
            & (df["time_label"] == time_label)
            & df["qc_ok"]
            & (df["Height"] <= zmax)
        ]

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
        ax_ws.set_ylim(0, zmax)
        ax_ws.set_xlim(left=0)
        ax_ws.set_title(time_label, fontsize=9, fontweight="bold")
        ax_ws.set_xlabel("WS (m s$^{-1}$)", fontsize=9)
        # sharey=True 下 set_yticklabels([]) 会清掉整组共享刻度文字；只用 tick_params 控制可见性
        if not showed_height_ticks:
            ax_ws.set_ylabel("Height (m)", fontsize=9)
            ax_ws.tick_params(axis="y", labelleft=True)
            showed_height_ticks = True
        else:
            ax_ws.tick_params(axis="y", labelleft=False)

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
        ax_wd.set_ylim(0, zmax)
        ax_wd.set_xlim(0, 360)
        ax_wd.set_xticks([0, 90, 180, 270, 360])
        ax_wd.set_xticklabels(["N", "E", "S", "W", "N"], fontsize=8)
        ax_wd.tick_params(axis="y", labelleft=False)
        ax_wd.set_xlabel("WD (°)", fontsize=9)
        if i_h == len(hours) - 1:
            ax_wd.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1.0))

    if not plotted:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Wind Speed & Direction Profiles - Site {site} on 2025-09-{day:02d} "
        f"(synoptic UTC {','.join(f'{h:02d}00' for h in hours)}; zmax={zmax} m)",
        fontsize=13,
        fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hour_tag = "-".join(f"{h:02d}00" for h in hours)
    if zmax == DEFAULT_ZMAX:
        fname = f"fig1_ws_wd_09{day:02d}_{site}_{hour_tag}.png"
    else:
        fname = f"fig1_ws_wd_09{day:02d}_{site}_{hour_tag}_zmax{int(zmax)}.png"
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_ws_wd_station_profiles_by_station(
    df: pd.DataFrame,
    out_dir: Path,
    sites: list[str] | None = None,
    days: list[int] | None = None,
    utc_hours: list[int] | None = None,
    zmax: int = DEFAULT_ZMAX,
) -> list[Path]:
    selected_sites = sites or sorted(df["obtid"].dropna().unique())
    selected_days = days or metric_utc_days()
    selected_hours = utc_hours or list(cfg.SYNOPTIC_UTC_HOURS)

    outputs = []
    for site in selected_sites:
        for day in selected_days:
            save_path = plot_one_station_day(
                df, site, day, out_dir, hours=selected_hours, zmax=zmax,
            )
            if save_path is not None:
                outputs.append(save_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-station WS/WD vertical profiles at synoptic UTC hours.",
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="Merged LiDAR/WRF/OpenFOAM CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory override. Default: by_station (zmax=1000) or by_station/z{zmax}.",
    )
    parser.add_argument(
        "--zmax",
        type=int,
        default=DEFAULT_ZMAX,
        help=f"Y-axis height max (m). Default {DEFAULT_ZMAX}; non-default saves under by_station/z{{zmax}}.",
    )
    parser.add_argument("--stations", nargs="+", default=None, help="Station IDs, e.g. GAW103 GAW111.")
    parser.add_argument("--days", nargs="+", type=int, default=None, help="UTC day numbers in September 2025.")
    parser.add_argument(
        "--utc-hours",
        nargs="+",
        type=int,
        default=None,
        help="Synoptic UTC hours to plot, default: 0 6 12 18.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_style()
    out_dir = resolve_out_dir(args.zmax, args.out_dir)
    df = quality_control(load_and_preprocess(args.csv))
    outputs = plot_ws_wd_station_profiles_by_station(
        df,
        out_dir,
        sites=args.stations,
        days=args.days,
        utc_hours=args.utc_hours,
        zmax=args.zmax,
    )
    print(f"Saved {len(outputs)} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
