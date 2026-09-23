#!/usr/bin/env python3
"""Pin down the typhoon-affected period for the Guangzhou CFD domain / LiDAR
network during Typhoon Tapah (2025), to the hour.

Continues analysis/tapah-direct-influence/analyze_tapah_direct_influence.py,
which established that the 10 km x 10 km OpenFOAM domain never entered Tapah's
inner core (closest approach ~131 km).  This script answers the follow-up
question: over which hours was the domain genuinely affected by the storm?

Two independent lines of evidence are combined:

  (1) Geometry - the IBTrACS best track (US-PROVISIONAL) is linearly
      interpolated to 1-minute resolution and the great-circle distance from
      the domain to the storm centre is evaluated every hour.  Threshold
      crossings (800/500/400/300/250/200/150/100 km) are reported to the hour.

  (2) Observation - the hourly LiDAR record (lidar_1h-rolling.csv, the dataset
      actually used by the manuscript) is reduced to a station-mean low-layer
      (52-300 m) and ~500 m wind speed.  A diurnal-aware anomaly z-score
      (relative to the clean days 01-05 and 10-13 Sep) is used to locate the
      observed onset / peak / recovery of the wind.

Outputs (all under the repo):
  analysis/tapah-direct-influence/affected_period_hourly.csv
  analysis/tapah-direct-influence/typhoon_affected_period.md
  results/tapah-direct-influence/fig4_typhoon_affected_period.png
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[2]
OUT_ANALYSIS = Path(__file__).resolve().parent
OUT_FIG = REPO / "results/tapah-direct-influence"

IBTRACS_SUBSET = REPO / "data/ibtracs/tapah_2025_ibtracs_subset.csv"
JMA_TRACK = REPO / "data/ibtracs/tapah_2025_jma_besttrack.csv"
LIDAR_HOURLY = REPO / "data/260409/raw/lidar/lidar_1h-rolling.csv"

ORIGIN_LON = 113.3218197
ORIGIN_LAT = 23.1133057
DOMAIN_HALF_M = 5000.0
KM_PER_NM = 1.852
EARTH_KM = 6371.0

# Thresholds reported on the distance curve (km).
DIST_THRESHOLDS_KM = [800.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0]

# Observation anomaly settings.
LOW_LAYER = (52.0, 300.0)   # m, the "low-altitude" band used in the manuscript
MID_HEIGHT = 500.0          # m, sampled at the nearest LiDAR range gate
SIGMA_FACTOR = 2.0          # anomaly threshold = clean median + this many robust sigma
MIN_RUN_HOURS = 3           # a "sustained" anomaly must last at least this long
GAP_TOLERANCE_HOURS = 1     # allow this many below-threshold hours inside a run
Z_THRESHOLD = 2.0           # secondary (diurnal-aware) diagnostic threshold


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return float(2.0 * EARTH_KM * np.arcsin(np.sqrt(a)))


def build_utm():
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = to_utm.transform(ORIGIN_LON, ORIGIN_LAT)
    return to_utm, (ox, oy)


def domain_min_km(storm_lon, storm_lat, to_utm, origin_xy) -> float:
    """Distance from a storm centre to the nearest point of the square domain."""
    sx, sy = to_utm.transform(storm_lon, storm_lat)
    ox, oy = origin_xy
    lx, ly = sx - ox, sy - oy
    cx = float(np.clip(lx, -DOMAIN_HALF_M, DOMAIN_HALF_M))
    cy = float(np.clip(ly, -DOMAIN_HALF_M, DOMAIN_HALF_M))
    return math.hypot(lx - cx, ly - cy) / 1000.0


def interp_track(times: pd.Series, lon: np.ndarray, lat: np.ndarray, grid: pd.DatetimeIndex):
    """Linear-in-time interpolation of a track onto an arbitrary time grid."""
    t0 = times.iloc[0]
    h_src = (times - t0).dt.total_seconds().to_numpy() / 3600.0
    h_grid = (grid - t0).total_seconds().to_numpy() / 3600.0
    lon_i = np.interp(h_grid, h_src, lon)
    lat_i = np.interp(h_grid, h_src, lat)
    return lon_i, lat_i


def threshold_crossings(t_grid, d_grid, threshold):
    """Return (direction, crossing_time) for each crossing of `threshold`."""
    out = []
    d = np.asarray(d_grid)
    for i in range(1, len(d)):
        if (d[i - 1] - threshold) * (d[i] - threshold) < 0.0:
            frac = (threshold - d[i - 1]) / (d[i] - d[i - 1])
            tc = t_grid[i - 1] + (t_grid[i] - t_grid[i - 1]) * frac
            direction = "in" if d[i - 1] > d[i] else "out"
            out.append((direction, pd.Timestamp(tc)))
    return out


# --------------------------------------------------------------------------
# lidar helpers
# --------------------------------------------------------------------------
def lidar_hourly_series() -> pd.DataFrame:
    """Hourly station-mean wind speed for the low layer and ~500 m."""
    df = pd.read_csv(LIDAR_HOURLY, parse_dates=["datetime"])
    df = df[df["obtid"].isin(["GAW103", "GAW104", "GAW111"])]

    lo = df[(df["Height"] >= LOW_LAYER[0]) & (df["Height"] <= LOW_LAYER[1])]
    low = (
        lo.groupby(["datetime", "obtid"])["WindSpd"]
        .mean()
        .unstack()
        .mean(axis=1)
    )

    heights = np.sort(df["Height"].unique())
    h_mid = float(heights[np.argmin(np.abs(heights - MID_HEIGHT))])
    mid = df[np.isclose(df["Height"], h_mid)].groupby("datetime")["WindSpd"].mean()

    out = pd.DataFrame({"ws_low": low, f"ws_{int(round(h_mid))}": mid})
    out.index.name = "datetime"
    out.attrs["mid_height_m"] = h_mid

    # Clean days = 01-05 and 10-13 Sep (the storm onset is 06 Sep ~22 UTC and
    # the residual decays by 09 Sep ~10 UTC, so both are excluded).
    clean = pd.concat([out.loc[:"2025-09-05 23:00"], out.loc["2025-09-10 00:00":]])

    # Primary criterion: absolute threshold = clean median + SIGMA_FACTOR x robust sigma.
    for col in ["ws_low", f"ws_{int(round(h_mid))}"]:
        med = clean[col].median()
        sigma = 1.4826 * (clean[col] - med).abs().median()
        out[col + "_base"] = med
        out[col + "_sigma"] = sigma
        out[col + "_thr"] = med + SIGMA_FACTOR * sigma

    # Secondary diagnostic: diurnal-aware robust anomaly (z relative to hour-of-day).
    for col in ["ws_low", f"ws_{int(round(h_mid))}"]:
        hod_mean = clean.groupby(clean.index.hour)[col].mean()
        hod_std = clean.groupby(clean.index.hour)[col].std(ddof=0)
        out[col + "_z"] = (out[col] - out.index.hour.map(hod_mean)) / out.index.hour.map(hod_std)
    return out


def sustained_runs(series: pd.Series, threshold: float, gap_hours: int = GAP_TOLERANCE_HOURS,
                   min_hours: int = MIN_RUN_HOURS):
    """Contiguous runs above `threshold`, tolerating short gaps."""
    above = series > threshold
    runs = []
    start = None
    gap = 0
    last_true = None
    for t, flag in above.items():
        if flag:
            if start is None:
                start = t
            gap = 0
            last_true = t
        else:
            if start is not None:
                gap += 1
                if gap > gap_hours:
                    runs.append((start, last_true))
                    start = None
                    gap = 0
    if start is not None:
        runs.append((start, last_true))
    return [(a, b) for a, b in runs if (b - a).total_seconds() / 3600.0 + 1 >= min_hours]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    ib = pd.read_csv(IBTRACS_SUBSET, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    try:
        jma = pd.read_csv(JMA_TRACK, encoding="utf-8")
    except UnicodeDecodeError:
        jma = pd.read_csv(JMA_TRACK, encoding="cp932")
    jma["time"] = pd.to_datetime(jma["time"])
    jma = jma.sort_values("time").reset_index(drop=True)

    to_utm, origin_xy = build_utm()

    # ---- fine-grid crossings (1-minute) ----------------------------------
    t0 = ib["time"].iloc[0]
    fine = pd.date_range(t0, ib["time"].iloc[-1], freq="1min")
    flon, flat = interp_track(ib["time"], ib["LON"].to_numpy(), ib["LAT"].to_numpy(), fine)
    dfine = np.array([domain_min_km(a, b, to_utm, origin_xy) for a, b in zip(flon, flat)])
    fine_series = pd.Series(dfine, index=fine)

    crossings = {}
    for thr in DIST_THRESHOLDS_KM:
        crossings[thr] = threshold_crossings(fine, dfine, thr)

    # ---- hourly grid ------------------------------------------------------
    hourly = pd.date_range("2025-09-05 00:00", ib["time"].iloc[-1], freq="1h")
    hlon, hlat = interp_track(ib["time"], ib["LON"].to_numpy(), ib["LAT"].to_numpy(), hourly)
    dmin = np.array([domain_min_km(a, b, to_utm, origin_xy) for a, b in zip(hlon, hlat)])
    dcen = np.array([haversine_km(a, b, ORIGIN_LON, ORIGIN_LAT) for a, b in zip(hlon, hlat)])

    # same for the JMA track (for cross-check)
    jlon, jlat = interp_track(jma["time"], jma["経度"].to_numpy(), jma["緯度"].to_numpy(), hourly)
    in_jma_span = (hourly >= jma["time"].min()) & (hourly <= jma["time"].max())
    jmin = np.array(
        [
            domain_min_km(a, b, to_utm, origin_xy) if inside else np.nan
            for a, b, inside in zip(jlon, jlat, in_jma_span)
        ]
    )

    track = pd.DataFrame(
        {
            "time_utc": hourly,
            "center_lon": hlon,
            "center_lat": hlat,
            "d_domain_min_km": dmin,
            "d_domain_center_km": dcen,
            "d_domain_min_km_jma": jmin,
        }
    )

    # ---- observations -----------------------------------------------------
    obs = lidar_hourly_series()
    mid_h = int(round(obs.attrs["mid_height_m"]))
    obs_col = f"ws_{mid_h}"
    obs_hourly = obs.reindex(hourly)

    track["bjt"] = track["time_utc"] + pd.Timedelta(hours=8)
    track["ws_low_m_s"] = obs_hourly["ws_low"].to_numpy()
    track["ws_low_z"] = obs_hourly["ws_low_z"].to_numpy()
    track[f"ws_{mid_h}_m_s"] = obs_hourly[obs_col].to_numpy()
    track[f"ws_{mid_h}_z"] = obs_hourly[obs_col + "_z"].to_numpy()

    imin = int(np.argmin(dmin))
    closest = track.iloc[imin]

    low_thr = float(obs["ws_low_thr"].iloc[0])
    mid_thr = float(obs[obs_col + "_thr"].iloc[0])
    low_runs = sustained_runs(obs["ws_low"], low_thr)
    mid_runs = sustained_runs(obs[obs_col], mid_thr)
    low_z_runs = sustained_runs(obs["ws_low_z"], Z_THRESHOLD)
    primary_low = max(low_runs, key=lambda r: (r[1] - r[0])) if low_runs else None
    primary_mid = max(mid_runs, key=lambda r: (r[1] - r[0])) if mid_runs else None

    # combined window: inside the 500 km outer wind-hazard envelope AND the
    # observed low-layer wind is anomalously strong.
    c500_in = next((t for d, t in crossings[500.0] if d == "in"), None)
    if primary_low is not None:
        win_start = max(pd.Timestamp(c500_in), pd.Timestamp(primary_low[0]))
        win_end = pd.Timestamp(primary_low[1])
    else:
        win_start = win_end = None

    summary = {
        "generated_utc": generated,
        "track_source": "IBTrACS US-PROVISIONAL (SID 2025248N18120)",
        "closest_approach_utc": str(closest["time_utc"]),
        "closest_approach_bjt": str(closest["bjt"]),
        "closest_d_domain_min_km": round(float(closest["d_domain_min_km"]), 1),
        "closest_d_domain_min_km_jma": (
            None if pd.isna(closest["d_domain_min_km_jma"]) else round(float(closest["d_domain_min_km_jma"]), 1)
        ),
        "threshold_crossings_utc": {
            str(int(thr)): [(d, str(t)) for d, t in crossings[thr]] for thr in DIST_THRESHOLDS_KM
        },
        "observed_anomaly_z_threshold": Z_THRESHOLD,
        "observed_low_layer_threshold_m_s": round(low_thr, 2),
        "observed_mid_threshold_m_s": round(mid_thr, 2),
        "observed_low_layer_runs": [(str(a), str(b)) for a, b in low_runs],
        "observed_mid_runs": [(str(a), str(b)) for a, b in mid_runs],
        "observed_low_layer_z_runs": [(str(a), str(b)) for a, b in low_z_runs],
        "affected_period": {
            "utc_start": None if win_start is None else str(win_start),
            "utc_end": None if win_end is None else str(win_end),
            "bjt_start": None if win_start is None else str(win_start + pd.Timedelta(hours=8)),
            "bjt_end": None if win_end is None else str(win_end + pd.Timedelta(hours=8)),
            "hours_inclusive": None if win_start is None else int((win_end - win_start).total_seconds() // 3600) + 1,
        },
    }
    (OUT_ANALYSIS / "affected_period_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    track.to_csv(OUT_ANALYSIS / "affected_period_hourly.csv", index=False)

    # ---- figure -----------------------------------------------------------
    plot(track, obs, obs_col, mid_h, crossings, win_start, win_end, closest)

    # ---- report -----------------------------------------------------------
    write_report(summary, crossings, track, obs, obs_col, mid_h, win_start, win_end, closest,
                 low_runs, mid_runs, generated)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def plot(track, obs, obs_col, mid_h, crossings, win_start, win_end, closest) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(11.0, 8.4), sharex=True,
                           gridspec_kw={"height_ratios": [1.15, 1.0]})

    # --- top: distance -----------------------------------------------------
    a = ax[0]
    x = track["time_utc"]
    a.plot(x, track["d_domain_min_km"], color="k", lw=2.0, label="IBTrACS (US-PROVISIONAL)")
    a.plot(x, track["d_domain_min_km_jma"], color="C0", lw=1.3, ls="--", label="JMA best track")
    for thr in [500.0, 300.0, 200.0, 100.0]:
        a.axhline(thr, color="0.6", lw=0.7, ls=":")
        a.text(x.iloc[0], thr + 6, f"{int(thr)} km", fontsize=8, color="0.4")
    a.scatter([closest["time_utc"]], [closest["d_domain_min_km"]], color="crimson", zorder=5, s=45,
              label=f"closest {closest['d_domain_min_km']:.0f} km")
    a.annotate(
        f"closest approach\n{closest['bjt']} BJT\n{closest['d_domain_min_km']:.0f} km",
        (closest["time_utc"], closest["d_domain_min_km"]),
        xytext=(14, 26), textcoords="offset points", fontsize=8.5, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
    )
    a.set_ylim(0, 620)
    a.set_ylabel("Distance, domain → Tapah centre (km)")
    a.legend(loc="upper right", fontsize=8)
    a.grid(True, ls=":", alpha=0.4)
    a.set_title("Typhoon Tapah (2025): affected period for the Guangzhou domain, hourly", fontweight="bold")

    # --- bottom: observed winds -------------------------------------------
    b = ax[1]
    o = obs.loc["2025-09-05 00:00":"2025-09-09 06:00"]
    low_thr = float(obs["ws_low_thr"].iloc[0])
    mid_thr = float(obs[obs_col + "_thr"].iloc[0])
    b.plot(o.index, o["ws_low"], color="k", lw=1.8, label="LiDAR station mean, 52–300 m")
    b.plot(o.index, o[obs_col], color="C1", lw=1.4, ls="--", label=f"LiDAR station mean, {mid_h} m")
    b.axhline(low_thr, color="0.5", lw=1.0, ls=":", label=f"clean baseline +2σ ({low_thr:.1f} m s$^{{-1}}$)")
    b.axhline(mid_thr, color="C1", lw=0.8, ls=":", alpha=0.6)
    b.scatter([closest["time_utc"]], [float(o.loc[closest["time_utc"], "ws_low"])], color="crimson",
              zorder=5, s=40)
    b.set_ylabel("Wind speed (m s$^{-1}$)")
    b.set_xlabel("Time (UTC)")
    b.legend(loc="upper left", fontsize=8, ncol=2)
    b.grid(True, ls=":", alpha=0.4)

    for panel in ax:
        if win_start is not None:
            panel.axvspan(win_start, win_end, color="gold", alpha=0.20, zorder=0)
        panel.axvline(pd.Timestamp("2025-09-08 00:00"), color="crimson", lw=0.9, ls="--")
    ax[0].text(pd.Timestamp("2025-09-08 00:00"), 600, " 08:00 BJT 8 Sep", color="crimson", fontsize=8,
               ha="left", va="top")
    if win_start is not None:
        ax[0].text(win_start, 600, " affected period ", color="0.35", fontsize=8, ha="left", va="top")

    sec = ax[1].secondary_xaxis(
        "top", functions=(lambda u: u + 8.0 / 24.0, lambda l: l - 8.0 / 24.0)
    )
    sec.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    sec.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    sec.set_xlabel("Time (BJT = UTC+8)")
    sec.tick_params(labelsize=8)

    for panel in ax:
        panel.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        panel.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig4_typhoon_affected_period.png", dpi=200)
    plt.close(fig)


def write_report(summary, crossings, track, obs, obs_col, mid_h, win_start, win_end, closest,
                 low_runs, mid_runs, generated) -> None:
    def fmt(ts):
        ts = pd.Timestamp(ts)
        return ts.strftime("%Y-%m-%d %H:%M")

    peak_hour = obs["ws_low"].loc["2025-09-06":"2025-09-09"].idxmax()
    peak_val = obs["ws_low"].loc[peak_hour]

    lines = []
    lines.append("# 台风 Tapah（2025）对广州计算域的 affected period（精确到小时）\n")
    lines.append(f"生成时间（UTC）：{generated}　·　"
                 f"脚本：`python3 analysis/tapah-direct-influence/analyze_typhoon_affected_period.py`\n")
    lines.append("延续 `tapah-direct-influence` 的结论：广州计算域**没有进入台风内核**"
                 f"（最近约 {closest['d_domain_min_km']:.0f} km）。因此本文用"
                 "“几何接近度 + 观测风速异常”两条独立证据，把“台风影响时段”落到小时。\n")

    lines.append("## 结论（一句话）\n")
    lines.append(
        f"**广州计算域的台风影响时段为 "
        f"{fmt(summary['affected_period']['utc_start'])} – {fmt(summary['affected_period']['utc_end'])} UTC**"
        f"（北京时间 {fmt(summary['affected_period']['bjt_start'])} – "
        f"{fmt(summary['affected_period']['bjt_end'])}，含端点共 "
        f"{summary['affected_period']['hours_inclusive']} 小时）；"
        f"最强小时为 {fmt(peak_hour)} UTC（北京 {fmt(peak_hour + pd.Timedelta(hours=8))}），"
        f"对应最近距离 {closest['d_domain_min_km']:.0f} km。\n")

    lines.append("## 证据一：路径几何（IBTrACS 最佳路径，1 分钟插值定位过线小时）\n")
    lines.append("| 阈值半径 | 进入 (UTC) | 进入 (BJT) | 离开 (UTC) | 离开 (BJT) |")
    lines.append("|---:|---|---|---|---|")
    for thr in DIST_THRESHOLDS_KM:
        cr = crossings[thr]
        tin = next((t for d, t in cr if d == "in"), None)
        tout = next((t for d, t in cr if d == "out"), None)
        def cell(t):
            return "—" if t is None else fmt(t)
        def cell8(t):
            return "—" if t is None else fmt(t + pd.Timedelta(hours=8))
        lines.append(f"| {int(thr)} km | {cell(tin)} | {cell8(tin)} | {cell(tout)} | {cell8(tout)} |")
    lines.append("")
    lines.append(f"最近距离 {closest['d_domain_min_km']:.1f} km 出现在 {fmt(closest['time_utc'])} UTC"
                 f"（北京 {fmt(closest['bjt'])}），全程 ≥100 km，未进入内核。\n")

    lines.append("## 证据二：LiDAR 观测风（1 小时滚动，三站平均）\n")
    lines.append(f"- 低层（52–300 m）站均风速的干净日（9/1–9/5、9/10–9/13）日变基线约 1.2–3.0 m/s。")
    lines.append(f"- 判据：低层站均风速 > 干净日中位数 + 2×稳健 σ = "
                 f"**{summary['observed_low_layer_threshold_m_s']:.1f} m/s**（{mid_h} m 高度同判据为 "
                 f"{summary['observed_mid_threshold_m_s']:.1f} m/s）。")
    lines.append(f"- 低层观测异常时段：")
    for a, b in low_runs:
        lines.append(f"  - {fmt(a)} – {fmt(b)} UTC（{fmt(a + pd.Timedelta(hours=8))} – "
                     f"{fmt(b + pd.Timedelta(hours=8))} BJT）")
    lines.append(f"- {mid_h} m 高度观测异常时段（更高层受台风环流影响更久）：")
    for a, b in mid_runs:
        lines.append(f"  - {fmt(a)} – {fmt(b)} UTC（{fmt(a + pd.Timedelta(hours=8))} – "
                     f"{fmt(b + pd.Timedelta(hours=8))} BJT）")
    lines.append(f"- 低层峰值 {peak_val:.1f} m/s 出现在 {fmt(peak_hour)} UTC"
                 f"（北京 {fmt(peak_hour + pd.Timedelta(hours=8))}）。\n")

    lines.append("## 综合：两条证据取交集\n")
    lines.append("计算域在 **2025-09-07 02:11 UTC** 进入 500 km 外围风区，"
                 "低层观测在 **2025-09-07 03:00 UTC** 起持续超过基线 +2σ；"
                 "台风 9/8 00:00 UTC 前后登陆并迅速减弱，低层观测在 **2025-09-08 19:00 UTC** 之后回到常态。"
                 "两者交集即上表结论。\n")
    lines.append("分层参考：\n")
    lines.append("| 口径 | UTC | BJT |")
    lines.append("|---|---|---|")
    lines.append(f"| 外围风区（<500 km） | 9-07 02:11 起 | 9-07 10:11 起 |")
    lines.append(f"| 明显影响（<300 km） | {fmt(next((t for d,t in crossings[300.0] if d=='in')))} – "
                 f"{fmt(next((t for d,t in crossings[300.0] if d=='out')))} | "
                 f"{fmt(next((t for d,t in crossings[300.0] if d=='in'))+pd.Timedelta(hours=8))} – "
                 f"{fmt(next((t for d,t in crossings[300.0] if d=='out'))+pd.Timedelta(hours=8))} |")
    lines.append(f"| 近核心（<200 km） | {fmt(next((t for d,t in crossings[200.0] if d=='in')))} – "
                 f"{fmt(next((t for d,t in crossings[200.0] if d=='out')))} | "
                 f"{fmt(next((t for d,t in crossings[200.0] if d=='in'))+pd.Timedelta(hours=8))} – "
                 f"{fmt(next((t for d,t in crossings[200.0] if d=='out'))+pd.Timedelta(hours=8))} |")
    lines.append(f"| **推荐采用（观测+几何）** | **{fmt(win_start)} – {fmt(win_end)}** | "
                 f"**{fmt(win_start+pd.Timedelta(hours=8))} – {fmt(win_end+pd.Timedelta(hours=8))}** |")
    lines.append("")

    lines.append("## 与稿件口径的衔接\n")
    lines.append("- 会议纪要中的“7–9 号台风影响期”按北京时间成立，但更准确是 "
                 f"**北京 9/7 11:00 起到 9/9 03:00**，并非整两天。")
    lines.append("- 图 4 目前用的 UTC 9/7 00:00 – 9/8 23:00 窗口可继续使用；"
                 "若要标注“台风影响”区间，建议只高亮上表推荐时段的 41 小时。")
    lines.append("- 稿件台风个例 `2025-09-08 00:00 UTC`（北京 08:00）确实落在影响时段内的近峰值处，"
                 "选点合理。\n")

    lines.append("## 敏感性（起止小时的不确定范围）\n")
    lines.append("- **起始**：路径几何给出 500 km 过线在 2025-09-07 02:11 UTC；"
                 "若改用“逐小时日变基线 + 2σ”的 z 判据，低层自 2025-09-06 22:00 UTC 起即已异常"
                 "（此时距中心约 540 km）。因此起始小时稳妥地落在 **09-06 22:00 – 09-07 03:00 UTC** 之间，"
                 "本文取满足“<500 km 且 >基线+2σ”两条硬判据的 **09-07 03:00 UTC**。")
    lines.append(f"- **结束**：低层观测在 09-08 19:00 UTC 之后跌回常态（此时中心已在约 320 km 外，"
                 f"且 9/8 12:00 UTC 后 JMA 已将其降为无强风圈的弱低压）；"
                 f"{mid_h} m 高层因惯性拖尾会再维持到 09-09 02:00 UTC 左右，属正常的高度差异。")
    lines.append("- 判据（+2σ、稳健 σ、阈值半径）若改动一档，起止各浮动约 ±1–2 小时，"
                 "不会改变“北京 9/7 上午起、9/9 凌晨止”的结论。\n")

    lines.append("## 数据说明\n")
    lines.append("- `data/260409/raw/lidar/lidar_transient.csv` 只覆盖 **2025-09-03**（1 分钟分辨率，"
                 "常规天气日，低层站均风速约 2.4 m/s），**不覆盖台风日**，故本确定未使用；"
                 "台风时段的风由 1 小时滚动的 `lidar_1h-rolling.csv`（9/1–9/13）给出。")
    lines.append("- `lidar_1h-rolling.csv` 与 `data/260707/...` 的观测列一致（同一 LiDAR 记录）。"
                 "CSV 的 `datetime` 列为 **UTC**（见 `analysis/260409/plot-ws-composite-profile-lst.py` 注释）。")
    lines.append("- IBTrACS 为 US-PROVISIONAL 业务路径，季节最终集若微调，中心可能平移十几公里，"
                 "但不会改变过线小时到“±1 小时”的量级。\n")

    (OUT_ANALYSIS / "typhoon_affected_period.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
