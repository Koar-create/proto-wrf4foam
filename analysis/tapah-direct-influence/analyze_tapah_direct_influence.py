#!/usr/bin/env python3
"""Quantify whether the Guangzhou CFD domain / LiDAR sites were under
direct tropical-cyclone influence during Tapah (2025).

Writes tables under analysis/tapah-direct-influence/ and figures under
results/tapah-direct-influence/. Does not touch docs/scs-wrf-of-manuscript.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[2]
IBTRACS_CSV = REPO / "data/ibtracs/ibtracs.last3years.list.v04r01.csv"
JMA_CSV = REPO / "data/ibtracs/jma_table2025.csv"
OUT_ANALYSIS = Path(__file__).resolve().parent
OUT_FIG = REPO / "results/tapah-direct-influence"
OUT_DATA = REPO / "data/ibtracs"

SID = "2025248N18120"
JMA_NAME = "TAPAH"
CASE_UTC = pd.Timestamp("2025-09-08 00:00:00")

ORIGIN_LON = 113.3218197
ORIGIN_LAT = 23.1133057
DOMAIN_HALF_M = 5000.0
NM_KM = 1.852
EARTH_KM = 6371.0

SITES = {
    "GAW103": (113.331446, 23.110176),
    "GAW104": (113.326053, 23.116321),
    "GAW111": (113.322620, 23.113718),
}

# JMA 8-azimuth long-axis direction (format_csv.html): 1=NE ... 8=N, 9=concentric.
JMA_DIR_DEG = {1: 45.0, 2: 90.0, 3: 135.0, 4: 180.0, 5: 225.0, 6: 270.0, 7: 315.0, 8: 0.0}

THRESHOLDS_KM = {
    "rmw_proxy_100km": 100.0,
    "inner_region_200km": 200.0,
    "cma_common_300km": 300.0,
    "outer_region_500km": 500.0,
    "hko_signal1_800km": 800.0,
}


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * EARTH_KM * np.arcsin(np.sqrt(a)))


def bearing_deg(lon1, lat1, lon2, lat2) -> float:
    """Forward azimuth, degrees clockwise from north."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def quadrant_from_bearing(brng: float) -> str:
    if brng < 90.0:
        return "NE"
    if brng < 180.0:
        return "SE"
    if brng < 270.0:
        return "SW"
    return "NW"


def destination(lon, lat, az_deg, dist_km) -> tuple[float, float]:
    """Destination point given start, azimuth (from N), distance km."""
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(az_deg)
    ang = dist_km / EARTH_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lon2), math.degrees(lat2))


def numeric(val):
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return np.nan
    try:
        v = float(val)
    except (TypeError, ValueError):
        return np.nan
    if v in (-9999.0, -999.0):
        return np.nan
    return v


def usa_r_in_quadrant(row, prefix: str, quad: str) -> float:
    col = f"{prefix}_{quad}"
    v = numeric(row.get(col, np.nan))
    return np.nan if np.isnan(v) else v * NM_KM


def jma_offset_circle(lon, lat, direction, long_nm, short_nm):
    """JMA 30/50-kt wind area: offset circle, radii in nautical miles.

    Center is shifted (L-S)/2 along the long-axis azimuth; radius is (L+S)/2.
    """
    L = numeric(long_nm)
    S = numeric(short_nm)
    if np.isnan(L) or np.isnan(S) or L <= 0.0:
        return None
    d = int(numeric(direction) if not np.isnan(numeric(direction)) else 9)
    if d == 9 or abs(L - S) < 1e-6:
        return {"clon": lon, "clat": lat, "radius_km": L * NM_KM, "offset_km": 0.0, "dir": d}
    az = JMA_DIR_DEG.get(d)
    if az is None:
        return {"clon": lon, "clat": lat, "radius_km": L * NM_KM, "offset_km": 0.0, "dir": d}
    offset_km = 0.5 * (L - S) * NM_KM
    radius_km = 0.5 * (L + S) * NM_KM
    clon, clat = destination(lon, lat, az, offset_km)
    return {"clon": clon, "clat": clat, "radius_km": radius_km, "offset_km": offset_km, "dir": d}


def inside_circle(lon, lat, circle) -> bool | None:
    if circle is None:
        return None
    d = haversine_km(circle["clon"], circle["clat"], lon, lat)
    return bool(d <= circle["radius_km"] + 1e-6)


def load_ibtracs() -> pd.DataFrame:
    df = pd.read_csv(IBTRACS_CSV, skiprows=[1], low_memory=False)
    tap = df[df["SID"].astype(str) == SID].copy()
    tap["time"] = pd.to_datetime(tap["ISO_TIME"])
    for c in [
        "LAT",
        "LON",
        "USA_LAT",
        "USA_LON",
        "USA_WIND",
        "USA_PRES",
        "USA_SSHS",
        "USA_RMW",
        "USA_ROCI",
        "USA_R34_NE",
        "USA_R34_SE",
        "USA_R34_SW",
        "USA_R34_NW",
        "USA_R50_NE",
        "USA_R50_SE",
        "USA_R50_SW",
        "USA_R50_NW",
        "USA_R64_NE",
        "USA_R64_SE",
        "USA_R64_SW",
        "USA_R64_NW",
        "TOKYO_LAT",
        "TOKYO_LON",
        "TOKYO_GRADE",
        "TOKYO_WIND",
        "TOKYO_PRES",
        "TOKYO_R30_DIR",
        "TOKYO_R30_LONG",
        "TOKYO_R30_SHORT",
        "TOKYO_R50_DIR",
        "TOKYO_R50_LONG",
        "TOKYO_R50_SHORT",
        "CMA_LAT",
        "CMA_LON",
        "CMA_CAT",
        "CMA_WIND",
        "CMA_PRES",
        "DIST2LAND",
        "LANDFALL",
        "WMO_WIND",
        "WMO_PRES",
    ]:
        if c in tap.columns:
            tap[c] = pd.to_numeric(tap[c], errors="coerce")
    return tap.sort_values("time").reset_index(drop=True)


def load_jma() -> pd.DataFrame:
    raw = pd.read_csv(JMA_CSV, encoding="cp932")
    tap = raw[raw["台風名"].astype(str).str.upper() == JMA_NAME].copy()
    tap["time"] = pd.to_datetime(
        dict(year=tap["年"], month=tap["月"], day=tap["日"], hour=tap["時（UTC）"])
    )
    return tap.sort_values("time").reset_index(drop=True)


def domain_corners_lonlat():
    to_lonlat = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = to_utm.transform(ORIGIN_LON, ORIGIN_LAT)
    corners = []
    for x, y in [
        (-DOMAIN_HALF_M, -DOMAIN_HALF_M),
        (DOMAIN_HALF_M, -DOMAIN_HALF_M),
        (DOMAIN_HALF_M, DOMAIN_HALF_M),
        (-DOMAIN_HALF_M, DOMAIN_HALF_M),
        (0.0, 0.0),
    ]:
        lon, lat = to_lonlat.transform(ox + x, oy + y)
        corners.append((float(lon), float(lat)))
    labels = ["SW", "SE", "NE", "NW", "center"]
    return dict(zip(labels, corners)), to_utm, (ox, oy)


def domain_minmax_km(storm_lon, storm_lat, to_utm, origin_xy) -> tuple[float, float]:
    sx, sy = to_utm.transform(storm_lon, storm_lat)
    ox, oy = origin_xy
    lx, ly = sx - ox, sy - oy
    cx = float(np.clip(lx, -DOMAIN_HALF_M, DOMAIN_HALF_M))
    cy = float(np.clip(ly, -DOMAIN_HALF_M, DOMAIN_HALF_M))
    dmin = math.hypot(lx - cx, ly - cy) / 1000.0
    corners = [
        (-DOMAIN_HALF_M, -DOMAIN_HALF_M),
        (DOMAIN_HALF_M, -DOMAIN_HALF_M),
        (DOMAIN_HALF_M, DOMAIN_HALF_M),
        (-DOMAIN_HALF_M, DOMAIN_HALF_M),
    ]
    dmax = max(math.hypot(lx - x, ly - y) for x, y in corners) / 1000.0
    return dmin, dmax


def build_distance_table(ib, jma, corners, to_utm, origin_xy) -> pd.DataFrame:
    jma_by_time = {pd.Timestamp(t): r for t, r in zip(jma["time"], jma.to_dict("records"))}
    rows = []
    for rec in ib.to_dict("records"):
        t = pd.Timestamp(rec["time"])
        slon, slat = float(rec["LON"]), float(rec["LAT"])
        dmin, dmax = domain_minmax_km(slon, slat, to_utm, origin_xy)
        dcen = haversine_km(slon, slat, ORIGIN_LON, ORIGIN_LAT)
        brng = bearing_deg(slon, slat, ORIGIN_LON, ORIGIN_LAT)
        quad = quadrant_from_bearing(brng)
        r34_q = usa_r_in_quadrant(rec, "USA_R34", quad)
        r50_q = usa_r_in_quadrant(rec, "USA_R50", quad)
        r64_q = usa_r_in_quadrant(rec, "USA_R64", quad)
        r34s = [usa_r_in_quadrant(rec, "USA_R34", q) for q in ("NE", "SE", "SW", "NW")]
        r34_max = np.nanmax(r34s) if np.any(np.isfinite(r34s)) else np.nan
        rmw = numeric(rec.get("USA_RMW")) * NM_KM if pd.notna(rec.get("USA_RMW")) else np.nan
        roci = numeric(rec.get("USA_ROCI")) * NM_KM if pd.notna(rec.get("USA_ROCI")) else np.nan

        tokyo_lon = numeric(rec.get("TOKYO_LON"))
        tokyo_lat = numeric(rec.get("TOKYO_LAT"))
        c30 = None
        c50 = None
        if pd.notna(tokyo_lon) and pd.notna(tokyo_lat):
            c30 = jma_offset_circle(
                tokyo_lon,
                tokyo_lat,
                rec.get("TOKYO_R30_DIR"),
                rec.get("TOKYO_R30_LONG"),
                rec.get("TOKYO_R30_SHORT"),
            )
            c50 = jma_offset_circle(
                tokyo_lon,
                tokyo_lat,
                rec.get("TOKYO_R50_DIR"),
                rec.get("TOKYO_R50_LONG"),
                rec.get("TOKYO_R50_SHORT"),
            )
        jma_row = jma_by_time.get(t)
        if jma_row is not None:
            c30 = jma_offset_circle(
                float(jma_row["経度"]),
                float(jma_row["緯度"]),
                jma_row["30KT長径方向"],
                jma_row["30KT長径"],
                jma_row["30KT短径"],
            )
            c50 = jma_offset_circle(
                float(jma_row["経度"]),
                float(jma_row["緯度"]),
                jma_row["50KT長径方向"],
                jma_row["50KT長径"],
                jma_row["50KT短径"],
            )

        site_d = {
            name: haversine_km(slon, slat, lon, lat) for name, (lon, lat) in SITES.items()
        }
        row = {
            "time_utc": t,
            "lat": slat,
            "lon": slon,
            "usa_wind_kt": numeric(rec.get("USA_WIND")),
            "usa_pres_hpa": numeric(rec.get("USA_PRES")),
            "usa_sshs": numeric(rec.get("USA_SSHS")),
            "tokyo_wind_kt": numeric(rec.get("TOKYO_WIND")),
            "tokyo_grade": numeric(rec.get("TOKYO_GRADE")),
            "tokyo_pres_hpa": numeric(rec.get("TOKYO_PRES")),
            "cma_wind_kt": numeric(rec.get("CMA_WIND")),
            "cma_cat": numeric(rec.get("CMA_CAT")),
            "cma_pres_hpa": numeric(rec.get("CMA_PRES")),
            "dist2land_km": numeric(rec.get("DIST2LAND")),
            "landfall_km": numeric(rec.get("LANDFALL")),
            "bearing_to_domain_deg": brng,
            "quadrant_to_domain": quad,
            "d_domain_min_km": dmin,
            "d_domain_max_km": dmax,
            "d_domain_center_km": dcen,
            "d_GAW103_km": site_d["GAW103"],
            "d_GAW104_km": site_d["GAW104"],
            "d_GAW111_km": site_d["GAW111"],
            "usa_r34_quad_km": r34_q,
            "usa_r34_max_km": r34_max,
            "usa_r50_quad_km": r50_q,
            "usa_r64_quad_km": r64_q,
            "usa_rmw_km": rmw,
            "usa_roci_km": roci,
            "in_usa_r34_quad": (dmin <= r34_q) if pd.notna(r34_q) else None,
            "in_usa_r34_max": (dmin <= r34_max) if pd.notna(r34_max) else None,
            "in_usa_r50_quad": (dmin <= r50_q) if pd.notna(r50_q) else None,
            "in_usa_r64_quad": (dmin <= r64_q) if pd.notna(r64_q) else None,
            "in_usa_rmw": (dmin <= rmw) if pd.notna(rmw) else None,
            "in_usa_2rmw": (dmin <= 2.0 * rmw) if pd.notna(rmw) else None,
            "in_usa_roci": (dmin <= roci) if pd.notna(roci) else None,
            "in_jma_r30": inside_circle(ORIGIN_LON, ORIGIN_LAT, c30),
            "in_jma_r50": inside_circle(ORIGIN_LON, ORIGIN_LAT, c50),
            "jma_r30_radius_km": None if c30 is None else c30["radius_km"],
            "jma_r50_radius_km": None if c50 is None else c50["radius_km"],
        }
        for k, thr in THRESHOLDS_KM.items():
            row[f"in_{k}"] = bool(dmin <= thr)
        rows.append(row)
    return pd.DataFrame(rows)


def yesno(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return "YES" if bool(v) else "NO"


def verdicts_at(df: pd.DataFrame, t: pd.Timestamp, loc_col: str, loc_name: str) -> list[dict]:
    r = df.loc[df["time_utc"] == t].iloc[0]
    d = float(r[loc_col])
    items = [
        ("A1 inner core: distance ≤ RMW (JTWC/USA)", r["usa_rmw_km"], r["in_usa_rmw"] if loc_col.startswith("d_domain") else (d <= r["usa_rmw_km"] if pd.notna(r["usa_rmw_km"]) else None)),
        ("A2 inner-core envelope: distance ≤ 2×RMW", None if pd.isna(r["usa_rmw_km"]) else 2.0 * r["usa_rmw_km"], r["in_usa_2rmw"] if loc_col.startswith("d_domain") else (d <= 2.0 * r["usa_rmw_km"] if pd.notna(r["usa_rmw_km"]) else None)),
        ("A3 direct-hit proxy: distance ≤ 100 km", 100.0, d <= 100.0),
        ("A4 inner region (rainfall papers): distance ≤ 200 km", 200.0, d <= 200.0),
        ("B1 gale: inside quadrant-specific R34 (JTWC/USA)", r["usa_r34_quad_km"], (d <= r["usa_r34_quad_km"]) if pd.notna(r["usa_r34_quad_km"]) else None),
        ("B2 gale, liberal: inside max-quadrant R34", r["usa_r34_max_km"], (d <= r["usa_r34_max_km"]) if pd.notna(r["usa_r34_max_km"]) else None),
        ("B3 JMA 30-kt strong-wind area (offset circle)", r["jma_r30_radius_km"], r["in_jma_r30"]),
        ("B4 storm-force: inside quadrant R50", r["usa_r50_quad_km"], (d <= r["usa_r50_quad_km"]) if pd.notna(r["usa_r50_quad_km"]) else None),
        ("B5 hurricane-force: inside quadrant R64", r["usa_r64_quad_km"], (d <= r["usa_r64_quad_km"]) if pd.notna(r["usa_r64_quad_km"]) else None),
        ("B6 inside ROCI (outermost closed isobar)", r["usa_roci_km"], (d <= r["usa_roci_km"]) if pd.notna(r["usa_roci_km"]) else None),
        ("C1 fixed 300 km", 300.0, d <= 300.0),
        ("C2 outer region / wind-hazard 500 km", 500.0, d <= 500.0),
        ("C3 HKO Signal No. 1 distance 800 km", 800.0, d <= 800.0),
    ]
    out = []
    for name, thr, flag in items:
        out.append(
            {
                "location": loc_name,
                "definition": name,
                "threshold_km": None if thr is None or (isinstance(thr, float) and np.isnan(thr)) else round(float(thr), 1),
                "distance_km": round(d, 1),
                "verdict": yesno(flag),
            }
        )
    return out


def plot_track(ib, dist, corners, case_row, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    lon = ib["LON"].to_numpy()
    lat = ib["LAT"].to_numpy()
    wind = ib["USA_WIND"].to_numpy()
    ax.plot(lon, lat, color="0.55", lw=1.2, zorder=2)
    sc = ax.scatter(
        lon,
        lat,
        c=np.where(np.isfinite(wind), wind, 20.0),
        cmap="YlOrRd",
        vmin=20,
        vmax=70,
        s=28,
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("JTWC/USA wind (kt)")

    # Case-time wind circles around 00 UTC center.
    clon, clat = float(case_row["lon"]), float(case_row["lat"])
    for radius, ls, lab in [
        (case_row["usa_rmw_km"], ":", "RMW"),
        (case_row["usa_r34_quad_km"], "--", "R34 NE"),
        (case_row["usa_r34_max_km"], "-.", "R34 max"),
        (200.0, (0, (3, 2)), "200 km"),
        (500.0, (0, (1, 2)), "500 km"),
    ]:
        if radius is None or (isinstance(radius, float) and np.isnan(radius)):
            continue
        th = np.linspace(0, 2 * np.pi, 361)
        # local km to deg (small-circle approx)
        ax.plot(
            clon + (radius / (111.32 * np.cos(np.radians(clat)))) * np.cos(th),
            clat + (radius / 110.57) * np.sin(th),
            ls=ls,
            color="0.2",
            lw=0.9,
            label=f"{lab} ({radius:.0f} km)",
            zorder=1,
        )

    # Domain box (tiny).
    box = np.array([corners["SW"], corners["SE"], corners["NE"], corners["NW"], corners["SW"]])
    ax.plot(box[:, 0], box[:, 1], color="C0", lw=1.4, zorder=4, label="CFD domain")
    ax.scatter(
        [ORIGIN_LON],
        [ORIGIN_LAT],
        marker="s",
        s=28,
        color="C0",
        zorder=5,
    )
    for name, (slon, slat) in SITES.items():
        ax.scatter([slon], [slat], s=18, color="C0", marker="o", zorder=5)
        ax.annotate(name.replace("GAW", ""), (slon, slat), xytext=(4, 3), textcoords="offset points", fontsize=7)

    ax.scatter([clon], [clat], marker="*", s=120, color="crimson", zorder=6, label="00 UTC 8 Sep (case)")
    # JMA 00 UTC center
    ax.annotate(
        "landfall ~Taishan\n00 UTC 8 Sep",
        (clon, clat),
        xytext=(-70, -28),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_xlim(109.2, 121.0)
    ax.set_ylim(16.8, 25.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Tapah (2025) IBTrACS track vs Guangzhou CFD domain")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_distance(dist: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    t = dist["time_utc"]
    ax.plot(t, dist["d_domain_min_km"], color="k", lw=2.0, label="CFD domain (nearest)")
    ax.fill_between(t, dist["d_domain_min_km"], dist["d_domain_max_km"], color="0.8", alpha=0.8, label="domain min–max")
    ax.plot(t, dist["d_GAW103_km"], ls="--", lw=1.1, label="GAW103")
    ax.plot(t, dist["d_GAW104_km"], ls="-.", lw=1.1, label="GAW104")
    ax.plot(t, dist["d_GAW111_km"], ls=":", lw=1.3, label="GAW111")
    ax.plot(t, dist["usa_r34_quad_km"], color="C3", lw=1.2, label="R34 in domain quadrant")
    ax.plot(t, dist["usa_rmw_km"], color="C1", lw=1.0, label="RMW")
    for y, lab, c in [
        (100, "100 km", "0.4"),
        (200, "200 km", "0.3"),
        (500, "500 km", "0.2"),
    ]:
        ax.axhline(y, color=c, ls="--", lw=0.7, alpha=0.8)
        ax.text(t.iloc[0], y + 8, lab, fontsize=8, color=c)
    ax.axvline(CASE_UTC, color="crimson", lw=1.0, ls="--")
    ax.text(CASE_UTC, 820, "case\n08 BJT", color="crimson", fontsize=8, ha="center", va="bottom")
    ax.set_ylim(0, 900)
    ax.set_ylabel("Distance to Tapah center (km)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Distance from Guangzhou targets to Tapah center")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, ls=":", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prd_zoom(ib, dist, corners, case_row, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.plot(ib["LON"], ib["LAT"], color="0.4", lw=1.5)
    ax.scatter(ib["LON"], ib["LAT"], s=18, c="0.3", zorder=3)
    clon, clat = float(case_row["lon"]), float(case_row["lat"])
    th = np.linspace(0, 2 * np.pi, 361)
    for radius, color, lab in [
        (case_row["usa_rmw_km"], "C1", "RMW"),
        (case_row["usa_r34_quad_km"], "C3", f"R34 {case_row['quadrant_to_domain']}"),
        (200.0, "C0", "200 km"),
    ]:
        if radius is None or (isinstance(radius, float) and np.isnan(radius)):
            continue
        ax.plot(
            clon + (radius / (111.32 * np.cos(np.radians(clat)))) * np.cos(th),
            clat + (radius / 110.57) * np.sin(th),
            color=color,
            lw=1.2,
            label=f"{lab} ({radius:.0f} km)",
        )
    box = np.array([corners["SW"], corners["SE"], corners["NE"], corners["NW"], corners["SW"]])
    ax.plot(box[:, 0], box[:, 1], color="navy", lw=2.0, label="CFD 10 km domain")
    ax.scatter([ORIGIN_LON], [ORIGIN_LAT], marker="s", s=40, color="navy", zorder=5)
    for slon, slat in SITES.values():
        ax.scatter([slon], [slat], s=22, color="navy", zorder=5)
    ax.annotate(
        "GAW103/104/111",
        (ORIGIN_LON, ORIGIN_LAT),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=8,
    )
    ax.scatter([clon], [clat], marker="*", s=160, color="crimson", zorder=6, label="Tapah 00 UTC")
    ax.annotate("Taishan / Xiachuan\nlandfall", (clon, clat), xytext=(8, -22), textcoords="offset points", fontsize=8)
    ax.set_xlim(111.8, 114.2)
    ax.set_ylim(21.2, 23.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Pearl River Delta at 00 UTC 8 Sep 2025")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    ib = load_ibtracs()
    jma = load_jma()
    corners, to_utm, origin_xy = domain_corners_lonlat()
    dist = build_distance_table(ib, jma, corners, to_utm, origin_xy)

    subset_path = OUT_DATA / "tapah_2025_ibtracs_subset.csv"
    ib.to_csv(subset_path, index=False)
    jma_path = OUT_DATA / "tapah_2025_jma_besttrack.csv"
    jma.to_csv(jma_path, index=False)
    dist_path = OUT_ANALYSIS / "distance_timeseries.csv"
    dist.to_csv(dist_path, index=False)

    case = dist.loc[dist["time_utc"] == CASE_UTC].iloc[0]
    imin = int(dist["d_domain_min_km"].idxmin())
    closest = dist.iloc[imin]

    locations = [
        ("d_domain_min_km", "CFD domain (nearest point)"),
        ("d_domain_center_km", "CFD domain center"),
        ("d_GAW103_km", "GAW103"),
        ("d_GAW104_km", "GAW104"),
        ("d_GAW111_km", "GAW111"),
    ]
    verdict_rows = []
    for col, name in locations:
        verdict_rows.extend(verdicts_at(dist, CASE_UTC, col, name))
    verdict_df = pd.DataFrame(verdict_rows)
    verdict_path = OUT_ANALYSIS / "verdicts_20250908_00utc.csv"
    verdict_df.to_csv(verdict_path, index=False)

    closest_rows = []
    tclose = closest["time_utc"]
    for col, name in locations:
        closest_rows.extend(verdicts_at(dist, tclose, col, name))
    closest_df = pd.DataFrame(closest_rows)
    closest_df.to_csv(OUT_ANALYSIS / "verdicts_closest_approach.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ibtracs_sid": SID,
        "ibtracs_file": str(IBTRACS_CSV.relative_to(REPO)),
        "jma_file": str(JMA_CSV.relative_to(REPO)),
        "jma_id": "2516",
        "track_type": "US-PROVISIONAL",
        "origin_lon": ORIGIN_LON,
        "origin_lat": ORIGIN_LAT,
        "domain_half_m": DOMAIN_HALF_M,
        "sites": {k: {"lon": v[0], "lat": v[1]} for k, v in SITES.items()},
        "corners": {k: {"lon": v[0], "lat": v[1]} for k, v in corners.items()},
        "case": {
            "time_utc": str(CASE_UTC),
            "time_bjt": "2025-09-08 08:00 BJT",
            "center_lon": float(case["lon"]),
            "center_lat": float(case["lat"]),
            "bearing_deg": float(case["bearing_to_domain_deg"]),
            "quadrant": str(case["quadrant_to_domain"]),
            "usa_wind_kt": None if pd.isna(case["usa_wind_kt"]) else float(case["usa_wind_kt"]),
            "tokyo_wind_kt": None if pd.isna(case["tokyo_wind_kt"]) else float(case["tokyo_wind_kt"]),
            "tokyo_grade": None if pd.isna(case["tokyo_grade"]) else int(case["tokyo_grade"]),
            "cma_wind_kt": None if pd.isna(case["cma_wind_kt"]) else float(case["cma_wind_kt"]),
            "d_domain_min_km": float(case["d_domain_min_km"]),
            "d_domain_max_km": float(case["d_domain_max_km"]),
            "d_domain_center_km": float(case["d_domain_center_km"]),
            "d_GAW103_km": float(case["d_GAW103_km"]),
            "d_GAW104_km": float(case["d_GAW104_km"]),
            "d_GAW111_km": float(case["d_GAW111_km"]),
            "usa_r34_quad_km": None if pd.isna(case["usa_r34_quad_km"]) else float(case["usa_r34_quad_km"]),
            "usa_r34_max_km": None if pd.isna(case["usa_r34_max_km"]) else float(case["usa_r34_max_km"]),
            "usa_rmw_km": None if pd.isna(case["usa_rmw_km"]) else float(case["usa_rmw_km"]),
            "usa_roci_km": None if pd.isna(case["usa_roci_km"]) else float(case["usa_roci_km"]),
            "jma_r30_radius_km": None if pd.isna(case["jma_r30_radius_km"]) else float(case["jma_r30_radius_km"]),
        },
        "closest_approach": {
            "time_utc": str(tclose),
            "center_lon": float(closest["lon"]),
            "center_lat": float(closest["lat"]),
            "d_domain_min_km": float(closest["d_domain_min_km"]),
            "d_GAW103_km": float(closest["d_GAW103_km"]),
            "d_GAW104_km": float(closest["d_GAW104_km"]),
            "d_GAW111_km": float(closest["d_GAW111_km"]),
        },
    }
    (OUT_ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    plot_track(ib, dist, corners, case, OUT_FIG / "fig1_tapah_track_domain.png")
    plot_distance(dist, OUT_FIG / "fig2_distance_timeseries.png")
    plot_prd_zoom(ib, dist, corners, case, OUT_FIG / "fig3_prd_zoom_00utc.png")

    print("case 00 UTC")
    print(case[["lat", "lon", "bearing_to_domain_deg", "quadrant_to_domain",
                "d_domain_min_km", "d_domain_max_km", "d_domain_center_km",
                "d_GAW103_km", "d_GAW104_km", "d_GAW111_km",
                "usa_r34_quad_km", "usa_r34_max_km", "usa_rmw_km", "usa_roci_km",
                "jma_r30_radius_km", "in_usa_r34_quad", "in_jma_r30"]].to_string())
    print("\nclosest")
    print(closest[["time_utc", "lat", "lon", "d_domain_min_km", "d_GAW103_km"]].to_string())
    print("\nwrote", dist_path)
    print("wrote", verdict_path)


if __name__ == "__main__":
    main()
