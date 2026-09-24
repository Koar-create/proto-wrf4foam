#!/usr/bin/env python3
"""Downscaling-induced local wind ratio for manuscript section 3.2.

R = V_CFD / V_WRF is a custom ratio. The denominator is the 1 km WRF wind speed
at the same horizontal location, not a no-building CFD rerun, so this is not
Blocken's amplification factor.

R' = V_CFD / mean(V_CFD on interior fluid cells at the same height) is the
within-domain reference. It does not inherit a WRF bias.

Zone detection and percentiles use the interior only: |x| and |y| <= 2000 m
(a 500 m band inside each lateral boundary is dropped). A fast zone is a
contiguous patch of smoothed R' above the threshold. R is summarized inside
those patches and is not used to draw them.

Horizontal slices are the ParaView ResampleToImage product:
1000 x 1000 points on [-2500, 2500] m (about 5 m). Building footprints from
the UTM49 shapefile are masked before smoothing, thresholding, or any mean.
A footprint is solid at a slice only when the shapefile height jzgd reaches
that slice; air above a lower roof stays in the fluid set. vtkValidPointMask
is applied as well, so unsampled resample points are not treated as calm air.

Weather labels follow the manuscript cut: 1-6 September regular (6 September
is 3-hourly and stays in the regular aggregate), 7-9 September typhoon.
Snapshot hours are serially correlated. Regime ranges describe that realized
time variation. They are not an iid sampling distribution.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_UTIL = os.path.join(_REPO, "util")
if _UTIL not in sys.path:
    sys.path.insert(0, _UTIL)

import visualize_WRF_CFD_xy_two_panel as base  # noqa: E402
from convert_lonlat_xy_origin import origin_utm49n, xy_to_lonlat  # noqa: E402

HEIGHTS = (30, 60, 120)
GRID_N = 1000
XY_MIN = -2500.0
XY_MAX = 2500.0
# Inclusive linspace used by ResampleToImage SamplingDimensions=1000.
DX = (XY_MAX - XY_MIN) / (GRID_N - 1)
CELL_AREA = DX * DX

# Guard against a near-zero WRF denominator. 0.2 m/s keeps calm hours in the
# sample; those hours are still reported with n_ratio so the floor is visible.
WRF_SPEED_FLOOR = 0.2
# Drop a 500 m band inside each lateral boundary. Zone detection and
# percentiles use only the remaining interior (|x| and |y| <= 2000 m).
MARGIN_M = 500.0
BASE_SMOOTH_M = 25.0
BASE_THRESHOLD = 1.2
BASE_MIN_AREA_M2 = 25.0 * CELL_AREA  # ~25 cells, about 25 m x 25 m
DECEL_THRESHOLD = 0.8
SMOOTH_WEIGHT_FLOOR = 0.5  # kernel must be at least half fluid

# One-at-a-time departures from the baseline. Not a full factorial.
SENSITIVITY = (
    {"setting": "baseline", "smooth_m": 25.0, "threshold": 1.2, "min_area_m2": BASE_MIN_AREA_M2},
    {"setting": "smooth_15m", "smooth_m": 15.0, "threshold": 1.2, "min_area_m2": BASE_MIN_AREA_M2},
    {"setting": "thr_1.1", "smooth_m": 25.0, "threshold": 1.1, "min_area_m2": BASE_MIN_AREA_M2},
    {"setting": "thr_1.3", "smooth_m": 25.0, "threshold": 1.3, "min_area_m2": BASE_MIN_AREA_M2},
    {"setting": "minarea_250m2", "smooth_m": 25.0, "threshold": 1.2, "min_area_m2": 250.0},
    {"setting": "minarea_2500m2", "smooth_m": 25.0, "threshold": 1.2, "min_area_m2": 2500.0},
)

BJT = timezone(timedelta(hours=8))
TYPHOON_START = pd.Timestamp("2025-09-07 00:00:00", tz="UTC")

OUT_DIR = os.path.join(_REPO, "results", "wrf_openfoam", "local_wind_ratio")
MASK_NPZ = os.path.join(OUT_DIR, "building_fluid_mask.npz")
SNAPSHOT_CSV = os.path.join(OUT_DIR, "snapshot_stats.csv")
ZONES_CSV = os.path.join(OUT_DIR, "zones.csv")
SENS_SNAPSHOT_CSV = os.path.join(OUT_DIR, "sensitivity_snapshots.csv")
REGIME_CSV = os.path.join(OUT_DIR, "regime_summary.csv")
SENS_CSV = os.path.join(OUT_DIR, "sensitivity_summary.csv")
CASE_LIST = os.path.join(OUT_DIR, "case_list.txt")

AUTOCORR_NOTE = (
    "min/max and percentiles across snapshots describe the realized time "
    "variation of a serially correlated sequence, not an iid sampling distribution"
)


def grid_axis() -> np.ndarray:
    return np.linspace(XY_MIN, XY_MAX, GRID_N)


def interior_mask() -> np.ndarray:
    """True inside the analysis square, 500 m clear of each lateral boundary."""
    axis = grid_axis()
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    limit = XY_MAX - MARGIN_M
    return (np.abs(xx) <= limit) & (np.abs(yy) <= limit)


def xy_to_ij(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.rint((x - XY_MIN) / DX).astype(np.int32)
    iy = np.rint((y - XY_MIN) / DX).astype(np.int32)
    return iy, ix


def kernel_size(smooth_m: float) -> int:
    n = int(round(smooth_m / DX))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return n


def regime_of(ts: pd.Timestamp) -> str:
    """Manuscript cut: through 6 September regular, from 7 September typhoon."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return "typhoon" if ts >= TYPHOON_START else "regular"


def iter_target_cases(root: str) -> list[str]:
    """Converged production cases for 1-9 September, using hours that exist."""
    spec = {
        "20250901": list(range(24)),
        "20250902": list(range(24)),
        "20250903": list(range(24)),
        "20250904": list(range(24)),
        "20250905": list(range(24)),
        "20250906": [0, 3, 6, 9, 12, 15, 18, 21],
        "20250907": [h for h in range(24) if h not in (13, 14, 16, 17)],
        "20250908": list(range(24)),
        "20250909": [0, 3, 6, 9, 12, 15, 18, 21],
    }
    cases = []
    for day, hours in spec.items():
        for hour in hours:
            name = f"{day}_{hour:02d}00_two_boundaries_as_outlet"
            case_dir = os.path.join(root, name)
            if os.path.isdir(os.path.join(case_dir, "5000")):
                cases.append(case_dir)
    return cases


def build_building_masks(shp_path: str) -> dict[int, np.ndarray]:
    """True where the point lies inside a building solid at that slice height."""
    import shapefile
    from shapely import contains_xy, union_all
    from shapely.geometry import Polygon

    ox, oy = origin_utm49n()
    reader = shapefile.Reader(shp_path, encoding="gbk")
    by_height: dict[int, list] = {h: [] for h in HEIGHTS}
    n_used = {h: 0 for h in HEIGHTS}

    for shp, rec in zip(reader.shapes(), reader.records()):
        if shp.shapeType not in (5, 15, 25):
            continue
        height_m = float(rec.as_dict().get("jzgd") or 0.0)
        rings = base._rings_from_shape(shp)
        if not rings:
            continue
        local_rings = []
        for ring in rings:
            xy = np.column_stack([ring[:, 0] - ox, ring[:, 1] - oy])
            if xy[:, 0].max() < XY_MIN or xy[:, 0].min() > XY_MAX:
                continue
            if xy[:, 1].max() < XY_MIN or xy[:, 1].min() > XY_MAX:
                continue
            local_rings.append(xy)
        if not local_rings:
            continue
        for h in HEIGHTS:
            if height_m + 1e-6 < h:
                continue
            for xy in local_rings:
                poly = Polygon(xy)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                geoms = getattr(poly, "geoms", None)
                if geoms is None:
                    by_height[h].append(poly)
                else:
                    by_height[h].extend(g for g in geoms if not g.is_empty)
            n_used[h] += 1

    axis = grid_axis()
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    masks = {}
    for h in HEIGHTS:
        polys = by_height[h]
        print(f"  height {h} m: {n_used[h]} buildings, {len(polys)} rings")
        if not polys:
            masks[h] = np.zeros((GRID_N, GRID_N), dtype=bool)
            continue
        merged = union_all(polys)
        inside = contains_xy(merged, xx.ravel(), yy.ravel())
        masks[h] = np.asarray(inside, dtype=bool).reshape(GRID_N, GRID_N)
        print(f"    masked cells: {int(masks[h].sum())}")
    return masks


def save_masks(masks: dict[int, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"inside_building_{h}m": masks[h] for h in HEIGHTS}
    payload["dx_m"] = np.array([DX])
    np.savez_compressed(path, **payload)
    print(f"Wrote {path}")


def load_masks(path: str) -> dict[int, np.ndarray]:
    with np.load(path) as data:
        return {h: data[f"inside_building_{h}m"].astype(bool) for h in HEIGHTS}


def load_cfd_grid(csv_path: str) -> dict[str, np.ndarray]:
    """Scatter the resampled CSV onto the 1000x1000 grid. Same columns as load_cfd_csv."""
    cfd = base.load_cfd_csv(csv_path)
    mask_col = pd.read_csv(csv_path, usecols=["vtkValidPointMask"])
    valid_pts = mask_col["vtkValidPointMask"].to_numpy() == 1
    if len(valid_pts) != len(cfd["x"]):
        raise RuntimeError(f"Row count mismatch in {csv_path}")

    iy, ix = xy_to_ij(cfd["x"], cfd["y"])
    ok = (iy >= 0) & (iy < GRID_N) & (ix >= 0) & (ix < GRID_N)
    if not np.all(ok):
        raise RuntimeError(f"Coordinates outside the 5 km grid: {csv_path}")

    speed = np.full((GRID_N, GRID_N), np.nan, dtype=np.float64)
    sampled = np.zeros((GRID_N, GRID_N), dtype=bool)
    speed[iy, ix] = cfd["wind_speed"]
    sampled[iy, ix] = valid_pts
    return {"wind_speed": speed, "sampled": sampled}


def smooth_fluid(field: np.ndarray, fluid: np.ndarray, smooth_m: float) -> np.ndarray:
    """Box mean on fluid cells only. Kernel size is a fixed physical length."""
    size = kernel_size(smooth_m)
    values = np.where(fluid, field, 0.0)
    weights = fluid.astype(np.float64)
    num = ndimage.uniform_filter(values, size=size, mode="constant", cval=0.0)
    den = ndimage.uniform_filter(weights, size=size, mode="constant", cval=0.0)
    out = np.full(field.shape, np.nan, dtype=np.float64)
    keep = fluid & (den >= SMOOTH_WEIGHT_FLOOR)
    out[keep] = num[keep] / den[keep]
    return out


def label_zones(
    field: np.ndarray,
    threshold: float,
    min_area_m2: float,
    above: bool,
    companion: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    if above:
        candidate = np.isfinite(field) & (field > threshold)
    else:
        candidate = np.isfinite(field) & (field < threshold)
    labels, nlab = ndimage.label(candidate, structure=np.ones((3, 3), dtype=int))
    min_cells = max(1, int(round(min_area_m2 / CELL_AREA)))
    axis = grid_axis()
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    kept = np.zeros(field.shape, dtype=bool)
    zones = []
    for lab in range(1, nlab + 1):
        sel = labels == lab
        n_cells = int(sel.sum())
        if n_cells < min_cells:
            continue
        kept[sel] = True
        vals = field[sel]
        zone = {
            "n_cells": n_cells,
            "area_m2": n_cells * CELL_AREA,
            "peak": float(np.nanmax(vals) if above else np.nanmin(vals)),
            "mean": float(np.nanmean(vals)),
            "centroid_x_m": float(xx[sel].mean()),
            "centroid_y_m": float(yy[sel].mean()),
            "mean_companion": float("nan"),
            "peak_companion": float("nan"),
        }
        if companion is not None:
            cvals = companion[sel]
            cvals = cvals[np.isfinite(cvals)]
            if cvals.size:
                zone["mean_companion"] = float(np.mean(cvals))
                zone["peak_companion"] = float(np.max(cvals) if above else np.min(cvals))
        zones.append(zone)
    return kept, zones


def wrf_speed_on_grid(cfd_dir: str, height: int) -> np.ndarray:
    wrf_nc, _, _ = base.infer_paths(cfd_dir, height)
    wrf_nc = base.resolve_existing_wrf_nc_path(wrf_nc)
    if not os.path.isfile(wrf_nc):
        raise FileNotFoundError(wrf_nc)
    # Crop wider than the 5 km CFD square so the interpolator covers the corners.
    wrf = base.extract_wrf_xy(
        wrf_nc,
        height,
        target_lat=base.TARGET_LAT,
        target_lon=base.TARGET_LON,
        lat_tol=0.04,
        lon_tol=0.04,
    )
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    ox, oy = origin_utm49n()
    east, north = to_utm.transform(wrf["lon"].ravel(), wrf["lat"].ravel())
    src = np.column_stack([east - ox, north - oy])
    u = wrf["u"].ravel()
    v = wrf["v"].ravel()
    finite = np.isfinite(u) & np.isfinite(v)
    fu = LinearNDInterpolator(src[finite], u[finite])
    fv = LinearNDInterpolator(src[finite], v[finite])
    axis = grid_axis()
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ui = fu(xx, yy)
    vi = fv(xx, yy)
    speed = np.sqrt(ui * ui + vi * vi)
    return speed


def _percentile(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def zone_summary(zones: list[dict]) -> dict[str, float]:
    empty = {
        "n_zones": 0,
        "area_m2": 0.0,
        "peak_of_peaks": float("nan"),
        "median_zone_peak": float("nan"),
        "mean_zone_mean": float("nan"),
        "area_weighted_mean": float("nan"),
        "mean_zone_mean_companion": float("nan"),
        "area_weighted_mean_companion": float("nan"),
        "peak_of_peaks_companion": float("nan"),
    }
    if not zones:
        return empty
    peaks = np.array([z["peak"] for z in zones], dtype=float)
    means = np.array([z["mean"] for z in zones], dtype=float)
    areas = np.array([z["area_m2"] for z in zones], dtype=float)
    comp = np.array([z["mean_companion"] for z in zones], dtype=float)
    comp_peak = np.array([z["peak_companion"] for z in zones], dtype=float)
    area_sum = float(areas.sum())
    finite_comp = np.isfinite(comp) & (areas > 0)
    return {
        "n_zones": len(zones),
        "area_m2": area_sum,
        "peak_of_peaks": float(np.nanmax(peaks)),
        "median_zone_peak": float(np.nanmedian(peaks)),
        "mean_zone_mean": float(np.nanmean(means)),
        "area_weighted_mean": float(np.sum(means * areas) / area_sum) if area_sum else float("nan"),
        "mean_zone_mean_companion": float(np.nanmean(comp)) if finite_comp.any() else float("nan"),
        "area_weighted_mean_companion": (
            float(np.sum(comp[finite_comp] * areas[finite_comp]) / areas[finite_comp].sum())
            if finite_comp.any() else float("nan")
        ),
        "peak_of_peaks_companion": float(np.nanmax(comp_peak)) if np.isfinite(comp_peak).any() else float("nan"),
    }


def analyze_case(
    cfd_dir: str,
    height: int,
    inside_building: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    csv_path = os.path.join(cfd_dir, "postProcessing", f"{height}m.csv")
    grid = load_cfd_grid(csv_path)
    fluid = grid["sampled"] & ~inside_building & interior_mask()
    v_cfd = grid["wind_speed"]
    v_wrf = wrf_speed_on_grid(cfd_dir, height)

    fluid_speed = v_cfd[fluid & np.isfinite(v_cfd)]
    domain_mean = float(np.mean(fluid_speed)) if fluid_speed.size else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full(v_cfd.shape, np.nan, dtype=np.float64)
        ratio_p = np.full(v_cfd.shape, np.nan, dtype=np.float64)
        usable = (
            fluid
            & np.isfinite(v_cfd)
            & np.isfinite(v_wrf)
            & (v_wrf >= WRF_SPEED_FLOOR)
        )
        ratio[usable] = v_cfd[usable] / v_wrf[usable]
        if np.isfinite(domain_mean) and domain_mean > 0:
            ratio_p[usable] = v_cfd[usable] / domain_mean

    ts = pd.Timestamp(base.parse_timestamp_from_cfd_dir(cfd_dir).replace("_", " "))
    ts = ts.tz_localize("UTC")
    bjt = ts.tz_convert(BJT)

    raw = ratio[np.isfinite(ratio)]
    raw_p = ratio_p[np.isfinite(ratio_p)]

    snap = {
        "time_utc": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
        "time_bjt": bjt.strftime("%Y-%m-%d %H:%M:%S%z"),
        "regime": regime_of(ts),
        "case": os.path.basename(cfd_dir),
        "height_m": height,
        "n_sampled": int(grid["sampled"].sum()),
        "n_fluid": int(fluid.sum()),
        "n_building_cells": int(inside_building.sum()),
        "n_sampled_inside_building": int((grid["sampled"] & inside_building).sum()),
        "n_wrf_nonfinite": int((fluid & ~np.isfinite(v_wrf)).sum()),
        "n_wrf_below_floor": int((fluid & np.isfinite(v_wrf) & (v_wrf < WRF_SPEED_FLOOR)).sum()),
        "n_ratio": int(usable.sum()),
        "domain_mean_cfd_m_s": domain_mean,
        "wrf_speed_floor_m_s": WRF_SPEED_FLOOR,
        "p50_R": _percentile(raw, 50),
        "p98_R": _percentile(raw, 98),
        "p99_R": _percentile(raw, 99),
        "max_R": float(np.max(raw)) if raw.size else float("nan"),
        "p50_Rp": _percentile(raw_p, 50),
        "p98_Rp": _percentile(raw_p, 98),
        "p99_Rp": _percentile(raw_p, 99),
        "max_Rp": float(np.max(raw_p)) if raw_p.size else float("nan"),
    }

    sens_rows = []
    zone_rows = []
    for spec in SENSITIVITY:
        sm_r = smooth_fluid(ratio, usable, spec["smooth_m"])
        sm_p = smooth_fluid(ratio_p, usable, spec["smooth_m"])
        # Zones are detected on the interior Λ' field. Λ is reported inside them.
        acc_p, zones_p = label_zones(
            sm_p, spec["threshold"], spec["min_area_m2"], above=True, companion=sm_r,
        )
        _, zones_d = label_zones(sm_p, DECEL_THRESHOLD, spec["min_area_m2"], above=False)
        sp = zone_summary(zones_p)
        sd = zone_summary(zones_d)
        sens_rows.append(
            {
                "time_utc": snap["time_utc"],
                "regime": snap["regime"],
                "height_m": height,
                "setting": spec["setting"],
                "smooth_m": spec["smooth_m"],
                "threshold": spec["threshold"],
                "min_area_m2": spec["min_area_m2"],
                "p98_R": snap["p98_R"],
                "p98_Rp": snap["p98_Rp"],
                "n_accel_R": sp["n_zones"],
                "mean_zone_mean_R": sp["mean_zone_mean_companion"],
                "median_zone_peak_R": float("nan"),
                "peak_of_peaks_R": sp["peak_of_peaks_companion"],
                "area_weighted_mean_R": sp["area_weighted_mean_companion"],
                "zone_area_m2": sp["area_m2"],
                "n_accel_Rp": sp["n_zones"],
                "mean_zone_mean_Rp": sp["mean_zone_mean"],
                "peak_of_peaks_Rp": sp["peak_of_peaks"],
                "n_decel_R": sd["n_zones"],
                "mean_zone_mean_decel_R": sd["mean_zone_mean"],
                "jaccard_accel_R_Rp": float("nan"),
            }
        )
        if spec["setting"] != "baseline":
            continue
        snap.update(
            {
                "smooth_m": spec["smooth_m"],
                "threshold": spec["threshold"],
                "min_area_m2": spec["min_area_m2"],
                "n_interior": int(usable.sum()),
                "n_accel_zones": sp["n_zones"],
                "accel_zone_area_m2": sp["area_m2"],
                "accel_mean_of_zone_means": sp["mean_zone_mean_companion"],
                "accel_area_weighted_mean": sp["area_weighted_mean_companion"],
                "accel_median_zone_peak": float("nan"),
                "accel_peak_of_peaks": sp["peak_of_peaks_companion"],
                "n_accel_zones_Rp": sp["n_zones"],
                "accel_mean_of_zone_means_Rp": sp["mean_zone_mean"],
                "accel_area_weighted_mean_Rp": sp["area_weighted_mean"],
                "accel_peak_of_peaks_Rp": sp["peak_of_peaks"],
                "jaccard_accel_R_Rp": float("nan"),
                "n_decel_zones": sd["n_zones"],
                "decel_mean_of_zone_means": sd["mean_zone_mean"],
                "decel_peak_of_peaks": sd["peak_of_peaks"],
            }
        )
        for kind, zones in (("accel_Rp", zones_p), ("decel_Rp", zones_d)):
            for zi, z in enumerate(zones, start=1):
                lon, lat = xy_to_lonlat(z["centroid_x_m"], z["centroid_y_m"])
                zone_rows.append(
                    {
                        "time_utc": snap["time_utc"],
                        "regime": snap["regime"],
                        "height_m": height,
                        "kind": kind,
                        "zone_id": zi,
                        "peak": z["peak"],
                        "mean": z["mean"],
                        "peak_R": z["peak_companion"],
                        "mean_R": z["mean_companion"],
                        "area_m2": z["area_m2"],
                        "n_cells": z["n_cells"],
                        "centroid_x_m": z["centroid_x_m"],
                        "centroid_y_m": z["centroid_y_m"],
                        "centroid_lon": lon,
                        "centroid_lat": lat,
                    }
                )
    return snap, zone_rows, sens_rows


def _lag1(series: pd.Series, times: pd.Series) -> float:
    """Lag-1 correlation using only pairs separated by about one hour."""
    df = pd.DataFrame({"t": pd.to_datetime(times, utc=True), "v": series}).dropna()
    df = df.sort_values("t")
    if len(df) < 3:
        return float("nan")
    dt_h = df["t"].diff().dt.total_seconds() / 3600.0
    paired = dt_h.between(0.5, 1.5)
    a = df["v"].shift(1)[paired]
    b = df["v"][paired]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_summaries(snapshot_csv: str, sens_csv: str) -> None:
    snap = pd.read_csv(snapshot_csv)
    snap["time_utc"] = pd.to_datetime(snap["time_utc"], utc=True)
    rows = []
    for (regime, height), g in snap.groupby(["regime", "height_m"], sort=True):
        g = g.sort_values("time_utc")
        rows.append(
            {
                "regime": regime,
                "height_m": int(height),
                "n_snapshots": int(len(g)),
                "time_start_utc": g["time_utc"].iloc[0].strftime("%Y-%m-%d %H:%M:%S%z"),
                "time_end_utc": g["time_utc"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S%z"),
                "mean_p98_R": float(g["p98_R"].mean()),
                "median_p98_R": float(g["p98_R"].median()),
                "min_p98_R": float(g["p98_R"].min()),
                "max_p98_R": float(g["p98_R"].max()),
                "mean_p99_R": float(g["p99_R"].mean()),
                "min_p99_R": float(g["p99_R"].min()),
                "max_p99_R": float(g["p99_R"].max()),
                "mean_accel_zone_mean_R": float(g["accel_mean_of_zone_means"].mean()),
                "min_accel_zone_mean_R": float(g["accel_mean_of_zone_means"].min()),
                "max_accel_zone_mean_R": float(g["accel_mean_of_zone_means"].max()),
                "mean_n_accel_zones": float(g["n_accel_zones"].mean()),
                "mean_p98_Rp": float(g["p98_Rp"].mean()),
                "median_p98_Rp": float(g["p98_Rp"].median()),
                "min_p98_Rp": float(g["p98_Rp"].min()),
                "max_p98_Rp": float(g["p98_Rp"].max()),
                "mean_jaccard_R_Rp": float(g["jaccard_accel_R_Rp"].mean()),
                "lag1_corr_p98_R": _lag1(g["p98_R"], g["time_utc"]),
                "interpretation": AUTOCORR_NOTE,
            }
        )
    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(REGIME_CSV, index=False)
    print(f"Wrote {REGIME_CSV} ({len(regime_df)} rows)")

    sens = pd.read_csv(sens_csv)
    sens_rows = []
    for (setting, regime, height), g in sens.groupby(
        ["setting", "regime", "height_m"], sort=True
    ):
        sens_rows.append(
            {
                "setting": setting,
                "regime": regime,
                "height_m": int(height),
                "n_snapshots": int(len(g)),
                "smooth_m": float(g["smooth_m"].iloc[0]),
                "threshold": float(g["threshold"].iloc[0]),
                "min_area_m2": float(g["min_area_m2"].iloc[0]),
                "mean_p98_R": float(g["p98_R"].mean()),
                "median_p98_R": float(g["p98_R"].median()),
                "min_p98_R": float(g["p98_R"].min()),
                "max_p98_R": float(g["p98_R"].max()),
                "mean_n_accel_R": float(g["n_accel_R"].mean()),
                "mean_zone_mean_R": float(g["mean_zone_mean_R"].mean()),
                "mean_peak_of_peaks_R": float(g["peak_of_peaks_R"].mean()),
                "min_peak_of_peaks_R": float(g["peak_of_peaks_R"].min()),
                "max_peak_of_peaks_R": float(g["peak_of_peaks_R"].max()),
                "mean_n_accel_Rp": float(g["n_accel_Rp"].mean()),
                "mean_zone_mean_Rp": float(g["mean_zone_mean_Rp"].mean()),
                "mean_peak_of_peaks_Rp": float(g["peak_of_peaks_Rp"].mean()),
                "mean_jaccard_accel_R_Rp": float(g["jaccard_accel_R_Rp"].mean()),
                "interpretation": AUTOCORR_NOTE,
            }
        )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(SENS_CSV, index=False)
    print(f"Wrote {SENS_CSV} ({len(sens_df)} rows)")


def _append_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.isfile(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def done_keys(path: str) -> set[tuple[str, int]]:
    if not os.path.isfile(path):
        return set()
    df = pd.read_csv(path, usecols=["case", "height_m"])
    # snapshot file stores case; if an older partial file lacks it, fall back
    return set(zip(df["case"].astype(str), df["height_m"].astype(int)))


def run_all(root: str, limit: int | None) -> None:
    if not os.path.isfile(MASK_NPZ):
        raise SystemExit(f"Missing {MASK_NPZ}; run build-mask first")
    masks = load_masks(MASK_NPZ)
    cases = iter_target_cases(root)
    if limit is not None:
        cases = cases[:limit]
    finished = done_keys(SNAPSHOT_CSV)
    os.chdir(_REPO)
    n_skip = n_run = n_miss = 0
    for case_dir in cases:
        name = os.path.basename(case_dir)
        for height in HEIGHTS:
            csv_path = os.path.join(case_dir, "postProcessing", f"{height}m.csv")
            # Skip files still being written by pvbatch. A finished slice is ~88 MB.
            if not os.path.isfile(csv_path) or os.path.getsize(csv_path) < 80_000_000:
                n_miss += 1
                continue
            if (name, height) in finished:
                n_skip += 1
                continue
            print(f"Analyze {name} {height} m", flush=True)
            snap, zones, sens = analyze_case(case_dir, height, masks[height])
            _append_rows(ZONES_CSV, zones)
            _append_rows(SENS_SNAPSHOT_CSV, sens)
            _append_rows(SNAPSHOT_CSV, [snap])
            finished.add((name, height))
            n_run += 1
    print(f"Analyzed {n_run}, skipped {n_skip}, missing CSV {n_miss}")


def write_case_list(root: str) -> None:
    cases = iter_target_cases(root)
    os.makedirs(OUT_DIR, exist_ok=True)
    pending = []
    for case_dir in cases:
        need = [
            h
            for h in HEIGHTS
            if not os.path.isfile(os.path.join(case_dir, "postProcessing", f"{h}m.csv"))
        ]
        if need:
            foam = os.path.join(case_dir, "myExpxx.foam")
            pending.append(foam + " " + " ".join(str(h) for h in need))
    with open(CASE_LIST, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pending))
        if pending:
            fh.write("\n")
    print(f"Wrote {CASE_LIST} ({len(pending)} cases still need slices)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-mask", help="Rasterize height-aware building footprints")
    b.add_argument("--shp", default=None)

    sub.add_parser("list-cases", help="Write foam paths that still lack 30/60/120 m CSV")

    r = sub.add_parser("run", help="Compute R, R', zones, and sensitivity rows")
    r.add_argument("--limit", type=int, default=None, help="Only the first N target cases")

    sub.add_parser("summarize", help="Regime and sensitivity tables from the snapshot CSVs")

    a = sub.add_parser("all", help="build-mask, run, summarize")
    a.add_argument("--limit", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = os.path.join(_REPO, "steady_experiments_finer_ABL")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    if args.cmd == "build-mask":
        shp = args.shp or os.path.join(_REPO, base.DEFAULT_SHP)
        print(f"Shapefile: {shp}")
        masks = build_building_masks(shp)
        save_masks(masks, MASK_NPZ)
    elif args.cmd == "list-cases":
        write_case_list(root)
    elif args.cmd == "run":
        run_all(root, args.limit)
    elif args.cmd == "summarize":
        write_summaries(SNAPSHOT_CSV, SENS_SNAPSHOT_CSV)
    elif args.cmd == "all":
        shp = os.path.join(_REPO, base.DEFAULT_SHP)
        if not os.path.isfile(MASK_NPZ):
            save_masks(build_building_masks(shp), MASK_NPZ)
        run_all(root, args.limit)
        if os.path.isfile(SNAPSHOT_CSV):
            write_summaries(SNAPSHOT_CSV, SENS_SNAPSHOT_CSV)
    else:
        raise SystemExit(args.cmd)


if __name__ == "__main__":
    main()
