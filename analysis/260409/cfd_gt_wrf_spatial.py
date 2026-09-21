#!/usr/bin/env python3
"""Spatial diagnostics for hours where CFD > WRF at 52-300 m.

1. WRF cartesian NC: lidar-site WS vs inlet-face WS (z=200 m) for every hour.
2. Example hour 2025-09-09 06:00 UTC: CFD 100 m slice vs WRF at 100 m.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "cfd_gt_wrf"
OUT.mkdir(parents=True, exist_ok=True)

SITES = {
    "GAW103": (975.0, -320.0),
    "GAW104": (450.0, 350.0),
    "GAW111": (75.0, 30.0),
}
FACE_STRIP = 200.0  # m from domain edge


def find_nc(dt: pd.Timestamp) -> Path | None:
    stem = f"auxhist2_d03_{dt.strftime('%Y-%m-%d_%H:%M:%S')}_1h-rolling_cartesian.nc"
    roots = [REPO / "W_myExp05" / "auxhist2", REPO / "W_myExp03" / "auxhist2"]
    names = [stem, stem.replace(":", "%3A")]
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                return p
    return None


def wrf_spatial_at_z(nc_path: Path, z_target: float) -> dict:
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        x = ds["x_rel"].values.squeeze().astype(float)
        y = ds["y_rel"].values.squeeze().astype(float)
        z = ds["z"].values.squeeze().astype(float)
        u = ds["U"].values.squeeze().astype(float)
        v = ds["V"].values.squeeze().astype(float)
        ws = ds["WS"].values.squeeze().astype(float)
    iz = int(np.argmin(np.abs(z - z_target)))
    u2 = u[iz]
    v2 = v[iz]
    ws2 = ws[iz]
    xx, yy = np.meshgrid(x, y)
    xmax = float(np.nanmax(np.abs(x)))
    ymax = float(np.nanmax(np.abs(y)))
    faces = {
        "west": np.abs(xx + xmax) <= FACE_STRIP,
        "east": np.abs(xx - xmax) <= FACE_STRIP,
        "south": np.abs(yy + ymax) <= FACE_STRIP,
        "north": np.abs(yy - ymax) <= FACE_STRIP,
    }
    site = {}
    for name, (sx, sy) in SITES.items():
        i = int(np.argmin(np.abs(x - sx)))
        j = int(np.argmin(np.abs(y - sy)))
        site[name] = {
            "ws": float(ws2[j, i]),
            "u": float(u2[j, i]),
            "v": float(v2[j, i]),
        }
    su = float(np.mean([d["u"] for d in site.values()]))
    sv = float(np.mean([d["v"] for d in site.values()]))
    # inlet: outward normal velocity < 0
    outward = {"west": (-1, 0), "east": (1, 0), "south": (0, -1), "north": (0, 1)}
    inlets = [p for p, (nx, ny) in outward.items() if su * nx + sv * ny < 0]
    face_ws = {p: float(np.nanmean(ws2[mask])) for p, mask in faces.items()}
    inlet_mask = np.zeros_like(ws2, dtype=bool)
    for p in inlets:
        inlet_mask |= faces[p]
    return {
        "z": float(z[iz]),
        "site_mean_ws": float(np.mean([d["ws"] for d in site.values()])),
        "site_min_ws": float(np.min([d["ws"] for d in site.values()])),
        "domain_mean_ws": float(np.nanmean(ws2)),
        "domain_p10_ws": float(np.nanpercentile(ws2, 10)),
        "domain_p50_ws": float(np.nanpercentile(ws2, 50)),
        "inlets": ",".join(inlets),
        "inlet_mean_ws": float(np.nanmean(ws2[inlet_mask])) if inlet_mask.any() else np.nan,
        **{f"face_{p}": face_ws[p] for p in faces},
        "u_site": su,
        "v_site": sv,
    }


def scan_all_hours() -> pd.DataFrame:
    hourly = pd.read_csv(OUT / "hourly_3station_layers.csv", parse_dates=["datetime"])
    rows = []
    missing = 0
    for dt in hourly["datetime"]:
        nc = find_nc(pd.Timestamp(dt))
        if nc is None:
            missing += 1
            continue
        try:
            d200 = wrf_spatial_at_z(nc, 200.0)
            d900 = wrf_spatial_at_z(nc, 900.0)
        except Exception as exc:
            print(f"FAIL {dt}: {exc}")
            continue
        rec = {"datetime": dt}
        rec.update({f"{k}_z200": v for k, v in d200.items()})
        rec["inlet_minus_site_z200"] = d200["inlet_mean_ws"] - d200["site_mean_ws"]
        rec["inlet_minus_site_z900"] = d900["inlet_mean_ws"] - d900["site_mean_ws"]
        rec["inlets_z900"] = d900["inlets"]
        rec["site_mean_ws_z900"] = d900["site_mean_ws"]
        rec["inlet_mean_ws_z900"] = d900["inlet_mean_ws"]
        rows.append(rec)
    print(f"spatial scan: {len(rows)} hours, missing nc={missing}")
    spat = pd.DataFrame(rows)
    spat["datetime"] = pd.to_datetime(spat["datetime"])
    merged = hourly.merge(spat, on="datetime", how="left")
    return merged


def nearest_cfd(df: pd.DataFrame, x: float, y: float) -> pd.Series:
    d2 = (df["x"] - x) ** 2 + (df["y"] - y) ** 2
    return df.iloc[int(d2.to_numpy().argmin())]


def example_100m() -> dict:
    csv = (
        REPO
        / "steady_experiments_finer_ABL"
        / "20250909_0600_two_boundaries_as_outlet"
        / "postProcessing"
        / "100m.csv"
    )
    cfd = pd.read_csv(
        csv,
        usecols=["U:0", "U:1", "Coords:0", "Coords:1", "vtkValidPointMask"],
    )
    cfd = cfd[cfd["vtkValidPointMask"] == 1].copy()
    cfd["x"] = cfd["Coords:0"]
    cfd["y"] = cfd["Coords:1"]
    cfd["ws"] = np.hypot(cfd["U:0"], cfd["U:1"])
    # valid fluid points only; buildings are mask=0 already
    inner = cfd[(cfd["x"].abs() <= 2500) & (cfd["y"].abs() <= 2500)]
    sites = {}
    for name, (sx, sy) in SITES.items():
        row = nearest_cfd(cfd, sx, sy)
        sites[name] = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "ws": float(row["ws"]),
            "u": float(row["U:0"]),
            "v": float(row["U:1"]),
        }
    # river-ish corridor: |x|<400 around GAW111, vs building-dense east
    river = inner[inner["x"].abs() <= 400]
    east_urban = inner[inner["x"] >= 600]
    west = inner[inner["x"] <= -600]
    # WRF at 100 m
    nc = find_nc(pd.Timestamp("2025-09-09 06:00:00"))
    wrf = wrf_spatial_at_z(nc, 100.0) if nc is not None else {}
    wrf200 = wrf_spatial_at_z(nc, 200.0) if nc is not None else {}
    return {
        "cfd_n": int(len(cfd)),
        "cfd_xy_range": (float(cfd["x"].min()), float(cfd["x"].max()), float(cfd["y"].min()), float(cfd["y"].max())),
        "cfd_mean_ws_all": float(cfd["ws"].mean()),
        "cfd_p50_ws_all": float(cfd["ws"].median()),
        "cfd_mean_inner5km": float(inner["ws"].mean()),
        "cfd_p50_inner5km": float(inner["ws"].median()),
        "cfd_mean_river_x400": float(river["ws"].mean()) if len(river) else np.nan,
        "cfd_mean_east_x600": float(east_urban["ws"].mean()) if len(east_urban) else np.nan,
        "cfd_mean_west_x-600": float(west["ws"].mean()) if len(west) else np.nan,
        "cfd_sites": sites,
        "cfd_site_mean": float(np.mean([d["ws"] for d in sites.values()])),
        "wrf_z100": wrf,
        "wrf_z200": wrf200,
    }


def main() -> None:
    merged = scan_all_hours()
    merged.to_csv(OUT / "hourly_with_wrf_inlet_site.csv", index=False)

    valid = merged.dropna(subset=["inlet_minus_site_z200"])
    print("\n=== WRF inlet vs lidar-site (z≈200 m) vs CFD-WRF low-layer ===")
    print(f"n with NC = {len(valid)}")
    gt = valid[valid["d_cfd_wrf"] > 0]
    le = valid[valid["d_cfd_wrf"] <= 0]
    for label, g in [("CFD>WRF", gt), ("CFD<=WRF", le), ("all", valid)]:
        if len(g) == 0:
            continue
        print(
            f"{label:10s} n={len(g):3d}  "
            f"WRF_site200={g['site_mean_ws_z200'].mean():.2f}  "
            f"WRF_inlet200={g['inlet_mean_ws_z200'].mean():.2f}  "
            f"inlet-site={g['inlet_minus_site_z200'].mean():+.2f}  "
            f"dCFD-WRF={g['d_cfd_wrf'].mean():+.2f}  "
            f"ws_wrf_low={g['ws_wrf'].mean():.2f}"
        )
    r = valid["d_cfd_wrf"].corr(valid["inlet_minus_site_z200"])
    r2 = valid["d_cfd_wrf"].corr(valid["ws_wrf"])
    r3 = valid["d_cfd_wrf"].corr(valid["site_mean_ws_z200"])
    print(f"corr(d_cfd_wrf, inlet-site z200) = {r:.3f}")
    print(f"corr(d_cfd_wrf, ws_wrf 52-300)   = {r2:.3f}")
    print(f"corr(d_cfd_wrf, WRF site z200)   = {r3:.3f}")

    print("\nBy campaign:")
    for camp, g0 in valid.groupby("campaign"):
        ggt = g0[g0["d_cfd_wrf"] > 0]
        gle = g0[g0["d_cfd_wrf"] <= 0]
        print(
            f"  {camp}: CFD>WRF inlet-site={ggt['inlet_minus_site_z200'].mean():+.2f} "
            f"(n={len(ggt)}); CFD<=WRF inlet-site={gle['inlet_minus_site_z200'].mean():+.2f} "
            f"(n={len(gle)})"
        )

    print("\nExample hour in merged table:")
    ex = valid[valid["datetime"] == pd.Timestamp("2025-09-09 06:00:00")]
    cols = [
        "campaign",
        "ws_obs",
        "ws_wrf",
        "ws_cfd",
        "d_cfd_wrf",
        "site_mean_ws_z200",
        "inlet_mean_ws_z200",
        "inlet_minus_site_z200",
        "inlets_z200",
        "domain_mean_ws_z200",
        "face_south_z200",
        "face_west_z200",
        "face_east_z200",
        "face_north_z200",
    ]
    with pd.option_context("display.max_columns", 40, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(ex[cols].to_string(index=False))

    print("\nLargest 8 CFD-WRF hours with inlet contrast:")
    top = valid.sort_values("d_cfd_wrf", ascending=False).head(8)
    show = [
        "campaign",
        "datetime",
        "lst_hour",
        "ws_obs",
        "ws_wrf",
        "ws_cfd",
        "d_cfd_wrf",
        "site_mean_ws_z200",
        "inlet_mean_ws_z200",
        "inlet_minus_site_z200",
        "inlets_z200",
        "d_cfd_wrf_aloft",
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(top[show].to_string(index=False))

    print("\n=== CFD 100 m slice vs WRF z=100, 2025-09-09 06:00 UTC ===")
    info = example_100m()
    for k, v in info.items():
        if k not in {"wrf_z100", "wrf_z200", "cfd_sites"}:
            print(f"  {k}: {v}")
    print("  CFD nearest-cell at lidar XY:")
    for name, d in info["cfd_sites"].items():
        print(f"    {name}: {d}")
    print("  WRF z100:", info["wrf_z100"])
    print("  WRF z200:", info["wrf_z200"])


if __name__ == "__main__":
    main()
