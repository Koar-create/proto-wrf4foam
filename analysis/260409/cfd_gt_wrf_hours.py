#!/usr/bin/env python3
"""Find hours where WRF-to-OpenFOAM (CFD) exceeds standalone WRF.

Focus: 52-300 m (manuscript layer) and 800-1000 m (aloft, near forcing).
Campaigns: 260409 (1-5 Sep) and 260707 (6-13 Sep).
Outputs: results/cfd_gt_wrf/*.csv and a stdout report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "cfd_gt_wrf"
OUT.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = {
    "260409": REPO / "data/260409/processed/merged_lidar_simulation_final.csv",
    "260707": REPO / "data/260707/processed/merged_lidar_simulation_final.csv",
}

LOW = (52.0, 300.0)
ALOFT = (800.0, 1000.0)
LST_OFFSET = pd.Timedelta(hours=8)
SITES = ("GAW103", "GAW104", "GAW111")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["lst"] = df["datetime"] + LST_OFFSET
    df["lst_hour"] = df["lst"].dt.hour
    df["is_day"] = df["lst_hour"].between(7, 18)
    return df


def layer_means(df: pd.DataFrame, zmin: float, zmax: float, by: list[str]) -> pd.DataFrame:
    sub = df[(df["Height"] >= zmin) & (df["Height"] <= zmax)].copy()
    g = sub.groupby(by, observed=True)
    out = g.agg(
        n=("ws_obs", "size"),
        ws_obs=("ws_obs", "mean"),
        ws_wrf=("ws_wrf", "mean"),
        ws_cfd=("ws_cfd", "mean"),
        k_wrf=("k_wrf", "mean"),
        k_cfd=("k_cfd", "mean"),
        u_wrf=("u_wrf", "mean"),
        v_wrf=("v_wrf", "mean"),
        u_cfd=("u_cfd", "mean"),
        v_cfd=("v_cfd", "mean"),
        u_obs=("u_obs", "mean"),
        v_obs=("v_obs", "mean"),
    ).reset_index()
    out["d_cfd_wrf"] = out["ws_cfd"] - out["ws_wrf"]
    out["wrf_bias"] = out["ws_wrf"] - out["ws_obs"]
    out["cfd_bias"] = out["ws_cfd"] - out["ws_obs"]
    out["wd_wrf"] = (np.degrees(np.arctan2(-out["u_wrf"], -out["v_wrf"])) + 360) % 360
    out["wd_cfd"] = (np.degrees(np.arctan2(-out["u_cfd"], -out["v_cfd"])) + 360) % 360
    out["wd_obs"] = (np.degrees(np.arctan2(-out["u_obs"], -out["v_obs"])) + 360) % 360
    return out


def wd_sector(wd: pd.Series) -> pd.Series:
    bins = np.array([0, 45, 135, 225, 315, 360])
    labels = ["N", "E", "S", "W", "N2"]
    cat = pd.cut(wd, bins=bins, labels=labels, include_lowest=True, right=False)
    return cat.replace({"N2": "N"})


def summarize_campaign(name: str, df: pd.DataFrame) -> dict:
    df = df[df["obtid"].isin(SITES)].copy()
    hours_low = layer_means(df, *LOW, by=["datetime"])
    hours_aloft = layer_means(df, *ALOFT, by=["datetime"])
    site_low = layer_means(df, *LOW, by=["datetime", "obtid"])

    hours_low["lst"] = hours_low["datetime"] + LST_OFFSET
    hours_low["lst_hour"] = hours_low["lst"].dt.hour
    hours_low["is_day"] = hours_low["lst_hour"].between(7, 18)
    hours_low["wd_sector"] = wd_sector(hours_low["wd_wrf"])
    hours_aloft = hours_aloft.rename(
        columns={
            "ws_obs": "ws_obs_aloft",
            "ws_wrf": "ws_wrf_aloft",
            "ws_cfd": "ws_cfd_aloft",
            "d_cfd_wrf": "d_cfd_wrf_aloft",
            "wrf_bias": "wrf_bias_aloft",
            "cfd_bias": "cfd_bias_aloft",
        }
    )
    merged = hours_low.merge(
        hours_aloft[
            [
                "datetime",
                "ws_obs_aloft",
                "ws_wrf_aloft",
                "ws_cfd_aloft",
                "d_cfd_wrf_aloft",
                "wrf_bias_aloft",
                "cfd_bias_aloft",
            ]
        ],
        on="datetime",
        how="left",
    )

    n = len(merged)
    gt = merged[merged["d_cfd_wrf"] > 0]
    gt_strong = merged[merged["d_cfd_wrf"] > 0.3]
    gt_aloft = merged[merged["d_cfd_wrf_aloft"] > 0]
    both = merged[(merged["d_cfd_wrf"] > 0) & (merged["d_cfd_wrf_aloft"] > 0)]
    low_only = merged[(merged["d_cfd_wrf"] > 0) & (merged["d_cfd_wrf_aloft"] <= 0)]

    # WRF already weaker than obs among CFD>WRF hours
    wrf_under_in_gt = gt[gt["wrf_bias"] < 0] if len(gt) else gt
    cfd_closer = (
        gt[gt["cfd_bias"].abs() < gt["wrf_bias"].abs()] if len(gt) else gt
    )

    print("=" * 72)
    print(f"CAMPAIGN {name}  n_hours={n}  sites={list(SITES)}  layer=52-300 m")
    print("=" * 72)
    print(f"  mean(ws_cfd - ws_wrf)                 = {merged['d_cfd_wrf'].mean():+.3f} m/s")
    print(f"  median(ws_cfd - ws_wrf)               = {merged['d_cfd_wrf'].median():+.3f} m/s")
    print(f"  hours CFD > WRF                       = {len(gt)} / {n}  ({100*len(gt)/n:.1f}%)")
    print(f"  hours CFD - WRF > 0.3 m/s             = {len(gt_strong)} / {n}")
    print(f"  hours CFD > WRF also in 800-1000 m    = {len(gt_aloft)} / {n}")
    print(f"  CFD>WRF in BOTH low and aloft         = {len(both)} / {n}")
    print(f"  CFD>WRF low ONLY (aloft CFD<=WRF)     = {len(low_only)} / {n}")
    if n:
        print(f"  WRF underestimates obs (all hours)    = {(merged['wrf_bias']<0).sum()} / {n}")
    if len(gt):
        print(f"  among CFD>WRF: WRF already < obs      = {len(wrf_under_in_gt)} / {len(gt)}")
        print(f"  among CFD>WRF: |CFD bias| < |WRF bias|= {len(cfd_closer)} / {len(gt)}")
        print(f"  among CFD>WRF: mean WRF ws            = {gt['ws_wrf'].mean():.2f} m/s")
        print(f"  among CFD<=WRF: mean WRF ws           = {merged.loc[merged['d_cfd_wrf']<=0,'ws_wrf'].mean():.2f} m/s")
        print(f"  among CFD>WRF: daytime fraction       = {gt['is_day'].mean():.2f}")
        print(f"  among CFD<=WRF: daytime fraction      = {merged.loc[merged['d_cfd_wrf']<=0,'is_day'].mean():.2f}")
        print("  CFD>WRF by LST hour:")
        print(gt.groupby("lst_hour").size().reindex(range(24), fill_value=0).to_string())
        print("  CFD>WRF by WRF wd sector:")
        print(gt.groupby("wd_sector", observed=True).size().to_string())

    print("\n  Top 12 hours with largest CFD-WRF (52-300 m, 3-station mean):")
    cols = [
        "datetime",
        "lst",
        "ws_obs",
        "ws_wrf",
        "ws_cfd",
        "d_cfd_wrf",
        "wrf_bias",
        "cfd_bias",
        "d_cfd_wrf_aloft",
        "wd_wrf",
        "lst_hour",
    ]
    top = merged.sort_values("d_cfd_wrf", ascending=False).head(12)
    with pd.option_context("display.max_columns", 20, "display.width", 160, "display.float_format", "{:.2f}".format):
        print(top[cols].to_string(index=False))

    # per-station agreement: how many of 3 sites have CFD>WRF
    site_flag = site_low.assign(gt=site_low["d_cfd_wrf"] > 0)
    n_sites_gt = site_flag.groupby("datetime")["gt"].sum().rename("n_sites_cfd_gt_wrf")
    merged = merged.merge(n_sites_gt, on="datetime", how="left")
    if len(gt):
        gt2 = merged[merged["d_cfd_wrf"] > 0]
        print("\n  among 3-station-mean CFD>WRF hours, how many individual sites agree:")
        print(gt2["n_sites_cfd_gt_wrf"].value_counts().sort_index().to_string())

    merged["campaign"] = name
    site_low["campaign"] = name
    site_low["lst"] = site_low["datetime"] + LST_OFFSET
    return {
        "hourly": merged,
        "site_low": site_low,
        "n": n,
        "n_gt": len(gt),
        "n_both": len(both),
        "n_low_only": len(low_only),
    }


def profile_one_hour(df: pd.DataFrame, utc: str) -> pd.DataFrame:
    sub = df[(df["datetime"] == pd.Timestamp(utc)) & df["obtid"].isin(SITES)].copy()
    sub["zbin"] = pd.cut(sub["Height"], bins=np.arange(0, 1050, 50))
    agg = sub.groupby(["obtid", "zbin"], observed=True).agg(
        z=("Height", "mean"),
        ws_obs=("ws_obs", "mean"),
        ws_wrf=("ws_wrf", "mean"),
        ws_cfd=("ws_cfd", "mean"),
    ).reset_index()
    agg["d"] = agg["ws_cfd"] - agg["ws_wrf"]
    return agg


def maybe_inlet_vs_site(utc: str) -> dict | None:
    """Compare WRF cartesian WS at lidar sites vs domain-edge mean at ~200 m and ~900 m."""
    dt = pd.Timestamp(utc)
    stem = f"auxhist2_d03_{dt.strftime('%Y-%m-%d_%H:%M:%S')}_1h-rolling_cartesian.nc"
    candidates = [
        REPO / "W_myExp05" / "auxhist2" / stem,
        REPO / "W_myExp03" / "auxhist2" / stem,
        REPO / "W_myExp05" / "auxhist2" / stem.replace(":", "%3A"),
        REPO / "W_myExp03" / "auxhist2" / stem.replace(":", "%3A"),
    ]
    nc_path = next((p for p in candidates if p.is_file()), None)
    if nc_path is None:
        return None
    try:
        import xarray as xr
    except ImportError:
        return None

    sites = {
        "GAW103": (975.0, -320.0),
        "GAW104": (450.0, 350.0),
        "GAW111": (75.0, 30.0),
    }
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        x = ds["x_rel"].values.squeeze().astype(float)
        y = ds["y_rel"].values.squeeze().astype(float)
        z = ds["z"].values.squeeze().astype(float)
        ws = ds["WS"].values.squeeze().astype(float)  # (z, y, x) expected
        if ws.ndim != 3:
            return None
        # guess dim order: (nz, ny, nx)
        if ws.shape[1] == y.size and ws.shape[2] == x.size:
            pass
        elif ws.shape[0] == y.size and ws.shape[1] == x.size:
            ws = np.moveaxis(ws, 2, 0)
        else:
            return {"nc": str(nc_path), "shape": list(ws.shape), "note": "unexpected WS shape"}

        def nearest_z(z0):
            return int(np.argmin(np.abs(z - z0)))

        iz200 = nearest_z(200.0)
        iz900 = nearest_z(900.0)
        # domain-edge strip: outer 500 m on each side, interior 8 km
        edge = 500.0
        on_edge = (np.abs(x) >= (np.nanmax(np.abs(x)) - edge))[:, None] | (
            np.abs(y) >= (np.nanmax(np.abs(y)) - edge)
        )[None, :]
        # x is 1d, y is 1d; build 2d mask (ny, nx)
        xx, yy = np.meshgrid(x, y)
        xmax = float(np.nanmax(np.abs(x)))
        ymax = float(np.nanmax(np.abs(y)))
        edge_mask = (np.abs(xx) >= xmax - edge) | (np.abs(yy) >= ymax - edge)
        interior_mask = (np.abs(xx) < 1500.0) & (np.abs(yy) < 1500.0)

        def mean_ws(iz, mask):
            sl = ws[iz]
            return float(np.nanmean(sl[mask]))

        site_ws_200 = {}
        site_ws_900 = {}
        for name, (sx, sy) in sites.items():
            i = int(np.argmin(np.abs(x - sx)))
            j = int(np.argmin(np.abs(y - sy)))
            site_ws_200[name] = float(ws[iz200, j, i])
            site_ws_900[name] = float(ws[iz900, j, i])

        return {
            "nc": str(nc_path),
            "z200": float(z[iz200]),
            "z900": float(z[iz900]),
            "edge_ws_200": mean_ws(iz200, edge_mask),
            "edge_ws_900": mean_ws(iz900, edge_mask),
            "interior_ws_200": mean_ws(iz200, interior_mask),
            "interior_ws_900": mean_ws(iz900, interior_mask),
            "site_mean_200": float(np.mean(list(site_ws_200.values()))),
            "site_mean_900": float(np.mean(list(site_ws_900.values()))),
            "site_ws_200": site_ws_200,
            "site_ws_900": site_ws_900,
        }


def main() -> None:
    all_hourly = []
    all_site = []
    summaries = {}
    for name, path in CAMPAIGNS.items():
        df = load(path)
        print(f"\nLoaded {name}: {path.name}  rows={len(df)}  times={df['datetime'].nunique()}  sites={sorted(df['obtid'].unique())}")
        s = summarize_campaign(name, df)
        summaries[name] = s
        all_hourly.append(s["hourly"])
        all_site.append(s["site_low"])

        # example hour requested
        if name == "260707":
            utc = "2025-09-09 06:00:00"
            print("\n--- requested figure hour 2025-09-09 06:00 UTC = 14:00 LST ---")
            h = s["hourly"]
            row = h[h["datetime"] == pd.Timestamp(utc)]
            if len(row):
                with pd.option_context("display.max_columns", 30, "display.width", 180, "display.float_format", "{:.3f}".format):
                    print(row.to_string(index=False))
            site = s["site_low"]
            site_row = site[site["datetime"] == pd.Timestamp(utc)]
            print("\n  per-station 52-300 m:")
            with pd.option_context("display.float_format", "{:.3f}".format):
                print(
                    site_row[
                        ["obtid", "ws_obs", "ws_wrf", "ws_cfd", "d_cfd_wrf", "wrf_bias", "cfd_bias"]
                    ].to_string(index=False)
                )
            prof = profile_one_hour(df, utc)
            print("\n  binned profiles (50 m) per station, selected heights:")
            for obtid in SITES:
                p = prof[prof["obtid"] == obtid]
                sel = p[p["z"].between(50, 1000)]
                print(f"  {obtid}:")
                with pd.option_context("display.float_format", "{:.2f}".format):
                    print(sel[["z", "ws_obs", "ws_wrf", "ws_cfd", "d"]].to_string(index=False))

    hourly = pd.concat(all_hourly, ignore_index=True)
    site = pd.concat(all_site, ignore_index=True)
    hourly.to_csv(OUT / "hourly_3station_layers.csv", index=False)
    site.to_csv(OUT / "site_low_52_300.csv", index=False)
    gt = hourly[hourly["d_cfd_wrf"] > 0].sort_values(["campaign", "d_cfd_wrf"], ascending=[True, False])
    gt.to_csv(OUT / "hours_cfd_gt_wrf_low.csv", index=False)
    print(f"\nWrote {OUT / 'hourly_3station_layers.csv'}")
    print(f"Wrote {OUT / 'hours_cfd_gt_wrf_low.csv'}")

    # WRF spatial contrast for example + a typical decelerating hour
    print("\n=== WRF cartesian NC: domain-edge vs lidar-site WS ===")
    for utc, label in [
        ("2025-09-09 06:00:00", "anomalous CFD>WRF (260707 example)"),
        ("2025-09-08 06:00:00", "same LST previous day"),
        ("2025-09-02 00:00:00", "typical night 260409 08:00 LST"),
        ("2025-09-02 09:00:00", "manuscript WRF-under + CFD-worse 17:00 LST"),
    ]:
        info = maybe_inlet_vs_site(utc)
        print(f"\n{label}  {utc}")
        if info is None:
            print("  (cartesian NC not found or unreadable)")
        else:
            for k, v in info.items():
                if k != "nc":
                    print(f"  {k}: {v}")
            print(f"  nc: {info['nc']}")


if __name__ == "__main__":
    main()
