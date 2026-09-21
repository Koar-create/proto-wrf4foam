#!/usr/bin/env python3
"""Hours when OpenFOAM low-layer wind exceeds standalone WRF.

Uses the current merged tables only. Writes results/openfoam_faster_than_wrf/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "openfoam_faster_than_wrf"
OUT.mkdir(parents=True, exist_ok=True)

TABLES = {
    "1-5 Sep": REPO / "data/260409/processed/merged_lidar_simulation_final.csv",
    "6-9 Sep": REPO / "data/260707/processed/merged_lidar_simulation_final.csv",
}
SITES = ("GAW103", "GAW104", "GAW111")
LOW = (52.0, 300.0)
ALOFT = (800.0, 1000.0)
LST = pd.Timedelta(hours=8)
SITE_XY = {
    "GAW103": (975.0, -320.0),
    "GAW104": (450.0, 350.0),
    "GAW111": (75.0, 30.0),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[df["obtid"].isin(SITES)].copy()
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    return df


def layer_mean(df: pd.DataFrame, z0: float, z1: float, keys: list[str]) -> pd.DataFrame:
    sub = df[(df["Height"] >= z0) & (df["Height"] <= z1)]
    g = sub.groupby(keys, observed=True)
    out = g.agg(
        n=("ws_wrf", "size"),
        ws_obs=("ws_obs", "mean"),
        ws_wrf=("ws_wrf", "mean"),
        ws_of=("ws_cfd", "mean"),
        k_wrf=("k_wrf", "mean"),
        k_of=("k_cfd", "mean"),
        u_wrf=("u_wrf", "mean"),
        v_wrf=("v_wrf", "mean"),
        u_of=("u_cfd", "mean"),
        v_of=("v_cfd", "mean"),
        u_obs=("u_obs", "mean"),
        v_obs=("v_obs", "mean"),
    ).reset_index()
    out["d_of_wrf"] = out["ws_of"] - out["ws_wrf"]
    out["wrf_bias"] = out["ws_wrf"] - out["ws_obs"]
    out["of_bias"] = out["ws_of"] - out["ws_obs"]
    out["wd_wrf"] = (np.degrees(np.arctan2(-out["u_wrf"], -out["v_wrf"])) + 360) % 360
    return out


def wd_sector(wd: pd.Series) -> pd.Series:
    cat = pd.cut(wd, bins=[0, 45, 135, 225, 315, 360], labels=["N", "E", "S", "W", "N2"], include_lowest=True, right=False)
    return cat.replace({"N2": "N"})


def find_nc(dt: pd.Timestamp) -> Path | None:
    stem = f"auxhist2_d03_{dt.strftime('%Y-%m-%d_%H:%M:%S')}_1h-rolling_cartesian.nc"
    for root in (REPO / "W_myExp05" / "auxhist2", REPO / "W_myExp03" / "auxhist2"):
        for name in (stem, stem.replace(":", "%3A")):
            p = root / name
            if p.is_file():
                return p
    return None


def wrf_at_z(nc: Path, z_target: float) -> dict:
    with xr.open_dataset(nc, mask_and_scale=False) as ds:
        x = ds["x_rel"].values.squeeze().astype(float)
        y = ds["y_rel"].values.squeeze().astype(float)
        z = ds["z"].values.squeeze().astype(float)
        u = ds["U"].values.squeeze().astype(float)
        v = ds["V"].values.squeeze().astype(float)
        ws = ds["WS"].values.squeeze().astype(float)
    iz = int(np.argmin(np.abs(z - z_target)))
    u2, v2, ws2 = u[iz], v[iz], ws[iz]
    xx, yy = np.meshgrid(x, y)
    xmax, ymax = float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y)))
    strip = 200.0
    faces = {
        "west": np.abs(xx + xmax) <= strip,
        "east": np.abs(xx - xmax) <= strip,
        "south": np.abs(yy + ymax) <= strip,
        "north": np.abs(yy - ymax) <= strip,
    }
    site_ws = []
    su = sv = 0.0
    for sx, sy in SITE_XY.values():
        i = int(np.argmin(np.abs(x - sx)))
        j = int(np.argmin(np.abs(y - sy)))
        site_ws.append(float(ws2[j, i]))
        su += float(u2[j, i])
        sv += float(v2[j, i])
    su /= 3
    sv /= 3
    outward = {"west": (-1, 0), "east": (1, 0), "south": (0, -1), "north": (0, 1)}
    inlets = [p for p, (nx, ny) in outward.items() if su * nx + sv * ny < 0]
    inlet_mask = np.zeros_like(ws2, dtype=bool)
    for p in inlets:
        inlet_mask |= faces[p]
    return {
        "z": float(z[iz]),
        "site_ws": float(np.mean(site_ws)),
        "inlet_ws": float(np.nanmean(ws2[inlet_mask])) if inlet_mask.any() else np.nan,
        "domain_ws": float(np.nanmean(ws2)),
        "inlets": ",".join(inlets),
    }


def profile_sanity(df: pd.DataFrame, utc: str) -> pd.DataFrame:
    sub = df[(df["datetime"] == pd.Timestamp(utc)) & df["obtid"].isin(SITES)].copy()
    sub["zbin"] = pd.cut(sub["Height"], bins=np.arange(0, 1050, 50))
    agg = (
        sub.groupby(["obtid", "zbin"], observed=True)
        .agg(z=("Height", "mean"), ws_obs=("ws_obs", "mean"), ws_wrf=("ws_wrf", "mean"), ws_of=("ws_cfd", "mean"))
        .reset_index()
    )
    agg["d"] = agg["ws_of"] - agg["ws_wrf"]
    return agg


def main() -> None:
    hourly_all = []
    site_all = []
    for period, path in TABLES.items():
        df = load(path)
        print("=" * 72)
        print(f"{period}  {path.name}  rows={len(df)}  hours={df['datetime'].nunique()}")
        # guard: reject vertically uniform OpenFOAM columns (initial-field signature)
        chk = df.groupby(["datetime", "obtid"])["ws_cfd"].std().reset_index(name="ws_std")
        flat = chk[chk["ws_std"] < 0.05]
        print(f"  vertically-flat OpenFOAM columns (std<0.05): {len(flat)}")
        if len(flat):
            print(flat.to_string(index=False))

        low = layer_mean(df, *LOW, ["datetime"])
        aloft = layer_mean(df, *ALOFT, ["datetime"]).rename(
            columns={
                "ws_obs": "ws_obs_aloft",
                "ws_wrf": "ws_wrf_aloft",
                "ws_of": "ws_of_aloft",
                "d_of_wrf": "d_of_wrf_aloft",
                "wrf_bias": "wrf_bias_aloft",
                "of_bias": "of_bias_aloft",
            }
        )
        h = low.merge(
            aloft[
                [
                    "datetime",
                    "ws_obs_aloft",
                    "ws_wrf_aloft",
                    "ws_of_aloft",
                    "d_of_wrf_aloft",
                    "wrf_bias_aloft",
                    "of_bias_aloft",
                ]
            ],
            on="datetime",
        )
        h["period"] = period
        h["lst"] = h["datetime"] + LST
        h["lst_hour"] = h["lst"].dt.hour
        h["wd_sector"] = wd_sector(h["wd_wrf"])
        site = layer_mean(df, *LOW, ["datetime", "obtid"])
        site["period"] = period
        n_agree = (
            site.assign(faster=site["d_of_wrf"] > 0).groupby("datetime")["faster"].sum().rename("n_sites_faster")
        )
        h = h.merge(n_agree, on="datetime")

        n = len(h)
        faster = h[h["d_of_wrf"] > 0]
        print(f"  mean OpenFOAM minus WRF (52-300 m) = {h['d_of_wrf'].mean():+.3f} m/s")
        print(f"  hours OpenFOAM > WRF               = {len(faster)} / {n}")
        print(f"  of those, also faster at 800-1000 m= {(faster['d_of_wrf_aloft']>0).sum()}")
        print(f"  of those, WRF already < obs        = {(faster['wrf_bias']<0).sum()}")
        print(f"  of those, |OF bias| < |WRF bias|   = {(faster['of_bias'].abs()<faster['wrf_bias'].abs()).sum()}")
        print(f"  mean WRF ws | OF>WRF / OF<=WRF     = {faster['ws_wrf'].mean() if len(faster) else np.nan:.2f} / {h.loc[h['d_of_wrf']<=0,'ws_wrf'].mean():.2f}")
        if len(faster):
            print("  LST hour counts:")
            print(faster.groupby("lst_hour").size().reindex(range(24), fill_value=0).to_string())
            print("  WRF wd sector:")
            print(faster.groupby("wd_sector", observed=True).size().to_string())
            print("  sites agreeing (1/2/3):")
            print(faster["n_sites_faster"].value_counts().sort_index().to_string())
            print("  hours ranked by OpenFOAM minus WRF:")
            cols = ["datetime", "lst", "ws_obs", "ws_wrf", "ws_of", "d_of_wrf", "wrf_bias", "of_bias", "d_of_wrf_aloft", "wd_wrf", "n_sites_faster"]
            with pd.option_context("display.width", 180, "display.float_format", "{:.2f}".format):
                print(faster.sort_values("d_of_wrf", ascending=False)[cols].to_string(index=False))

        h["ws_wrf_bin"] = pd.cut(h["ws_wrf"], bins=[0, 1.5, 3, 5, 8, 30], labels=["<1.5", "1.5-3", "3-5", "5-8", ">8"])
        hourly_all.append(h)
        site_all.append(site)

        if period == "6-9 Sep":
            utc = "2025-09-09 06:00:00"
            print("\n--- 2025-09-09 06:00 UTC = 14:00 LST ---")
            row = h[h["datetime"] == pd.Timestamp(utc)]
            with pd.option_context("display.max_columns", 40, "display.width", 200, "display.float_format", "{:.3f}".format):
                print(row.to_string(index=False))
            srow = site[site["datetime"] == pd.Timestamp(utc)]
            print("per station 52-300 m:")
            print(srow[["obtid", "ws_obs", "ws_wrf", "ws_of", "d_of_wrf", "wrf_bias", "of_bias"]].to_string(index=False, float_format="{:.3f}".format))
            prof = profile_sanity(df, utc)
            for obtid in SITES:
                p = prof[prof["obtid"] == obtid]
                print(f"{obtid} binned profile:")
                print(p[["z", "ws_obs", "ws_wrf", "ws_of", "d"]].to_string(index=False, float_format="{:.2f}".format))

    hourly = pd.concat(hourly_all, ignore_index=True)
    site = pd.concat(site_all, ignore_index=True)
    hourly.to_csv(OUT / "hourly_3station.csv", index=False)
    site.to_csv(OUT / "site_52_300.csv", index=False)
    faster = hourly[hourly["d_of_wrf"] > 0].sort_values(["period", "d_of_wrf"], ascending=[True, False])
    faster.to_csv(OUT / "hours_openfoam_faster.csv", index=False)

    print("\n=== pooled, by WRF 52-300 m speed ===")
    g = hourly.groupby("ws_wrf_bin", observed=True)
    tab = pd.DataFrame({
        "n": g.size(),
        "n_faster": g.apply(lambda x: int((x["d_of_wrf"] > 0).sum()), include_groups=False),
        "frac": g.apply(lambda x: float((x["d_of_wrf"] > 0).mean()), include_groups=False),
        "mean_d": g["d_of_wrf"].mean(),
        "mean_wrf": g["ws_wrf"].mean(),
    })
    print(tab.round(3).to_string())
    print(f"corr(OpenFOAM minus WRF, WRF ws) = {hourly['d_of_wrf'].corr(hourly['ws_wrf']):.3f}")

    print("\n=== WRF cartesian z=200 m: inlet vs lidar site ===")
    targets = [
        "2025-09-09 06:00:00",
        "2025-09-03 23:00:00",
        "2025-09-04 23:00:00",
        "2025-09-08 06:00:00",
        "2025-09-02 09:00:00",
    ]
    for utc in targets:
        nc = find_nc(pd.Timestamp(utc))
        print(f"\n{utc}")
        if nc is None:
            print("  nc missing")
            continue
        d200 = wrf_at_z(nc, 200.0)
        d900 = wrf_at_z(nc, 900.0)
        print(f"  z200 site={d200['site_ws']:.2f} inlet={d200['inlet_ws']:.2f} domain={d200['domain_ws']:.2f} inlets={d200['inlets']}")
        print(f"  z900 site={d900['site_ws']:.2f} inlet={d900['inlet_ws']:.2f} domain={d900['domain_ws']:.2f}")
        row = hourly[hourly["datetime"] == pd.Timestamp(utc)]
        if len(row):
            r = row.iloc[0]
            print(f"  table 52-300: obs={r.ws_obs:.2f} wrf={r.ws_wrf:.2f} of={r.ws_of:.2f} d={r.d_of_wrf:+.2f}")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
