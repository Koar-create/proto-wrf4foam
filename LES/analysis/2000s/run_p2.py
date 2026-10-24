#!/usr/bin/env python3
"""P2: M3 layered verification (probes + LiDAR) and station-level virtual DBS."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from align import interp_column, lidar_gates_in_cfd_range
from lesio import load_les_probes, load_urans_probes, read_vtp_points_U, vtp_path
from paths import FIGS, LES_WIN_A, MERGED_HOURLY, OUT, PROBE_Z, STATIONS, URANS_WIN
from sampling import load_lidar, longitudinal, sample_match_mc, station_gate, timestamps_rel_s, window
from stats import band_variance, batch_mean_sigma_ci, default_lag_edges, integral_time_scale, structure_function
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import C_LES as LESC, C_OBS, C_RANS, C_URANS, GRID, INK, MUTED, SOLID, apply, panel_label, savefig
from vdbs import HALF_ANGLES, station_dbs_error_snapshot

apply()


def _win_stats(t, U, t0, t1):
    sel = (t >= t0) & (t <= t1)
    u = U[sel]
    um = u.mean(0)
    up = u - um
    return dict(
        Umean=um,
        sig=up.std(0, ddof=1),
        ws_s=np.hypot(u[..., 0], u[..., 1]).mean(0),
        ws_v=np.hypot(um[:, 0], um[:, 1]),
        sig_w=up[:, :, 2].std(0, ddof=1),
        n=int(sel.sum()),
        t=t[sel],
        u=u,
    )


def _ci_sigma(t, U, j, t0, t1):
    sel = (t >= t0) & (t <= t1)
    uh = np.hypot(U[sel, j, 0], U[sel, j, 1])
    dt = float(np.median(np.diff(t[sel])))
    tau = integral_time_scale(uh, dt)
    ul = U[sel, j, 0]
    bm_u = batch_mean_sigma_ci(ul, dt, tau)
    bm_h = batch_mean_sigma_ci(uh, dt, tau)
    return bm_u, bm_h


def obs_profile(d, station):
    g = d[d.obtid == station]
    rows = []
    for H, gg in g.groupby("Height"):
        z = float(gg.zAGL.iloc[0])
        if z < 60 or z > 600:
            continue
        ul, _ = longitudinal(gg.U.values, gg.V.values)
        rows.append(dict(
            Height=H, z=z, n=len(gg),
            S=gg.WindSpd.mean(),
            S25=gg.WindSpd.quantile(0.25), S75=gg.WindSpd.quantile(0.75),
            Sv=np.hypot(gg.U.mean(), gg.V.mean()),
            su=float(np.std(ul, ddof=1)),
            sw=float(gg.VerticalSpd.std(ddof=1)),
        ))
    return pd.DataFrame(rows)


def load_hourly_cfd_wrf():
    if not Path(MERGED_HOURLY).is_file():
        return None
    d = pd.read_csv(MERGED_HOURLY, parse_dates=["datetime"])
    d = d[(d.datetime == "2025-09-01 10:00:00") | (d.datetime == "2025-09-01 10:00")]
    return d


def fig_f02(les, ur, lidar_w, hourly):
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 5.6), sharey=True)
    tL, UL = les["t"], les["U"]
    tU, UU = ur["t"], ur["U"]
    stL = _win_stats(tL, UL, *LES_WIN_A)
    stU = _win_stats(tU, UU, *URANS_WIN)
    z_src = np.asarray(PROBE_Z, float)

    for k, name in enumerate(STATIONS):
        o = obs_profile(lidar_w, name)
        i0, i1 = STATIONS[name]["i0"], STATIONS[name]["i1"]
        ax = axs[0, k]
        ax.fill_betweenx(o.z, o.S25, o.S75, color=GRID, lw=0, label="LiDAR IQR" if k == 0 else None)
        ax.plot(o.S, o.z, color=C_OBS, lw=1.6, label="LiDAR scalar mean" if k == 0 else None)
        # LES window mean + CI of the mean via sigma/sqrt(n_eff) is not A-grade;
        # plot window scalar mean at probe z with batch-mean CI on the mean speed.
        ws = stL["ws_s"][i0:i1]
        ci = []
        for iz in range(11):
            bm_u, bm_h = _ci_sigma(tL, UL, i0 + iz, *LES_WIN_A)
            ci.append(bm_h["half"] if np.isfinite(bm_h["half"]) else 0.0)
        ax.errorbar(ws, z_src, xerr=ci, fmt="o", ms=3, color=LESC, ecolor=LESC, elinewidth=0.8, capsize=0,
                    label="LES 1400–2000 s ±95 % CI" if k == 0 else None)
        ax.plot(stU["ws_s"][i0:i1], z_src, color=C_URANS, lw=1.4, ls="-.",
                label="URANS 1200–1800 s" if k == 0 else None)
        if hourly is not None:
            hh = hourly[hourly.obtid == name] if "obtid" in hourly.columns else None
            if hh is not None and len(hh):
                zobs = hh.Height + STATIONS[name]["h0"] if "Height" in hh.columns else None
                if zobs is not None and "ws_cfd" in hh.columns:
                    ax.plot(hh.ws_cfd, zobs, color=C_RANS, lw=1.4, ls="--",
                            label="RANS hourly" if k == 0 else None)
                if zobs is not None and "ws_wrf" in hh.columns:
                    ax.plot(hh.ws_wrf, zobs, color=MUTED, lw=1.0, ls=":",
                            label="WRF hourly" if k == 0 else None)
        ax.set_title(name, loc="left")
        ax.set_xlim(0, 6)
        if k == 0:
            ax.set_ylabel("z AGL (m)")
        ax.set_xlabel(r"window $U_h$ (m s$^{-1}$)")

        ax = axs[1, k]
        ax.plot(o.su, o.z, color=C_OBS, lw=1.6, label=r"LiDAR $\sigma_u$" if k == 0 else None)
        ax.plot(o.sw, o.z, color=C_OBS, lw=1.0, ls=":", label=r"LiDAR $\sigma_w$" if k == 0 else None)
        su = stL["sig"][i0:i1, 0]
        sw = stL["sig_w"][i0:i1]
        ci_u = []
        for iz in range(11):
            bm_u, _ = _ci_sigma(tL, UL, i0 + iz, *LES_WIN_A)
            ci_u.append(bm_u["half"] if np.isfinite(bm_u["half"]) else 0.0)
        ax.errorbar(su, z_src, xerr=ci_u, fmt="o", ms=3, color=LESC, elinewidth=0.8,
                    label=r"LES resolved $\sigma_u$" if k == 0 else None)
        ax.plot(stU["sig"][i0:i1, 0], z_src, color=C_URANS, lw=1.4, ls="-.",
                label=r"URANS resolved $\sigma_u$" if k == 0 else None)
        ax.set_xlim(0, 1.8)
        if k == 0:
            ax.set_ylabel("z AGL (m)")
        ax.set_xlabel(r"$\sigma_u$ (solid) · $\sigma_w$ (dotted) (m s$^{-1}$)")
        axs[0, k].set_ylim(0, 650)
        panel_label(axs[0, k], "abc"[k])
        panel_label(axs[1, k], "def"[k])
    fig.suptitle("LiDAR 09:30–10:30 UTC  ·  LES 1400–2000 s  ·  URANS 1200–1800 s  (transient-window)",
                 x=0.01, ha="left", fontsize=8, color=INK)
    H = [
        Line2D([], [], color=C_OBS, lw=1.6),
        Patch(color=GRID),
        Line2D([], [], color=C_OBS, lw=1.0, ls=":"),
        Line2D([], [], color=C_RANS, lw=1.4, ls="--"),
        Line2D([], [], color=C_URANS, lw=1.4, ls="-."),
        Line2D([], [], color=MUTED, lw=1.0, ls=":"),
        Line2D([], [], color=LESC, marker="o", ms=3, lw=0.8),
    ]
    L = [
        r"LiDAR mean / $\sigma_u$", "LiDAR IQR", r"LiDAR $\sigma_w$",
        "RANS hourly", "URANS 1200–1800 s", "WRF hourly",
        r"LES 1400–2000 s ±95 % CI",
    ]
    fig.legend(H, L, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6.4, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    savefig(fig, FIGS / "F02_layered_profiles.png")
    plt.close(fig)


def fig_f03_f08(les, ur, lidar_w):
    edges = default_lag_edges()
    cent = np.sqrt(edges[:-1] * edges[1:])
    tL, UL = les["t"], les["U"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
    rows = []
    for k, name in enumerate(STATIONS):
        ax = axs[k]
        g = station_gate(lidar_w, name, 104.0)
        t_obs = (g.datetime - g.datetime.iloc[0]).dt.total_seconds().to_numpy()
        ul, _ = longitudinal(g.U.values, g.V.values)
        D, n = structure_function(t_obs, ul, edges)
        ok = np.isfinite(D)
        ax.loglog(cent[ok], D[ok], color=C_OBS, lw=0, marker="o", ms=3.2, label="LiDAR 09:30–10:30")

        j = STATIONS[name]["i0"] + 2
        sel = (tL >= LES_WIN_A[0]) & (tL <= LES_WIN_A[1])
        tt = tL[sel]
        uu = UL[sel, j, 0]
        Dl, _ = structure_function(tt[::5], uu[::5], edges)
        okl = np.isfinite(Dl) & (cent <= 300)
        ax.loglog(cent[okl], Dl[okl], color=LESC, lw=1.6, label="LES 0.2 s probe")

        t_rel = timestamps_rel_s(g)
        reals = sample_match_mc(tL, UL[:, j, 0], t_rel, n=80, rng=k + 7)
        Ds = []
        for tq, uq in reals:
            Dm, _ = structure_function(tq, uq, edges, min_n=15)
            Ds.append(Dm)
        if Ds:
            Dmed = np.nanmedian(np.vstack(Ds), 0)
            okm = np.isfinite(Dmed) & (cent <= 300)
            ax.loglog(cent[okm], Dmed[okm], color=LESC, lw=0, marker="D", ms=2.6, mfc="none", mew=0.8,
                      label="LES at LiDAR timestamps")

        ax.axvspan(1.5, 120, color=GRID, lw=0, alpha=0.7)
        ax.axvline(600, color=MUTED, lw=0.7, ls="--")
        ax.set_xlim(1.5, 3600)
        ax.set_ylim(0.03, 5)
        ax.set_title(f"{name} · {104 + STATIONS[name]['h0']:.0f} m AGL", loc="left")
        ax.set_xlabel("lag τ (s)")
        if k == 0:
            ax.set_ylabel(r"$D_{uu}(\tau)$ (m$^2$ s$^{-2}$)")
        if k == 2:
            ax.legend(loc="lower right", fontsize=6.0)
        panel_label(ax, "abc"[k])

        # band variance table at four gates
        for H in (65, 104, 195, 299):
            gg = station_gate(lidar_w, name, H)
            if len(gg) < 30:
                continue
            tsec = timestamps_rel_s(gg)
            ul2, _ = longitudinal(gg.U.values, gg.V.values)
            b_obs = band_variance(tsec, ul2)
            # LES sampled
            zA = H + STATIONS[name]["h0"]
            # nearest probe
            iz = int(np.argmin(np.abs(np.asarray(PROBE_Z) - zA)))
            j2 = STATIONS[name]["i0"] + iz
            reals2 = sample_match_mc(tL, UL[:, j2, 0], tsec, n=80, rng=20 + k)
            mids = [band_variance(tq, uq)["mid"] for tq, uq in reals2]
            los = [band_variance(tq, uq)["lo"] for tq, uq in reals2]
            rows.append(dict(
                station=name, gate=H, zAGL=zA, iz=iz,
                obs_lo=b_obs["lo"], obs_mid=b_obs["mid"], obs_hi=b_obs["hi"], obs_tot=b_obs["tot"],
                les_mid_med=float(np.nanmedian(mids)) if mids else np.nan,
                les_lo_med=float(np.nanmedian(los)) if los else np.nan,
            ))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    savefig(fig, FIGS / "F03_structure_functions.png")
    plt.close(fig)

    obs = pd.DataFrame(rows)
    obs.to_csv(OUT / "T4_band_variance.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.0), sharey=False)
    B = ["#9ec5f4", "#3987e5", "#184f95"]
    for k, name in enumerate(STATIONS):
        ax = axs[k]
        o = obs[obs.station == name]
        if o.empty:
            continue
        y = np.arange(len(o))
        for j, col in enumerate(("obs_lo", "obs_mid", "obs_hi")):
            left = o[["obs_lo", "obs_mid", "obs_hi"]].values[:, :j].sum(1)
            ax.barh(y + 0.2, o[col].values, left=left, height=0.36, color=B[j], edgecolor="#fcfcfb", lw=1.0)
        ax.barh(y - 0.2, o.les_mid_med.values, height=0.36, color=LESC, edgecolor="#fcfcfb", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{z:.0f} m" for z in o.zAGL])
        ax.set_title(name, loc="left")
        ax.set_xlabel(r"variance of $u$ (m$^2$ s$^{-2}$)")
        ax.grid(axis="y", visible=False)
        if k == 0:
            ax.set_ylabel("z AGL (upper bar: LiDAR · lower bar: LES)")
    H = [
        Patch(color="#9ec5f4"), Patch(color="#3987e5"), Patch(color="#184f95"),
        Patch(color=LESC),
    ]
    L = ["LiDAR < 120 s", "LiDAR 120–600 s", "LiDAR > 600 s (within 1 h)",
         "LES sampled 120–600 s"]
    fig.legend(H, L, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    savefig(fig, FIGS / "F08_variance_budget.png")
    plt.close(fig)
    return obs


def run_vdbs_stations():
    rows = []
    times = [t for t in range(1000, 2001, 50)]
    for z in (30, 60, 100, 120):
        print(f"  vDBS z={z}", flush=True)
        for t in times:
            pts, vec = read_vtp_points_U(vtp_path(z, t))
            for ang in HALF_ANGLES:
                rows.extend(station_dbs_error_snapshot(pts, vec, z, ang, t))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T5_virtual_dbs_stations.csv", index=False)
    return df


def fig_f05_routes():
    from lesio import load_all_routes
    from uav import route_envelope, route_hovmoller
    from paths import TIMES as T
    pack = load_all_routes(roof=None)
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.0), gridspec_kw=dict(height_ratios=[1.25, 1]))
    for k, (route, z, lab) in enumerate([(1, 30, "x along route (m)"), (2, 30, "y along route (m)")]):
        key = f"r{route}_z{z}"
        U = pack[f"{key}_U"]
        axis = pack[f"{key}_x"] if route == 1 else pack[f"{key}_y"]
        spd = route_hovmoller(U)
        solid = pack[f"{key}_solid"]
        ax = axs[0, k]
        im = ax.pcolormesh(axis, np.asarray(T), spd, cmap="Blues", vmin=0, vmax=3, shading="nearest")
        if route == 2:
            for g in axis[solid]:
                ax.axvspan(g - 10, g + 10, color="#6f6e69", lw=0, zorder=2)
        ax.axhline(1000, color=INK, lw=0.6, ls="--")
        ax.set_title(f"Route {route} · 30 m", loc="left")
        ax.set_xlabel(lab)
        if k == 0:
            ax.set_ylabel("LES time t (s)")
        ax.grid(False)
        panel_label(ax, "ab"[k])
        env = route_envelope(spd, np.asarray(T), 1000.0)
        ax2 = axs[1, k]
        ax2.fill_between(axis, env["p05"], env["p95"], color=LESC, alpha=0.22, lw=0,
                         label="LES P5–P95" if k == 0 else None)
        ax2.plot(axis, env["p50"], color=LESC, lw=1.4,
                 label="LES median, t ≥ 1000 s" if k == 0 else None)
        ax2.set_xlabel(lab)
        if k == 0:
            ax2.set_ylabel(r"$|U_h|$ (m s$^{-1}$)")
        ax2.set_ylim(0, 4.0)
        panel_label(ax2, "cd"[k])
    axs[1, 0].legend(loc="upper right", fontsize=6.2)
    H_b = [Patch(color=SOLID)]
    axs[0, 1].legend(
        H_b, ["building solid (no sample)"], loc="upper left", fontsize=6.0,
        frameon=True, fancybox=False, edgecolor="none", facecolor="white", framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.02, 0.92, 1))
    cax = fig.add_axes([0.93, 0.50, 0.012, 0.42])
    fig.colorbar(im, cax=cax).set_label(r"$|U_h|$ (m s$^{-1}$)")
    savefig(fig, FIGS / "F05_route_time.png")
    plt.close(fig)


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    print("[P2] load caches …", flush=True)
    les = load_les_probes()
    ur = load_urans_probes()
    lidar_w = window(load_lidar())
    hourly = load_hourly_cfd_wrf()
    print("[P2] F02 …", flush=True)
    fig_f02(les, ur, lidar_w, hourly)
    print("[P2] F03 / F08 …", flush=True)
    fig_f03_f08(les, ur, lidar_w)
    print("[P2] F05 routes …", flush=True)
    fig_f05_routes()
    print("[P2] virtual DBS stations (21 frames × 4 z × 3 cones) …", flush=True)
    run_vdbs_stations()
    print(f"[P2] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
