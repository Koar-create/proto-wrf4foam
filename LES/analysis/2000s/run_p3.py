#!/usr/bin/env python3
"""P3: M5 UAV exposure along the two sampled routes (does not require slice .nc)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from lesio import load_all_routes, load_les_probes, load_rans_horizontal
from paths import FIGS, OUT, STATIONS, TIMES
from matplotlib.lines import Line2D

from style import C_LES, C_RANS, INK, INK2, apply, panel_label, savefig
from uav import (
    along_track_delta,
    exceedance_fraction,
    lambda_field,
    resolved_gust_factor,
    route_envelope,
    route_hovmoller,
    spatial_structure_function,
)

apply()


_RANS = {}


def rans_on_route(z, x, y):
    z = int(z)
    if z not in _RANS:
        xs, ys, U, K = load_rans_horizontal(z)
        _RANS[z] = (xs, ys, U, K)
    xs, ys, U, K = _RANS[z]
    fu = RegularGridInterpolator((ys, xs), U[:, :, 0], bounds_error=False, fill_value=np.nan)
    fv = RegularGridInterpolator((ys, xs), U[:, :, 1], bounds_error=False, fill_value=np.nan)
    fk = RegularGridInterpolator((ys, xs), K, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([y, x])
    u, v = fu(pts), fv(pts)
    return np.hypot(u, v), fk(pts)


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    pack = load_all_routes(roof=None)
    t = np.asarray(TIMES, float)
    sel = t >= 1000
    rows_l, rows_e = [], []

    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.6), sharey="row")
    for ic, z in enumerate((30, 60, 120)):
        for route in (1, 2):
            key = f"r{route}_z{z}"
            x, y = pack[f"{key}_x"], pack[f"{key}_y"]
            spd = route_hovmoller(pack[f"{key}_U"])
            solid = pack[f"{key}_solid"]
            rans_s, rans_k = rans_on_route(z, x, y)
            rans_s[solid] = np.nan
            lam = lambda_field(spd, rans_s[None, :])
            lam[:, solid] = np.nan
            lam_a = lam[sel]
            med = np.nanmedian(lam_a, 0)
            p95 = np.nanpercentile(lam_a, 95, 0)
            rows_l.append(dict(
                route=route, z=z,
                median=float(np.nanmedian(med)),
                p95=float(np.nanmedian(p95)),
                p95_over_p50=float(np.nanmedian(p95 / np.maximum(med, 1e-3))),
                n_fluid=int((~solid).sum()),
            ))
            env = route_envelope(spd, t, 1000.0)
            axis = x if route == 1 else y
            # encounter |ΔU| P95 at r=100 m, t>=1000
            dU = []
            for i in np.where(sel)[0]:
                dU.append(along_track_delta(spd[i], 100.0))
            dU = np.concatenate(dU) if dU else np.array([np.nan])
            side = pack[f"{key}_U"][sel, :, 1 if route == 1 else 0]
            w = pack[f"{key}_U"][sel, :, 2]
            rows_e.append(dict(
                route=route, z=z,
                Uh_p50=float(np.nanmedian(env["p50"])),
                Uh_p95=float(np.nanmedian(env["p95"])),
                Uh_route_max_p95=float(np.nanpercentile(np.nanmax(spd[sel], 1), 95)),
                rans_mean=float(np.nanmean(rans_s)),
                dU100_p95=float(np.nanpercentile(np.abs(dU), 95)),
                side_p95=float(np.nanpercentile(np.abs(side), 95)),
                w_gt_0p5=exceedance_fraction(w, 0.5),
                w_gt_1=exceedance_fraction(w, 1.0),
            ))
        # plot Lambda P95 along both routes at this z in top row? keep one figure: median Λ vs along-track
        ax = axs[0, ic]
        for route, ls in ((1, "-"), (2, "--")):
            key = f"r{route}_z{z}"
            x, y = pack[f"{key}_x"], pack[f"{key}_y"]
            spd = route_hovmoller(pack[f"{key}_U"])
            rans_s, _ = rans_on_route(z, x, y)
            lam = lambda_field(spd, rans_s[None, :])
            axis = x if route == 1 else y
            ax.plot(axis, np.nanpercentile(lam[sel], 95, 0), color=C_LES, lw=1.2, ls=ls,
                    label=f"Route {route}" if ic == 0 else None)
        ax.axhline(1, color=INK2, lw=0.6)
        ax.set_title(f"z = {z} m", loc="left")
        ax.set_xlabel("along-route coord (m)")
        if ic == 0:
            ax.set_ylabel(r"P95 $\Lambda_{LES}$")
        panel_label(ax, "abc"[ic])
        ax = axs[1, ic]
        for route, ls in ((1, "-"), (2, "--")):
            key = f"r{route}_z{z}"
            spd = route_hovmoller(pack[f"{key}_U"])
            axis = pack[f"{key}_x"] if route == 1 else pack[f"{key}_y"]
            env = route_envelope(spd, t, 1000.0)
            rans_s, _ = rans_on_route(z, pack[f"{key}_x"], pack[f"{key}_y"])
            ax.plot(axis, env["p95"], color=C_LES, lw=1.2, ls=ls,
                    label=fr"LES P95 $|U_h|$" if (ic == 0 and route == 1) else None)
            ax.plot(axis, rans_s, color=C_RANS, lw=1.0, ls=ls,
                    label="RANS steady" if (ic == 0 and route == 1) else None)
        ax.set_xlabel("along-route coord (m)")
        if ic == 0:
            ax.set_ylabel(r"$|U_h|$ (m s$^{-1}$)")
        panel_label(ax, "def"[ic])
    H = [
        Line2D([], [], color=INK, lw=1.2, ls="-"),
        Line2D([], [], color=INK, lw=1.2, ls="--"),
        Line2D([], [], color=C_LES, lw=1.6),
        Line2D([], [], color=C_RANS, lw=1.2),
    ]
    L = ["Route 1 (river)", "Route 2 (canyon)", r"LES P95 $|U_h|$", "RANS steady"]
    fig.legend(H, L, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, FIGS / "F06_transient_lambda_routes.png")
    plt.close(fig)

    pd.DataFrame(rows_l).to_csv(OUT / "T6_lambda_routes.csv", index=False)
    pd.DataFrame(rows_e).to_csv(OUT / "T7_route_exposure.csv", index=False)

    # probe gust factors (B-grade)
    les = load_les_probes()
    tL, UL = les["t"], les["U"]
    grow = []
    for name, meta in STATIONS.items():
        for iz, z in enumerate((60, 100, 120, 200, 400)):
            j = meta["i0"] + list(__import__("paths").PROBE_Z).index(z)
            ws = np.hypot(UL[:, j, 0], UL[:, j, 1])
            g30 = resolved_gust_factor(tL, ws, 30.0, 600.0, 2000.0)
            g60 = resolved_gust_factor(tL, ws, 60.0, 600.0, 2000.0)
            grow.append(dict(station=name, z=z, G30=g30, G60=g60))
    pd.DataFrame(grow).to_csv(OUT / "T7b_resolved_gust_factor.csv", index=False)
    print(f"[P3] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
