#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
判据 (4)：水平平均剖面（中心 |x|,|y|<2500 m）与 WRF、参考 simpleFoam 对比。
输出 <case>/check/profiles_<time>.csv 与 .png。

用法: python util/check_bbsf_profiles.py <case> <time> --ref-case <simpleFoam case> --nc <bbsf.nc>
"""
import argparse, os, sys
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import init_bbsf_internal_fields as I


def vec(path):
    d = open(path, "rb").read()
    s, e = I.split_internal(d)
    b = d[s:e].decode()
    body = b[b.index("\n(\n") + 3:b.rindex("\n)\n;")].replace("(", " ").replace(")", " ")
    return np.fromstring(body, sep=" ").reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("time")
    ap.add_argument("--ref-case", required=True); ap.add_argument("--ref-time", default="5000")
    ap.add_argument("--nc", required=True); ap.add_argument("--half", type=float, default=2500)
    a = ap.parse_args()
    case, ref = os.path.abspath(a.case), os.path.abspath(a.ref_case)
    cx, cy, cz = (I.read_scalar_list(f"{ref}/0/{c}") for c in ("Cx", "Cy", "Cz"))
    U = np.linalg.norm(vec(f"{case}/{a.time}/U"), axis=1)
    Ur = np.linalg.norm(vec(f"{ref}/{a.ref_time}/U"), axis=1)
    T = I.read_scalar_list(f"{case}/{a.time}/T")
    k = I.read_scalar_list(f"{case}/{a.time}/k")
    core = (abs(cx) < a.half) & (abs(cy) < a.half)
    ds = xr.open_dataset(a.nc)
    sel = ds.sel(x_rel=slice(-a.half, a.half), y_rel=slice(-a.half, a.half))
    wz = sel.z.values
    wU = np.sqrt(sel.U ** 2 + sel.V ** 2).mean(("x_rel", "y_rel")).values
    wT = sel.Theta_v.mean(("x_rel", "y_rel")).values
    wk = sel.TKE_PBL.mean(("x_rel", "y_rel")).values
    edges = np.array([0, 10, 20, 50, 100, 200, 300, 450, 600, 800, 1000, 1250, 1500, 1750, 2000])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = core & (cz >= lo) & (cz < hi)
        w = (wz >= lo) & (wz < hi)
        rows.append((0.5 * (lo + hi), U[s].mean(), Ur[s].mean(), wU[w].mean(), T[s].mean(), wT[w].mean(),
                     k[s].mean(), wk[w].mean()))
    rows = np.array(rows)
    out = os.path.join(case, "check"); os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"profiles_{a.time}")
    np.savetxt(base + ".csv", rows, delimiter=",", header="z,U_bbsf,U_simpleFoam,U_WRF,T_bbsf,thv_WRF,k_bbsf,TKE_WRF",
               comments="")
    print("z   U_bbsf U_ref U_WRF | T_bbsf thv_WRF | k_bbsf TKE_WRF")
    for r in rows:
        print(f"{r[0]:6.0f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f} | {r[4]:7.2f} {r[5]:7.2f} | {r[6]:6.3f} {r[7]:6.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    z = rows[:, 0]
    ax[0].plot(rows[:, 1], z, "o-", label="BBSF"); ax[0].plot(rows[:, 2], z, "s--", label="simpleFoam")
    ax[0].plot(rows[:, 3], z, "^:", label="WRF"); ax[0].set_xlabel("|U| [m/s]"); ax[0].legend(); ax[0].set_ylabel("z [m]")
    ax[1].plot(rows[:, 4], z, "o-", label="BBSF T"); ax[1].plot(rows[:, 5], z, "^:", label="WRF theta_v"); ax[1].set_xlabel("T [K]"); ax[1].legend()
    ax[2].plot(rows[:, 6], z, "o-", label="BBSF k"); ax[2].plot(rows[:, 7], z, "^:", label="WRF TKE"); ax[2].set_xlabel("k [m2/s2]"); ax[2].legend()
    fig.suptitle(f"{os.path.basename(case)} t={a.time}")
    fig.savefig(base + ".png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
