#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
buoyantBoussinesqSimpleFoam 物理有效性检查（不读场文件，仅用 postProcessing，快速）。

判据（用户确认）:
 (1) |U|max <= 2 x 入口最大风速
 (2) T 在 [入口θv最小-2, max(入口θv最大, 地表估算)+3] K
 (3) 残差平台 + 进出口质量通量闭合
 (4) 水平平均剖面趋势（需重构场，由 --profiles 另行比较，本脚本不判）

用法: python util/check_bbsf_physics.py <case> --nc <bbsf.nc> [--tsfc K]
"""
import argparse, glob, os, re
import numpy as np, xarray as xr


def last_block(path, field):
    """返回 fieldMinMax.dat 中 field 的最后一行 (time, min, max)。"""
    rows = []
    for l in open(path):
        if l.startswith("#"):
            continue
        p = l.split("\t")
        if len(p) >= 7 and p[1].strip() == field:
            rows.append((float(p[0]), float(p[2]), float(p[5])))
    return rows


def phi_sums(case):
    out = {}
    for n in ["west", "east", "south", "north", "upper"]:
        fs = sorted(glob.glob(f"{case}/postProcessing/phi_{n}/*/surfaceFieldValue.dat"))
        v = []
        for f in fs:
            for l in open(f):
                if not l.startswith("#") and l.strip():
                    v.append(float(l.split()[1]))
        out[n] = v[-1] if v else np.nan
    return out


def residual_trend(case):
    fs = sorted(glob.glob(f"{case}/postProcessing/solverInfo/*/solverInfo.dat"))
    if not fs:
        return {}
    hdr = None
    rows = []
    for l in open(fs[-1]):
        if l.startswith("# Time"):
            hdr = l[1:].split()
        elif not l.startswith("#") and l.strip():
            rows.append(l.split())
    res = {}
    for i, h in enumerate(hdr or []):
        if h.endswith("_initial"):
            a = np.array([float(r[i]) for r in rows if len(r) > i])
            if len(a) >= 200:
                res[h[:-8]] = (a[-1], a[-100:].mean(), a[-200:-100].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--nc", required=True)
    ap.add_argument("--tsfc", type=float, default=None, help="地表估算温度上界 [K]")
    a = ap.parse_args()
    ds = xr.open_dataset(a.nc)
    ds = ds.where(ds.z <= 2100, drop=True)
    umax_in = float(np.sqrt(ds.U ** 2 + ds.V ** 2 + ds.W ** 2).max())
    tmin_in, tmax_in = float(ds.Theta_v.min()), float(ds.Theta_v.max())
    tsfc = a.tsfc if a.tsfc is not None else tmax_in
    lo, hi = tmin_in - 2, max(tmax_in, tsfc) + 3
    f = glob.glob(f"{a.case}/postProcessing/fieldMinMax1/*/fieldMinMax.dat")
    f = sorted(f)[-1]
    U = last_block(f, "mag(U)")[-1]
    T = last_block(f, "T")[-1]
    print(f"time {U[0]:.0f}")
    ok1 = U[2] <= 2 * umax_in
    print(f"(1) |U|max={U[2]:.2f} vs 2*inlet max {2*umax_in:.2f}  -> {'PASS' if ok1 else 'FAIL'}")
    ok2 = lo <= T[1] and T[2] <= hi
    print(f"(2) T in [{T[1]:.2f},{T[2]:.2f}] vs allowed [{lo:.2f},{hi:.2f}] -> {'PASS' if ok2 else 'FAIL'}")
    s = phi_sums(a.case)
    net = sum(v for v in s.values())
    inflow = sum(abs(v) for k, v in s.items() if v < 0)
    print("phi sums:", {k: f"{v:.3g}" for k, v in s.items()}, f"net/inflow={net/max(inflow,1e-30):.2e}")
    r = residual_trend(a.case)
    plateau = True
    for k, (last, m1, m0) in r.items():
        flat = abs(m1 - m0) / max(m0, 1e-30) < 0.2
        plateau &= flat
        print(f"  res {k}: last={last:.2e} mean(last100)={m1:.2e} mean(prev100)={m0:.2e} {'flat' if flat else 'trending'}")
    ok3 = plateau and abs(net / max(inflow, 1e-30)) < 1e-2 and bool(r)
    print(f"(3) -> {'PASS' if ok3 else 'FAIL/INCOMPLETE'}")
    print("(4) 需用重构场的水平平均剖面与 WRF 比较")


if __name__ == "__main__":
    main()
