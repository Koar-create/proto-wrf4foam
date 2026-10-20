#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 buoyantBoussinesqSimpleFoam 算例初始化内场（热启动）。

1. U/k/epsilon/nut：把 --ref-case/<--ref-time>/<field> 的 internalField 块
   原样替换进 <case>/0/<field>（保留 0/ 中的边界条件与 $kInlet 等宏）。
2. T：按 *_bbsf.nc 水平平均 θ(z)（--temp-var）在参考算例 0/Cz 上插值，写非均匀内场。
3. p_rgh：用同一平均剖面按静力平衡积分（与 construct_OF_boundary_arrays_bbsf.py 的出口一致）。
   beta、TRef 取自 <case>/constant/transportProperties。

用法:
  python util/init_bbsf_internal_fields.py --case experiments/20250901_1000_buoyFoam \\
      --ref-case steady_experiments_finer_ABL/20250901_1000_two_boundaries_as_outlet \\
      --input-nc W_myExp03/auxhist2/auxhist2_d03_2025-09-01_10:00:00_1h-rolling_cartesian_bbsf.nc
"""
import argparse
import importlib.util
import os
import re

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "cons", os.path.join(HERE, "construct_OF_boundary_arrays_bbsf.py"))
cons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cons)


def split_internal(data: bytes):
    """返回 (internalField 块起点, 终点)。块以 'internalField' 开头，以 ')\n;' 结束。"""
    s = data.index(b"internalField")
    if data[s:s + 80].split(b"\n")[0].split()[1] == b"uniform":
        e = data.index(b";", s) + 1
        return s, e
    e = data.index(b"\n)\n;", s) + 4
    return s, e


def internal_block(path):
    d = open(path, "rb").read()
    s, e = split_internal(d)
    return d[s:e]


def replace_internal(path, block):
    d = open(path, "rb").read()
    s, e = split_internal(d)
    open(path, "wb").write(d[:s] + block + d[e:])


def read_scalar_list(path):
    d = open(path, "rb").read()
    s, e = split_internal(d)
    blk = d[s:e].decode()
    n = int(re.search(r"\n(\d+)\n\(", blk).group(1))
    body = blk[blk.index("\n(\n") + 3:blk.rindex("\n)\n;")]
    arr = np.fromstring(body, sep="\n")
    assert arr.size == n, (arr.size, n)
    return arr


def scalar_block(vals):
    return ("internalField   nonuniform List<scalar> \n%d\n(\n" % len(vals)
            + "\n".join(f"{v:.6g}" for v in vals) + "\n)\n;").encode()


def read_tp(case, key):
    txt = open(os.path.join(case, "constant", "transportProperties")).read()
    return float(re.search(rf"^{key}\s+(?:\w+\s+\[[^\]]*\]\s+)?([-+0-9.eE]+);", txt, re.M).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--ref-case", required=True)
    ap.add_argument("--ref-time", default="5000")
    ap.add_argument("--input-nc", required=True)
    ap.add_argument("--temp-var", default="Theta_v")
    a = ap.parse_args()
    case, ref = os.path.abspath(a.case), os.path.abspath(a.ref_case)

    for f in ("U", "k", "epsilon", "nut"):
        blk = internal_block(os.path.join(ref, a.ref_time, f))
        replace_internal(os.path.join(case, "0", f), blk)
        print(f"[ok] {f} internalField <- {a.ref_time}/{f}")

    beta, TRef = read_tp(case, "beta"), read_tp(case, "TRef")
    ds = xr.open_dataset(a.input_nc)
    ds = ds.where(ds.z <= cons.MAX_HEIGHT, drop=True)
    z = ds.z.values
    Tm = ds[a.temp_var].mean(dim=("y_rel", "x_rel")).values
    prgh = cons.hydrostatic_prgh(z, Tm, beta, TRef)
    print(f"mean {a.temp_var}(z): {Tm.min():.2f}..{Tm.max():.2f} K; p_rgh profile {prgh.min():.2f}..{prgh.max():.2f}")

    cz = read_scalar_list(os.path.join(ref, "0", "Cz"))
    print(f"cells={cz.size}, z range {cz.min():.1f}..{cz.max():.1f}")
    T = np.interp(cz, z, Tm)
    P = np.interp(cz, z, prgh)
    replace_internal(os.path.join(case, "0", "T"), scalar_block(T))
    replace_internal(os.path.join(case, "0", "p_rgh"), scalar_block(P))
    print("[ok] T, p_rgh internalField written")


if __name__ == "__main__":
    main()
