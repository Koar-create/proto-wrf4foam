#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
construct_OF_boundary_arrays.py 的 buoyantBoussinesqSimpleFoam 版本。

输入为 convert_wrf_coord_and_do_3D_interp_bbsf.py 生成的 *_cartesian_bbsf.nc
（Theta 为干位温，z 为离地高度）。

相对原脚本的修正 / 补充：
  1. 输入改为 *_bbsf.nc，避免读入 Theta 偏低约 7.5 K 的旧产物。
  2. 温度变量可选 Theta（干位温）或 Theta_v（虚位温，默认，计入湿度对浮力的贡献）。
  3. 新增出口静力平衡 p_rgh 剖面 boundaryData/<patch>/0/p_rgh（运动学单位 m2/s2）：
         p_rgh(z) = g * ( rhok(z)*z - int_0^z rhok dz' ),  rhok = 1 - beta*(T - TRef)
     出口若直接给 p_rgh=0，在层结大气中会引入 O(10-100) m2/s2 的虚假压力差。
  4. 新增 0/T、0/p_rgh、0/alphat，按 constant/polyMesh/boundary 的 patch 生成边界条件；
     地面可用 --hfx 施加 WRF 显热通量（atmTurbulentHeatFluxTemperature，Cp0 = rho*cp）。
  5. 写 constant/transportProperties（nu、beta、TRef、Pr、Prt 一致）与 constant/g。
     nu 默认 1.5e-5：v2412 的 incompressible 求解器只读 transportProperties。
  6. boundaryData 写到 constant/boundaryData（原脚本写到算例根目录）。

用法（在 OpenFOAM 算例根目录下运行）:
  python util/construct_OF_boundary_arrays_bbsf.py "09-01_10:00" \\
      --inlets east south --outlets west north --hfx 98.75
  python util/construct_OF_boundary_arrays_bbsf.py --input-nc /path/to/foo_cartesian_bbsf.nc ...
"""
import argparse
import os
import re
import sys

import numpy as np
import xarray as xr

G = 9.81
CP = 1004.0
RD = 287.0
P0 = 1.0e5
KAPPA = RD / CP
CMU = 0.09
KAPPA_VK = 0.41
L_CFD_MIN = 10.0
MAX_HEIGHT = 2100.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRF_AUXHIST_EXP03 = os.path.join(_REPO_ROOT, "W_myExp03", "auxhist2")
WRF_AUXHIST_EXP05 = os.path.join(_REPO_ROOT, "W_myExp05", "auxhist2")
WRF_EXP05_START = (2025, 9, 7)
DEFAULT_WRF_YEAR = 2025
NC_STEM_TEMPLATE = "auxhist2_d03_{wrf_iso}_1h-rolling_cartesian"
NC_SUFFIX = "_bbsf.nc"


# ---------------------------------------------------------------------------
# 输入路径
# ---------------------------------------------------------------------------
def parse_input_date_to_wrf_iso(input_date, year=DEFAULT_WRF_YEAR):
    s = input_date.strip()
    m = re.fullmatch(r"(?:(\d{4})-)?(\d{2})-(\d{2})_(\d{2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        yr = int(m.group(1) or year)
        mo, dy, hh, mm = (int(m.group(i)) for i in range(2, 6))
        return f"{yr:04d}-{mo:02d}-{dy:02d}_{hh:02d}:{mm:02d}:{int(m.group(6) or 0):02d}"
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", s)
    if m:
        yr, mo, dy, hh, mm = (int(x) for x in m.groups())
        return f"{yr:04d}-{mo:02d}-{dy:02d}_{hh:02d}:{mm:02d}:00"
    raise ValueError(f"无法解析时刻标签: {input_date!r}")


def wrf_auxhist_dir_for_time(wrf_iso):
    ymd = tuple(int(x) for x in re.match(r"(\d{4})-(\d{2})-(\d{2})_", wrf_iso).groups())
    return WRF_AUXHIST_EXP05 if ymd >= WRF_EXP05_START else WRF_AUXHIST_EXP03


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_date", nargs="?", metavar="MM-DD_HH:MM")
    ap.add_argument("--input-nc", metavar="PATH", help="显式指定 *_cartesian_bbsf.nc")
    ap.add_argument("--wrf-dir", default=None, help="auxhist2 目录（默认按日期选 W_myExp03/05）")
    ap.add_argument("--temp-var", choices=["Theta_v", "Theta"], default="Theta_v",
                    help="写入 T 的变量（默认 Theta_v）")
    ap.add_argument("--inlets", nargs="+", default=["east", "south"])
    ap.add_argument("--outlets", nargs="+", default=["west", "north"])
    ap.add_argument("--ground", default="lower", help="地面 patch 名")
    ap.add_argument("--walls", nargs="+", default=["lower", "buildings"],
                    help="使用 alphat 壁函数的 patch")
    ap.add_argument("--hfx", type=float, default=None,
                    help="地面显热通量 [W/m2]；不给则地面绝热 (zeroGradient)")
    ap.add_argument("--z0", type=float, default=0.1, help="alphat 壁函数 z0 [m]，与 nut 壁函数一致")
    ap.add_argument("--nu", type=float, default=1.5e-5)
    ap.add_argument("--Pr", type=float, default=0.71)
    ap.add_argument("--Prt", type=float, default=0.74)
    ap.add_argument("--case-dir", default=os.getcwd())
    ap.add_argument("--overwrite-thermal", action="store_true",
                    help="覆盖已存在的 0/T、0/p_rgh、0/alphat、transportProperties、g")
    args = ap.parse_args()
    if not args.input_nc and not args.input_date:
        ap.error("请提供 input_date，或使用 --input-nc。")
    return args


def resolve_input_nc(args):
    if args.input_nc:
        return os.path.abspath(args.input_nc)
    wrf_iso = parse_input_date_to_wrf_iso(args.input_date)
    aux_dir = args.wrf_dir or wrf_auxhist_dir_for_time(wrf_iso)
    return os.path.join(aux_dir, NC_STEM_TEMPLATE.format(wrf_iso=wrf_iso) + NC_SUFFIX)


# ---------------------------------------------------------------------------
# OpenFOAM 文本
# ---------------------------------------------------------------------------
def get_header(cls, location, obj):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2412                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


FOOTER = "\n// ************************************************************************* //\n"


def write_list(path, cls, location, obj, values, fmt):
    with open(path, "w") as f:
        f.write(get_header(cls, location, obj))
        f.write(f"\n{len(values)}\n(\n")
        for v in values:
            f.write(fmt(v) + "\n")
        f.write(")\n")


def patch_block(name, lines):
    return f"    {name}\n    {{\n" + "".join(f"        {l}\n" for l in lines) + "    }\n"


def read_patches(case_dir):
    path = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    text = open(path).read()
    body = text[text.index("(", text.index("// *")) + 1:]
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"\n\s*(\w+)\s*\n\s*\{[^}]*?\btype\s+(\w+);", body)]


def write_if_allowed(path, text, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f"  [Skip] {path} 已存在（用 --overwrite-thermal 覆盖）")
        return False
    with open(path, "w") as f:
        f.write(text)
    print(f"  [Write] {path}")
    return True


# ---------------------------------------------------------------------------
# 物理
# ---------------------------------------------------------------------------
def epsilon_from_k(k, z):
    return max((CMU ** 0.75) * (k ** 1.5) / np.clip(KAPPA_VK * z, L_CFD_MIN, 100.0), 1e-8)


def hydrostatic_prgh(z, T, beta, TRef):
    """单柱运动学 p_rgh（p(z=0)=0，hRef=0，g=(0 0 -9.81)）。z 需升序。"""
    rk = 1.0 - beta * (T - TRef)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rk[1:] + rk[:-1]) * np.diff(z))])
    return G * (rk * z - cum)


def main():
    args = parse_args()
    input_nc = resolve_input_nc(args)
    if not os.path.exists(input_nc):
        sys.exit(f"Error: File {input_nc} not found.（先运行 convert_wrf_coord_and_do_3D_interp_bbsf.py）")
    print(f"Resolved input: {input_nc}")
    ds = xr.open_dataset(input_nc)
    if "Theta_v" not in ds or not ds.attrs.get("z_reference", "").startswith("height above"):
        sys.exit("Error: 输入不是 *_bbsf.nc（缺少 Theta_v 或离地高度标记）。")
    ds = ds.where(ds.z <= MAX_HEIGHT, drop=True)

    case_dir = os.path.abspath(args.case_dir)
    zero_dir = os.path.join(case_dir, "0")
    const_dir = os.path.join(case_dir, "constant")
    tvar = args.temp_var

    # --- 全域平均 ---
    k_3d = np.maximum(ds["TKE_PBL"].values, 1e-6)
    z_3d = np.broadcast_to(ds["z"].values[:, None, None], k_3d.shape)
    eps_3d = np.maximum((CMU ** 0.75) * k_3d ** 1.5 / np.clip(KAPPA_VK * z_3d, L_CFD_MIN, 100.0), 1e-8)
    U_mean = tuple(float(np.nanmean(ds[v].values)) for v in ("U", "V", "W"))
    k_mean = float(np.nanmean(k_3d))
    eps_mean = float(np.nanmean(eps_3d))
    nut_mean = CMU * k_mean ** 2 / eps_mean
    TRef = float(np.nanmean(ds[tvar].values))
    beta = 1.0 / TRef
    alphat_mean = nut_mean / args.Prt
    T_sfc = float(ds[tvar].sel(z=0).mean())
    rho_sfc = float((P0 * ds["Exner"].sel(z=0) ** (1 / KAPPA) / (RD * ds["TK"].sel(z=0))).mean())
    Cp0 = rho_sfc * CP

    print("\n=== Domain-mean / reference values ===")
    print(f"U_mean   = ({U_mean[0]:.6g} {U_mean[1]:.6g} {U_mean[2]:.6g})")
    print(f"k_mean   = {k_mean:.6g}\neps_mean = {eps_mean:.6g}\nnut_mean = {nut_mean:.6g}")
    print(f"T var    = {tvar}\nTRef     = {TRef:.4f} K\nbeta     = {beta:.6e} 1/K")
    print(f"rho_sfc  = {rho_sfc:.4f} kg/m3\nCp0      = rho*cp = {Cp0:.1f} J/m3/K")

    # --- 更新已有 0/ 中的 U/k/epsilon/nut 初值（与原脚本相同） ---
    def update(fp, pat, rep):
        if not os.path.exists(fp):
            print(f"  [Warning] {fp} not found, skip.")
            return
        txt = open(fp).read()
        new, n = re.subn(pat, rep, txt, flags=re.MULTILINE)
        if n:
            open(fp, "w").write(new)
            print(f"  [Success] Updated {os.path.basename(fp)}")
        else:
            print(f"  [Warning] Pattern not found in {os.path.basename(fp)}")

    print(f"\n=== Updating {zero_dir} ===")
    update(os.path.join(zero_dir, "epsilon"), r"^epsilonInlet\s+[^;]+;", f"epsilonInlet  {eps_mean:.6e};")
    update(os.path.join(zero_dir, "k"), r"^kInlet\s+[^;]+;", f"kInlet  {k_mean:.6g};")
    update(os.path.join(zero_dir, "nut"), r"^internalField\s+uniform\s+[^;]+;",
           f"internalField   uniform {nut_mean:.6g};")
    update(os.path.join(zero_dir, "U"), r"^internalField\s+uniform\s+\([^)]+\);",
           f"internalField   uniform ({U_mean[0]:.6g} {U_mean[1]:.6g} {U_mean[2]:.6g});")

    # --- boundaryData ---
    boundaries = {
        "west": ("x_rel", 0), "east": ("x_rel", -1),
        "south": ("y_rel", 0), "north": ("y_rel", -1),
    }
    out_root = os.path.join(const_dir, "boundaryData")
    z = ds.z.values
    prgh_ranges = {}
    for patch, (sdim, idx) in boundaries.items():
        print(f"Processing {patch} boundary (H <= {MAX_HEIGHT}m)...")
        sl = ds.isel({sdim: idx})
        hdim = "y_rel" if sdim == "x_rel" else "x_rel"
        fixed = float(ds[sdim].values[idx])
        other = sl[hdim].values
        Tcol = sl[tvar].transpose("z", hdim).values
        prgh = np.stack([hydrostatic_prgh(z, Tcol[:, j], beta, TRef) for j in range(len(other))], axis=1)
        prgh_ranges[patch] = (prgh.min(), prgh.max())

        pts, uvw, ks, eps, ts, ps = [], [], [], [], [], []
        Uv = sl.U.transpose("z", hdim).values
        Vv = sl.V.transpose("z", hdim).values
        Wv = sl.W.transpose("z", hdim).values
        Kv = sl.TKE_PBL.transpose("z", hdim).values
        for i_z, zz in enumerate(z):
            for i_o, oo in enumerate(other):
                pts.append((fixed, oo, zz) if sdim == "x_rel" else (oo, fixed, zz))
                uvw.append((Uv[i_z, i_o], Vv[i_z, i_o], Wv[i_z, i_o]))
                kk = max(Kv[i_z, i_o], 1e-6)
                ks.append(kk)
                eps.append(epsilon_from_k(kk, zz))
                ts.append(Tcol[i_z, i_o])
                ps.append(prgh[i_z, i_o])

        pdir = os.path.join(out_root, patch)
        tdir = os.path.join(pdir, "0")
        os.makedirs(tdir, exist_ok=True)
        loc = f"constant/boundaryData/{patch}"
        write_list(os.path.join(pdir, "points"), "vectorField", loc, "points", pts,
                   lambda p: f"({p[0]:.4f} {p[1]:.4f} {p[2]:.4f})")
        loc0 = loc + "/0"
        write_list(os.path.join(tdir, "U"), "vectorAverageField", loc0, "U", uvw,
                   lambda u: f"({u[0]:.4f} {u[1]:.4f} {u[2]:.4f})")
        write_list(os.path.join(tdir, "k"), "scalarAverageField", loc0, "k", ks, lambda v: f"{v:.6f}")
        write_list(os.path.join(tdir, "epsilon"), "scalarAverageField", loc0, "epsilon", eps, lambda v: f"{v:.6e}")
        write_list(os.path.join(tdir, "T"), "scalarAverageField", loc0, "T", ts, lambda v: f"{v:.4f}")
        write_list(os.path.join(tdir, "p_rgh"), "scalarAverageField", loc0, "p_rgh", ps, lambda v: f"{v:.6f}")
    for p, (a, b) in prgh_ranges.items():
        print(f"  p_rgh hydrostatic range on {p}: {a:.3f} .. {b:.3f} m2/s2")

    # --- 0/T、0/p_rgh、0/alphat ---
    patches = read_patches(case_dir)
    names = [n for n, _ in patches]
    for n in args.inlets + args.outlets + [args.ground] + args.walls:
        if n not in names:
            sys.exit(f"Error: patch {n!r} 不在 constant/polyMesh/boundary 中：{names}")
    print(f"\n=== Writing thermal fields (patches: {names}) ===")

    def bf_T(n):
        if n in args.inlets:
            return ["type            timeVaryingMappedFixedValue;", "offset          constant 0;",
                    "outOfBounds     clamp;", "setAverage      false;", "value           uniform $TInlet;"]
        if n in args.outlets:
            return ["type            inletOutlet;", "inletValue      uniform $TInlet;",
                    "value           uniform $TInlet;"]
        if n == args.ground and args.hfx is not None:
            return ["type            atmTurbulentHeatFluxTemperature;", "heatSource      flux;",
                    "alphaEff        alphaEff;", f"Cp0             {Cp0:.2f};",
                    f"q               uniform {args.hfx:.4f};", f"value           uniform {T_sfc:.4f};"]
        return ["type            zeroGradient;"]

    def bf_prgh(n):
        if n in args.outlets:
            return ["type            timeVaryingMappedFixedValue;", "offset          constant 0;",
                    "outOfBounds     clamp;", "setAverage      false;", "value           uniform 0;"]
        return ["type            fixedFluxPressure;", "rho             rhok;", "value           uniform 0;"]

    def bf_alphat(n):
        if n in args.walls:
            return ["type            atmAlphatkWallFunction;", f"Cmu             {CMU};", "kappa           0.4;",
                    f"Pr              {args.Pr};", f"Prt             uniform {args.Prt};",
                    f"z0              uniform {args.z0};", "value           uniform 0;"]
        return ["type            calculated;", "value           uniform 0;"]

    def field_text(obj, dims, preamble, internal, fn):
        return (get_header("volScalarField", "0", obj) + preamble +
                f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n" +
                "".join(patch_block(n, fn(n)) for n in names) + "}\n" + FOOTER)

    t_written = write_if_allowed(os.path.join(zero_dir, "T"),
                                 field_text("T", "[0 0 0 1 0 0 0]", f"\nTInlet          {TRef:.4f};\n",
                                            "uniform $TInlet", bf_T), args.overwrite_thermal)
    if not t_written:
        update(os.path.join(zero_dir, "T"), r"^TInlet\s+[^;]+;", f"TInlet          {TRef:.4f};")
    write_if_allowed(os.path.join(zero_dir, "p_rgh"),
                     field_text("p_rgh", "[0 2 -2 0 0 0 0]", "", "uniform 0", bf_prgh), args.overwrite_thermal)
    write_if_allowed(os.path.join(zero_dir, "alphat"),
                     field_text("alphat", "[0 2 -1 0 0 0 0]", "", f"uniform {alphat_mean:.6g}", bf_alphat),
                     args.overwrite_thermal)

    # --- constant/ ---
    tp = (get_header("dictionary", "constant", "transportProperties") +
          f"\ntransportModel  Newtonian;\n\nnu              {args.nu};\n\n"
          f"beta            {beta:.6e};\nTRef            {TRef:.4f};\n"
          f"Pr              {args.Pr};\nPrt             {args.Prt};\n" + FOOTER)
    write_if_allowed(os.path.join(const_dir, "transportProperties"), tp, args.overwrite_thermal)
    gtxt = (get_header("uniformDimensionedVectorField", "constant", "g") +
            "\ndimensions      [0 1 -2 0 0 0 0];\nvalue           (0 0 -9.81);\n" + FOOTER)
    write_if_allowed(os.path.join(const_dir, "g"), gtxt, args.overwrite_thermal)

    print("\n提示: fvOptions 建议加入 atmBuoyancyTurbSource（beta 与上面一致），"
          f"limitTurbulenceViscosity 的 c 以 nu={args.nu} 计算上限。")
    print("Finished! OpenFOAM boundaryData and thermal fields have been created.")


if __name__ == "__main__":
    main()
