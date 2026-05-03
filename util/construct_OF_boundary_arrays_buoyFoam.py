#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys
import numpy as np
import xarray as xr

# 引入环境配置
HOME = os.environ.get('HOME')

project_path = f"{HOME}/WRF-OpenFOAM-Coupling/W_myExp03"

if len(sys.argv) != 2:
    print(f"Error: Missing input date argument.")
    print(f"Usage: python {sys.argv[0]} <mm-dd_HH%3AMM>")
    print(f"Example: python {sys.argv[0]} \"09-01_00:00\"")
    sys.exit(1)

input_date = sys.argv[1]

# --- 1. 读取已经处理好的拉伸网格 NC 文件 ---
input_nc = f"{project_path}/auxhist2/auxhist2_d03_2025-{input_date}:00_1h-rolling_cartesian.nc"
if not os.path.exists(input_nc):
    print(f"Error: File {input_nc} not found.")
    sys.exit(1)
ds = xr.open_dataset(input_nc)

# --- 新增：定义截断高度，并进行全局高度截断 ---
MAX_HEIGHT = 2100.0  # 截断高度 (米)，统一在这里定义
ds = ds.where(ds.z <= MAX_HEIGHT, drop=True)

# --- 1.5 计算整个 CFD 计算域的 U / T / k / epsilon 均值 ---

def compute_domain_mean_fields(ds):
    """
    基于整个插值后的 CFD 笛卡尔域，计算：
    - U_mean_vec: 全域平均速度向量 (Ux_mean, Uy_mean, Uz_mean)
    - T_mean:     全域平均温度（如果 ds 里有 T）
    - k_mean:     全域平均 k (= TKE_PBL)
    - eps_mean:   全域平均 epsilon，与 boundaryData 生成公式一致
    """
    Cmu = 0.09
    kappa = 0.41
    L_CFD_min = 10.0

    # --- U mean ---
    Ux_mean = float(np.nanmean(ds["U"].values))
    Uy_mean = float(np.nanmean(ds["V"].values))
    Uz_mean = float(np.nanmean(ds["W"].values))
    U_mean_vec = (Ux_mean, Uy_mean, Uz_mean)

    # --- T mean ---
    T_mean = float(np.nanmean(ds["TK"].values))  # ds["Theta"].values

    # --- k mean ---
    k_3d = np.maximum(ds["TKE_PBL"].values, 1e-6)
    k_mean = float(np.nanmean(k_3d))

    # --- epsilon mean ---
    z_1d = ds["z"].values
    z_3d = np.broadcast_to(z_1d[:, None, None], k_3d.shape)
    mixing_length = np.clip(kappa * z_3d, L_CFD_min, 100.0)

    eps_3d = (Cmu**0.75) * (k_3d**1.5) / mixing_length
    eps_3d = np.maximum(eps_3d, 1e-8)
    eps_mean = float(np.nanmean(eps_3d))

    return U_mean_vec, T_mean, k_mean, eps_mean

# 计算全域均值
U_mean_vec, T_mean_domain, k_mean_domain, eps_mean_domain = compute_domain_mean_fields(ds)

# --- 新增：计算自洽的初始湍流粘度 (nut) 和湍流热扩散率 (alphat) ---
Cmu_const = 0.09
Prt_const = 0.85  # 必须与 transportProperties 和 0/alphat 保持一致
nut_mean_domain = Cmu_const * (k_mean_domain**2) / eps_mean_domain
alphat_mean_domain = nut_mean_domain / Prt_const

print("\n=== Domain-mean initial fields for OpenFOAM 0/ ===")
print()
print(f"epsilon_mean_domain = {eps_mean_domain:.6g}")
print()
print(f"k_mean_domain       = {k_mean_domain:.6g}")
print()
print(f"nut_mean_domain     = {nut_mean_domain:.6g}")
print()
print(f"U_mean_domain       = ({U_mean_vec[0]:.6g} {U_mean_vec[1]:.6g} {U_mean_vec[2]:.6g})")
print()
print(f"T_mean_domain       = {T_mean_domain:.4f}")
print()
print(f"alphat_mean_domain  = {alphat_mean_domain:.6g}")
print()

# ========================================================================= #
# --- 新增：自动替换 steady_experiments/0/ 下文件的初始场 -------------------- #
# ========================================================================= #
steady_case_dir = os.getcwd()
of_zero_dir = os.path.join(steady_case_dir, "0")

def update_of_dictionary(filepath, pattern, replacement):
    """读取 OpenFOAM 字典文件，使用正则替换指定内容，并写回"""
    if not os.path.exists(filepath):
        print(f"  [Warning] File not found: {filepath}, skipping update.")
        return
    with open(filepath, 'r') as f:
        content = f.read()
    
    # re.subn 返回替换后的新字符串和发生替换的次数
    # [^;]+ 表示匹配到分号前的内容，这样能完好保留后面的注释部分 (如 // 1.5;)
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    
    if count > 0:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  [Success] Updated {os.path.basename(filepath)} in {of_zero_dir}")
    else:
        print(f"  [Warning] Pattern not found in {os.path.basename(filepath)}")

print(f"\n=== Updating OpenFOAM 0/ dictionaries in {steady_case_dir} ===")

# 1. 替换 epsilon 文件中的 epsilonInlet 
eps_file = os.path.join(of_zero_dir, "epsilon")
update_of_dictionary(eps_file, r"^epsilonInlet\s+[^;]+;", f"epsilonInlet  {eps_mean_domain:.6e};")

# 2. 替换 k 文件中的 kInlet
k_file = os.path.join(of_zero_dir, "k")
update_of_dictionary(k_file, r"^kInlet\s+[^;]+;", f"kInlet  {k_mean_domain:.6g};")

# 3. 替换 nut 文件中的 internalField
nut_file = os.path.join(of_zero_dir, "nut")
update_of_dictionary(nut_file, r"^internalField\s+uniform\s+[^;]+;", f"internalField   uniform {nut_mean_domain:.6g};")

# 4. 替换 U 文件中的 internalField uniform (...)
u_file = os.path.join(of_zero_dir, "U")
update_of_dictionary(u_file, r"^internalField\s+uniform\s+\([^)]+\);", f"internalField   uniform ({U_mean_vec[0]:.6g} {U_mean_vec[1]:.6g} {U_mean_vec[2]:.6g});")

# 5. 替换 T 文件中的 TInlet (配合 buoyantBoussinesqSimpleFoam)
t_file = os.path.join(of_zero_dir, "T")
update_of_dictionary(t_file, r"^TInlet\s+[^;]+;", f"TInlet          {T_mean_domain:.4f};")

# 6. 替换 alphat 文件中的 internalField
alphat_file = os.path.join(of_zero_dir, "alphat")
update_of_dictionary(alphat_file, r"^internalField\s+uniform\s+[^;]+;", f"internalField   uniform {alphat_mean_domain:.6g};")
# ========================================================================= #

# --- 2. 配置部分 ---
MAX_HEIGHT = 2100.0  # 截断高度 (米)，建议比计算域高度(2000)稍高一点作为缓冲
output_dir = "boundaryData"
# 这里的定义要与你插值脚本中的网格范围一致
boundaries = {
    "west":  {"slice_dim": "x_rel", "index":  0,   "fixed_coord": "x"},
    "east":  {"slice_dim": "x_rel", "index": -1,   "fixed_coord": "x"},
    "south": {"slice_dim": "y_rel", "index":  0,   "fixed_coord": "y"},
    "north": {"slice_dim": "y_rel", "index": -1,   "fixed_coord": "y"},
}

# OpenFOAM 文件头模板
def get_header(cls, location, obj):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  8
     \\\\/     M anipulation  |
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

# --- 3. 循环处理每个边界 ---
for patch_name, cfg in boundaries.items():
    print(f"Processing {patch_name} boundary (H <= {MAX_HEIGHT}m)...")
    
    # 提取切片 (Slicing)
    # 切片提取边界数据 (z, target_dim)
    # 例如 west 边界，我们会得到 (z, y_rel) 的二维面
    # isel(x_rel=0) 取的是 x_rel 维度上的第 0 个数据
    if cfg["slice_dim"] == "x_rel":
        ds_slice = ds.isel(x_rel=cfg["index"])
        # 此时 ds_slice 的维度应该是 (z, y_rel)
        dim_horizontal = "y_rel"
    else:
        ds_slice = ds.isel(y_rel=cfg["index"])
        # 此时 ds_slice 的维度应该是 (z, x_rel)
        dim_horizontal = "x_rel"
        
    # 提取坐标
    z_coords = ds_slice.z.values
    other_coords = ds_slice[dim_horizontal].values

    # 生成 OpenFOAM 格式的点阵坐标
    # OpenFOAM 坐标顺序通常是 (x, y, z)
    pts = []
    u_vals = []
    k_vals = []
    eps_vals = []
    t_vals = []  # 新增：用于存储位温数据

    # 遍历面上的每一个网格点
    # 注意：这里的顺序必须在 points 和 data 文件中完全同步
    for i_z, z in enumerate(z_coords):
        for i_o, other in enumerate(other_coords):
            # 根据边界类型分配 X 和 Y
            # 确定坐标 (x, y, z)
            if cfg["slice_dim"] == "x_rel":
                x_val = ds.x_rel.values[cfg["index"]]
                y_val = other
            else:
                x_val = other
                y_val = ds.y_rel.values[cfg["index"]]
            
            pts.append((x_val, y_val, z))
            
            # 提取物理量
            u = ds_slice.U.values[i_z, i_o]
            v = ds_slice.V.values[i_z, i_o]
            w = ds_slice.W.values[i_z, i_o]
            tke = ds_slice.TKE_PBL.values[i_z, i_o]
            tk = ds_slice.TK.values[i_z, i_o]  # 提取绝对位温
            
            u_vals.append((u, v, w))
            t_vals.append(tk)                     # 保存到位温列表
            
            # --- 湍流换算 ---
            k = max(tke, 1e-6)
            k_vals.append(k)
            # epsilon 的简单估计公式: eps = Cmu^0.75 * k^1.5 / L
            Cmu = 0.09
            kappa = 0.41
            # L_ABL = kappa * z (von Karman 混合长度), 上限 100m 防止自由大气过大
            # L_CFD_min: CFD 近地面网格的代表性单元高度下限（约 10m）
            # 当 z 很小时（如 z=5m），kappa*z=2.05m 远小于 CFD 分辨率，
            # 会导致边界注入极高的 epsilon（~2.6e-2），CFD 内部场无法解析该梯度，
            # 引发 k-epsilon 生成-耗散失衡，k 单调积累直至发散。
            # 解决：L 取 kappa*z 与 CFD 代表性高度（10m）中的较大值。
            L_CFD_min = 10.0  # m，对应 CFD 近地面单元的代表性高度
            mixing_length = np.clip(kappa * z, L_CFD_min, 100.0)
            eps = (Cmu**0.75) * (k**1.5) / mixing_length
            eps_vals.append(max(eps, 1e-8))


    # --- 4. 写入文件 ---
    patch_dir = os.path.join(output_dir, patch_name)
    # 时间文件夹，这里假设是 0 时刻，后续你可以根据循环处理多个时刻
    time_dir = os.path.join(patch_dir, "0")
    os.makedirs(time_dir, exist_ok=True)
    
    num_points = len(pts)
    
    # A. 写入 points 文件
    points_path = os.path.join(patch_dir, "points")
    with open(points_path, "w") as f:
        f.write(get_header("vectorField", f"constant/boundaryData/{patch_name}", "points"))
        f.write(f"\n{num_points}\n(\n")
        for p in pts:
            # 保留4位小数，清爽且够用
            f.write(f"({p[0]:.4f} {p[1]:.4f} {p[2]:.4f})\n")
        f.write(")\n")

    # B. 写入 0/U 文件
    # 注意：这里我们直接生成在 boundaryData 里的 0 文件夹下
    u_path = os.path.join(time_dir, "U")
    with open(u_path, "w") as f:
        # 注意 class 是 vectorAverageField 或者 vectorField，这里用 data 格式
        f.write(get_header("vectorAverageField", f"constant/boundaryData/{patch_name}/0", "U"))
        f.write(f"\n{num_points}\n(\n")
        for u in u_vals:
            f.write(f"({u[0]:.4f} {u[1]:.4f} {u[2]:.4f})\n")
        f.write(")\n")
    
    # C. --- 写入 0/k ---
    k_path = os.path.join(time_dir, "k")
    with open(k_path, "w") as f:
        # k 是标量场 (scalarAverageField 或 scalarField)
        f.write(get_header("scalarAverageField", f"constant/boundaryData/{patch_name}/0", "k"))
        f.write(f"\n{num_points}\n(\n")
        for k in k_vals:
            f.write(f"{k:.6f}\n")
        f.write(")\n")
    
    # D. --- 写入 0/epsilon ---
    eps_path = os.path.join(time_dir, "epsilon")
    with open(eps_path, "w") as f:
        f.write(get_header("scalarAverageField", f"constant/boundaryData/{patch_name}/0", "epsilon"))
        f.write(f"\n{num_points}\n(\n")
        for e in eps_vals:
            # !!! 修改点 2: 改用科学计数法 .6e !!!
            f.write(f"{e:.6e}\n")
        f.write(")\n")

    # E. --- 新增：写入 0/T ---
    t_path = os.path.join(time_dir, "T")
    with open(t_path, "w") as f:
        # T 是标量场 (scalarAverageField)
        f.write(get_header("scalarAverageField", f"constant/boundaryData/{patch_name}/0", "T"))
        f.write(f"\n{num_points}\n(\n")
        for t_val in t_vals:
            # 保留4位小数即可
            f.write(f"{t_val:.4f}\n")
        f.write(")\n")

print("\nFinished! OpenFOAM boundaryData has been created.")
