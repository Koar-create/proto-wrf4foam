#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

def read_foam_field(filepath, is_vector=False):
    """鲁棒地解析 OpenFOAM 格式的 boundaryData"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data = []
    in_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("/*"): 
            continue
        if not in_data:
            if line == "(": 
                in_data = True
        else:
            if line == ")": 
                break
            if is_vector:
                if line.startswith("("):
                    vals = line.strip("()").split()
                    data.append([float(vals[0]), float(vals[1]), float(vals[2])])
            else:
                data.append(float(line))
    return np.array(data)

def analyze_patch(patch_name, patch_dir):
    print(f"\n" + "-"*50)
    print(f"👉 正在分析边界 Patch: {patch_name.upper()}")
    print("-"*50)
    
    pts_file = os.path.join(patch_dir, "points")
    U_file = os.path.join(patch_dir, "0", "U")
    k_file = os.path.join(patch_dir, "0", "k")
    eps_file = os.path.join(patch_dir, "0", "epsilon")
    
    pts = read_foam_field(pts_file, is_vector=True)
    U = read_foam_field(U_file, is_vector=True)
    k = read_foam_field(k_file, is_vector=False)
    eps = read_foam_field(eps_file, is_vector=False)
    
    if any(v is None for v in [pts, U, k, eps]):
        print(f"⚠️  错误: 在 {patch_name} 目录下找不到完整的 points, U, k, 或 epsilon 文件。")
        return

    # 1. 按高度 (Z) 进行水平平均剖面计算
    Z = np.round(pts[:, 2], decimals=1) # 容忍浮点误差
    z_unique = np.unique(Z)
    
    k_prof, eps_prof, U_mag_prof = [], [], []
    for z in z_unique:
        mask = (Z == z)
        k_prof.append(np.mean(k[mask]))
        eps_prof.append(np.mean(eps[mask]))
        # 水平风速大小 (忽略垂直的 W，更符合气象学 U_h)
        U_mag_prof.append(np.mean(np.linalg.norm(U[mask][:, :2], axis=1))) 
        
    k_prof = np.array(k_prof)
    eps_prof = np.array(eps_prof)
    U_mag_prof = np.array(U_mag_prof)
    
    # 2. 物理指标计算
    # 2.1 高空平均 TKE (大于 500m)
    mask_aloft = z_unique > 500
    if np.any(mask_aloft):
        k_aloft_mean = np.mean(k_prof[mask_aloft])
    else:
        k_aloft_mean = np.mean(k_prof)
        
    k_max = np.max(k_prof)
    
    # 2.2 湍流混合长度 Lt = C_mu^0.75 * k^1.5 / epsilon
    C_mu = 0.09
    Lt_prof = (C_mu**0.75) * (k_prof**1.5) / (eps_prof + 1e-15)
    Lt_max = np.max(Lt_prof)
    
    # 2.3 低空急流 (LLJ) 探测
    # 如果在 50m 到 600m 之间出现了显著的速度极大值，说明可能发生脱耦
    mask_llj = (z_unique > 50) & (z_unique < 600)
    U_llj_max = np.max(U_mag_prof[mask_llj]) if np.any(mask_llj) else 0
    U_top = np.mean(U_mag_prof[z_unique > 1500]) if np.any(z_unique > 1500) else U_llj_max
    is_llj = U_llj_max > (U_top * 1.1) and U_llj_max > 3.0 # 低空极大值大于高空10%
    
    # 3. 打印物理指标
    print(f"📊 [气象学特征提取]")
    print(f"   • 最大湍流动能 (k_max):       {k_max:.4f} m^2/s^2")
    print(f"   • 500m以上高空平均 k:       {k_aloft_mean:.4f} m^2/s^2")
    print(f"   • 最大湍流混合长度 (Lt_max): {Lt_max:.1f} m")
    print(f"   • 是否出现低空急流 (LLJ):    {'是 (强风切变)' if is_llj else '否'}")
    
    # 4. 判决与背书
    print(f"\n🔍 [WRF->RANS 物理适用性判决]")
    
    # 判决逻辑（基于 WRF 降尺度 RANS 的典型阈值）
    if k_aloft_mean < 0.05 and k_max < 0.6:
        print("   ⛔ 判定结果: 强稳定层结 (Strongly Stable ABL) / '夜间僵尸局'")
        print("   📖 物理背书: 高空 TKE 极低，湍流被浮力极度抑制。在 OpenFOAM 中会导致 ν_t 趋近于零。")
        print("   ⚠️  CFD 预警: 流场退化为准层流，城市建筑尾流将发生大规模的非物理剥离，导致压力方程 (p) 极易发散。纯中性 simpleFoam 在此数据下几乎必死，建议剔除！")
    
    elif k_aloft_mean > 0.2 or k_max > 1.5:
        print("   🔥 判定结果: 不稳定层结 (Unstable / Convective ABL)")
        print("   📖 物理背书: 湍流深厚且旺盛，高空维持了良好的湍流动能储备。")
        print("   ✅ CFD 预警: 非常适合稳态 RANS 计算。丰富的湍流粘度将加速流场平滑与方程收敛，是极其优质的驱动数据。")
        
    else:
        print("   ⚖️  判定结果: 中性/弱稳定层结 (Neutral / Weakly Stable ABL)")
        print("   📖 物理背书: 大气处于过渡态，保有合理的混合长度和适度的剪切。")
        print("   ✅ CFD 预警: 只要网格和松弛因子设置得当，simpleFoam 通常可以顺利求得稳态解。可放心使用。")

def main():
    if len(sys.argv) < 3:
        print("用法: python diagnose_stability_diag.py <case_dir> <patch1> [patch2 ...]")
        print("示例: python diagnose_stability_diag.py ./steady_experiments/20250903_1200 east south")
        sys.exit(1)
        
    case_dir = sys.argv[1]
    patches = sys.argv[2:]
    
    print("="*65)
    print(" 🌪️  WRF 入流边界大气稳定度诊断 (CFD Applicability Verifier)")
    print("="*65)
    
    # 定位 boundaryData 目录
    bd_dir = os.path.join(case_dir, "constant", "boundaryData")
    if not os.path.exists(bd_dir):
        # 兼容如果用户直接传了 constant 目录或 boundaryData 目录
        if "boundaryData" in case_dir:
            bd_dir = case_dir
        else:
            print(f"❌ 找不到 boundaryData 目录，请检查路径: {bd_dir}")
            sys.exit(1)
            
    for patch in patches:
        patch_dir = os.path.join(bd_dir, patch)
        if not os.path.exists(patch_dir):
            print(f"⚠️  警告: 找不到指定的边界文件夹 {patch_dir}")
            continue
        analyze_patch(patch, patch_dir)
        
    print("\n" + "="*65)

if __name__ == "__main__":
    main()
