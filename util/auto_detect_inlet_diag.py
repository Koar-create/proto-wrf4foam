#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

def read_foam_vector_field(filepath):
    """鲁棒地解析 OpenFOAM 格式的边界 U 场"""
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
            if line.startswith("("):
                vals = line.strip("()").split()
                data.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(data)

def analyze_boundary_flux(bd_dir):
    print("="*65)
    print(" 🌬️  WRF 边界风向与出入口自动识别 (Boundary Flux Detector)")
    print("="*65)
    
    patches = ['west', 'east', 'south', 'north']
    
    # 存储各个面的平均风速分量
    wind_data = {}
    
    for patch in patches:
        u_file = os.path.join(bd_dir, patch, "0", "U")
        U_array = read_foam_vector_field(u_file)
        
        if U_array is not None and len(U_array) > 0:
            # 计算该边界的平均 U (X方向) 和 V (Y方向) 分量
            mean_u = np.mean(U_array[:, 0])
            mean_v = np.mean(U_array[:, 1])
            wind_data[patch] = {'U': mean_u, 'V': mean_v}
        else:
            print(f"⚠️  警告: 找不到或无法读取 {patch} 边界的 0/U 数据。")

    if not wind_data:
        print("❌ 错误: 未能在任何边界读取到风速数据。")
        return

    print("\n📊 [宏观风速分析]")
    # 计算全域平均宏观风
    avg_U = np.mean([v['U'] for v in wind_data.values()])
    avg_V = np.mean([v['V'] for v in wind_data.values()])
    print(f"   • 全域平均纬向风速 (U, 西-东): {avg_U:+.3f} m/s")
    print(f"   • 全域平均经向风速 (V, 南-北): {avg_V:+.3f} m/s")
    
    # 确定主导风向
    wind_dir_str = ""
    if avg_U > 0 and avg_V > 0: wind_dir_str = "西南风 (Southwesterly)"
    elif avg_U < 0 and avg_V > 0: wind_dir_str = "东南风 (Southeasterly)"
    elif avg_U > 0 and avg_V < 0: wind_dir_str = "西北风 (Northwesterly)"
    else: wind_dir_str = "东北风 (Northeasterly)"
    print(f"   • 气象学宏观风向判别: {wind_dir_str}\n")

    print("🎯 [OpenFOAM 边界属性判定]")
    
    inlets = []
    outlets = []

    # 物理逻辑判定
    for patch, vel in wind_data.items():
        is_inlet = False
        u, v = vel['U'], vel['V']
        
        if patch == 'west' and u > 0:
            is_inlet = True
        elif patch == 'east' and u < 0:
            is_inlet = True
        elif patch == 'south' and v > 0:
            is_inlet = True
        elif patch == 'north' and v < 0:
            is_inlet = True
            
        if is_inlet:
            inlets.append(patch)
            print(f"   🟢 {patch.upper():<5} 边界: 入气口 (Inlet)   [法向通量指向域内]")
        else:
            outlets.append(patch)
            print(f"   🔴 {patch.upper():<5} 边界: 出气口 (Outlet)  [法向通量指向域外]")

    print("\n🛠️ [0/U 配置文件推荐行动 (Action Items)]")
    for patch in inlets:
        print(f"   - {patch}: 设置为 timeVaryingMappedFixedValue (强迫入流)")
    for patch in outlets:
        print(f"   - {patch}: 设置为 inletOutlet, inletValue uniform (0 0 0) (防御性自由出流)")
        
    print("\n" + "="*65)

def main():
    if len(sys.argv) != 2:
        print("用法: python auto_detect_inlet.py <boundaryData_路径>")
        print("示例: python auto_detect_inlet.py ./steady_experiments_finer_ABL/20250903_1200_two_boundaries_as_outlet/constant/boundaryData")
        sys.exit(1)
        
    bd_dir = sys.argv[1]
    
    if not os.path.exists(bd_dir):
        print(f"❌ 找不到指定的 boundaryData 目录: {bd_dir}")
        sys.exit(1)
            
    analyze_boundary_flux(bd_dir)

if __name__ == "__main__":
    main()
