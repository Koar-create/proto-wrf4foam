#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import re
import numpy as np

# --- 配置 ---
BASE_PATH = sys.argv[1]
# BASE_PATH = f"{HOME}/WRF-OpenFOAM-Coupling/steady_experiments/20250902_1200/constant/boundaryData"
PATCHES = ["west", "east", "south", "north"] # 你想检查的边界
FIELDS = ["U", "k", "epsilon", "T"] # 你想检查的变量 (T可选)

# --- 阈值设置 (用于报警) ---
THRESHOLDS = {
    "U_max": 100.0,       # 风速上限 (m/s)
    "k_min": 0.0,         # k 下限 (不能为负)
    "epsilon_min": 0.0,   # epsilon 下限 (不能为负)
    "T_min": 250.0,       # 温度下限 (K) - 如果有T的话
    "T_max": 350.0        # 温度上限 (K)
}

def parse_foam_file(filepath, field_name):
    """
    简单的文本解析器，提取括号内的数据
    """
    if not os.path.exists(filepath):
        return None, "File Not Found"

    with open(filepath, 'r') as f:
        content = f.read()

    # 1. 寻找数据块：找到 start "(" 和 end ")"
    # OpenFOAM 文件的头部之后，会有一个数量，然后是数据列表
    # 例如: 
    # 4444
    # (
    # ...
    # )
    
    # 查找最后一个孤立的 '(' 和 ')' 包裹的内容，这通常是数据主体
    # 这种简单的正则适用于标准的 boundaryData 文件
    match = re.search(r'\n\s*(\d+)\s*\n\s*\(\s*\n([\s\S]+?)\n\s*\)', content)
    
    if not match:
        return None, "Parse Error (Format mismatch)"
    
    count = int(match.group(1))
    raw_data = match.group(2).strip()
    
    # 2. 解析数值
    try:
        if field_name == "U":
            # 向量: (10.2 0 0) -> 去掉括号 -> split
            # 正则替换掉 '(' and ')'
            cleaned = raw_data.replace('(', '').replace(')', '')
            data = np.fromstring(cleaned, sep=' ')
            if data.size != count * 3:
                return None, f"Size Mismatch (Expected {count*3}, got {data.size})"
            data = data.reshape(count, 3)
        else:
            # 标量: 0.02
            data = np.fromstring(raw_data, sep=' ')
            if data.size != count:
                return None, f"Size Mismatch (Expected {count}, got {data.size})"
                
    except Exception as e:
        return None, f"Numpy Parse Error: {str(e)}"

    return data, "OK"

def check_field(patch, field, data):
    issues = []
    stats = {}
    
    # 1. 基础检查：NaN 和 Inf
    if np.any(np.isnan(data)):
        issues.append("❌ CRITICAL: Contains NaN (Not a Number)!")
    if np.any(np.isinf(data)):
        issues.append("❌ CRITICAL: Contains Inf (Infinity)!")

    # 2. 统计值
    if field == "U":
        # 计算合速度
        mag = np.linalg.norm(data, axis=1)
        stats['min'] = np.min(mag)
        stats['max'] = np.max(mag)
        stats['mean'] = np.mean(mag)
        
        if stats['max'] > THRESHOLDS["U_max"]:
            issues.append(f"⚠️ Warning: Max velocity {stats['max']:.2f} > {THRESHOLDS['U_max']} m/s. Check units!")
        if stats['max'] == 0.0:
            issues.append(f"⚠️ Warning: Max velocity is 0.0 (Still air?).")
            
    else:
        # 标量
        stats['min'] = np.min(data)
        stats['max'] = np.max(data)
        stats['mean'] = np.mean(data)

        # 负值检查 (k, epsilon 绝对不能为负)
        if field in ["k", "epsilon"]:
            if stats['min'] < 0:
                issues.append(f"❌ CRITICAL: Negative values found! Min: {stats['min']}")
            elif stats['min'] == 0:
                issues.append(f"⚠️ Warning: Zero values found (potential division by zero).")
        elif field == "T":
            if stats['min'] < THRESHOLDS["T_min"]:
                issues.append(
                    f"⚠️ Warning: T min {stats['min']:.2f} < {THRESHOLDS['T_min']} K"
                )
            if stats['max'] > THRESHOLDS["T_max"]:
                issues.append(
                    f"⚠️ Warning: T max {stats['max']:.2f} > {THRESHOLDS['T_max']} K"
                )

            T_ref = 300.0
            dTmax = np.max(np.abs(data - T_ref))
            if dTmax > 30.0:
                issues.append(
                    f"⚠️ Warning: max |T-TRef| = {dTmax:.2f} K, large for Boussinesq"
                )
            elif dTmax > 20.0:
                issues.append(
                    f"ℹ️ Notice: max |T-TRef| = {dTmax:.2f} K"
                )

    return stats, issues

def main():
    print(f"{'='*60}")
    print(f" 🩺 OpenFOAM Boundary Data Health Check")
    print(f"{'='*60}")
    
    found_issues = False

    for patch in PATCHES:
        for field in FIELDS:
            # 构建路径: constant/boundaryData/west/0/U
            filepath = os.path.join(BASE_PATH, patch, "0", field)
            
            # 这里的打印是为了对齐好看
            prefix = f"[{patch}/{field}]".ljust(20)
            
            data, status = parse_foam_file(filepath, field)
            
            if status != "OK":
                if "File Not Found" in status:
                    # 文件不存在通常是因为没生成，跳过即可
                    continue
                print(f"{prefix} 🔴 {status}")
                found_issues = True
                continue

            # 执行数值检查
            stats, issues = check_field(patch, field, data)
            
            if issues:
                found_issues = True
                print(f"{prefix} 🔴 FAILED Checks")
                for issue in issues:
                    print(f"    └─ {issue}")
                print(f"    └─ Stats: Min={stats['min']:.4e}, Max={stats['max']:.4e}, Mean={stats['mean']:.4e}")
            else:
                print(f"{prefix} 🟢 PASS (Min: {stats['min']:.2e} | Max: {stats['max']:.2e})")

    print(f"{'='*60}")
    if found_issues:
        print("⚠️  检测到潜在问题，建议在运行 simpleFoam 前修正！")
    else:
        print("✅  所有文件数值看起来很健康！Ready to run.")

if __name__ == "__main__":
    main()