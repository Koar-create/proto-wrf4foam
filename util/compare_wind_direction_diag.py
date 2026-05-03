#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比控制组与实验组域内平均水平风向的脚本。
用法: python compare_wind_direction.py steady_experiments_finer_ABL/20250902_0000_two_boundaries_as_outlet
"""

import os
import sys
import math
import pandas as pd

def get_meteorological_angle(u, v):
    """
    根据 U 和 V 分量计算气象学风向（风吹来的方向）。
    正北为0度，顺时针旋转（东为90度，南为180度，西为270度）。
    """
    # math.atan2(y, x)，这里使用 (-u, -v) 因为气象学风向是“风的来向”
    angle = math.atan2(-u, -v) * (180.0 / math.pi)
    if angle < 0:
        angle += 360.0
    return angle

def get_compass_direction(angle):
    """
    将 0-360 度的角度映射为 8 个主要方位的中文描述。
    """
    directions = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]
    # 偏移 22.5 度以便于按 45 度区间划分
    idx = int(((angle + 22.5) % 360) // 45)
    return directions[idx]

def calculate_domain_average_uv(csv_file_path):
    """
    分块读取CSV，计算全域平均的 U:0 和 U:1 分量。
    """
    chunksize = 100000
    sum_u = 0.0
    sum_v = 0.0
    total_count = 0
    
    try:
        chunk_iter = pd.read_csv(csv_file_path, chunksize=chunksize)
        for chunk in chunk_iter:
            if 'U:0' in chunk.columns and 'U:1' in chunk.columns:
                u_values = chunk['U:0'].astype(float)
                v_values = chunk['U:1'].astype(float)
                
                sum_u += u_values.sum()
                sum_v += v_values.sum()
                total_count += len(u_values)
                
        if total_count > 0:
            return sum_u / total_count, sum_v / total_count
        else:
            return None, None
    except Exception as e:
        print(f"读取或处理数据时出错: {e}")
        return None, None

def analyze_wind_direction(experiment_dir):
    """
    分析并对比两组实验数据的风向。
    """
    # 构建对照组（控制组）路径
    control_csv = os.path.join(experiment_dir, "postProcessing", "100m.csv")
    
    # 构建实验组（fvOpt_sensitivity_run）路径
    # 移除末尾可能的斜杠再拼接，以防止路径解析错误
    base_dir = experiment_dir.rstrip('/')
    experiment_csv = os.path.join(f"{base_dir}-fvOpt_sensitivity_run", "postProcessing", "100m.csv")
    
    # 检查文件存在性
    if not os.path.exists(control_csv):
        print(f"错误: 对照组文件不存在 - {control_csv}")
        sys.exit(1)
        
    if not os.path.exists(experiment_csv):
        print(f"警告: 实验组文件不存在 - {experiment_csv}")
        print("请检查 fvOpt_sensitivity_run 目录是否已正确生成并包含数据。程序退出。")
        sys.exit(1)
        
    print(f"正在分析对照组: {control_csv}")
    ctrl_u, ctrl_v = calculate_domain_average_uv(control_csv)
    
    print(f"正在分析实验组: {experiment_csv}")
    exp_u, exp_v = calculate_domain_average_uv(experiment_csv)
    
    if None in (ctrl_u, ctrl_v, exp_u, exp_v):
        print("错误: 无法完成风向计算，请检查CSV文件中是否包含 U:0 和 U:1 列。")
        sys.exit(1)
        
    # 计算角度和文本描述
    ctrl_angle = get_meteorological_angle(ctrl_u, ctrl_v)
    ctrl_direction = get_compass_direction(ctrl_angle)
    
    exp_angle = get_meteorological_angle(exp_u, exp_v)
    exp_direction = get_compass_direction(exp_angle)
    
    # 打印最终对比结果
    print("-" * 50)
    print(f"对照组平均向量 (U, V) : ({ctrl_u:.4f}, {ctrl_v:.4f}) m/s")
    print(f"对照组域内平均风向   : {ctrl_angle:.2f}° [{ctrl_direction}]")
    print("")
    print(f"实验组平均向量 (U, V) : ({exp_u:.4f}, {exp_v:.4f}) m/s")
    print(f"实验组域内平均风向   : {exp_angle:.2f}° [{exp_direction}]")
    print("-" * 50)
    
    if ctrl_direction == exp_direction:
        print(f"结论: 对照组与实验组风向 相同 (均为 {ctrl_direction})。")
    else:
        print(f"结论: 对照组与实验组风向 不同 (对照组为 {ctrl_direction}，实验组为 {exp_direction})。")

def main():
    # 参数检查
    if len(sys.argv) != 2:
        print("用法: python compare_wind_direction.py <experiment_directory>")
        print("示例: python compare_wind_direction.py steady_experiments_finer_ABL/20250902_0000_two_boundaries_as_outlet")
        sys.exit(1)
        
    experiment_dir = sys.argv[1]
    analyze_wind_direction(experiment_dir)

if __name__ == "__main__":
    main()
