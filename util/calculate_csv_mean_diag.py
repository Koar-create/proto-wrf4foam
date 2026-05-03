#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 steady_experiments_finer_ABL 实验目录下 postProcessing/100m.csv 文件中 (U:0^2 + U:1^2)**0.5 的均值
用法: python calculate_csv_mean.py steady_experiments_finer_ABL/20250903_0000
"""

import os
import sys
import pandas as pd
import numpy as np

def calculate_wind_speed_mean(csv_file_path):
    """
    计算CSV文件中 (U:0^2 + U:1^2)**0.5 的均值
    
    参数:
        csv_file_path: CSV文件路径
        
    返回:
        float: (U:0^2 + U:1^2)**0.5 的均值，如果计算失败则返回 None
    """
    print(f"正在读取文件: {csv_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件不存在 - {csv_file_path}")
        return None
    
    try:
        # 读取CSV文件
        # 文件有100万行，使用chunksize分块读取以提高内存效率
        chunksize = 100000
        chunk_iter = pd.read_csv(csv_file_path, chunksize=chunksize)
        
        # 初始化累加器
        wind_speed_total_sum = 0.0
        total_count = 0
        
        for i, chunk in enumerate(chunk_iter):
            # 检查必要的列是否存在
            if 'U:0' in chunk.columns and 'U:1' in chunk.columns:
                # 提取 U:0 和 U:1 列
                u0_values = chunk['U:0'].astype(float)
                u1_values = chunk['U:1'].astype(float)
                
                # 计算每行的风速: sqrt(U:0^2 + U:1^2)
                wind_speed_values = np.sqrt(u0_values**2 + u1_values**2)
                
                wind_speed_total_sum += wind_speed_values.sum()
                total_count += len(wind_speed_values)
                print(f"  已处理块 {i+1}, 累计行数: {total_count}")
            else:
                missing_cols = []
                if 'U:0' not in chunk.columns:
                    missing_cols.append('U:0')
                if 'U:1' not in chunk.columns:
                    missing_cols.append('U:1')
                print(f"警告: 第 {i+1} 块中未找到列: {', '.join(missing_cols)}")
        
        # 计算均值
        if total_count > 0:
            wind_speed_mean = wind_speed_total_sum / total_count
            return wind_speed_mean
        else:
            print("错误: 未找到足够的数据计算风速")
            return None
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python calculate_csv_mean.py <experiment_directory>")
        print("示例: python calculate_csv_mean.py steady_experiments_finer_ABL/20250903_0000")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    # 构建CSV文件路径
    csv_file_path = os.path.join(experiment_dir, "postProcessing", "100m.csv")
    
    # 计算风速均值
    wind_speed_mean = calculate_wind_speed_mean(csv_file_path)
    
    if wind_speed_mean is not None:
        # 使用常规小数格式输出，保留6位小数
        print(f"\n风速 (U:0^2 + U:1^2)**0.5 均值: {wind_speed_mean:.6f}")
    else:
        print("计算风速均值失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
