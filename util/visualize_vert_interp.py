#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化垂直插值前后的风速剖面
"""
import os
import sys
import numpy as np
import pandas as pd

# 使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 不显示图形，只保存
import matplotlib.pyplot as plt

# 设置中文字体（如果需要）
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_interpolation_comparison():
    """
    可视化插值前后的风速剖面对比
    """
    # 文件路径
    input_csv = "海岸城0612_wind_profile.csv"
    interp_csv = "海岸城0612_wind_profile_interp.csv"
    
    # 检查文件是否存在
    for f in [input_csv, interp_csv]:
        if not os.path.exists(f):
            print(f"错误: 文件 '{f}' 不存在")
            sys.exit(1)
    
    print("读取数据文件...")
    
    # 读取原始数据
    df_original = pd.read_csv(input_csv)
    df_interp = pd.read_csv(interp_csv)
    
    # 提取高度数据
    height_col = df_original.columns[0]
    heights_original = df_original[height_col].values.astype(float)
    heights_interp = df_interp[height_col].values.astype(float)
    
    # 选择第一个时间点进行可视化（也可以选择其他时间点）
    # 注意：列名包含b'前缀，需要正确处理
    time_col = df_original.columns[1]  # 第一个时间点
    
    print(f"可视化时间点: {time_col}")
    
    # 提取风速数据
    ws_original = df_original[time_col].values.astype(float)
    ws_interp = df_interp[time_col].values.astype(float)
    
    # 对原始数据做高度<=3000的条件筛选
    mask = heights_original <= 3000
    heights_original_filtered = heights_original[mask]
    ws_original_filtered = ws_original[mask]
    
    print(f"原始数据: {len(heights_original_filtered)} 个点 (高度≤3000m)")
    print(f"插值数据: {len(heights_interp)} 个点 (0-3000m)")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(3, 6))
    
    # 绘制原始数据（筛选后）
    ax.plot(ws_original_filtered, heights_original_filtered, 
            'o-', color='blue', linewidth=1.5, markersize=4, 
            label='Original (≤3000m)', alpha=0.7)
    
    # 绘制插值数据
    ax.plot(ws_interp, heights_interp, 
            'd-', color='red', linewidth=2, 
            label='Interpolated (0-3000m)', alpha=0.2)
    
    # 设置图形属性
    ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax.set_ylabel('Height AGL (m)', fontsize=10)
    
    # 提取时间字符串（去掉b'前缀和'后缀）
    time_str = time_col
    if time_str.startswith("b'") and time_str.endswith("'"):
        time_str = time_str[2:-1]
    
    ax.set_title(f'Vertical Wind Profile\n{time_str}', fontsize=11, pad=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # 设置y轴范围
    ax.set_ylim(0, 3000)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_png = "vertical_profile_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"图形已保存: {output_png}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    # 打印一些统计信息
    print("\n统计信息:")
    print(f"原始数据 (≤3000m):")
    print(f"  高度范围: {heights_original_filtered[0]:.1f}m - {heights_original_filtered[-1]:.1f}m")
    print(f"  风速范围: {np.nanmin(ws_original_filtered):.2f} - {np.nanmax(ws_original_filtered):.2f} m/s")
    print(f"  数据点数: {len(heights_original_filtered)}")
    
    print(f"\n插值数据 (0-3000m):")
    print(f"  高度范围: {heights_interp[0]:.1f}m - {heights_interp[-1]:.1f}m")
    print(f"  风速范围: {np.nanmin(ws_interp):.2f} - {np.nanmax(ws_interp):.2f} m/s")
    print(f"  数据点数: {len(heights_interp)}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("垂直插值可视化脚本")
    print("=" * 60)
    
    try:
        visualize_interpolation_comparison()
        print("\n可视化完成!")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()