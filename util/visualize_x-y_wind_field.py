#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化 steady_experiments_finer_ABL 实验目录下 postProcessing/100m.csv 的2-D水平风场
用法: python visualize_wind_field.py steady_experiments_finer_ABL/20250903_0000
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata

def visualize_wind_field(csv_file_path, output_title):
    print(f"正在读取文件: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件不存在 - {csv_file_path}")
        return False
    
    # === 1. 基础设置与字体语言 ===
    # 禁用中文，全局使用 serif 字体，字号调大
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    try:
        # 分块读取 CSV
        chunksize = 100000
        chunk_iter = pd.read_csv(csv_file_path, chunksize=chunksize)
        
        all_x, all_y, all_u0, all_u1, all_wind_speed = [], [], [], [], []
        
        for i, chunk in enumerate(chunk_iter):
            required_cols = ['Coords:0', 'Coords:1', 'U:0', 'U:1']
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            
            if missing_cols:
                print(f"警告: 第 {i+1} 块中未找到列: {', '.join(missing_cols)}")
                continue
            
            x_coords = chunk['Coords:0'].astype(float)
            y_coords = chunk['Coords:1'].astype(float)
            u0_values = chunk['U:0'].astype(float)
            u1_values = chunk['U:1'].astype(float)
            wind_speed = np.sqrt(u0_values**2 + u1_values**2)
            
            all_x.extend(x_coords.values)
            all_y.extend(y_coords.values)
            all_u0.extend(u0_values.values)
            all_u1.extend(u1_values.values)
            all_wind_speed.extend(wind_speed.values)
            
            print(f"  已处理块 {i+1}, 累计点数: {len(all_x)}")
        
        if len(all_x) == 0:
            print("错误: 未找到足够的数据进行可视化")
            return False
        
        x_array = np.array(all_x)
        y_array = np.array(all_y)
        u0_array = np.array(all_u0)
        u1_array = np.array(all_u1)
        wind_speed_array = np.array(all_wind_speed)
        
        print(f"数据准备完成，总共 {len(x_array)} 个点")
        
        # === 修复 Quiver 采样 Bug ===
        # 使用 griddata 生成完美的均匀二维网格以绘制箭头，告别一维切片带来的竖条纹
        print("正在插值生成均匀矢量网格...")
        # 设定网格密度，40x40 能够在视觉上比较清爽
        grid_resolution = 40 
        grid_x, grid_y = np.mgrid[x_array.min():x_array.max():complex(0, grid_resolution), 
                                  y_array.min():y_array.max():complex(0, grid_resolution)]
        
        # 对超大数据集轻微下采样后插值，提升计算速度 (控制在约10万点级别去插值)
        sub = max(1, len(x_array) // 100000)
        grid_u0 = griddata((x_array[::sub], y_array[::sub]), u0_array[::sub], (grid_x, grid_y), method='linear')
        grid_u1 = griddata((x_array[::sub], y_array[::sub]), u1_array[::sub], (grid_x, grid_y), method='linear')
        
        # 展平供 quiver 使用
        x_quiver = grid_x.flatten()
        y_quiver = grid_y.flatten()
        u0_quiver = grid_u0.flatten()
        u1_quiver = grid_u1.flatten()
        
        # 绘图初始化
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # === 3. 色彩管理 ===
        # 标量场使用 viridis (若有压力场请另外配置 RdBu_r，严禁 jet)
        hexbin = ax.hexbin(x_array, y_array, C=wind_speed_array,
                           gridsize=120, cmap='viridis', reduce_C_function=np.mean,
                           alpha=0.85, edgecolors='none')
        
        cbar = plt.colorbar(hexbin, ax=ax, label='Wind Speed (m/s)', pad=0.02)
        
        # 计算 Quiver 参数
        mean_wind_speed = np.mean(wind_speed_array)
        # scale_factor = 40.0 / mean_wind_speed if mean_wind_speed > 0 else 40.0
        scale_factor = 80
        arrow_color = 'white' if mean_wind_speed < 0.5 else 'black'
            
        quiver_plot = ax.quiver(x_quiver, y_quiver, u0_quiver, u1_quiver,
                                color=arrow_color, alpha=0.85, 
                                scale=scale_factor,   # 移除 scale_units='inches'，释放长度限制
                                width=0.002,          # [箭杆粗细]：设得很细，保证画面不拥挤
                                headwidth=3.5,        # [箭头宽度]：默认值是3，保持在3~4之间，箭头就不会臃肿
                                headlength=5,         # [箭头长度]：控制箭头尖锐程度，5~6比较锐利
                                headaxislength=4,     # [箭头尾部凹陷]：比 headlength 略小一点，尾部呈锋利的倒V型
                                minshaft=2)           # 当风速极小（箭杆短于箭头2倍）时，不画箭杆，只画箭头，避免变成难看的方块
        # quiver_plot = ax.quiver(x_quiver, y_quiver, u0_quiver, u1_quiver,
        #                         color=arrow_color, alpha=0.85, scale=scale_factor,
        #                         # width=3.3e-3)
        #                         scale_units='inches', width=0.005, headwidth=6,
        #                         headlength=7, headaxislength=5, minshaft=2)
        
        ax.quiverkey(quiver_plot, X=0.95, Y=0.98, U=1.0, label='1 m/s', 
                     labelpos='E', coordinates='axes', color=arrow_color,
                     fontproperties={'family': 'serif', 'size': 12, 'weight': 'bold'})
        
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title(f'2-D Horizontal Wind Field: {output_title}', fontweight='bold', pad=15)
        
        # === 4. 物理比例 ===
        # 确保城市几何结构在 X/Y 轴上等比显示，不发生拉伸畸变
        ax.set_aspect('equal')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 保存图像
        output_dir = os.path.dirname(csv_file_path)
        output_filename = f"wind_field_{output_title.replace('/', '_')}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # === 5. 布局自动优化 & DPI 设置 ===
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已成功保存至: {output_path} (DPI=300)")
        
        try:
            plt.show()
        except:
            pass
        
        plt.close()
        return True
            
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_wind_field.py <experiment_directory>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    csv_file_path = os.path.join(experiment_dir, "postProcessing", "100m.csv")
    
    if experiment_dir.startswith('steady_experiments_finer_ABL/'):
        output_title = experiment_dir[len('steady_experiments_finer_ABL/'):]
    else:
        output_title = experiment_dir
    
    success = visualize_wind_field(csv_file_path, output_title)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()