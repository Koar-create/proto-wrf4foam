#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"wrong usage! correct usage: python plot_residuals.py /path/to/logs")
    sys.exit()

Dir = sys.argv[1]

# --- 1. 配置区域 ---
# 你的实验路径
# Dir = f"{os.environ['HOME']}/WRF-OpenFOAM-Coupling/myExp11/logs"
OutputName = "monitor_residuals.png"

# 只保留动力学相关的残差
residual_files = ['p_0', 'Ux_0', 'Uy_0', 'Uz_0']
courant_file = 'CourantMax_0'

# --- 2. 全局绘图风格设置 (解决字体太小问题) ---
plt.rcParams.update({
    'font.size': 14,          # 全局基础字号
    'axes.titlesize': 18,     # 标题字号
    'axes.labelsize': 16,     # 轴标签字号
    'xtick.labelsize': 14,    # X轴刻度字号
    'ytick.labelsize': 14,    # Y轴刻度字号
    'legend.fontsize': 14,    # 图例字号
    'lines.linewidth': 2      # 线宽
})

# --- 3. 准备画布 ---
# figsize=(宽, 高)，增加高度以防挤压
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# === 绘制残差 (上图) ===
print(f"Processing Residuals in: {Dir}")
has_res_data = False

for fname in residual_files:
    fpath = os.path.join(Dir, fname)
    if os.path.exists(fpath):
        try:
            # skiprows=1: 跳过第一行，防止因首行格式不同导致的报错
            # usecols=(0,1): 强制只读前两列 (Time, Value)
            data = np.loadtxt(fpath, skiprows=1) 
            
            # 确保数据不仅仅是空的
            if data.ndim > 1 and data.shape[0] > 0:
                ax1.plot(data[:, 0], data[:, 1], label=fname)
                has_res_data = True
        except Exception as e:
            print(f"Warning: Could not read {fname}: {e}")

if has_res_data:
    ax1.set_yscale('log')
    ax1.set_ylabel('Initial Residual')
    ax1.set_title('Residuals (p, U)')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend(loc='upper right', framealpha=0.9)
else:
    ax1.text(0.5, 0.5, "No valid residual data found", ha='center', transform=ax1.transAxes)

# === 绘制 Courant Number (下图) ===
cpath = os.path.join(Dir, courant_file)
print(f"Processing Courant: {cpath}")

if os.path.exists(cpath):
    try:
        # 关键修复: skiprows=1 跳过没有时间戳的第一行
        c_data = np.loadtxt(cpath, skiprows=1)
        
        if c_data.ndim > 1 and c_data.shape[0] > 0:
            ax2.plot(c_data[:, 0], c_data[:, 1], 'r-', label='Co Max')
            
            # 绘制警戒线
            ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Limit = 1.0')
            
            ax2.set_ylabel('Courant Number (Max)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Stability Monitor')
            ax2.grid(True, which="both", ls="-", alpha=0.4)
            ax2.legend(loc='upper left')
            
            # 检测是否已经炸网
            max_co = np.max(c_data[:, 1])
            if max_co > 100:
                # 如果数值太大，自动切换为对数坐标，否则看不清前面的变化
                print(f"警告: Courant 数极大 ({max_co:.1e})，切换为对数坐标显示。")
                ax2.set_yscale('log')
                ax2.text(0.5, 0.9, f"CRASHED: Max Co = {max_co:.1e}", 
                         transform=ax2.transAxes, ha='center', color='red', weight='bold')
    except Exception as e:
        print(f"Error reading Courant file: {e}")
else:
    ax2.text(0.5, 0.5, "CourantMax_0 not found", ha='center', transform=ax2.transAxes)

# --- 4. 调整布局并保存 ---
# rect 参数用于给标题留出空间，同时避免 xlabel 被切掉
plt.tight_layout()
# 如果 tight_layout 依然切掉底部，可以使用 subplots_adjust 手动增加底部边距
# plt.subplots_adjust(bottom=0.15) 

save_path = os.path.join(Dir, OutputName)
plt.savefig(save_path, dpi=150)
print(f"Done. Plot saved to: {save_path}")
plt.close()
