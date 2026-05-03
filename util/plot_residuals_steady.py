#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"wrong usage! correct usage: python plot_residuals.py /path/to/logs")
    sys.exit()

Dir = sys.argv[1]
input_date = Dir.split('/')[-2]

# --- 1. 配置区域 ---
# 你的实验路径
# Dir = f"{os.environ['HOME']}/WRF-OpenFOAM-Coupling/myExp11/logs"
OutputName = f"monitor_residuals_{input_date}.png"

residual_files = ['p_0', 'Ux_0', 'Uy_0', 'Uz_0', 'k_0', 'epsilon_0']

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
fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

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
    ax1.legend(loc='best', framealpha=0.9)
else:
    ax1.text(0.5, 0.5, "No valid residual data found", ha='center', transform=ax1.transAxes)

# --- 4. 调整布局并保存 ---
# rect 参数用于给标题留出空间，同时避免 xlabel 被切掉
plt.tight_layout()
# 如果 tight_layout 依然切掉底部，可以使用 subplots_adjust 手动增加底部边距
# plt.subplots_adjust(bottom=0.15) 

save_path = os.path.join(Dir, OutputName)
plt.savefig(save_path, dpi=150)
print(f"Done. Plot saved to: {save_path}")
plt.close()
