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

OutputName = f"monitor_residuals_{input_date}.png"

residual_files = ['p_rgh_0', 'T_0', 'Ux_0', 'Uy_0', 'Uz_0', 'k_0', 'epsilon_0']

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})

# 分组：上图单独画 p_rgh，下图画其余变量
top_group = ['p_rgh_0']
bottom_group = ['T_0', 'Ux_0', 'Uy_0', 'Uz_0', 'k_0', 'epsilon_0']

print(f"Processing Residuals in: {Dir}")

# 两个子图，共享 x 轴
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(13, 9),
    sharex=True,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08}
)

has_top_data = False
has_bottom_data = False

def plot_group(ax, file_list):
    has_data = False
    for fname in file_list:
        fpath = os.path.join(Dir, fname)
        if os.path.exists(fpath):
            try:
                data = np.loadtxt(fpath, skiprows=1)
                if data.ndim > 1 and data.shape[0] > 0:
                    ax.plot(data[:, 0], data[:, 1], label=fname)
                    has_data = True
            except Exception as e:
                print(f"Warning: Could not read {fname}: {e}")
    return has_data

has_top_data = plot_group(ax1, top_group)
has_bottom_data = plot_group(ax2, bottom_group)

# 上图
if has_top_data:
    ax1.set_yscale('log')
    ax1.set_ylabel('Initial Residual')
    ax1.set_title('Residuals (p, U)')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend(loc='best', framealpha=0.9)
else:
    ax1.text(0.5, 0.5, "No valid p_rgh data found", ha='center', va='center', transform=ax1.transAxes)

# 下图
if has_bottom_data:
    ax2.set_yscale('log')
    ax2.set_ylabel('Initial Residual')
    ax2.set_xlabel('Iteration / Time Step')
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend(loc='best', ncol=2, framealpha=0.9)
else:
    ax2.text(0.5, 0.5, "No valid residual data found", ha='center', va='center', transform=ax2.transAxes)

save_path = os.path.join(Dir, OutputName)
plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Done. Plot saved to: {save_path}")
plt.close()