import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

OUTPUT_DIR = os.path.join("steady_experiments_finer_ABL", "patch_verification")

def parse_openfoam_file(filepath, is_vector=False):
    """
    解析 OpenFOAM 边界场数据文件
    自动跳过 Header，提取 '(' 和 ')' 之间的纯数据
    """
    if not os.path.exists(filepath):
        print(f"Error: 找不到文件 {filepath}")
        sys.exit(1)
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data = []
    in_data = False
    for line in lines:
        line = line.strip()
        if not in_data:
            if line == "(":  # 找到数据块开始标记
                in_data = True
        else:
            if line == ")":  # 数据块结束
                break
            if is_vector:
                # 向量文件，去除括号并按空格分割: (-0.0002 0.0005 0.0097)
                vals = line.replace('(', '').replace(')', '').split()
                if len(vals) == 3:
                    data.append([float(v) for v in vals])
            else:
                # 标量文件
                data.append(float(line))
    return np.array(data)

def main():
    if len(sys.argv) < 3:
        print("用法: python plot_boundary.py /path/to/boundaryData [west|east|south|north]")
        sys.exit(1)
        
    target_dir = sys.argv[1]
    unique_flag = target_dir.split(os.sep)[-3]
    patch = sys.argv[2]
    
    # 构建文件路径
    points_file = os.path.join(target_dir, patch, "points")
    k_file = os.path.join(target_dir, patch, "0", "k")
    eps_file = os.path.join(target_dir, patch, "0", "epsilon")
    U_file = os.path.join(target_dir, patch, "0", "U")
    
    print(f"正在读取 {patch} 边界数据...")
    points = parse_openfoam_file(points_file, is_vector=True)
    k = parse_openfoam_file(k_file, is_vector=False)
    epsilon = parse_openfoam_file(eps_file, is_vector=False)
    U = parse_openfoam_file(U_file, is_vector=True)
    
    # ---------------- 智能坐标轴识别逻辑 ----------------
    # 计算点云在 X, Y, Z 三个维度的方差
    variance = np.var(points, axis=0)
    # 选取方差最大的两个维度作为 2D 绘图的平面轴
    plot_axes = np.argsort(variance)[1:] 
    
    # 习惯上，高度 Z 轴 (索引 2) 只要存在于切面中，我们通常将它用作图像的 Y 轴 (垂直方向)
    if 2 in plot_axes:
        horizontal_axis = plot_axes[0] if plot_axes[1] == 2 else plot_axes[1]
        x_coord = points[:, horizontal_axis]
        y_coord = points[:, 2]
        xlabel = ['X', 'Y', 'Z'][horizontal_axis]
        ylabel = 'Z'
    else:
        # 如果切面是水平的 (XY 平面，比如 bottom 边界)
        x_coord = points[:, plot_axes[0]]
        y_coord = points[:, plot_axes[1]]
        xlabel = ['X', 'Y', 'Z'][plot_axes[0]]
        ylabel = ['X', 'Y', 'Z'][plot_axes[1]]

    # ---------------- 数据计算 ----------------
    # U_horizon = sqrt(Ux^2 + Uy^2)
    U_horizon = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    # w 即 U 的 z 分量
    w = U[:, 2]

    # 生成非结构网格的三角剖分用于平滑 Shading 渲染
    triang = mtri.Triangulation(x_coord, y_coord)

    # ---------------- 开始绘图 ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Boundary Shading Visualization: {patch.capitalize()}', fontsize=16)

    # 绘图通用配置函数
    def plot_shading(ax, data, title, cmap):
        # 使用 tripcolor 和 gouraud shading 实现平滑插值的高质量标量着色
        c = ax.tripcolor(triang, data, shading='gouraud', cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(f'{xlabel} Coordinate')
        ax.set_ylabel(f'{ylabel} Coordinate')
        # 强制长宽比一致，避免网格形状被拉伸变形
        ax.set_aspect('equal', adjustable='box') 
        fig.colorbar(c, ax=ax)

    print("正在渲染图像...")
    # 1. 湍动能 k
    plot_shading(axs[0, 0], k, 'k (Turbulent Kinetic Energy)', 'viridis')

    # 2. 湍流耗散率 epsilon
    plot_shading(axs[0, 1], epsilon, 'epsilon (Turbulent Dissipation)', 'plasma')

    # 3. 水平速度大小 U_horizon
    plot_shading(axs[1, 0], U_horizon, 'U_horizon (Horizontal Velocity Magnitude)', 'inferno')

    # 4. 垂直速度分量 w
    plot_shading(axs[1, 1], w, 'w (Z-component of U)', 'coolwarm')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 导出图片并显示
    out_file = os.path.join(OUTPUT_DIR, f"shading_{unique_flag}_{patch}.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"可视化完成，已保存至: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()