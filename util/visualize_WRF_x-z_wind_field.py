#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化 WRF 输出文件的 X-Z 垂直剖面风场
基于 y=800m 的纬度线（lat=23.1211944444）提取垂直剖面

用法: python visualize_WRF_x-z_wind_field.py <nc_file_path> [--lat LAT] [--lon LON] [--output OUTPUT]

示例:
    python visualize_WRF_x-z_wind_field.py W_myExp03/auxhist2/tmp/auxhist2_d03_2025-09-03_04:00:00_tmp.nc
    python visualize_WRF_x-z_wind_field.py W_myExp03/auxhist2/tmp/auxhist2_d03_2025-09-03_04:00:00_tmp.nc --lat 23.1211944444 --lon 113.321102778
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wrf import getvar, destagger, interplevel

def extract_wrf_xz_section(nc_file_path, target_lat=23.1211944444, target_lon=113.321102778, 
                          lat_tol=0.004, lon_tol=0.0225000225, max_height=3000):
    """
    从 WRF NetCDF 文件中提取 X-Z 垂直剖面
    
    参数:
        nc_file_path: NetCDF 文件路径
        target_lat: 目标纬度 (y=800m 对应的纬度)
        target_lon: 目标经度 (y=800m 对应的经度)
        lat_tol: 纬度容差
        lon_tol: 经度容差
        max_height: 最大高度限制 (米)
    
    返回:
        xarray Dataset 包含提取的剖面数据
    """
    print(f"正在读取 WRF 文件: {nc_file_path}")
    
    if not os.path.exists(nc_file_path):
        raise FileNotFoundError(f"错误: 文件不存在 - {nc_file_path}")
    
    # 打开数据集
    ds = xr.open_dataset(nc_file_path)
    print(f"数据集维度: {dict(ds.dims)}")
    print(f"可用变量: {list(ds.data_vars.keys())}")
    
    # === 1. 提取 y=800m 的纬度线条件 ===
    print(f"提取纬度线: lat={target_lat}±{lat_tol}, lon={target_lon}±{lon_tol}")
    condi_latlon = ((target_lat - lat_tol <= ds.XLAT) & 
                    (ds.XLAT <= target_lat + lat_tol) &
                    (target_lon - lon_tol <= ds.XLONG) & 
                    (ds.XLONG <= target_lon + lon_tol))
    
    # === 2. 计算高度场 ===
    print("计算高度场...")
    PH = ds['PH']
    PHB = ds['PHB']
    H_stag = (PH + PHB) / 9.81  # 位势高度转换为几何高度
    H = destagger(H_stag, stagger_dim=0)  # 解除交错网格
    
    # 将高度添加到数据集
    ds['H'] = xr.DataArray(H, coords=ds.T.coords, dims=ds.T.dims)
    
    # 高度限制条件
    condi_height = ds.H <= max_height
    condi_total = condi_latlon & condi_height
    
    # === 3. 计算风速场 ===
    print("计算风速场...")
    U = destagger(ds.U, stagger_dim=2)  # 东向风分量
    V = destagger(ds.V, stagger_dim=1)  # 北向风分量
    
    ds['U_destaggered'] = xr.DataArray(U, dims=ds.T.dims, coords=ds.T.coords)
    ds['V_destaggered'] = xr.DataArray(V, dims=ds.T.dims, coords=ds.T.coords)
    ds['WS'] = np.sqrt(ds.U_destaggered**2 + ds.V_destaggered**2)  # 风速大小
    
    # === 4. 提取剖面数据 ===
    print("提取剖面数据...")
    wind_speed_2d = ds['WS'].where(condi_total, drop=True).squeeze()
    height_2d = ds['H'].where(condi_total, drop=True).squeeze()
    
    # 创建经度坐标的2D网格
    # 使用第一层的经度数据扩展到所有高度层
    lon_2d = xr.concat(
        [ds['XLONG'].where(condi_total.sel(bottom_top=0), drop=True).squeeze() 
         for _ in range(wind_speed_2d.shape[0])], 
        dim='bottom_top'
    )
    
    print(f"提取的剖面尺寸: {wind_speed_2d.shape}")
    print(f"风速范围: [{wind_speed_2d.min().values:.2f}, {wind_speed_2d.max().values:.2f}] m/s")
    print(f"高度范围: [{height_2d.min().values:.0f}, {height_2d.max().values:.0f}] m")
    print(f"经度范围: [{lon_2d.min().values:.5f}, {lon_2d.max().values:.5f}] °E")
    
    # 创建结果数据集
    result_ds = xr.Dataset({
        'wind_speed': wind_speed_2d,
        'height': height_2d,
        'longitude': lon_2d,
        'U': ds['U_destaggered'].where(condi_total, drop=True).squeeze(),
        'V': ds['V_destaggered'].where(condi_total, drop=True).squeeze()
    })
    
    return result_ds, target_lat, target_lon

def visualize_wrf_xz_section(result_ds, target_lat, target_lon, output_path=None, 
                           cfd_domain_top=2000, nc_file_path=None):
    """
    可视化 WRF X-Z 垂直剖面
    
    参数:
        result_ds: 包含剖面数据的 xarray Dataset
        target_lat: 目标纬度
        target_lon: 目标经度
        output_path: 输出图像路径 (如果为None则自动生成)
        cfd_domain_top: CFD域顶部高度 (米)
        nc_file_path: 原始NetCDF文件路径 (用于生成默认输出文件名)
    """
    # === 1. 基础设置与字体语言 ===
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # 提取数据
    lon_2d = result_ds['longitude']
    height_2d = result_ds['height']
    wind_speed_2d = result_ds['wind_speed']
    
    # === 2. 创建图形 ===
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # === 3. 绘制风速场 ===
    # 使用 perceptually uniform 的色彩映射
    im = ax.pcolormesh(lon_2d, height_2d, wind_speed_2d, 
                      vmin=0, vmax=wind_speed_2d.max().values * 1.1,
                      cmap='viridis', shading='auto', alpha=0.9)
    
    cbar = plt.colorbar(im, ax=ax, label='Wind Speed (m/s)', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    # === 4. 标记目标点 ===
    ax.scatter(target_lon, target_lat, marker='x', color='red', s=100, 
               linewidth=2, label=f'Target point (lat={target_lat:.6f})')
    
    # === 5. 添加 CFD 域顶部参考线 ===
    if cfd_domain_top <= height_2d.max().values:
        ax.axhline(cfd_domain_top, color='black', alpha=0.8, linestyle=':', linewidth=2)
        ax.text(lon_2d.mean().values, cfd_domain_top + 50, 
                f'CFD domain top ({cfd_domain_top} m)', 
                color='black', alpha=0.8, fontsize=12,
                verticalalignment='bottom', horizontalalignment='center')
    
    # === 6. 坐标轴和标签 ===
    ax.set_xlabel('Longitude (°E)', fontweight='bold')
    ax.set_ylabel('Height (m)', fontweight='bold')
    
    # 设置合理的刻度
    lon_min, lon_max = lon_2d.min().values, lon_2d.max().values
    lon_ticks = np.linspace(lon_min, lon_max, 5)
    ax.set_xticks([round(tick, 5) for tick in lon_ticks])
    
    height_min, height_max = height_2d.min().values, height_2d.max().values
    height_ticks = np.linspace(height_min, height_max, 6)
    ax.set_yticks([round(tick, -2) for tick in height_ticks])  # 四舍五入到百位
    
    # === 7. 网格和图例 ===
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # === 8. 标题 ===
    title = f'WRF X-Z Vertical Wind Field at lat={target_lat:.6f}°N'
    ax.set_title(title, fontweight='bold', pad=15)
    
    # === 9. 保存图像 ===
    if output_path is None:
        if nc_file_path:
            # 从输入文件路径生成输出文件名
            base_name = os.path.basename(nc_file_path).replace('.nc', '')
            output_path = f"wrf_xz_section_lat{target_lat:.6f}_{base_name}.png"
        else:
            # 生成默认文件名
            output_path = f"wrf_xz_section_lat{target_lat:.6f}.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_path} (DPI=300)")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='可视化 WRF 输出文件的 X-Z 垂直剖面风场',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s W_myExp03/auxhist2/tmp/auxhist2_d03_2025-09-03_04:00:00_tmp.nc
  %(prog)s W_myExp03/auxhist2/tmp/auxhist2_d03_2025-09-03_04:00:00_tmp.nc --lat 23.1211944444 --lon 113.321102778 --output my_plot.png
        """
    )
    
    parser.add_argument('nc_file', help='WRF NetCDF 文件路径')
    parser.add_argument('--lat', type=float, default=23.1211944444,
                       help='目标纬度 (默认: 23.1211944444)')
    parser.add_argument('--lon', type=float, default=113.321102778,
                       help='目标经度 (默认: 113.321102778)')
    parser.add_argument('--lat_tol', type=float, default=0.004,
                       help='纬度容差 (默认: 0.004)')
    parser.add_argument('--lon_tol', type=float, default=0.0225000225,
                       help='经度容差 (默认: 0.0225000225)')
    parser.add_argument('--max_height', type=float, default=3000,
                       help='最大高度限制 (米) (默认: 3000)')
    parser.add_argument('--cfd_top', type=float, default=2000,
                       help='CFD域顶部高度 (米) (默认: 2000)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    try:
        # 提取剖面数据
        result_ds, target_lat, target_lon = extract_wrf_xz_section(
            args.nc_file, 
            target_lat=args.lat,
            target_lon=args.lon,
            lat_tol=args.lat_tol,
            lon_tol=args.lon_tol,
            max_height=args.max_height
        )
        
        # 可视化
        output_path = visualize_wrf_xz_section(
            result_ds, 
            target_lat, 
            target_lon,
            output_path=args.output,
            cfd_domain_top=args.cfd_top,
            nc_file_path=args.nc_file
        )
        
        print(f"处理完成! 输出文件: {output_path}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()