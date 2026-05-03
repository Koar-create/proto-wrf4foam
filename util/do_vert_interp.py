#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import xarray as xr
from pyproj import Proj
from scipy.interpolate import griddata
from wrf import getvar, destagger, interplevel

HOME = os.environ.get('HOME')

def generate_stretched_z(z_max, dz_bottom, stretch_factor, dz_max_limit):
    """
    生成下密上疏的垂直网格。
    z_max: 域的最大几何高度 (m)
    dz_bottom: 最底层的网格厚度 (m)
    stretch_factor: 拉伸系数 (e.g., 1.1)
    dz_max_limit: 允许的最大层间距 (m)
    """
    # 以第一层网格中心为起点
    z = [0]  # [dz_bottom / 2.0]
    dz = dz_bottom
    while z[-1] < z_max:
        dz = min(dz * stretch_factor, dz_max_limit)
        z.append(z[-1] + dz)
    return np.array(z)

project_path = f"{HOME}/WRF-OpenFOAM-Coupling/W_myExp03"
lon0, lat0 = 113.32, 23.115


if __name__ == '__main__':
    # --- Part 1: Input Arguments Parsing ---
    if len(sys.argv) != 2:
        print(f"Error: Missing input date argument.")
        print(f"Usage: python {sys.argv[0]} <mm-dd_HH%3AMM>")
        print(f"Example: python {sys.argv[0]} \"09-01_00:00\"")
        sys.exit(1)
    
    input_date = sys.argv[1]

    fname = f"auxhist2_d03_2025-{input_date}:00_tmp.nc"  # f"wrfout_d03_2025-{input_date}:00"
    ds = xr.open_dataset(f"{project_path}/auxhist2/tmp/{fname}", engine='netcdf4').squeeze()  # f"{project_path}/WRF/run/{fname}"
    ds_wrfinputd03 = xr.open_dataset(f"{project_path}/WRF/run/wrfinput_d03", engine='netcdf4').squeeze()
    ds = ds.assign_coords(XLAT=ds_wrfinputd03['XLAT'], XLONG=ds_wrfinputd03['XLONG'])

    # --- Part 2: Destaggering and Basic Math ---
    PH = ds['PH']
    PHB = ds['PHB']
    H_stag = (PH + PHB) / 9.81
    H = destagger(H_stag, stagger_dim=0)
    W = destagger(ds['W'], stagger_dim=0)
    U = destagger(ds.U, stagger_dim=2)
    V = destagger(ds.V, stagger_dim=1)
    ds['H'] = xr.DataArray(H, coords=ds.T.coords, dims=ds.T.dims)
    ds['W'] = xr.DataArray(W, coords=ds.T.coords, dims=ds.T.dims)
    ds['U'] = xr.DataArray(U, coords=ds.T.coords, dims=ds.T.dims)
    ds['V'] = xr.DataArray(V, coords=ds.T.coords, dims=ds.T.dims)
    ds['WS'] = np.sqrt(ds['U'] ** 2 + ds['V'] ** 2)  # wind speed

    # 如果你想把它们加回原来的 dataset
    ds_ = ds.copy()
    for coord in ['Times', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'XTIME']:
        if coord in ds_.coords:
            ds_ = ds_.drop_vars(coord, errors='ignore')
    ds_ = ds_[['U', 'V', 'W', 'WS', 'H']]

    # --- Part 4: Vertical Interpolation & Surface NaN Solver ---
    print("Running Vertical Interpolation...")
    
    # 1. 定义拉伸目标高度 Z
    z_tgt = generate_stretched_z(z_max=3000, dz_bottom=10, stretch_factor=1, dz_max_limit=50)
    
    # 获取水平插值后的 WRF 物理高度场 (bottom_top, south_north, west_east)
    H_horiz = ds_['H'].values
    # 提取 WRF 最底层的物理高度，用于 Log law 计算参考基准 Z_ref
    H_min_horiz = H_horiz[0, :, :] 
    
    final_vars = {}
    var_list = ['U', 'V', 'W', 'WS']
    z0 = 1.478  # 地表粗糙度，可根据实际地形更改
    
    for var in var_list:
        data_horiz = ds_[var].values
        var_vert = np.zeros((len(z_tgt), len(ds_.south_north), len(ds_.west_east)))
        
        # 逐层调用 wrf.interplevel
        for idx, z_val in enumerate(z_tgt):
            var_vert[idx, :, :] = interplevel(data_horiz, H_horiz, z_val).values
            
        # === 处理近地面 NaN (Surface NaN Solver) ===
        # 将结果转回 DataArray 方便操作
        da_vert = xr.DataArray(var_vert, dims=['z', 'south_north', 'west_east'], coords={'z': z_tgt, 'XLAT': ds_.XLAT, 'XLONG': ds_.XLONG})
        # 向后填充 (backfill)：用 WRF 最底层的有效值填满下方因为坐标不够深产生的 NaN
        da_bfilled = da_vert.bfill(dim='z')
        
        if var in ['U', 'V', 'WS']:
            # 对水平速度场应用 Log Law 修正
            Z_3d = np.broadcast_to(z_tgt[:, None, None], var_vert.shape)
            H_min_3d = np.broadcast_to(H_min_horiz[None, :, :], var_vert.shape)
            
            # 确保对数计算安全
            Z_3d_safe = np.maximum(Z_3d, z0 + 1e-3)
            H_min_safe = np.maximum(H_min_3d, z0 + 1e-3)
            
            # log_mult = ln(z/z0) / ln(z_ref/z0)
            log_mult = np.log(Z_3d_safe / z0) / np.log(H_min_safe / z0)
            
            # 找到原来是 NaN，且其目标高度低于 WRF 最底层高度的地方
            nan_mask = np.isnan(var_vert)
            apply_mask = nan_mask & (Z_3d < H_min_3d)
            
            # 替换：直接使用 backfill 获取的最底层有效值，乘以对数衰减系数
            var_vert[apply_mask] = da_bfilled.values[apply_mask] * log_mult[apply_mask]
            
            # 如果最顶层还有 NaN（超出 WRF 顶层），直接常数向上延伸
            var_vert = xr.DataArray(var_vert, dims=['z','south_north','west_east']).ffill(dim='z').values
        else:
            # 对于标量场 (W, TKE)，不适用 Log Law，直接沿用最底层数值 (bfill)
            var_vert = da_bfilled.ffill(dim='z').values
            
        final_vars[var] = (['z', 'south_north', 'west_east'], var_vert)

    # --- Part 5: Save Dataset ---
    ds_final = xr.Dataset(
        data_vars=final_vars, 
        coords={
            'z': z_tgt, 
            'south_north': ds_.south_north, 
            'west_east'  : ds_.west_east, 
            'XLAT': ds_.XLAT, 
            'XLONG': ds_.XLONG, 
        }
    )
    
    save_path = f"{project_path}/auxhist2/horiz_raw_z_interp/{fname.replace('tmp', '1h-rolling')}"
    ds_final.to_netcdf(save_path)
    
    print(f"Successfully saved to {save_path} !")
    print('Z levels created:', len(z_tgt), f"(Min: {z_tgt[0]:.1f}m, Max: {z_tgt[-1]:.1f}m)")
    print('U range:', round(ds_final['U'].min().item(), 2), 'to', round(ds_final['U'].max().item(), 2))
    print('V range:', round(ds_final['V'].min().item(), 2), 'to', round(ds_final['V'].max().item(), 2))
    print('WS range:', round(ds_final['WS'].min().item(), 2), 'to', round(ds_final['WS'].max().item(), 2))