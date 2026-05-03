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

def calculate_relative_xy(ds, lon0, lat0):
    """
    将 WRF 的经纬度网格转换为相对于参考点 (lon0, lat0) 的 (x, y) 米坐标。
    使用以参考点为中心的方位等距投影 (Azimuthal Equidistant)。
    """
    # 1. 定义投影
    # proj='aeqd': 方位等距投影
    # lat_0, lon_0: 投影中心（即你的参考点，这里会变成 (0,0)）
    # datum='WGS84': 地球椭球模型
    p = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84', units='m')
    
    # 2. 执行转换
    # p(lon, lat) 返回 x, y
    # 注意：pyproj 处理 numpy 数组非常快
    x_vec, y_vec = p(ds['XLONG'], ds['XLAT'])
    
    # 3. 将结果封装回 xarray (可选，方便后续绘图或计算)
    # 保持与原始经纬度相同的维度结构 (south_north, west_east)
    dims = ds['XLONG'].dims  # 通常是 ('south_north', 'west_east')
    
    da_x = xr.DataArray(x_vec, coords=ds['XLONG'].coords, dims=dims, name='x_rel')
    da_y = xr.DataArray(y_vec, coords=ds['XLONG'].coords, dims=dims, name='y_rel')

    return da_x, da_y

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

    TKE_PBL = destagger(ds['TKE_PBL'], stagger_dim=0)
    # make sure the k value wont be zero or negative, assign a very small number, in order to avoid division by zero error.
    TKE_PBL = np.maximum(TKE_PBL, 1e-6)
    ds['TKE_PBL'] = xr.DataArray(TKE_PBL, coords=ds.T.coords, dims=ds.T.dims)
    # --- 新增：提取并计算绝对位温 (Potential Temperature) ---
    # WRF中的 'T' 变量是扰动位温 (Theta - 300K)，需要加300还原
    Theta = ds['T'] + 300.0
    ds['Theta'] = xr.DataArray(Theta, coords=ds.T.coords, dims=ds.T.dims)
    # -----------------------------------------------------
        
    # 调用函数
    # 注意：确保 ds 中包含 XLAT 和 XLONG
    x_meters, y_meters = calculate_relative_xy(ds, lon0, lat0)

    '''
    lon_d03_l, lon_d03_r = ds.XLONG.isel(  west_east=0).max().item(), ds.XLONG.isel(  west_east=-1).min().item()
    lat_d03_b, lat_d03_t =  ds.XLAT.isel(south_north=0).max().item(),  ds.XLAT.isel(south_north=-1).min().item()
    
    print("计算完成！")
    print(f"参考点 (0,0) 附近的转换结果:")
    
    # 验证：找到离参考点最近的网格点，看看它的 x, y 是否接近 0
    dist_sq = x_meters**2 + y_meters**2
    min_idx = dist_sq.values.argmin()
    # 使用 unravel_index 获取多维索引
    y_idx, x_idx = np.unravel_index(min_idx, dist_sq.shape)

    print(f"网格中最近点的索引: ({y_idx}, {x_idx})")
    print(f"该点的 x 坐标: {x_meters[y_idx, x_idx].values:.2f} 米")
    print(f"该点的 y 坐标: {y_meters[y_idx, x_idx].values:.2f} 米")
    '''

    # 如果你想把它们加回原来的 dataset
    ds['x_rel'] = x_meters
    ds['y_rel'] = y_meters
    ds_ = ds.copy()
    ds_ = ds_.assign_coords(x_rel=ds_['x_rel'], y_rel=ds_['y_rel']).drop_vars(
        ['Times', 'XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'XTIME'], errors='ignore'
    )
    ds_ = ds_[['U', 'V', 'W', 'WS', 'H', 'TKE_PBL', 'Theta']]

    # --- Part 3: Horizontal Interpolation ---
    x_tgt = np.arange(-5000, 5000 + 1, 100)
    y_tgt = np.arange(-5000, 5000 + 1, 100)
    grid_x, grid_y = np.meshgrid(x_tgt, y_tgt)

    # 定义核心插值函数 (这是给 apply_ufunc 调用的)
    # 这个函数只处理单层的 2D 数据，xarray 会负责把它应用到每一层
    def regrid_solver(data, src_x, src_y):
        # 移除 NaN 值以避免报错 (可选，视数据质量而定)
        # 注意：griddata 需要 (N, 2) 形式的坐标点
        points = np.column_stack((src_x.flatten(), src_y.flatten()))
        values = data.flatten()
        return griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)

    # 4. 封装成通用函数
    def regrid_dataset(ds_in, grid_x_coord, grid_y_coord):
        """
        ds_in: 输入的 Dataset
        grid_x_coord: 输入数据的 X 坐标变量 (2D 数组)
        grid_y_coord: 输入数据的 Y 坐标变量 (2D 数组)
        """
        
        # 使用 apply_ufunc
        return xr.apply_ufunc(
            regrid_solver,            # 核心函数
            ds_in,                    # 输入数据 (可以是整个 Dataset 或 DataArray)
            grid_x_coord,             # 原始 X 坐标
            grid_y_coord,             # 原始 Y 坐标
            
            # 核心维度：这些维度会在运算中被消耗掉 (变成了新的 grid)
            # 假设你的原始维度名就是 'y_rel' 和 'x_rel'
            input_core_dims=[
                ['south_north', 'west_east'],  # ds_in 中变量的维度
                ['south_north', 'west_east'],  # grid_x_coord 的维度
                ['south_north', 'west_east']   # grid_y_coord 的维度
            ],
            
            # 输出维度：这些是产生的新维度
            output_core_dims=[['y_rel', 'x_rel']],
            
            # 开启向量化，允许处理额外的维度 (bottom_top 等)
            vectorize=True,
            
            # 传递给 numpy/scipy 的参数，告知输出形状不一致
            exclude_dims=set(('y_rel', 'x_rel')),
            
            # 必须指定输出的数据类型，通常是 float
            output_dtypes=[float]
        )

    # 5. 执行操作
    # 注意：你需要把坐标变量作为参数传进去，而不是让 apply_ufunc 在 ds 内部找
    # 这样可以确保坐标也被正确广播
    print("Running Horizontal Interpolation...")
    ds_horiz = regrid_dataset(ds_, ds_['x_rel'], ds_['y_rel'])

    # --- Part 4: Vertical Interpolation & Surface NaN Solver ---
    print("Running Vertical Interpolation...")
    
    # 1. 定义拉伸目标高度 Z
    z_tgt = generate_stretched_z(z_max=3000, dz_bottom=10, stretch_factor=1, dz_max_limit=50)
    
    # 获取水平插值后的 WRF 物理高度场 (bottom_top, y, x)
    H_horiz = ds_horiz['H'].values
    # 提取 WRF 最底层的物理高度，用于 Log law 计算参考基准 Z_ref
    H_min_horiz = H_horiz[0, :, :] 
    
    final_vars = {}
    var_list = ['U', 'V', 'W', 'WS', 'TKE_PBL', 'Theta']
    z0 = 1.478 # 地表粗糙度，可根据实际地形更改
    
    for var in var_list:
        data_horiz = ds_horiz[var].values
        var_vert = np.zeros((len(z_tgt), len(y_tgt), len(x_tgt)))
        
        # 逐层调用 wrf.interplevel
        for idx, z_val in enumerate(z_tgt):
            var_vert[idx, :, :] = interplevel(data_horiz, H_horiz, z_val).values
            
        # === 处理近地面 NaN (Surface NaN Solver) ===
        # 将结果转回 DataArray 方便操作
        da_vert = xr.DataArray(var_vert, dims=['z', 'y', 'x'], coords={'z': z_tgt})
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
            var_vert = xr.DataArray(var_vert, dims=['z','y','x']).ffill(dim='z').values
        else:
            # 对于标量场 (W, TKE)，不适用 Log Law，直接沿用最底层数值 (bfill)
            var_vert = da_bfilled.ffill(dim='z').values
            
        final_vars[var] = (['z', 'y_rel', 'x_rel'], var_vert)

    # --- Part 5: Save Dataset ---
    ds_final = xr.Dataset(
        data_vars=final_vars,
        coords={
            'z': z_tgt,
            'y_rel': y_tgt,
            'x_rel': x_tgt
        }
    )
    
    save_path = f"{project_path}/auxhist2/{fname.replace('tmp', '1h-rolling_cartesian')}"
    ds_final.to_netcdf(save_path)
    
    print(f"Successfully saved to {save_path} !")
    print('Z levels created:', len(z_tgt), f"(Min: {z_tgt[0]:.1f}m, Max: {z_tgt[-1]:.1f}m)")
    print('U range:', round(ds_final['U'].min().item(), 2), 'to', round(ds_final['U'].max().item(), 2))
    print('V range:', round(ds_final['V'].min().item(), 2), 'to', round(ds_final['V'].max().item(), 2))
    print('WS range:', round(ds_final['WS'].min().item(), 2), 'to', round(ds_final['WS'].max().item(), 2))