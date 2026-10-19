#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_wrf_coord_and_do_3D_interp.py 的 buoyantBoussinesqSimpleFoam 版本。

相对原脚本的修正：
  1. Theta = T + 300。WRF 输出流中的 T 恒为干位温扰动（USE_THETA_M=1 时湿位温另存为 THM），
     不能再除以 (1 + 1.61*QVAPOR)。
  2. 垂直插值使用离地高度：H_agl = (PH+PHB)/g - 地表位势高度，与 CFD 的 z=0 地面一致。
  3. 新增虚位温 Theta_v = Theta*(1 + 0.61*QVAPOR)，供浮力计算使用。
  4. TK 由 Theta 与插值后的 Exner 函数计算；WRF 最低层以下按静力关系外推 Exner，
     而不是把 TK 常数下延。
  5. 近地面对数律的 z0 改为命令行参数，默认 0.1 m，与 CFD 壁函数 atmNutkWallFunction 保持一致。
  6. 输出文件名带后缀（默认 _bbsf），不覆盖原脚本产物；可用 --out-dir 指定输出目录。

用法:
  python util/convert_wrf_coord_and_do_3D_interp_bbsf.py \\
      W_myExp03/auxhist2/tmp/auxhist2_d03_2025-09-01_10:00:00_tmp.nc [--out-dir DIR] [--z0 0.1]
"""
import argparse
import os
import numpy as np
import xarray as xr
from pyproj import Proj
from scipy.interpolate import griddata
from wrf import destagger, interplevel

G = 9.81
RD = 287.0
CP = 1004.0
KAPPA = RD / CP
P0 = 1.0e5

lon0, lat0 = 113.32, 23.115


def calculate_relative_xy(ds, lon0, lat0):
    """WRF 经纬度 -> 以 (lon0, lat0) 为中心的方位等距投影 (x, y) 米坐标。"""
    p = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84', units='m')
    x_vec, y_vec = p(ds['XLONG'], ds['XLAT'])
    dims = ds['XLONG'].dims
    da_x = xr.DataArray(x_vec, coords=ds['XLONG'].coords, dims=dims, name='x_rel')
    da_y = xr.DataArray(y_vec, coords=ds['XLONG'].coords, dims=dims, name='y_rel')
    return da_x, da_y


def generate_stretched_z(z_max, dz_bottom, stretch_factor, dz_max_limit):
    z = [0]
    dz = dz_bottom
    while z[-1] < z_max:
        dz = min(dz * stretch_factor, dz_max_limit)
        z.append(z[-1] + dz)
    return np.array(z)


def _cartesian_output_dir(nc_path):
    """WRF 产物若在 auxhist2/tmp/ 下，则输出到上一层 auxhist2/；否则与输入文件同目录。"""
    parent = os.path.dirname(os.path.abspath(nc_path))
    if os.path.basename(parent) == 'tmp':
        return os.path.dirname(parent)
    return parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('nc_path', help='WRF auxhist2 tmp NetCDF 文件')
    ap.add_argument('--out-dir', default=None, help='输出目录（默认与原脚本相同的 auxhist2/）')
    ap.add_argument('--suffix', default='_bbsf', help='输出文件名后缀（默认 _bbsf）')
    ap.add_argument('--z0', type=float, default=0.1, help='近地面对数律粗糙度 [m]，应与 CFD 壁函数 z0 一致')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nc_path = os.path.abspath(os.path.normpath(args.nc_path))
    if not os.path.isfile(nc_path):
        raise SystemExit(f"Error: file not found: {nc_path}")

    fname = os.path.basename(nc_path)
    ds = xr.open_dataset(nc_path, engine='netcdf4').squeeze()

    # --- 解交错、离地高度 ---
    H_stag = (ds['PH'] + ds['PHB']) / G
    H_sfc = H_stag.values[0]
    H_agl = destagger(H_stag, stagger_dim=0) - H_sfc[None, :, :]
    ds['H'] = xr.DataArray(H_agl, coords=ds.T.coords, dims=ds.T.dims)
    ds['HGT'] = xr.DataArray(H_sfc, coords=ds.T.isel(bottom_top=0).coords, dims=ds.T.dims[1:])
    ds['W'] = xr.DataArray(destagger(ds['W'], stagger_dim=0), coords=ds.T.coords, dims=ds.T.dims)
    ds['U'] = xr.DataArray(destagger(ds.U, stagger_dim=2), coords=ds.T.coords, dims=ds.T.dims)
    ds['V'] = xr.DataArray(destagger(ds.V, stagger_dim=1), coords=ds.T.coords, dims=ds.T.dims)
    ds['WS'] = np.sqrt(ds['U'] ** 2 + ds['V'] ** 2)

    TKE_PBL = np.maximum(destagger(ds['TKE_PBL'], stagger_dim=0), 1e-6)
    ds['TKE_PBL'] = xr.DataArray(TKE_PBL, coords=ds.T.coords, dims=ds.T.dims)

    # --- 位温、虚位温、Exner ---
    P_total = ds['P'] + ds['PB']
    ds['Theta'] = ds['T'] + 300.0
    ds['Theta_v'] = ds['Theta'] * (1.0 + 0.61 * ds['QVAPOR'])
    ds['Exner'] = (P_total / P0) ** KAPPA

    x_meters, y_meters = calculate_relative_xy(ds, lon0, lat0)
    ds['x_rel'] = x_meters
    ds['y_rel'] = y_meters
    ds_ = ds.assign_coords(x_rel=ds['x_rel'], y_rel=ds['y_rel']).drop_vars(
        ['Times', 'XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'XTIME'], errors='ignore'
    )
    ds_ = ds_[['U', 'V', 'W', 'WS', 'ZNT', 'H', 'HGT', 'TKE_PBL', 'Theta', 'Theta_v', 'QVAPOR', 'Exner']]

    # --- 水平插值 ---
    x_tgt = np.arange(-5000, 5000 + 1, 100)
    y_tgt = np.arange(-5000, 5000 + 1, 100)
    grid_x, grid_y = np.meshgrid(x_tgt, y_tgt)

    def regrid_solver(data, src_x, src_y):
        points = np.column_stack((src_x.flatten(), src_y.flatten()))
        return griddata(points, data.flatten(), (grid_x, grid_y), method='linear', fill_value=np.nan)

    def regrid_dataset(ds_in, gx, gy):
        return xr.apply_ufunc(
            regrid_solver, ds_in, gx, gy,
            input_core_dims=[['south_north', 'west_east']] * 3,
            output_core_dims=[['y_rel', 'x_rel']],
            vectorize=True,
            exclude_dims=set(('y_rel', 'x_rel')),
            output_dtypes=[float],
        )

    print("Running Horizontal Interpolation...")
    ds_horiz = regrid_dataset(ds_, ds_['x_rel'], ds_['y_rel'])

    # --- 垂直插值（离地高度）与近地面外推 ---
    print("Running Vertical Interpolation...")
    z_tgt = generate_stretched_z(z_max=3000, dz_bottom=10, stretch_factor=1, dz_max_limit=50)
    H_horiz = ds_horiz['H'].values
    H_min_horiz = H_horiz[0, :, :]
    Z_3d = np.broadcast_to(z_tgt[:, None, None], (len(z_tgt), len(y_tgt), len(x_tgt)))
    H_min_3d = np.broadcast_to(H_min_horiz[None, :, :], Z_3d.shape)
    below = Z_3d < H_min_3d
    z0 = args.z0

    final_vars = {}
    for var in ['U', 'V', 'W', 'WS', 'TKE_PBL', 'Theta', 'Theta_v', 'QVAPOR', 'Exner']:
        data_horiz = ds_horiz[var].values
        var_vert = np.zeros(Z_3d.shape)
        for idx, z_val in enumerate(z_tgt):
            var_vert[idx, :, :] = interplevel(data_horiz, H_horiz, z_val).values
        nan_mask = np.isnan(var_vert)
        bfilled = xr.DataArray(var_vert, dims=['z', 'y', 'x']).bfill(dim='z').values

        if var in ['U', 'V', 'WS']:
            log_mult = np.log(np.maximum(Z_3d, z0 + 1e-3) / z0) / np.log(np.maximum(H_min_3d, z0 + 1e-3) / z0)
            m = nan_mask & below
            var_vert[m] = bfilled[m] * log_mult[m]
        elif var == 'Exner':
            # 静力外推: dPi/dz = -g / (cp * theta)
            theta_low = final_vars['Theta'][1]
            m = nan_mask & below
            var_vert[m] = (bfilled + G / (CP * theta_low) * (H_min_3d - Z_3d))[m]
        else:
            var_vert = bfilled
        var_vert = xr.DataArray(var_vert, dims=['z', 'y', 'x']).bfill(dim='z').ffill(dim='z').values
        final_vars[var] = (['z', 'y_rel', 'x_rel'], var_vert)

    final_vars['TK'] = (['z', 'y_rel', 'x_rel'], final_vars['Theta'][1] * final_vars['Exner'][1])
    final_vars['HGT'] = (['y_rel', 'x_rel'], ds_horiz['HGT'].values)

    ds_final = xr.Dataset(data_vars=final_vars, coords={'z': z_tgt, 'y_rel': y_tgt, 'x_rel': x_tgt})
    ds_final.attrs.update({
        'theta_definition': 'dry potential temperature = WRF T + 300',
        'z_reference': 'height above local WRF terrain (AGL)',
        'log_law_z0_m': z0,
    })

    out_dir = args.out_dir or _cartesian_output_dir(nc_path)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname.replace('tmp', '1h-rolling_cartesian').replace('.nc', f'{args.suffix}.nc'))
    ds_final.to_netcdf(save_path)

    print(f"Successfully saved to {save_path} !")
    print('Z levels created:', len(z_tgt), f"(Min: {z_tgt[0]:.1f}m, Max: {z_tgt[-1]:.1f}m)")
    for v in ['WS', 'Theta', 'Theta_v', 'TK']:
        print(f"{v} range: {float(ds_final[v].min()):.2f} to {float(ds_final[v].max()):.2f}")
