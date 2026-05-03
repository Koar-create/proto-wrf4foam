import os
from pathlib import Path

import pandas as pd

# 配置路径
repo_root = Path(os.getcwd())
data_dir = repo_root / "data" / "260409" / "raw"
data_dir_opt = repo_root / "data" / "260413" / "processed"
output_file = data_dir_opt / "merged_lidar_simulation_final_nighttime_only.csv"

# 目标时次
target_times = [f"{date} {hr:02d}:00:00" for date in ["2025-09-01","2025-09-02","2025-09-03"] for hr in range(11, 24)]


# CFD 文件列表 (Reference vs Sensitivity)
cfd_files_ref = [f"CFD_lidar_simulation_{date}_{hr:02d}00_two_boundaries_as_outlet.csv" for date in ["20250901","20250902","20250903"] for hr in range(11, 24)]

cfd_files_sen = [f"CFD_lidar_simulation_{date}_{hr:02d}00_two_boundaries_as_outlet-fvOpt_sensitivity_run.csv" for date in ["20250901","20250902","20250903"] for hr in range(11, 24)]

def load_and_preprocess():
    print("Step 1: Loading WRF and Lidar data...")
    df_wrf = pd.read_csv(data_dir / "wrf" / "WRF_lidar_simulation_1h-rolling.csv")
    df_lidar = pd.read_csv(data_dir / "lidar" / "lidar_1h-rolling.csv")

    # 规范化时间格式
    df_wrf['datetime'] = pd.to_datetime(df_wrf['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_lidar['datetime'] = pd.to_datetime(df_lidar['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    unique_wrf_times = df_wrf['datetime'].unique()
    print(f"Total unique WRF times: {len(unique_wrf_times)}")
    print(f"Sample WRF times (first 10): {unique_wrf_times[:10]}")
    print(f"Sample WRF times (last 10): {unique_wrf_times[-10:]}")
    
    # Check if '2025-09-03 12:00:00' exists in unique_wrf_times
    test_time = "2025-09-03 12:00:00"
    if test_time in unique_wrf_times:
        print(f"MATCH: {test_time} found in WRF data.")
    else:
        print(f"MISS: {test_time} NOT found in WRF data.")

    # 过滤目标时次
    df_wrf = df_wrf[df_wrf['datetime'].isin(target_times)].copy()
    df_lidar = df_lidar[df_lidar['datetime'].isin(target_times)].copy()
    print(f"Target times: {target_times}")
    print(f"WRF shape after filter: {df_wrf.shape}")

    def process_cfd_files(file_list):
        cfd_list = []
        for f in file_list:
            tmp = pd.read_csv(data_dir / "cfd" / "control" / f)
            tmp['datetime'] = pd.to_datetime(tmp['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            cfd_list.append(tmp)
        return pd.concat(cfd_list, ignore_index=True)

    print("Step 2: Loading CFD Reference and Sensitivity data (S_p=-0.01 and -0.02)...")
    df_cfd_ref = process_cfd_files(cfd_files_ref)
    # sensitivity files live under raw/cfd/sensitivity
    df_cfd_sen = pd.concat(
        [
            pd.read_csv(data_dir / "cfd" / "sensitivity" / f).assign(
                datetime=lambda x: pd.to_datetime(x["datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            )
            for f in cfd_files_sen
        ],
        ignore_index=True,
    )
    print(f"CFD Ref: {df_cfd_ref.shape}, Sen (-0.01): {df_cfd_sen.shape}")

    # 为各表添加 "高度层索引"，确保逐层对齐
    print("Step 3: Aligning vertical layers by index...")
    for df in [df_wrf, df_lidar, df_cfd_ref, df_cfd_sen]:
        h_col = 'z_probe' if 'z_probe' in df.columns else 'Height'
        df.sort_values(by=['datetime', 'obtid', h_col], inplace=True)
        df['layer_idx'] = df.groupby(['datetime', 'obtid']).cumcount()

    # 开始合并
    print("Step 4: Merging tables...")
    
    # 1. 以 WRF 为基准
    df_wrf = df_wrf.rename(columns={
        'U_wrf': 'u_wrf', 'V_wrf': 'v_wrf', 'WS_wrf': 'ws_wrf', 'z_probe': 'Height'
    })
    
    # 2. 合并 CFD Reference
    df_cfd_ref_sub = df_cfd_ref[['datetime', 'obtid', 'layer_idx', 'U_cfd', 'V_cfd', 'W_cfd']].rename(columns={
        'U_cfd': 'u_cfd_ref', 'V_cfd': 'v_cfd_ref', 'W_cfd': 'w_cfd_ref'
    })
    merged = pd.merge(df_wrf, df_cfd_ref_sub, on=['datetime', 'obtid', 'layer_idx'], how='inner')

    # 3. 合并 CFD Sensitivity 1 (Sp = -0.01)
    df_cfd_sen_sub = df_cfd_sen[['datetime', 'obtid', 'layer_idx', 'U_cfd', 'V_cfd', 'W_cfd']].rename(columns={
        'U_cfd': 'u_cfd_sen', 'V_cfd': 'v_cfd_sen', 'W_cfd': 'w_cfd_sen'
    })
    merged = pd.merge(merged, df_cfd_sen_sub, on=['datetime', 'obtid', 'layer_idx'], how='inner')

    # 6. 合并 Lidar
    df_lidar_sub = df_lidar[['datetime', 'obtid', 'layer_idx', 'U', 'V', 'WindSpd']].rename(columns={
        'U': 'u_obs', 'V': 'v_obs', 'WindSpd': 'ws_obs'
    })
    merged = pd.merge(merged, df_lidar_sub, on=['datetime', 'obtid', 'layer_idx'], how='inner')

    # 整理最终列
    final_columns = [
        'datetime', 'obtid', 'Height', 'lon', 'lat', 
        'u_obs', 'v_obs', 'ws_obs', 
        'u_wrf', 'v_wrf', 'ws_wrf', 
        'u_cfd_ref', 'v_cfd_ref', 'w_cfd_ref',
        'u_cfd_sen', 'v_cfd_sen', 'w_cfd_sen',
        'ws_cfd_ref', 'ws_cfd_sen',
    ]
    
    # compute CFD wind speed (for downstream Hovmöller plotting)
    merged["ws_cfd_ref"] = (merged["u_cfd_ref"] ** 2 + merged["v_cfd_ref"] ** 2) ** 0.5
    merged["ws_cfd_sen"] = (merged["u_cfd_sen"] ** 2 + merged["v_cfd_sen"] ** 2) ** 0.5

    final_df = merged[final_columns].copy()
    
    print(f"Merge Complete! Final shape: {final_df.shape}")
    print(f"Saving to: {output_file}")
    final_df.to_csv(output_file, index=False)
    return final_df

if __name__ == "__main__":
    final_data = load_and_preprocess()
    print("Preview of first 5 rows:")
    print(final_data.head())
