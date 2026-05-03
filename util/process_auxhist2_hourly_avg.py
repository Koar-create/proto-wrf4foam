#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import xarray as xr

# Get user's HOME directory (commonly used for data paths)
HOME = os.environ.get('HOME')

# --- Project Path Configuration ---
# Define the root directory of the project
project_path = f"{HOME}/WRF-OpenFOAM-Coupling"
# Define the current working directory (parent directory of WRF simulation outputs)
work_dir = f"{project_path}/W_myExp03"
# Define the WRF run directory where auxhist2 files are typically located
wrf_run_dir = f"{work_dir}/WRF/run"  ## /archive
# Define the destination directory for the processed auxhist2 files (1-hour average)
destination_dir = f"{work_dir}/auxhist2/tmp"

# --- Generate Center Hour List ---
# Create a time range from 2025-09-03 00:00 to 2025-09-03 23:00, with an interval of 1 hour
# These points will serve as the center points for the 1-hour sliding average
center_hour_list = pd.date_range(start='2025-09-01 00:00', end='2025-09-06 23:00', freq='1h').to_list()

# --- Ensure Destination Directory Exists ---
# Create the destination directory if it does not already exist
os.makedirs(destination_dir, exist_ok=True)

# --- Loop through each center hour and perform 1-hour sliding average ---
for center_hour in center_hour_list:
    # Construct the destination filename, including the timestamp of the center hour
    fname_destination = f"auxhist2_d03_{center_hour.strftime('%Y-%m-%d_%H:%M:%S')}_tmp.nc"
    
    # Calculate the list of timestamps to be used for the 1-hour average
    # This list spans from 30 minutes before the center hour to 30 minutes after, with 10-minute intervals.
    # For example, for a center_hour of 00:00, the range is from 23:30 (previous day) to 00:30 (current day).
    time_used_list = pd.date_range(
        start=center_hour - pd.Timedelta(minutes=30), 
        end=center_hour + pd.Timedelta(minutes=30), 
        freq='10min').tolist()
    
    print(f"Current center hour: {center_hour}")
    print(f"Timestamps used for averaging: {time_used_list}")
    print('='*56) # Print a separator for readability
    
    # Read the auxhist2 files corresponding to each timestamp in time_used_list
    # xr.open_dataset is used to open NetCDF files
    # .squeeze() removes dimensions of size 1 (e.g., the time dimension for individual files)
    used_ds_list = [
        xr.open_dataset(
            f'{wrf_run_dir}/auxhist2_d03_' + time_used.strftime('%Y-%m-%d_%H:%M:%S'), engine='netcdf4'
        ).squeeze() for time_used in time_used_list
    ]
    
    # Concatenate all read datasets along the 'time' dimension, then calculate the mean across time
    # This step performs the core 1-hour sliding average operation.
    ds = xr.concat(used_ds_list, dim='time').mean('time')  
    
    # Delete the 'unlimited_dims' encoding, as it might cause compatibility issues when saving new NetCDF files
    del ds.encoding['unlimited_dims']
    
    # Save the averaged dataset to a new NetCDF file in the destination directory
    ds.to_netcdf( os.path.join(destination_dir, fname_destination) )

