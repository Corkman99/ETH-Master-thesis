#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday 10.03.2024

@author: mfroelich
"""

# Description
# This script loads the TX1day_decomposition data and selects two variables
# (eg. T_anom and one component) to compare correlations on subsets of point-
# wise data based on some quantiles. eg. we could compare, at every location, 
# the difference in correlation between lower 0.2 quantiles of the data, and 
# upper 0.8 quantiles of the data. 

# --------------------------------------------------------------------------- #
# -------------------------------- Preamble --------------------------------- #
# --------------------------------------------------------------------------- #
import xarray as xr
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #

indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
quantile_sep_l = 0.3
quantile_sep_u = 0.7
variables = ['T_anom','adiab']

print("Start: data load")
xr_in = xr.open_dataset(indir + '/' + infile,drop_variables=['adv','diab','res1','res2']).mean(dim='lev',skipna=True)

# Incredibly naive and inefficient, but should be okay
latitude = np.arange(-90,90.25,0.5)
longitude = np.arange(-180,180.25,0.5)
coor = np.array(np.meshgrid(latitude,longitude)).reshape(2,len(latitude)*len(longitude)).T

# saving dataframe
df_upper = pd.DataFrame(index=latitude,columns=longitude)
df_lower = pd.DataFrame(index=latitude,columns=longitude)

print("Start: processing")
for lat_i, lon_j in coor:
    # create a data frame for the data from that grid point
    xr_gp = xr_in[variables].sel(lat=lat_i,lon=lon_j)
    df = pd.DataFrame(columns=variables)
    for v in variables:
        df[v] = xr_gp[v].values
    df.dropna(inplace=True)

    # divide into upper and lower subsets
    up = df[df['T_anom'] >= df['T_anom'].quantile(q=quantile_sep_u)]
    lo = df[df['T_anom'] < df['T_anom'].quantile(q=quantile_sep_l)]
    
    # compute subset correlation and save in array (lat x lon)
    df_upper.at[lat_i,lon_j] = up['T_anom'].corr(up['adiab'])
    df_lower.at[lat_i,lon_j] = lo['T_anom'].corr(lo['adiab'])

# Assign dataframes to xarray type
final = xr.Dataset()
final['upper'] = xr.DataArray(df_upper,
                     dims=['lat','lon'],
                     coords=dict(
                         lat = latitude,
                         lon = longitude,
                         )
                     )
final['lower'] = xr.DataArray(df_upper,
                     dims=['lat','lon'],
                     coords=dict(
                         lat = latitude,
                         lon = longitude,
                         )
)
final['diff'] = xr.DataArray(abs(df_upper-df_lower),
                     dims=['lat','lon'],
                     coords=dict(
                         lat = latitude,
                         lon = longitude,
                         )
)

# saving
final.to_netcdf("~/Thesis/data/subset_correlations_adiab_30-70.nc")