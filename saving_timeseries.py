#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saving time-series data (from nc files) into csv for single locations

The data is saved in indir as yearly and per-level data. So, at each location,
for every yearly maxima, we have 3 files to look into (level=10,30,50). In each 
level, there are 8 time-series indexed by ts=0...7. Each time-series is indexed 
by 3 hourly intervals: -360, -357, - 354, ..., -3, 0 (121 points)
It contains many variables:

vars = ['T_anom','res1','seas','adv','adiab1','adiab2','adiab3','diab','res2',
'p_traj','lat_traj','lon_traj','dist_traj','age','dist','delta_p',
'gen_lat','gen_lon','gen_p','doy']

The adiab1,2,3 have to be summed to get the adiab component. Due to high dependence
between time-series (levels and time-intervals), we should average first over 'ts' 
and then over the three levels. How do we want to save the data? 

For a single location, the output is 42 time-series of length 121 ('trajtime'). We
can create a netcdf file with dimension [year, trajtime, lat, lon] with the variables
we are interested in. 
"""

# --------------------------------------------------------------------------- #
# -------------------------------- Preamble --------------------------------- #
# --------------------------------------------------------------------------- #
import xarray as xr
import pandas as pd

# --------------------------------------------------------------------------- #

indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/ncdf/TX1day/'
outdir = '/net/litho/atmosdyn2/mfroelich/'
vars_to_drop = ['traj_p','lat_traj','lon_traj','dist_traj','age','dist','delta_p','gen_lat','gen_lon','gen_p','doy']

# 1st for-loop : iterate over levels
levels = [10,30,50]
infiles_per_level = ['complete_budget_TX1day_era5_v10_' + str(i) for i in levels]

# 2nd for-loop : iterate over years 
years = range(1980,2021)

list_of_levels = []
for i, level in enumerate(infiles_per_level): 

    print(f'Loading level {levels[i]}')

    infile = [level + '_' + str(j) for j in years] # + '.nc'
    list_of_years = []

    for j, file in enumerate(infile):
        print(f'Loading year {years[j]}')
        xr_year = xr.open_dataset(indir + file, 
                                  drop_variables=vars_to_drop,
                                  chunks = {'lat': 361,'lon': 721,'trajtime':121,'ts':8}).mean(dim='ts',skipna=True)
        xr_year['adiab'] = xr_year['adiab1'] + xr_year['adiab2'] + xr_year['adiab3']
        xr_year['res'] = xr_year['res1'] + xr_year['seas']
        xr_year = xr_year.drop_vars(['res1','seas','adiab1','adiab2','adiab3'])
        list_of_years.append(xr_year)

    print('Ready to concat years')
    list_of_levels.append(xr.concat(list_of_years, pd.Index(list(years), name='year')))

print('Ready to concat levels')
final = xr.concat(list_of_levels,pd.Index(levels,name='level')).mean(dim='level',skipna=True)

print('Ready to save')
final.to_netcdf(outdir + 'TS_TX1day_mean-lvl')

#_Chunksizes