# correlation maps
# ... 

# Preamble
import xarray as xr
import numpy as np
import pandas as pd

indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = '~/Thesis/data/'

# Data load
xr_in = xr.open_dataset(indir + '/' + infile,
                        chunks = {'years' : 42,'lon': 242, 'lat': 182}).mean(dim='lev',skipna=True)


# residuals

# with res_years
res_years = xr.Dataset()
res_years['res1'] = xr_in['res1']
res_years['res2'] = xr_in['res2']
res_years['seas'] = xr_in['seas']
res_years['sum'] = xr_in['res1']+xr_in['res2']+xr_in['seas']
res_years.to_netcdf(outdir + "residuals.nc")

res_agg = xr.Dataset()
for var in ['res1','res2','seas']:
    res_agg[var+'_m'] = xr_in[var].mean(dim='years',skipna=True)
    res_agg[var+'_v'] = xr_in[var].var(dim='years',skipna=True)
res_agg['sum_m'] = res_years['sum'].mean(dim='years',skipna=True)
res_agg['sum_v'] = res_years['sum'].var(dim='years',skipna=True)
res_agg.to_netcdf(outdir + "residuals_agg.nc")

res_reg = xr.Dataset()
for var in ['res1','res2','seas']:
    res_reg[var] = xr_in[var].polyfit(deg=1,dim='years')['polyfit_coefficients'].sel(degree=1).drop_vars('degree')
res_reg['sum'] = res_years['sum'].polyfit(deg=1,dim='years')['polyfit_coefficients'].sel(degree=1).drop_vars('degree')
res_reg.to_netcdf(outdir + "residuals_reg.nc")