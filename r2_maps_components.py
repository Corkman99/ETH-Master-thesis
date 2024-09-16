"""
Timeseries Base Model Map

This script computes R2 values of predicting T_anom from previous values of T_anom or components. Using lag 9 (one day before) 
and window 3 (9h period) seems to be okay for most places on earth. Plot resulting R2.

We are doing Selection Bias tho... ie. by choosing minimum_sample = 30, we disregard locations that have faster processes.
Need to solve this somehow. Use only the start? use the middle 3 observations? ect. 
"""

import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs

import time

import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize
import cartopy.crs as ccrs

from sklearn.linear_model import LinearRegression

import warnings

"""
tot_time = time.time()

# Load data
xr_in = xr.open_dataset('/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl_diff',chunks={'lon':181,'lat':91,'trajtime':120,'year':42})

# Function definition
def ufunc(lag,window,min_samples):
    def compute_R2(target,ts):
        # determine column index of first non-zero / non-nan occurence for each row
        conditions = np.logical_and(~np.isnan(ts),ts != 0)
        m = np.argmax(conditions,axis=1) # vector of length nrow

        # drop rows that have series length less than min_samples
        drop = m<118-lag-window # bool vector of length nrow
        ts = ts[drop]
        target = target[drop]

        assert ts.shape == target.shape

        if min_samples > ts.shape[0]:
            #raise warnings.warn(f'Window and lag imply {ts.shape[0]} samples')
            return np.nan, ts.shape[0]

        # determine indicies to select
        ind = list(np.arange(-lag-window,-lag+1,1))
        
        # slice and reshape
        x = ts[:,ind]
        y = target[:,-1]

        if np.isnan(y).any():
            return np.nan, ts.shape[0]
        if np.isnan(x).any():
            return np.nan, ts.shape[0]

        # R2 calculation
        lm = LinearRegression()
        fit = lm.fit(x,y)
        return fit.score(x,y), ts.shape[0]
    return compute_R2

lag = int(sys.argv[1])
window = 7
min = 30

# apply function to data:
r2 = xr.full_like(xr_in,np.nan)
vars = ['T_anom','adv','adiab','diab']
for var in vars:
    start = time.time()
    print(f'Computing {var}')

    pre = xr.apply_ufunc(ufunc(lag,window,min),
                        xr_in['T_anom'],xr_in[var],
                        input_core_dims=[['year','trajtime'],['year','trajtime']],
                        output_core_dims=[[],[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float64','float64'])
    r2[var + '_r2'] = pre[0]
    r2[var +'_length'] = pre[1]
    r2 = r2.compute()

    elapsed = (time.time()-start)/60
    print(f'Elapsed: {elapsed} minutes')

# Plotting

outdir = 'plots/r2/'

lat_vals = r2.lat.values
lon_vals = r2.lon.values

colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, 10)))  # Using Viridis colormap for the range 0-10
cmap_r2 = ListedColormap(colors)

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

# labels/titles
title_string = "R2 T_anom for 3 Observations and " + str(3*lag) + "h Lag"
filename = 'all_diff_win3_lag' + str(lag)

rows=2
cols=2

def plot_func(pos,var,label,cmap,norm):
    val = r2[var+'_r2'].values
    ax = fig.add_subplot(pos, projection=crs)
    ax.coastlines(resolution='110m', color='black')
    ax.contourf(lon_vals, lat_vals, val, transform = trans, extend='neither',cmap = cmap,norm=norm)
    t=ax.text(0.025, 0.975,label, ha='left',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    ax.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_facecolor("lightgrey")

# Plot 1
# create figure
fig = plt.figure(constrained_layout=True,figsize=(14,8))
gs = GridSpec(rows, cols,
              wspace=0.05, hspace=0.05, figure=fig)

plot_func(gs[0,0],'T_anom','(a) T_anom',cmap_r2,Normalize(0,1))
plot_func(gs[0,1],'adv','(b) Adv',cmap_r2,Normalize(0,1))
plot_func(gs[1,0],'adiab','(c) Adiab',cmap_r2,Normalize(0,1))
plot_func(gs[1,1],'diab','(d) Diab',cmap_r2,Normalize(0,1))

cbar_ax = fig.add_axes([1.02, 0.25, 0.02, 0.5])
plt.colorbar(cm.ScalarMappable(cmap=cmap_r2,norm=Normalize(0,1)),cax=cbar_ax,label = 'R2', orientation='vertical')

fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()

elapsed = (time.time() - tot_time)/60
print(f'Total time: {elapsed} minutes')

"""

diff = sys.argv[3]

## Double components
tot_time = time.time()

# Load data
if diff:
    xr_in = xr.open_dataset('/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl_diff',chunks={'lon':181,'lat':91,'trajtime':121,'year':42})
    diff_num = 118
    add_str = '_diff'
else:
    xr_in = xr.open_dataset('/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl',chunks={'lon':181,'lat':91,'trajtime':121,'year':42})
    diff_num = 119
    add_str = ''

# Function definition
def ufunc(lag,window,min_samples,diff=119):
    def compute_R2(target,ts1,ts2):
        # determine column index of first non-zero / non-nan occurence for each row
        conditions = np.logical_and(~np.isnan(ts1),ts1 != 0)
        m = np.argmax(conditions,axis=1) # vector of length nrow

        # drop rows that have series length less than min_samples
        drop = m<diff-lag-window # bool vector of length nrow
        ts1 = ts1[drop]
        ts2 = ts2[drop]
        target = target[drop]

        assert ts1.shape == target.shape

        if min_samples > ts1.shape[0]:
            #raise warnings.warn(f'Window and lag imply {ts.shape[0]} samples')
            return np.nan, ts1.shape[0]

        # determine indicies to select
        ind = list(np.arange(-lag-window,-lag+1,1))
        
        # slice and reshape
        x = np.concatenate([ts1[:,ind],ts2[:,ind]],axis=1)
        y = target[:,-1]

        if np.isnan(y).any():
            return np.nan, ts1.shape[0]
        if np.isnan(x).any():
            return np.nan, ts1.shape[0]

        # R2 calculation
        lm = LinearRegression()
        fit = lm.fit(x,y)
        return fit.score(x,y), ts1.shape[0]
    return compute_R2

lag = int(sys.argv[1])
window = int(sys.argv[2])
min = 30

# apply function to data:
r2 = xr.full_like(xr_in,np.nan)
vars = [['T_anom','adv'],['T_anom','adiab'],['T_anom','diab'],
        ['adv','adiab'],['adv','diab'],['diab','adiab']]
for var in vars:
    start = time.time()
    print(f'Computing {var[0]} with {var[1]}')

    pre = xr.apply_ufunc(ufunc(lag,window,min,diff=diff_num),
                        xr_in['T_anom'],xr_in[var[0]],xr_in[var[1]],
                        input_core_dims=[['year','trajtime'],['year','trajtime'],['year','trajtime']],
                        output_core_dims=[[],[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float64','float64'])
    r2[var[0]+'_' +var[1] + '_r2'] = pre[0]
    r2[var[0]+'_' +var[1] +'_length'] = pre[1]
    r2 = r2.compute()

    elapsed = (time.time()-start)/60
    print(f'Elapsed: {elapsed} minutes')

# Plotting

outdir = 'plots/r2/'

lat_vals = r2.lat.values
lon_vals = r2.lon.values

colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, 10)))  # Using Viridis colormap for the range 0-10
cmap_r2 = ListedColormap(colors)

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

# labels/titles
title_string = "R2 T_anom for " + str(window) + ' ' + add_str + " Observations and " + str(3*lag) + "h Lag"
filename = 'doubles_win'+str(window)+'_lag' + str(lag) + add_str

rows=3
cols=2

def plot_func(pos,var,label,cmap,norm):
    val = r2[var[0]+'_' +var[1]+'_r2'].values
    ax = fig.add_subplot(pos, projection=crs)
    ax.coastlines(resolution='110m', color='black')
    ax.contourf(lon_vals, lat_vals, val, transform = trans, extend='neither',cmap = cmap,norm=norm)
    t=ax.text(0.025, 0.975,label, ha='left',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    ax.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_facecolor("lightgrey")

# Plot 1
# create figure
fig = plt.figure(constrained_layout=True,figsize=(14,8))
gs = GridSpec(rows, cols,
              wspace=0.05, hspace=0.05, figure=fig)

plot_func(gs[0,0],vars[0],'(a) T_anom, Adv',cmap_r2,Normalize(0,1))
plot_func(gs[0,1],vars[1],'(b) T_anom, Adiab',cmap_r2,Normalize(0,1))
plot_func(gs[1,0],vars[2],'(c) T_anom, Diab',cmap_r2,Normalize(0,1))
plot_func(gs[1,1],vars[3],'(d) Adv, Adiab',cmap_r2,Normalize(0,1))
plot_func(gs[2,0],vars[4],'(e) Adv, Diab',cmap_r2,Normalize(0,1))
plot_func(gs[2,1],vars[5],'(f) Diab, Adiab',cmap_r2,Normalize(0,1))

cbar_ax = fig.add_axes([1.02, 0.25, 0.02, 0.5])
plt.colorbar(cm.ScalarMappable(cmap=cmap_r2,norm=Normalize(0,1)),cax=cbar_ax,label = 'R2', orientation='vertical')

fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()

elapsed = (time.time() - tot_time)/60
print(f'Total time: {elapsed} minutes')