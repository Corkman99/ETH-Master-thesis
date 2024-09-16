"""
Timeseries Base Model Map

This script computes R2 values of predicting T_anom from previous values of T_anom or components. Using lag 9 (one day before) 
and window 3 (9h period) seems to be okay for most places on earth. Plot resulting R2.
"""

import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs

import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize
import cartopy.crs as ccrs

from sklearn.linear_model import LinearRegression

import warnings

# Load data
xr_in = xr.open_dataset('/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl',chunks={'lon':91,'lat':31,'trajtime':122,'years':42})

# Function definition
def ufunc(lag,window,min_samples):
    def compute_R2(ts):
        # determine column index of first non-zero / non-nan occurence for each row
        conditions = np.logical_and(~np.isnan(ts),ts != 0)
        m = np.argmax(conditions,axis=1)

        # drop years that have series length less than min_samples
        ts = ts[m<119-lag-window]
        if min_samples > ts.shape[0]:
            #raise warnings.warn(f'Window and lag imply {ts.shape[0]} samples')
            return np.nan, ts.shape[0]

        # determine indicies to select
        ind = list(np.arange(-lag-window,-lag+1,1))
        ind.append(120)
        
        # slice and reshape
        xy = ts[:,ind]
        x = xy[:,:-2]
        y = xy[:,-1]

        # R2 calculation
        lm = LinearRegression()
        fit = lm.fit(x,y)
        return fit.score(x,y), ts.shape[0]
    return compute_R2

lag = int(sys.argv[1])
print(lag)
window = 3
min = 30

# apply function to data:
pre = xr.apply_ufunc(ufunc(lag,window,min),
                    xr_in['T_anom'],
                    input_core_dims=[['year','trajtime']],
                    output_core_dims=[[],[]],
                    vectorize=True,
                    dask='parallelized', # since func converts to numpy.array
                    output_dtypes=['float64','float64'])

r2 = xr.Dataset({'r2':pre[0],'length':pre[1]}).compute()


# Plotting

outdir = 'plots/r2/'

lat_vals = r2.lat.values
lon_vals = r2.lon.values

colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, 10)))  # Using Viridis colormap for the range 0-10
cmap_r2 = ListedColormap(colors)

colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Using Viridis colormap for the range 0-10
cmap_len = ListedColormap(colors)

crs = ccrs.PlateCarree(0)

# labels/titles
labels = ['(a) R2','(b) Sample size']
title_string = "R2 of T_anom for 3 T_anom Observations and " + str(3*lag) + "h Lag"
filename = 'tanom_tanom_win3_lag' + str(lag)

rows=2
cols=1

def plot_func(pos,var,label,cmap,norm):
    val = r2[var].values
    ax = fig.add_subplot(pos, projection=crs)
    ax.coastlines(resolution='110m', color='black')
    ax.contourf(lon_vals, lat_vals, val, transform = crs, extend='neither',cmap = cmap,norm=norm)
    t=ax.text(0.975, 0.05,label, ha='right',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    ax.gridlines(crs=crs, draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_facecolor("lightgrey")

# Plot 1
# create figure
fig = plt.figure(constrained_layout=True,figsize=(8,8))
gs = GridSpec(rows, cols,
              wspace=0.05, hspace=0.05, figure=fig)

plot_func(gs[0,0],'r2',labels[0],cmap_r2,Normalize(0,1))
plot_func(gs[1,0],'length',labels[1],cmap_len,Normalize(min,42))

cbar_ax = fig.add_axes([1, 0.55, 0.02, 0.4])
cb = plt.colorbar(cm.ScalarMappable(cmap=cmap_r2,norm=Normalize(0,1)),cax=cbar_ax,label = 'R2', orientation='vertical')

cbar_ax = fig.add_axes([1, 0.05, 0.02, 0.4])
cb = plt.colorbar(cm.ScalarMappable(cmap=cmap_len,norm=Normalize(0,50)),cax=cbar_ax,label = 'Sample size', orientation='vertical')

fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()