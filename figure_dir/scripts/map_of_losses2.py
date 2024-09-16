import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm
import matplotlib.ticker as mticker

import cartopy.feature as cfeature
from shapely.vectorized import contains

def land_ocean(data):
    lons, lats = np.meshgrid(data.lon, data.lat)
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    land_geometries = list(land_feature.geometries())
    land_mask = np.zeros(lons.shape, dtype=bool)
    for geom in land_geometries:
        land_mask |= contains(geom, lons, lats)
    ocean_mask = ~land_mask
    return land_mask, ocean_mask

def plot_map(ax, data, label, cmap, norm, gridline_labels):
    """Plots a map with given data and custom options."""
    ax.contourf(data.lon, data.lat, data.values, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, extend='max', levels=bounds)
    ax.coastlines(resolution='110m',color='black')
    ax.gridlines(draw_labels=gridline_labels,ylocs=range(-60,61,30),linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0,1.1,label,ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

def plot_line(ax, data, ocean, land, transformed_lats):
    """Plots a line graph for given data over transformed latitudes, for ocean and land"""
    mse_ocean = data.where(ocean).mean(dim='lon')
    mse_land = data.where(land).mean(dim='lon')
    ax.plot(mse_ocean, transformed_lats, label='Ocean', color='blue',linewidth=2)
    ax.plot(mse_land, transformed_lats, label='Land', color='green',linewidth=2)
    ax.set_ylabel('')
    ax.set_xlabel('MSE Loss')
    ax.set_xscale('symlog', base=2, linthresh=0.5,linscale=0.75)
    ax.set_xticks([-0.5,0,0.5,1,2,4,8,16])
    ax.set_xticklabels(['-0.5','0','0.5','1','2','4','8','16'])
    ax.set_ylim([transformed_lats.min(), transformed_lats.max()])
    ax.set_yticks([])    
    
    #t=ax.text(0.05,0.95,'(b) Longitudinal mean MSE',ha='left',va='top', transform=ax.transAxes)
    #t.set_bbox(dict(facecolor='white', edgecolor='black'))

# Example usage of the functions
def plot_figure(xarray_data, cmap, norm, transformed_lats):
    """Creates a complex figure based on the structure described."""
    
    land, ocean = land_ocean(xarray_data)

    # Creating a figure and grid for the plots
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(6,6)# height_ratios=[1, 1])

    # Plot 2: MSELoss_tanom map (line 1, column 1)
    mse_loss_tanom_data = xarray_data['MSELoss_tanom']
    ax_map = fig.add_subplot(gs[0:3,0:5], projection=ccrs.Robinson(0))
    plot_map(ax_map, mse_loss_tanom_data, '(a) Full MSELoss', cmap, norm, gridline_labels=True)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax_map],
                 location='left', extend='max',label='MSE',
                 ticks=bounds,format=mticker.FixedFormatter([str(x) for x in bounds]))
    
    # Plot 3: Line plot (line 1, column 2)
    mse = xarray_data['MSELoss_tanom']
    ax_line = fig.add_subplot(gs[0:3,5])
    #ax_line = fig.add_axes([0.9, 0.535, 0.15, 0.35])
    plot_line(ax_line, mse, ocean, land, transformed_lats)
    box = ax_line.get_position()
    box.x0 = box.x0 - 0.1
    box.x1 = box.x1 - 0.1
    ax_line.set_position(box)

    # Plot 4: MSELoss_adv map (line 2, column 0)
    mse_loss_adv_data = xarray_data['MSEadv']
    ax_map2 = fig.add_subplot(gs[3:6,0:2], projection=ccrs.Robinson(0))
    plot_map(ax_map2, mse_loss_adv_data, '(b) Adv', cmap, norm, gridline_labels=False)

    # Plot 5: MSELoss_adiab map (line 2, column 1)
    mse_loss_adiab_data = xarray_data['MSEadiab']
    ax_map3 = fig.add_subplot(gs[3:6,2:4], projection=ccrs.Robinson(0))
    plot_map(ax_map3, mse_loss_adiab_data, '(c) Adiab', cmap, norm, gridline_labels=False)

    # Plot 6: MSELoss_diab map (line 2, column 2)
    mse_loss_diab_data = xarray_data['MSEdiab']
    ax_map4 = fig.add_subplot(gs[3:6,4:6], projection=ccrs.Robinson(0))
    plot_map(ax_map4, mse_loss_diab_data, '(d) Diab', cmap, norm, gridline_labels=False)

    plt.savefig('/home/mfroelich/Thesis/figure_dir/plots/map_median_mse_diff',bbox_inches='tight')
    plt.close()

data1 = xr.open_dataset('/home/mfroelich/Thesis/LSTM_results/baseline_inference').median(dim='year')
data2 = xr.open_dataset('/home/mfroelich/Thesis/LSTM_final/modelinference_final').median(dim='year')
data = data1 - data2

bounds = [0,0.25,0.5,0.75,1,1.5,3,6,12,24]
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))
cmap1 = ListedColormap(colors)
norm1 = BoundaryNorm(bounds,ncolors=cmap1.N,clip=False)

bounds = [0,0.2,0.4,0.6,0.8,1,2,4,8,16]
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds))))
cmap1 = ListedColormap([colors[0],colors[1],colors[2],colors[3],colors[4],
                        colors[5],colors[5],
                        colors[6],colors[6],
                        colors[7],colors[7],
                        colors[8],colors[8]])
norm1 = SymLogNorm(1,linscale=1.25,vmin=0,vmax=16,base=2)
transformed_lats = ccrs.Robinson(0).transform_points(ccrs.PlateCarree(), np.zeros(len(data.lat)), data.lat)[:, 1]

plot_figure(data,cmap1,norm1,transformed_lats)