import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point
from shapely.vectorized import contains
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm, PowerNorm
from matplotlib import cm

# Assume data is in the xarray dataset called `data`
data = xr.open_dataset('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_1e6lr/modelinference_48_0')

# Step 1: Load MSELoss_forecast and MSELoss_final
mse_forecast = data['MSELoss_forecast'].mean(dim='year')  # Averaged over year
mse_final = data['MSELoss_tanom'].mean(dim='year')

print(mse_forecast.max().values)
print(mse_forecast.median().values)
print(mse_forecast.min().values)
print('')
print(mse_final.max().values)
print(mse_final.median().values)
print(mse_final.min().values)

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

# Step 2: Create the lat/lon meshgrid for checking land/ocean
lons, lats = np.meshgrid(data.lon, data.lat)

# Step 3: Use cartopy's 'land' feature for land mask creation
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '110m')

# We need to concatenate all geometries (polygons) for land into one large geometry for fast processing
land_geometries = list(land_feature.geometries())

# Step 4: Use shapely.vectorized.contains to apply the land mask to the entire lat/lon grid
# Check which points in the lat/lon grid are contained within any of the land polygons
land_mask = np.zeros(lons.shape, dtype=bool)
for geom in land_geometries:
    land_mask |= contains(geom, lons, lats)

# Ocean mask is the inverse of the land mask
ocean_mask = ~land_mask

transformed_lats = crs.transform_points(ccrs.PlateCarree(), np.zeros(len(data.lat)), data.lat)[:, 1]

# Step 5: Initialize figure and axes
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2,1, height_ratios=[1, 1])

bounds = [0,0.25,0.5,0.75,1,1.5,3,6,12,24]
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))
cmap1 = ListedColormap(colors)
norm1 = BoundaryNorm(bounds,ncolors=cmap1.N,clip=False)

#cmap1 = 'Spectral_r'
#norm1 = LogNorm(vmin=0.05,vmax=40)


# all plots:
ax0 = fig.add_subplot(gs[0,0], projection=crs)
#ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0], projection=crs)
#ax3 = fig.add_subplot(gs[1,1])

# [0, 0] Contour plot for MSELoss_forecast
cf = ax0.contourf(data.lon, data.lat, mse_forecast, transform=trans, cmap=cmap1, norm=norm1, extend='max', levels=bounds)
ax0.coastlines(resolution='110m',color='black')
gl = ax0.gridlines(draw_labels={'left':True,'right':False,'top':True,'bottom':False},ylocs=range(-60,61,30),linewidth=1, color='gray', alpha=0.5, linestyle='--')
t=ax0.text(-0.1,1.1,'(a) Forecast MSELoss',ha='left',va='top', transform=ax0.transAxes)
t.set_bbox(dict(facecolor='white', edgecolor='black'))

# [0, 1] Line plot over land and ocean for MSELoss_forecast
mse_ocean = mse_forecast.where(ocean_mask).mean(dim='lon')
mse_land = mse_forecast.where(land_mask).mean(dim='lon')

ax1 = fig.add_axes([0.9, 0.535, 0.15, 0.35])
ax1.plot(mse_ocean, transformed_lats,   label='Ocean', color='blue',linewidth=2)
ax1.plot(mse_land, transformed_lats,   label='Land', color='green',linewidth=2)
ax1.set_ylabel('')
ax1.set_xlabel('')
ax1.legend()
ax1.set_xlim([0,40])
ax1.set_ylim([transformed_lats.min(), transformed_lats.max()])
ax1.set_yticks([])               # Remove y-axis ticks
ax1.tick_params(left=False)      # Disable y-axis tick marks


# [1, 0] Contour plot for MSELoss_final
cf = ax2.contourf(data.lon, data.lat, mse_final, transform=trans, cmap=cmap1, norm=norm1, extend='max', levels=bounds)
ax2.coastlines(resolution='110m',color='black')
gl = ax2.gridlines(draw_labels={'left':True,'right':False,'top':False,'bottom':True},ylocs=range(-60,61,30),linewidth=1, color='gray', alpha=0.5, linestyle='--')
t=ax2.text(-0.1,1.1,r'(a) Final $T^\prime$ MSELoss',ha='left',va='top', transform=ax2.transAxes)
t.set_bbox(dict(facecolor='white', edgecolor='black'))

# [1, 1] Line plot over land and ocean for MSELoss_final
mse_final_ocean = mse_final.where(ocean_mask).mean(dim='lon')
mse_final_land = mse_final.where(land_mask).mean(dim='lon')

ax3 = fig.add_axes([0.9, 0.11, 0.15, 0.35])
ax3.plot(mse_final_ocean, transformed_lats, label='Ocean', color='blue',linewidth=2)
ax3.plot(mse_final_land, transformed_lats, label='Land', color='green',linewidth=2)
ax3.set_ylabel('')
ax3.set_xlabel('MSE')
ax3.set_xlim([0,40])
ax3.set_ylim([transformed_lats.min(), transformed_lats.max()])
ax3.set_yticks([])               # Remove y-axis ticks
ax3.tick_params(left=False)      # Disable y-axis tick marks

cbar = fig.add_axes([0, 0.2, 0.025, 0.6])
clb = plt.colorbar(cm.ScalarMappable(cmap=cmap1,norm=norm1),cax=cbar,extend='max',
             label = None, orientation='vertical',boundaries = bounds) #,spacing='proportional')
clb.ax.set_title('MSE')

plt.savefig('/home/mfroelich/Thesis/figure_dir/plots/map_of_mse_48_0_03dropout',bbox_inches='tight')
plt.close()