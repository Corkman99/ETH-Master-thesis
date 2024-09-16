import xarray as xr
import pandas as pd
import time
import sys
import numpy as np
import math

# Function to calculate latitude length
def latitude_length(lat):
    lat = math.radians(lat)
    return (111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) - 0.0023 * math.cos(6 * lat)) / 1000

# Function to calculate longitude length
def longitude_length(lat):
    lat = math.radians(lat)
    return (111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)) / 1000

def average_longitude_distance(lat_min, lat_max, step_size=0.5):
    lat = lat_min
    total_distance = 0
    count = 0
    
    while lat <= lat_max:
        total_distance += longitude_length(lat)
        lat += step_size
        count += 1
    
    average_length = total_distance / count
    return average_length

# Function to compute the average latitude distance over a range of longitudes
def average_latitude_distance(lon_min, lon_max, step_size=0.5):
    lon = lon_min
    total_distance = 0
    count = 0
    
    while lon <= lon_max:
        total_distance += latitude_length(lon)
        lon += step_size
        count += 1
    
    average_length = total_distance / count
    return average_length

##### -------------------------------------------------------------------------

# dataset to be filled:
latitudes = np.arange(-90, 90.5, 0.5)
longitudes = np.arange(-180, 180.5, 0.5)

# Create an empty DataArray
ds = xr.DataArray(np.nan, coords=[latitudes, longitudes], dims=["lat", "lon"])

lat = len(ds['lat'])
lon = len(ds['lon'])

# generate indicies for validation set according to the following test regions
train_regions = {'T11':[-180,45,-90,90],'T13':[0,45,90,90],
           'T22':[-90,0,0,45],'T24':[90,0,180,45],
           'T31':[-180,-45,-90,0],'T33':[0,-45,90,0],
           'T42':[-90,-90,0,-45],'T44':[90,-90,180,-45]}

full_test_regions = {'T12':[-90,45,0,90],'T14':[90,45,180,90],
           'T21':[-180,0,-90,45],'T23':[0,0,90,45],
           'T32':[-90,-45,0,0],'T34':[90,-45,180,0],
           'T41':[-180,-90,-90,-45],'T43':[0,-90,90,-45]}

# Adjusted such that only inner validation region points are sampled.
# This ensures that testing is done outside of regions that are correlated. 
# We use formation distance to inform the gap that we leave between the validation 
# boundary and the testing boundary.

indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
xr_dist = xr.open_dataset(indir + '/' + infile).mean(dim='lev',skipna=True)['dist']

average_dist = []
for values in full_test_regions.values():
    lat_vals = np.arange(values[1],values[3],0.5)
    lon_vals = np.arange(values[0],values[2],0.5)
    average_dist.append(xr_dist.sel({'lat':list(lat_vals),'lon':list(lon_vals)},drop=True).mean().item()/1000)

average_long_dist = [average_longitude_distance(x[1],x[3]) for x in full_test_regions.values()] 
average_lat_dist = [average_latitude_distance(x[1],x[3]) for x in full_test_regions.values()]

decrease_lat = [np.ceil(max(average_dist)/(4*b)).item() for b in average_lat_dist] # max is 19.05944...
decrease_lon = [np.ceil(max(average_dist)/(4*b)).item() for b in average_long_dist] # max is 51.0895...

final_test_regions = dict()
for i, reg in enumerate(full_test_regions.keys()):
    list_a = full_test_regions[reg]
    list_b = [decrease_lon[i],decrease_lat[i],-decrease_lon[i],-decrease_lat[i]]
    final_test_regions[reg] = [a+b for a,b in zip(list_a,list_b)]

# Fill dataarray
for i, (key, box) in enumerate(train_regions.items()):
    min_lon, min_lat, max_lon, max_lat = box
    
    # Select the appropriate region and set the value
    ds.loc[min_lat:max_lat, min_lon:max_lon] = 0

for i, (key, box) in enumerate(final_test_regions.items()):
    min_lon, min_lat, max_lon, max_lat = box
    
    # Select the appropriate region and set the value
    ds.loc[min_lat:max_lat, min_lon:max_lon] = 1

# Plotting
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap

outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection=crs)

# Define the color map and levels
cmap = ListedColormap(['#BAE68E','#EBAB98'])

# Create a contourf plot
contour = ax.contourf(ds['lon'].values, ds['lat'].values, ds.values, levels=[-0.1,0.9,1.1], cmap=cmap, extend='neither',transform=trans)
ax.coastlines(resolution='110m', color='black')
ax.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_facecolor('lightgrey')
plt.tight_layout()
plt.savefig(outdir+'train_test', bbox_inches='tight')
plt.close()

