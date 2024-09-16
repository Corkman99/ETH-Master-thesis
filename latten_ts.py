import xarray as xr
import pandas as pd
import time
import sys
import numpy as np
import math

comp = sys.argv[1]

def flat_to_multi(flat):
    year_idx = flat // (361 * 721)
    remainder = flat % (361 * 721)
    lat_idx = remainder // 721
    lon_idx = remainder % 721
    return (int(year_idx),int(lat_idx),int(lon_idx)) # output is index, not value. ie. 1 is for 2001

def multi_to_flat(multi): # input is tuple (year,lat,lon) are indicies, not values. ie. for 2001, input 1
    return int((multi[0] * 361 * 721) + (multi[1] * 721) + multi[2])

def from_box(box_coords): # lon_min, lat_min, lon_max, lat_max
    """
    output list containing coordinates containing within box with width and height specified by box_coords
    """
    out = []
    for y in range(41):
        lat_start = int((box_coords[1] + 90) / 0.5)
        lat_end = int((box_coords[3] + 90) / 0.5)
        for lat in range(lat_start,lat_end+1):
            start = multi_to_flat((y,lat,(box_coords[0] + 180) / 0.5))
            end = multi_to_flat((y,lat,(box_coords[2] + 180) / 0.5))
            out.extend([x for x in range(start,end+1)])

    return out

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

file = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
ds = xr.open_dataset(file,chunks={'year':1,'lat':361,'lon':721,'trajtime':121})[comp]
ds = ds.fillna(0)

y = len(ds['year'])
lat = len(ds['lat'])
lon = len(ds['lon'])
traj = len(ds['trajtime'])
tot = y * lat * lon

# generate indicies for validation set according to the following test regions
regions = {'arctic':[-120,80,-60,90],
           'mid_lat_namerica' : [-150,45,-120,60],
           'mid_lat_europe' : [-15,35,30,50],
           'asia' : [80,20,120,35],
           'tropical_africa' : [10,-10,40,10],
           'tropical_pacific' : [-180,-15,-140,15],
           'tropical_pacific2' : [140,-15,180,15],
           'mid_lat_samerica' : [-90,-45,-55,-25],
           'storm_tracks_s' : [-40,-60,-65,0],
           'new_zealand' : [165,-48,180,-32],
           'antarctic' : [-170,-90,-60,-60]}

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

print(final_test_regions)
quit()

# Divide datasets
training_indicies = []
for values in train_regions.values():
    vals = from_box(values)
    training_indicies.extend(vals)

testing_indicies = []
for values in final_test_regions.values():
    vals = from_box(values)
    testing_indicies.extend(vals)

#This saves by yearly files for each component
now = time.time()
for year in range(y):

    ds_np = ds.isel({'year':year},drop=True).to_numpy()
    ds_np = ds_np.reshape(traj,-1).T
    ran = range(year*(lat*lon),(year+1)*(lat*lon))
    ds_np = pd.DataFrame(ds_np,index=ran,columns=[x for x in range(traj)])
    
    training_indicies_sub = [num for num in training_indicies if num in ran]
    ds_np_train = ds_np.loc[training_indicies_sub]
    ds_np_train.to_parquet(f'/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/train/{year}_{comp}.parquet',compression=None)
    del ds_np_train

    testing_indicies_sub = [num for num in testing_indicies if num in ran]
    ds_np_val = ds_np.loc[testing_indicies_sub]
    ds_np_val.to_parquet(f'/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/val/{year}_{comp}.parquet',compression=None)
    del ds_np_val

    del ds_np

print(f'Saved {comp} in {(time.time()-now)/60} min')
