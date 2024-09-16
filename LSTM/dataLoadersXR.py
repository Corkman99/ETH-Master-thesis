"""
Dataset for Training and Validation of Timeseries LSTM models

We give option to subsample specific locations and years, different window and forecast-steps values, 
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import xarray as xr
import math

class TX1day_train(Dataset):
    def __init__(self, 
                 address, #'/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
                 forecast_steps = 2,
                 year_subset = None, location_subset = None):
        
        self.fsteps = forecast_steps
        self.scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(3)]
        self.year_subset = year_subset

        ds = xr.open_dataset(address,chunks={'year':1,'lat':361,'lon':721,'trajtime':121})[['adv','adiab','diab']]
        ds = ds.fillna(0)
        train_regions = self.get_boxes()
        
        adv = []
        adiab = []
        diab = []
        idx = []
        i = 0
        for region_name, region_box in train_regions.items():
            min_lon, min_lat, max_lon, max_lat = region_box

            # Select region based on the given box and year_list
            subset = ds.sel(
                lon=np.arange(min_lon, max_lon,0.5),
                lat=np.arange(min_lat, max_lat,0.5),
                year=year_subset
            )

            # Flatten lat, lon, and years to create the "sample" dimension
            flattened_data = subset.stack(sample=('lat', 'lon', 'year')).transpose('sample', 'trajtime')

            adv.append(flattened_data['adv'].values)
            adiab.append(flattened_data['adiab'].values)
            diab.append(flattened_data['diab'].values)
            idx.extend(flattened_data['sample'].values)

            if location_subset is not None and i >= location_subset:
                break

        # Concatenate all regions along the "sample" dimension for each variable
        self.adv = np.concatenate(adv, axis=0)
        self.adiab = np.concatenate(adiab, axis=0)
        self.diab = np.concatenate(diab, axis=0)
        self.idx = np.array(idx)

        # Apply the scalers to each feature separately
        #self.adv = self.scalers[0].fit_transform(self.adv)
        #self.adiab = self.scalers[1].fit_transform(self.adiab)
        #self.diab = self.scalers[2].fit_transform(self.diab)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # Stack the variables together to form the feature array
        variables = np.stack((self.adv[idx,:], 
                              self.adiab[idx,:],
                              self.diab[idx,:]), axis=1)  # Shape: (121, 3)
        variables = torch.Tensor(variables)
        inputs = variables[:-self.fsteps, :]
        labels = variables[-self.fsteps:, :]

        return {'input': inputs, 'label': labels, 'idx':self.idx[idx]}

    def get_boxes(self):
        train_regions = {'T11':[-180,45,-90,90],'T13':[0,45,90,90],
                'T22':[-90,0,0,45],'T24':[90,0,180,45],
                'T31':[-180,-45,-90,0],'T33':[0,-45,90,0],
                'T42':[-90,-90,0,-45],'T44':[90,-90,180,-45]}
        return train_regions
    
    def idx_to_coords(self,index):
        """
        Maps the flattened index back to the original (lat, lon, year) coordinates.
        
        Parameters:
        index (int): The index in the flattened sample dimension.
        
        Returns:
        tuple: The corresponding (lat, lon, year) tuple.
        """
        if index >= len(self):
            raise IndexError("Index out of range for the mapped coordinates.")
        
        return [np.arange(-90,90,0.5)][self.idx[index][0]].item(), \
               [np.arange(-180,180,0.5)][self.idx[index][1]].item(), \
               [self.year_subset][self.idx[index][2]].item()

class TX1day_val(Dataset):
    def __init__(self, 
                 address, #'/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
                 forecast_steps = 2,
                 year_subset = None, location_subset = None):
        
        self.fsteps = forecast_steps
        self.scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(3)]
        self.year_subset = year_subset

        ds = xr.open_dataset(address,chunks={'year':1,'lat':361,'lon':721,'trajtime':121})[['adv','adiab','diab']]
        ds = ds.fillna(0)
        train_regions = self.get_boxes()
        
        adv = []
        adiab = []
        diab = []
        idx = []
        i = 0
        for region_name, region_box in train_regions.items():
            min_lon, min_lat, max_lon, max_lat = region_box

            # Select region based on the given box and year_list
            subset = ds.sel(
                lon=np.arange(min_lon, max_lon,0.5),
                lat=np.arange(min_lat, max_lat,0.5),
                year=year_subset
            )

            # Flatten lat, lon, and years to create the "sample" dimension
            flattened_data = subset.stack(sample=('lat', 'lon', 'year')).transpose('sample', 'trajtime')

            adv.append(flattened_data['adv'].values)
            adiab.append(flattened_data['adiab'].values)
            diab.append(flattened_data['diab'].values)
            idx.extend(flattened_data['sample'].values)

            if location_subset is not None and i >= location_subset:
                break

        # Concatenate all regions along the "sample" dimension for each variable
        self.adv = np.concatenate(adv, axis=0)
        self.adiab = np.concatenate(adiab, axis=0)
        self.diab = np.concatenate(diab, axis=0)
        self.idx = np.array(idx)

        # Apply the scalers to each feature separately
        #self.adv = self.scalers[0].fit_transform(self.adv)
        #self.adiab = self.scalers[1].fit_transform(self.adiab)
        #self.diab = self.scalers[2].fit_transform(self.diab)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # Stack the variables together to form the feature array
        variables = np.stack((self.adv[idx,:], 
                              self.adiab[idx,:],
                              self.diab[idx,:]), axis=1)  # Shape: (121, 3)
        variables = torch.Tensor(variables)
        inputs = variables[:-self.fsteps, :]
        labels = variables[-self.fsteps:, :]
        return {'input': inputs, 'label': labels, 'idx':self.idx[idx]}

    def get_boxes(self):
        test_regions = {'T12': [-77.0, 50.0, -13.0, 85.0], 'T14': [103.0, 50.0, 167.0, 85.0], 
                        'T21': [-174.0, 5.0, -96.0, 40.0], 'T23': [6.0, 5.0, 84.0, 40.0], 
                        'T32': [-84.0, -40.0, -6.0, -5.0], 'T34': [96.0, -40.0, 174.0, -5.0], 
                        'T41': [-167.0, -85.0, -103.0, -50.0], 'T43': [13.0, -85.0, 77.0, -50.0]}
        return test_regions
    
    def idx_to_coords(self,index):
        """
        Maps the flattened index back to the original (lat, lon, year) coordinates.
        
        Parameters:
        index (int): The index in the flattened sample dimension.
        
        Returns:
        tuple: The corresponding (lat, lon, year) tuple.
        """
        if index >= len(self):
            raise IndexError("Index out of range for the mapped coordinates.")
        
        return [np.arange(-90,90,0.5)][self.idx[index][0]].item(), \
               [np.arange(-180,180,0.5)][self.idx[index][1]].item(), \
               [self.year_subset][self.idx[index][2]].item()
