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

class TX1day_train(Dataset):
    def __init__(self, 
                 address, 
                 forecast_steps = 2,
                 year_subset = None, location_subset = None):
        
        self.fsteps = forecast_steps

        adv = []
        adiab = []
        diab = []
        if location_subset is not None:
            for y in year_subset:
                adv.append(pd.read_parquet(f'{address}{y}_adv.parquet').iloc[location_subset])
                adiab.append(pd.read_parquet(f'{address}{y}_adiab.parquet').iloc[location_subset])
                diab.append(pd.read_parquet(f'{address}{y}_diab.parquet').iloc[location_subset])
        else: 
            for y in year_subset:
                adv.append(pd.read_parquet(f'{address}{y}_adv.parquet'))
                adiab.append(pd.read_parquet(f'{address}{y}_adiab.parquet'))
                diab.append(pd.read_parquet(f'{address}{y}_diab.parquet'))

        self.adv = pd.concat(adv,axis=0)
        self.adiab = pd.concat(adiab,axis=0)
        self.diab = pd.concat(diab,axis=0)
        
        self.scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(3)]

        # Apply the scalers to each feature separately
        self.adv = self.scalers[0].fit_transform(self.adv.values)
        self.adiab = self.scalers[1].fit_transform(self.adiab.values)
        self.diab = self.scalers[2].fit_transform(self.diab.values)

    def __len__(self):
        return len(self.adv)

    def __getitem__(self, idx):
        # Stack the variables together to form the feature array
        variables = np.stack((self.adv[idx], 
                              self.adiab[idx],
                              self.diab[idx]), axis=1)  # Shape: (121, 3)
        variables = torch.Tensor(variables)
        inputs = variables[:-self.fsteps, :]
        labels = variables[-self.fsteps:, :]

        return {'input': inputs, 'label': labels, 'idx': idx}


class TX1day_val(Dataset):
    def __init__(self, 
                 address, 
                 forecast_steps=2,
                 year_subset=None, location_subset=None,
                 scalers=None):  # Added scalers parameter
        
        self.fsteps = forecast_steps

        adv = []
        adiab = []
        diab = []
        if location_subset is not None:
            for y in year_subset:
                adv.append(pd.read_parquet(f'{address}{y}_adv.parquet').iloc[location_subset])
                adiab.append(pd.read_parquet(f'{address}{y}_adiab.parquet').iloc[location_subset])
                diab.append(pd.read_parquet(f'{address}{y}_diab.parquet').iloc[location_subset])
        else: 
            for y in year_subset:
                adv.append(pd.read_parquet(f'{address}{y}_adv.parquet'))
                adiab.append(pd.read_parquet(f'{address}{y}_adiab.parquet'))
                diab.append(pd.read_parquet(f'{address}{y}_diab.parquet'))
                
        self.adv = pd.concat(adv, axis=0)
        self.adiab = pd.concat(adiab, axis=0)
        self.diab = pd.concat(diab, axis=0)

        # Store the scalers passed from the training DataLoader
        self.scalers = scalers

        # Apply the scalers to the entire test set
        self.adv = self.scalers[0].transform(self.adv.values)
        self.adiab = self.scalers[1].transform(self.adiab.values)
        self.diab = self.scalers[2].transform(self.diab.values)

    def __len__(self):
        return len(self.adv)

    def __getitem__(self, idx):
        # Generate numpy ndarray from data for the given idx
        variables = np.stack((self.adv[idx], 
                              self.adiab[idx],
                              self.diab[idx]), axis=1)  # Shape: (120, 3)

        variables = torch.Tensor(variables)
        inputs = variables[:-self.fsteps, :]
        labels = variables[-self.fsteps:, :]

        return {'input': inputs, 'label': labels, 'idx': idx}
    
def collater(batch):

    images = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    # Use default_collate to collate images, labels, and indices
    images = default_collate(images)
    labels = default_collate(labels)
    indices = default_collate(indices)
    
    # Return the collated batch as a dictionary
    return {
        'input': images,
        'label': labels,
        'idx': indices
    }
