import os
import torch
import numpy as np
import xarray as xr
from dataLoadersXR2 import TX1day_train, TX1day_test
from modelClass3 import CustomLSTM
from torch.utils.data import DataLoader
from torch import nn
import dask 
import joblib
import time

from dask.diagnostics import ProgressBar

address = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_results/'

model_path = os.path.join(result_dir, 'model_64_1e-06.pth')
#model = CustomLSTM(3,3,64,True,8)
#model.load_state_dict(model_path)
model = torch.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_2e6lr_verifydropout/models/model_48_1e-06.pth')
model.eval()  # Set the model to evaluation mode

n_cpus = 32 # Number of CPUs to use for parallelization

def mse(predictions, labels):
    return np.mean((predictions - labels) ** 2)

def lossing(inadv,inadiab,indiab,
         loss=mse,
         model=model): # truth and pred are np.arrays of shape (8,)
    
    labels = np.stack((inadv[-8:], inadiab[-8:], indiab[-8:]), axis=-1).reshape(8,1,-1) 
    input = torch.Tensor(np.stack((inadv[:113], inadiab[:113], indiab[:113]), axis=-1)).reshape(113,1,-1)

    with torch.no_grad():
        predictions, _ = model(input)  # Make predictions using the model

    predictions = predictions.numpy()
    return loss(predictions, labels).item(), loss(predictions[-1,:,:], labels[-1,:,:]).item(), loss(predictions[:,:,0],labels[:,:,0]), loss(predictions[:,:,1], labels[:,:,1]).item(), loss(predictions[:,:,2], labels[:,:,2]).item()

if __name__ == '__main__':

    start = time.time()

    ds = xr.open_dataset(address)[['adv','adiab','diab']]
    ds = ds.fillna(0)

    adv = ds['adv'].values
    adiab = ds['adiab'].values
    diab = ds['diab'].values

    ProgressBar().register()

    mse = xr.apply_ufunc(lossing,
                        adv,adiab,diab,
                        input_core_dims=[['trajtime'],['trajtime'],['trajtime']],
                        output_core_dims=[[], [], [], [], []],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=['float', 'float', 'float', 'float', 'float'])

    df = xr.Dataset({"MSELoss_forecast":mse[0],"MSELoss_tanom":mse[1],"MSEadv":mse[2],"MSEadiab":mse[3],"MSEdiab":mse[4]}).compute()

    # Save the xarray dataset to a file (NetCDF format)
    df.to_netcdf(result_dir+'trial_inference')
    print(f'Elapsed: {(time.time()-start)/3600} hours')
