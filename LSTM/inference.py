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

dask.config.set(**{'array.slicing.split_large_chunks': True})
torch.set_num_threads(60)

train_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
test_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_final/'

if __name__ == '__main__':

    # Load the saved model
    model_path = os.path.join(result_dir, 'models/model_64_1e-06.pth')
    model = CustomLSTM(3,3,64,True,8)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    scalers = [joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz'),
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz'),
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')]

    # Initialize the test dataset
    test_dataset = TX1day_test(
        address=test_dir, 
        forecast_steps=8, 
        year_subset=[x for x in range(1991, 2021)], 
        scalers=scalers  # Assuming scalers are available from training
    )

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define the loss function (MSE)
    loss_fn = nn.MSELoss()

    # Create an xarray dataset for storing MSE Loss
    ds = xr.Dataset(
        {
            "MSELoss_forecast": (("lat", "lon", "year"), np.full((361, 721, len(test_dataset.year_subset)), np.nan)),
            "MSELoss_tanom":(("lat", "lon", "year"), np.full((361, 721, len(test_dataset.year_subset)), np.nan)),
            "MSEadv":(("lat", "lon", "year"), np.full((361, 721, len(test_dataset.year_subset)), np.nan)),
            "MSEadiab":(("lat", "lon", "year"), np.full((361, 721, len(test_dataset.year_subset)), np.nan)),
            "MSEdiab":(("lat", "lon", "year"), np.full((361, 721, len(test_dataset.year_subset)), np.nan)),
        },
        coords={
            "lat": np.arange(-90, 90.5, 0.5),    # Latitude range
            "lon": np.arange(-180, 180.5, 0.5),  # Longitude range
            "year": test_dataset.year_subset,    # Years
        }
    )

    start = time.time()
    for batch in test_loader:
        inputs = batch['input'].squeeze().reshape(113,1,-1) # Shape: (121-forecast_steps,1, 3)
        labels = batch['label'].squeeze().reshape(8,1,-1) # Shape: (forecast_steps, 1, 3)
        idx = batch['idx']       # Tuple: (lat, lon, year)

        # Make prediction
        with torch.no_grad():
            predictions, _ = model(inputs)  # Make predictions using the model

        # location
        lat, lon, year = idx[0].item(), idx[1].item(), idx[2].item()

        ds['MSELoss_forecast'].loc[lat, lon, year] = loss_fn(predictions, labels).item()
        ds['MSELoss_tanom'].loc[lat, lon, year] = loss_fn(predictions[-1,:,:], labels[-1,:,:]).item()
        ds['MSEadv'].loc[lat, lon, year] = loss_fn(predictions[:,:,0], labels[:,:,0]).item()
        ds['MSEadiab'].loc[lat, lon, year] = loss_fn(predictions[:,:,1], labels[:,:,1]).item()
        ds['MSEdiab'].loc[lat, lon, year] = loss_fn(predictions[:,:,2], labels[:,:,2]).item()

    # Save the xarray dataset to a file (NetCDF format)
    ds.to_netcdf(result_dir+'modelinference_final')
    print(f'Elapsed: {(time.time()-start)/3600} hours')