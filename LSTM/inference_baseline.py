import numpy as np
import xarray as xr
from torch import nn
import time
import joblib
import dask

address = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_results/'

n_cpus = 60 # Number of CPUs to use for parallelization
dask.config.set(scheduler='threads', num_workers=n_cpus)

scalers = [joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz'),
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz'),
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')]

def loss(A,B):
    return ((A - B)**2).mean()

def lossing(truthadv,truthadiab,truthdiab,
         predadv,predadiab,preddiab,
         loss=loss): # truth and pred are np.arrays of shape (8,)
    truth = np.stack((truthadv, truthadiab, truthdiab), axis=-1)
    pred = np.stack((predadv, predadiab, preddiab), axis=-1)
    return loss(truth,pred), loss(truth[-1,:],pred[-1,:]), loss(truth[:,0],pred[:,0]), loss(truth[:,1],pred[:,1]), loss(truth[:,2],pred[:,2])

if __name__ == '__main__':

    start = time.time()

    ds = xr.open_dataset(address,chunks={'years': 21, 'lon': 91, 'lat': 181,'trajtime':121})[['adv','adiab','diab']]
    ds.coords['trajtime'] = np.arange(121)
    ds = ds.fillna(0)

    adv = xr.DataArray(scalers[0].transform(ds['adv'].values.reshape(-1,1)).reshape(ds['adv'].shape),
                                     coords=ds.coords,dims=ds.adv.dims)
    adiab = xr.DataArray(scalers[1].transform(ds['adiab'].values.reshape(-1,1)).reshape(ds['adv'].shape),
                                     coords=ds.coords,dims=ds.adv.dims)
    diab = xr.DataArray(scalers[2].transform(ds['diab'].values.reshape(-1,1)).reshape(ds['adv'].shape),
                                     coords=ds.coords,dims=ds.adv.dims)

    truth = {'trajtime':[113,114,115,116,117,118,119,120]}
    adv_truth = adv.loc[truth]
    adiab_truth = adiab.loc[truth]
    diab_truth = diab.loc[truth]

    pred = {'trajtime':list(np.repeat(112,8))} # feed-forward
    adv_pred = adv.loc[pred]
    adiab_pred = adiab.loc[pred]
    diab_pred = diab.loc[pred]

    # reset coordinates so they match
    adv_pred['trajtime'] = range(113,121)
    adiab_pred['trajtime'] = range(113,121)
    diab_pred['trajtime'] = range(113,121)

    mse = xr.apply_ufunc(lossing,
                        adv_truth,adiab_truth,diab_truth,
                        adv_pred,adiab_pred,diab_pred,
                        input_core_dims=[['trajtime'],['trajtime'],['trajtime'],
                                         ['trajtime'],['trajtime'],['trajtime']],
                        output_core_dims=[[], [], [], [], []],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=['float', 'float', 'float', 'float', 'float'])

    df = xr.Dataset({"MSELoss_forecast":mse[0],"MSELoss_tanom":mse[1],"MSEadv":mse[2],"MSEadiab":mse[3],"MSEdiab":mse[4]}).compute()

    # Save the xarray dataset to a file (NetCDF format)
    df.to_netcdf(result_dir+'baseline_inference_trans')
    print(f'Elapsed: {(time.time()-start)/3600} hours')