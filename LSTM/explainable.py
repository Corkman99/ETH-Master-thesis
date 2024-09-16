from captum.attr import IntegratedGradients
import os
import torch
import xarray as xr
import numpy as np
import joblib


result_dir = '/home/mfroelich/Thesis/LSTM_results/robust_25_all/'
test_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'

ds = xr.open_dataset(test_dir,chunks={'year':1,'lat':361,'lon':721,'trajtime':121})[['adv','adiab','diab']]
ds = ds.fillna(0)

adv_scaler = joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz')
adiab_scaler = joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz')
diab_scaler = joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')

model_path = os.path.join(result_dir, 'models/model_64_0.pth')
model = torch.load(model_path)

# Calc average event
# ----------------------------------

# Local level explanations
# ----------------------------------
# extract one example
example = ds.sel({'year':1990,'lat':45,'lon':-135},drop=True)
example = np.stack((example['adv'].values,example['adiab'].values,example['diab'].values),axis=0)

print(example.shape)


