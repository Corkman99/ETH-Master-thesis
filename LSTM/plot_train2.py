import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import joblib
import os
import xarray as xr
import random

from dataLoadersXR2 import TX1day_val
from modelClass3 import CustomLSTM

address = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_results/'

model_path = '/home/mfroelich/Thesis/LSTM_final/models/model_64_1e-06.pth'
model = CustomLSTM(3,3,64,True,8)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

scalers = [joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz'), 
           joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz'), 
           joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')]

ds = xr.open_dataset(address)[['adv','adiab','diab']]
ds = ds.fillna(0)

lat = 53.5
lon = 10
years = random.sample(range(1991,2021),9)
subselect = ds.sel({'lat':lat,'lon':lon,'year':years},drop=True)

def plot_train_examples(model, hidden, weight):
        fig, axes = plt.subplots(3,3, figsize=(20, 15))
        colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
        ax = 0

        with torch.no_grad():
            # 16 plots:
            for k, year in enumerate(years):
                
                year_vals =  subselect.sel({'year':year})
                array = np.stack((year_vals['adv'].values,year_vals['adiab'].values,year_vals['diab'].values),axis=-1)
                
                ins = array[range(113),:].reshape(113,1,3)
                outs = array[range(113,121),:].reshape(8,1,3)
                ypred, _ = model(torch.Tensor(np.stack((scalers[0].transform(ins[:,0,[0]]),
                                                       scalers[1].transform(ins[:,0,[1]]),
                                                       scalers[2].transform(ins[:,0,[2]])), axis=-1)))
                 
                i = k // 3  # Row index
                j = k % 3   # Column index
                ax = axes[i, j]

                x = ins.squeeze()
                y = outs.squeeze()
                pred = ypred.squeeze().detach()

                colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
                names = ['adv','adiab','diab']
                for l in range(3):
                    ax.plot([t for t in range(113)],x[:,l],color=colors[l])
                    ax.plot([t for t in range(113,121)],y[:,l],color=colors[l],label=names[l])
                    ax.plot([t for t in range(113,121)],scalers[l].inverse_transform(pred[:,l].reshape(-1,1)),color=colors[l+3])

                ax.set_title(f'{year}')
                
        axes[0,0].legend(loc='upper left')
        fig.savefig(result_dir+f'{lat}_{lon}.png',bbox_inches='tight')
        plt.close()

model_path = '/home/mfroelich/Thesis/LSTM_final/models/model_64_1e-06.pth'
model = CustomLSTM(3,3,64,True,8)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

plot_train_examples(model,64,1e-06)