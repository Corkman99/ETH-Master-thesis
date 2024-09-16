import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import joblib
import os
import random

from dataLoadersXR2 import TX1day_val
from modelClass3 import CustomLSTM

result_dir = '/home/mfroelich/Thesis/LSTM_final/'
train_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
forecast_steps = 8
location_subset=None
scalers = [joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz'), 
           joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz'), 
           joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')]

examples = TX1day_val(train_dir, forecast_steps = forecast_steps, year_subset = [2020], location_subset = location_subset, scalers = scalers)
example_locations = list(np.linspace(5000,60000,9).astype(int))
example_locations = random.sample(range(0,60000),9)

example_list = []
for idx in example_locations:
        item = examples.__getitem__(idx)
        example_list.append({'input':torch.Tensor(item['input']),'label':torch.Tensor(item['label']),'idx':item['idx']})

def plot_train_examples(model, hidden, weight):
        fig, axes = plt.subplots(3,3, figsize=(20, 15))
        colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
        ax = 0 
        criterion = nn.MSELoss()

        model.eval()
        with torch.no_grad():
            # 16 plots:
            for k, batch in enumerate(example_list):
                
                ins = batch['input'].reshape((batch['input'].shape[0],1,-1))
                outs = batch['label'].reshape((batch['label'].shape[0],1,-1))
                ypred_batch, _ = model(ins)
                ypred_batch = ypred_batch.reshape((batch['label'].shape[0],1,-1))
                loss = criterion(ypred_batch,outs)
                 
                i = k // 3  # Row index
                j = k % 3   # Column index
                ax = axes[i, j]

                x = ins.squeeze()
                y = outs.squeeze()
                pred = ypred_batch.squeeze().detach()

                colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
                for l in range(3):
                    ax.plot([t for t in range(121 - forecast_steps)],scalers[l].inverse_transform(x[:,l].reshape(-1,1)),color=colors[l])
                    ax.plot([t for t in range(121 - forecast_steps,121)],scalers[l].inverse_transform(y[:,l].reshape(-1,1)),color=colors[l],label='True')
                    ax.plot([t for t in range(121 - forecast_steps,121)],scalers[l].inverse_transform(pred[:,l].reshape(-1,1)),color=colors[l+3],label='Pred')

                ax.set_title(f'Location: {batch['idx']} - Loss: {loss.item():.3f}')
                
        ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        fig.savefig(result_dir+f'train_examples2_{hidden}_{weight}.png',bbox_inches='tight')
        plt.close()

model_path = '/home/mfroelich/Thesis/LSTM_final/models/model_64_1e-06.pth'
model = CustomLSTM(3,3,64,True,8)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

plot_train_examples(model,64,1e-06)