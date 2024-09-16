
from modelClass3 import CustomLSTM
import torch.nn as nn
import torch as th
import joblib
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from tint.attr import DynaMask
from tint.attr.models import MaskNet

th.manual_seed(16)

class ModifiedLSTMModel(CustomLSTM):
    def __init__(self,input_size, out_dim,hidden_size, init_zero, forecast_steps):
        super(ModifiedLSTMModel, self).__init__(input_size, out_dim,hidden_size, init_zero, forecast_steps)
    
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dtype=th.float,
                batch_first=True)
    
    def forward(self,seq):
        hidden, (_, _) = self.lstm(seq,(th.zeros(self.num_layers, seq.shape[0], self.hidden_size, dtype=th.float),
                                        th.zeros(self.num_layers, seq.shape[0], self.hidden_size, dtype=th.float))) # hidden has shape (seq_length,batch_size,hidden_size), hidden[-1].shape = (batch_size,hidden_size)
        out = self.dropout(hidden[:,-1,:])
        out = self.linear(out)
        out = out.view(seq.shape[0],self.forecast_steps,-1) # formatting
        return out

model_address = '/home/mfroelich/Thesis/LSTM_final/models/model_64_1e-06.pth'
data_address = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/figure_dir/plots/'


if __name__ == '__main__':

    model = ModifiedLSTMModel(3,3,64,True,8)
    model.load_state_dict(th.load(model_address))

    scalers = [joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdv.gz'), 
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerAdiab.gz'), 
            joblib.load('/home/mfroelich/Thesis/LSTM_results/scalerDiab.gz')]

    ds = xr.open_dataset(data_address)[['adv','adiab','diab']]
    ds = ds.fillna(0)

    lat = 38
    lon = 155
    years = range(2009,2021)
    subselect = ds.sel({'lat':lat,'lon':lon,'year':years},drop=True)

    adv = scalers[0].transform(subselect['adv'].values.reshape(-1,1)).reshape(12,121)
    adiab = scalers[1].transform(subselect['adiab'].values.reshape(-1,1)).reshape(12,121)
    diab = scalers[2].transform(subselect['diab'].values.reshape(-1,1)).reshape(12,121)

    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[1,1,1,1])#[0.2,1,0.2,1,0.2,1])
    colors = ['#9FE09F','#F09D97','#8BCBD6','#44DB00','#D63D34','#2575D6']
    names = ['adv','adiab','diab']
    labels = ['a','b','c','d','e','f','g','h','i','j','k','l']
    x_ticks = [0, 40, 80, 96, 112, 120]
    x_labels = ['15d', '10d', '5d', '3d', '24h', '0']

    model.eval() 

    in_data = []
    true_data = []
    for k, year in enumerate(years):
        array = np.stack((adv[k,:],adiab[k,:],diab[k,:]),axis=-1) # shape is (121,3)
        in_data.append(array[range(113),:])
        true_data.append(array[range(113,121),:])

    batched_in = np.stack(in_data,axis=0)
    truths = np.stack(true_data,axis=0)

    #explainer = DynaMask(model)
    #mask = MaskNet(model,perturbation='fade_moving_average_window',deletion_mode=True,initial_mask_coef = 0)
    
    #attr = explainer.attribute(th.Tensor(batched_in),mask_net=mask)
    
    with th.no_grad():
        ypred = model(th.Tensor(batched_in))

    for k, year in enumerate(years):

        x = batched_in[k,:,:]
        y = truths[k,:,:]
        pred = ypred[k,:,:].detach()
        #attr_to_plot = np.concatenate((attr[k].squeeze().permute(0,1).numpy(),np.zeros((8,3))),axis=0).transpose()

        #print(attr_to_plot.shape)
        i = k // 3  # Row index
        j = k % 3   # Column index
        
        #ax_imshow = fig.add_subplot(gs[2*i,j])
        #ax_imshow.imshow(attr_to_plot,cmap='binary',norm=Normalize(0.3,0.65),aspect=4,interpolation='none')
        #ax_imshow.grid(which='minor', color='w', linestyle='-', linewidth=2)
        #ax_imshow.tick_params(which='minor', bottom=False, left=False)
        #ax_imshow.axis('off')
        #ax_imshow.set_xticks([])
        #if j == 0:
        #    ax_imshow.set_yticks([0,1,2])
        #    ax_imshow.set_yticklabels(['Adv','Adiab','Diab'])
        #else:
        #    ax_imshow.set_yticks([])

        #for l in range(3):
        #    ax_imshow.plot([best[l],best[l]],[0,2],color=colors[l+3])

        #pos = ax_imshow.get_position()
        #new_pos = [pos.x0, pos.y0-0.03, pos.width, pos.height]
        #ax_imshow.set_position(new_pos)

        ax_line = fig.add_subplot(gs[i, j]) #2*i+1
        for l in range(3):
            ax_line.plot([t for t in range(113)],scalers[l].inverse_transform(x[:,l].reshape(-1,1)),color=colors[l])
            ax_line.plot([t for t in range(113,121)],scalers[l].inverse_transform(y[:,l].reshape(-1,1)),color=colors[l],label=names[l])
            ax_line.plot([t for t in range(113,121)],scalers[l].inverse_transform(pred[:,l].reshape(-1,1)),color=colors[l+3])
            #ax_line.plot([best[l],best[l]],color=colors[l+3],linestyle='dashed')
        t=ax_line.text(0.05,0.95,f'({labels[k]}) {year}',ha='left',va='top', transform=ax_line.transAxes)    
        t.set_bbox(dict(facecolor='white', edgecolor='black'))
        ax_line.set_xticks(x_ticks)
        ax_line.set_xlim([0,120])
        ax_line.set_xticklabels(x_labels)

    legend_ax = fig.add_axes([0.3, 0.05, 0.4, 0.05])  # [left, bottom, width, height] for the legend
    legend_ax.axis('off')  # Hide the axes for the empty legend container
    legend_lines = [plt.Line2D([0], [0], color=color, lw=2) for color in colors[-3:]]  # Custom lines for legend
    legend_ax.legend(legend_lines, names, loc='center', ncol=3, frameon=False)

    #cbar_ax = fig.add_axes([0.4, 0.01, 0.4, 0.02])  # Position for the colorbar: [left, bottom, width, height]
    #fig.colorbar(ScalarMappable(norm=Normalize(0.3,0.65), cmap='binary'), cax=cbar_ax, orientation='horizontal')
    fig.savefig(result_dir+f'explained_{lat}_{lon}.png',bbox_inches='tight')
    plt.close()