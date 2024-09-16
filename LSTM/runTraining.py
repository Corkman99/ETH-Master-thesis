"""
Script to run training
"""

import torch 
import numpy as np
import time 

from trainingClass import LSTMTrainer
from modelClass import CustomLSTM
from dataLoaders import TX1day_train, TX1day_val, collater
from dataLoadersXR import TX1day_train, TX1day_val

torch.manual_seed(16)
torch.set_num_threads(60)

#train_dir = '/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/train/'
#test_dir = '/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/val/'
train_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
test_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_results/high_lr/'

forecast_steps = 8
lr = 0.001
batch_size = 32
max_epochs = 5
patience = 1

param_hidden = [128]
param_l2 = [0.1,1e-5,1e-4,1e-3]
year_range = [x for x in range(1980,1986)]
example_locations = list(np.linspace(0,60000,9).astype(int))

if __name__ == "__main__":

    start = time.time()
    train_dataset = TX1day_train(train_dir, forecast_steps = forecast_steps, year_subset = year_range, location_subset = None)
    test_dataset = TX1day_val(test_dir, forecast_steps = forecast_steps, year_subset = year_range, location_subset = None) #, scalers=train_dataset.scalers)
    examples = TX1day_val(test_dir, forecast_steps = forecast_steps, year_subset = [2020], location_subset = None)

    trainer = LSTMTrainer(model_class=CustomLSTM,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        examples=examples,examples_idx=example_locations,
                        result_dir=result_dir,
                        forecast_steps=forecast_steps,init_zero=True,
                        batch_size=batch_size, lr=lr)
    
    print(f'Dataloaders and Training class initialized: elapsed {(time.time()-start):.0f} seconds')

    start = time.time()
    trainer.train(num_epochs=max_epochs, hidden_units_list=param_hidden, weight_decay_list=param_l2, patience=patience)
    print(f'Full training: elapsed {(time.time()-start)/3600:.0f} hours')

    # Plot cross-validation parameters
    trainer.plot_cross_heatmap(param_hidden,param_l2,'Average Validation MSELoss over K-folds','crossval.png')