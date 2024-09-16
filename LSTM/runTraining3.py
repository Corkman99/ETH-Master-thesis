"""
Script to run training
"""

import torch 
import numpy as np
import time 
import random

from trainingClass2 import LSTMTrainer
from modelClass3 import CustomLSTM
from dataLoadersXR2 import TX1day_train, TX1day_val

torch.manual_seed(16)
torch.set_num_threads(60)

#train_dir = '/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/train/'
#test_dir = '/net/litho/atmosdyn2/mfroelich/ML_data_format_notdiff/val/'
train_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
test_dir = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
result_dir = '/home/mfroelich/Thesis/LSTM_final/'

forecast_steps = 8
lr = 2e-6 # 0.000002 should be 2e-6
batch_size = 1
max_epochs = 15
patience = 3

param_hidden = [64]
param_l2 = [1e-6]

year_range = [x for x in range(1980,1991)]
location_subset = None
location_subset_val = None
example_locations = list(np.linspace(0,1000,9).astype(int))

if __name__ == "__main__":

    start = time.time()

    train_dataset = TX1day_train(train_dir, forecast_steps = forecast_steps, year_subset = year_range, location_subset = location_subset)
    test_dataset = TX1day_val(test_dir, forecast_steps = forecast_steps, year_subset = year_range, location_subset = location_subset_val, scalers=train_dataset.scalers)
    examples = TX1day_val(train_dir, forecast_steps = forecast_steps, year_subset = [1996], location_subset = None, scalers =train_dataset.scalers)

    trainer = LSTMTrainer(model_class=CustomLSTM,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        examples=examples,examples_idx=example_locations,
                        result_dir=result_dir,
                        forecast_steps=forecast_steps,init_zero=True,
                        batch_size=batch_size, lr=lr)
    
    np.save(f'{result_dir}losses/indicies_random_subsample.npy', np.array(location_subset))
    
    print(f'Dataloaders and Training class initialized: elapsed {(time.time()-start):.0f} seconds')

    start = time.time()
    trainer.train(num_epochs=max_epochs, hidden_units_list=param_hidden, weight_decay_list=param_l2, patience=patience)
    print(f'Full training: elapsed {(time.time()-start)/3600:.0f} hours')