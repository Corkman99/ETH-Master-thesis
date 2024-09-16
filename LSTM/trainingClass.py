"""
Time-series Models - LSTM Baseline

This script contains the classes, the training and testing functions as well as utility functions
for visualisation of TX1day timeseries prediction. 
"""

import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

from dataLoaders import TX1day_val

class LSTMTrainer:
    def __init__(self, model_class, result_dir, # '/home/mfroelich/Thesis/LSTM_results/'
                 train_dataset, test_dataset, examples, examples_idx,
                 input_size=3, out_dim=3,forecast_steps=8,init_zero=True,
                 batch_size=32, lr=0.005):
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.result_dir = result_dir
        self.input_size = input_size
        self.out_dim = out_dim
        self.forecast_steps = forecast_steps
        self.init_zero = init_zero

        self.batch_size = batch_size
        self.lr = lr

        self.final_losses = None
        self.best_params = None
        self.best_model = None

        self.example_list = []
        for idx in examples_idx:
            item = examples.__getitem__(idx)
            self.example_list.append({'input':torch.Tensor(item['input']),'label':torch.Tensor(item['label']),'idx':item['idx']})

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'losses'), exist_ok=True)

    def train(self, num_epochs, hidden_units_list, weight_decay_list,patience):
    
        self.final_losses = {'train':np.zeros((len(weight_decay_list),len(hidden_units_list))),
                           'validation':np.zeros((len(weight_decay_list),len(hidden_units_list)))}

        best_val_loss = float('inf')

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        for h, hidden_size in enumerate(hidden_units_list):
            for w, weight_decay in enumerate(weight_decay_list):
                
                start = time.time()
                print(f'[Hidden = {hidden_size}, L2 = {weight_decay}]')

                model = self.model_class(input_size=self.input_size, out_dim = self.out_dim,
                                         hidden_size=hidden_size, init_zero=self.init_zero, forecast_steps=self.forecast_steps)

                optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)
                criterion = nn.MSELoss()

                combi_train_loss = [] # contains epoch averages
                combi_val_loss = []

                for epoch in range(num_epochs):
                    ep_start = time.time()

                    model.train()
                    train_loss = []
                    for i, batch in enumerate(train_loader):
                        optimizer.zero_grad()
                        ins = batch['input'].squeeze().permute((1, 0, 2))
                        outs = batch['label'].squeeze().permute((1, 0, 2))
                        ypred_batch = model(ins)
                        loss = criterion(ypred_batch, outs)
                        loss.backward()
                        optimizer.step()

                        train_loss.append(loss.item())

                    model.eval()
                    val_loss = []
                    with torch.no_grad():
                        for i, batch in enumerate(val_loader):
                            ins = batch['input'].permute((1, 0, 2))
                            outs = batch['label'].permute((1, 0, 2))
                            ypred_batch = model(ins)
                            loss = criterion(ypred_batch, outs)

                            val_loss.append(loss.item())

                    combi_train_loss.append(np.mean(train_loss))
                    combi_val_loss.append(np.mean(val_loss))
                        
                    if combi_val_loss[-1] == min(combi_val_loss):
                        no_improve_epochs = 0  # Reset early stopping counter
                    else:
                        no_improve_epochs += 1

                    if no_improve_epochs >= patience:
                        print(f'[Hidden = {hidden_size}, L2 = {weight_decay}] || Early stopping at epoch {epoch + 1}')
                        break

                    # Print statements
                    print(f'Epoch {epoch+1} / {num_epochs} ({(time.time()-ep_start)/60:.1f} min) : Train = {combi_train_loss[-1]:.4f}, Val = {combi_val_loss[-1]:.4f})')
                    
                # Save losses for this combi
                np.save(os.path.join(self.result_dir, 'losses/', f'{hidden_size}_{weight_decay}_train_losses.npy'), combi_train_loss)
                np.save(os.path.join(self.result_dir, 'losses/', f'{hidden_size}_{weight_decay}_val_losses.npy'), combi_val_loss)
                torch.save(model, os.path.join(self.result_dir, 'models/', f'model_{hidden_size}_{weight_decay}.pth'))

                # Compute average losses for this combi
                self.final_losses['train'][w,h] = combi_train_loss[-1] # mean over num_epochs x folds
                self.final_losses['validation'][w,h] = combi_val_loss[-1]

                if combi_val_loss[-1] < best_val_loss:
                    best_val_loss = combi_val_loss[-1]
                    self.best_model = model.state_dict()
                    self.best_params = {'hidden_size': hidden_size, 'weight_decay': weight_decay}

                self.plot_train_examples(model,hidden_size,weight_decay)
                self.plot_val_examples(model,hidden_size,weight_decay)
                
                print(f'Elpased: {(time.time()-start)/60:.4f} min')
                print('')

        # Save the best model and its parameters
        torch.save(self.best_model, os.path.join(self.result_dir, 'models/', 'best_model.pth'))
        with open(os.path.join(self.result_dir, 'models/', 'best_params.txt'), 'w') as f:
            f.write(str(self.best_params))

    def test(self,model=None, address=None):
        # Load the best model
        if model is None:
            model = self.model_class(input_size=self.input_size, out_dim = self.out_dim, hidden_size=self.best_params['hidden_size'],
                                     init_zero=self.init_zero, forecast_steps=self.forecast_steps)
            model.load_state_dict(torch.load(os.path.join(self.result_dir, 'models', 'best_model.pth')))
        else: 
            model.load_state_dict(torch.load(address))

        model.eval()

        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        criterion = nn.MSELoss()

        losses = []
        predictions = []
        true_values = []
        indices = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                ins = batch['input'].unsqueeze(1)
                outs = batch['label'].unsqueeze(1)
                ypred_batch = model(ins)
                loss = criterion(ypred_batch, outs).item()

                losses.append(loss)
                predictions.append(ypred_batch.squeeze().numpy())
                true_values.append(outs.squeeze().numpy())
                indices.append(batch['idx'].item())

        losses = np.array(losses)
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        indices = np.array(indices)

        # Plot top 5 best and worst cases
        sorted_indices = np.argsort(losses)
        best_indices = sorted_indices[:5]
        worst_indices = sorted_indices[-5:]

        fig, axs = plt.subplots(2, 5, figsize=(20, 8))

        for i, idx in enumerate(best_indices):
            axs[0, i].plot(true_values[idx], label='True')
            axs[0, i].plot(predictions[idx], label='Predicted')
            axs[0, i].set_title(f'Idx: {indices[idx]}, Loss: {losses[idx]:.4f}')
            axs[0, i].legend()

        for i, idx in enumerate(worst_indices):
            axs[1, i].plot(true_values[idx], label='True')
            axs[1, i].plot(predictions[idx], label='Predicted')
            axs[1, i].set_title(f'Idx: {indices[idx]}, Loss: {losses[idx]:.4f}')
            axs[1, i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'test_results.png'))

        # Plot histogram of losses
        plt.figure()
        plt.hist(losses, bins=20, edgecolor='k')
        plt.title('Histogram of Test Losses')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.result_dir, 'test_loss_histogram.png'))

    # PLOTTING
    def plot_cross_heatmap(self, hidden_units_list, weight_decay_list, title, filename):
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.final_losses, annot=True, fmt=".4f", xticklabels=hidden_units_list, yticklabels=weight_decay_list, cmap="crest")
        plt.title(title)
        plt.xlabel('Hidden Units')
        plt.ylabel('Weight Decay')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, filename))
        plt.close()

    def plot_val_examples(self, model, hidden, weight):
        fig, axes = plt.subplots(3,3, figsize=(20, 15))
        colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
        ax = 0 
        criterion = nn.MSELoss()

        model.eval()
        with torch.no_grad():
            # 16 plots:
            for k, batch in enumerate(self.example_list):
                
                ins = batch['input'].unsqueeze(1)
                outs = batch['label'].unsqueeze(1)
                ypred_batch = model.forward(ins)
                loss = criterion(ypred_batch,outs)
                 
                i = k // 3  # Row index
                j = k % 3   # Column index
                ax = axes[i, j]

                x = ins.squeeze()
                y = outs.squeeze()
                pred = ypred_batch.squeeze().detach()

                colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
                for l in range(3):
                    ax.plot([t for t in range(121 - self.forecast_steps)],x[:,l],color=colors[l])
                    ax.plot([t for t in range(121 - self.forecast_steps,121)],y[:,l],color=colors[l],label='True')
                    ax.plot([t for t in range(121 - self.forecast_steps,121)],pred[:,l],color=colors[l+3],label='Pred')

                ax.set_title(f'Location: {batch['idx']} - Loss: {loss.item():.3f}')
                
        ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        fig.savefig(self.result_dir+f'examples_{hidden}_{weight}.png',bbox_inches='tight')

    def plot_train_examples(self, model, hidden, weight):
        fig, axes = plt.subplots(3,3, figsize=(20, 15))
        colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
        ax = 0 
        criterion = nn.MSELoss()
        
        examples = []
        example_locations = list(np.linspace(0,60000,9).astype(int))
        for idx in example_locations:
            item = self.train_dataset.__getitem__(idx)
            examples.append({'input':torch.Tensor(item['input']),'label':torch.Tensor(item['label']),'idx':item['idx']})

        model.eval()
        with torch.no_grad():
            # 16 plots:
            for k, batch in enumerate(examples):
                
                ins = batch['input'].unsqueeze(1)
                outs = batch['label'].unsqueeze(1)
                ypred_batch = model.forward(ins)
                loss = criterion(ypred_batch,outs)
                 
                i = k // 3  # Row index
                j = k % 3   # Column index
                ax = axes[i, j]

                x = ins.squeeze()
                y = outs.squeeze()
                pred = ypred_batch.squeeze().detach()

                colors = ['lightcoral','cornflowerblue','limegreen','red','blue','darkgreen']
                for l in range(3):
                    ax.plot([t for t in range(121 - self.forecast_steps)],x[:,l],color=colors[l])
                    ax.plot([t for t in range(121 - self.forecast_steps,121)],y[:,l],color=colors[l],label='True')
                    ax.plot([t for t in range(121 - self.forecast_steps,121)],pred[:,l],color=colors[l+3],label='Pred')

                ax.set_title(f'Location: {batch['idx']} - Loss: {loss.item():.3f}')
                
        ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        fig.savefig(self.result_dir+f'train_examples_{hidden}_{weight}.png',bbox_inches='tight')