'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

import numpy as np
import torch
from numpy import newaxis as na
from LRP_linear_layer import lrp_linear


class LSTM_bidi:
    
    def __init__(self, input, model_path='./model/'):
        """
        Load trained model from file.
        """
        
        # sequence
        self.input = input
        
        # model weights
        model = torch.load(model_path).state_dict()

        # LSTM left encoder
        self.Wxh_Left = model['lstm.weight_ih_l0'].numpy() # has weights w_ii, w_if, w_ic, w_io
        self.bxh_Left = model['lstm.bias_ih_l0'].numpy()
        self.Whh_Left = model['lstm.weight_hh_l0'].numpy() # has weights w_hi, w_hf, w_hc, w_ho 
        self.bhh_Left = model['lstm.bias_hh_l0'].numpy()

        # linear output layer
        self.Why_Left = model['linear.weight'].numpy()
        self.bhy_Left = model['linear.bias'].numpy()

        self.T      = self.input.shape[1]                      # sequence length
        self.e      = self.input.shape[-1]           # word embedding dimension
        self.d      = int(self.Wxh_Left.shape[0]/4)  # hidden layer dimension

        self.x              = self.input
  
        self.h_Left         = np.zeros((self.T+1, self.d))
        self.c_Left         = np.zeros((self.T+1, self.d))
                
        
    def set_input(self):
        """
        Build the numerical input sequence x from the word indices w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """

        self.h_Left         = np.zeros((self.T+1, self.d))
        self.c_Left         = np.zeros((self.T+1, self.d))
     
    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x was previously set)
        """
        d = self.d
        # gate indices (assuming the gate ordering in the LSTM weights is i,f,g,o):     
        idx    = np.hstack((np.arange(0,d), np.arange(d,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o separately
          
        # initialize
        self.gates_xh_Left  = np.zeros((self.T, 4*self.d))  
        self.gates_hh_Left  = np.zeros((self.T, 4*self.d)) 
        self.gates_pre_Left = np.zeros((self.T, 4*self.d))  # gates pre-activation
        self.gates_Left     = np.zeros((self.T, 4*self.d))  # gates activation
              
        for t in range(self.T): 
            self.gates_xh_Left[t]     = np.dot(self.Wxh_Left, self.x[t])        
            self.gates_hh_Left[t]     = np.dot(self.Whh_Left, self.h_Left[t-1]) 
            self.gates_pre_Left[t]    = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bxh_Left + self.bhh_Left
            self.gates_Left[t,idx]    = 1.0/(1.0 + np.exp(- self.gates_pre_Left[t,idx]))
            self.gates_Left[t,idx_g]  = np.tanh(self.gates_pre_Left[t,idx_g]) 
            self.c_Left[t]            = self.gates_Left[t,idx_f]*self.c_Left[t-1] + self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g]
            self.h_Left[t]            = self.gates_Left[t,idx_o]*np.tanh(self.c_Left[t])
        
        self.y_Left  = np.dot(self.Why_Left,  self.h_Left[-1]) + self.bhy_Left 
        self.s       = self.y_Left
        
        return self.s.copy() # prediction scores
     
              
    def backward(self, w, sensitivity_class):
        """
        Standard gradient backpropagation backward pass.
        Compute the hidden layer gradients by backpropagating a gradient of 1.0 for the class sensitivity_class
        """
        # forward pass
        self.set_input(w)
        self.forward() 
        
        C      = self.Why_Left.shape[0]   # number of classes
        idx    = np.hstack((np.arange(0,self.d), np.arange(2*self.d,4*self.d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0,self.d), np.arange(self.d,2*self.d), np.arange(2*self.d,3*self.d), np.arange(3*self.d,4*self.d) # indices of gates i,g,f,o separately
        
        # initialize
        self.dx               = np.zeros(self.x.shape)
        
        self.dh_Left          = np.zeros((self.T+1, self.d))
        self.dc_Left          = np.zeros((self.T+1, self.d))
        self.dgates_pre_Left  = np.zeros((self.T, 4*self.d))  # gates pre-activation
        self.dgates_Left      = np.zeros((self.T, 4*self.d))  # gates activation
               
        ds                    = np.zeros((C))
        ds[sensitivity_class] = 1.0
        dy_Left               = ds.copy()
        
        self.dh_Left[self.T-1]     = np.dot(self.Why_Left.self.T,  dy_Left)
        
        for t in reversed(range(self.T)): 
            self.dgates_Left[t,idx_o]    = self.dh_Left[t] * np.tanh(self.c_Left[t])  # do[t]
            self.dc_Left[t]             += self.dh_Left[t] * self.gates_Left[t,idx_o] * (1.-(np.tanh(self.c_Left[t]))**2) # dc[t]
            self.dgates_Left[t,idx_f]    = self.dc_Left[t] * self.c_Left[t-1]         # df[t]
            self.dc_Left[t-1]            = self.dc_Left[t] * self.gates_Left[t,idx_f] # dc[t-1]
            self.dgates_Left[t,idx_i]    = self.dc_Left[t] * self.gates_Left[t,idx_g] # di[t]
            self.dgates_Left[t,idx_g]    = self.dc_Left[t] * self.gates_Left[t,idx_i] # dg[t]
            self.dgates_pre_Left[t,idx]  = self.dgates_Left[t,idx] * self.gates_Left[t,idx] * (1.0 - self.gates_Left[t,idx]) # d ifo pre[t]
            self.dgates_pre_Left[t,idx_g]= self.dgates_Left[t,idx_g] *  (1.-(self.gates_Left[t,idx_g])**2) # d g pre[t]
            self.dh_Left[t-1]            = np.dot(self.Whh_Left.T, self.dgates_pre_Left[t])
            self.dx[t]                   = np.dot(self.Wxh_Left.T, self.dgates_pre_Left[t])
                    
        return self.dx.copy() 
    
                   
    def lrp(self, w, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(w)
        self.forward() 
        
        T      = len(self.w)
        d      = int(self.Wxh_Left.shape[0]/4)
        e      = self.E.shape[1] 
        C      = self.Why_Left.shape[0]  # number of classes

        idx  = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o separately
        
        # initialize
        Rx       = np.zeros(self.x.shape)
        
        Rh_Left  = np.zeros((T+1, d))
        Rc_Left  = np.zeros((T+1, d))
        Rg_Left  = np.zeros((T,   d)) # gate g only
        
        Rout_mask            = np.zeros((C))
        Rout_mask[LRP_class] = 1.0  
        
        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T-1]  = lrp_linear(self.h_Left[T-1],  self.Why_Left.T , self.bhy_Left, self.s, self.s*Rout_mask, d, eps, bias_factor, debug=False)
        
        for t in reversed(range(T)):
            Rc_Left[t]   += Rh_Left[t]
            Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,idx_f]*self.c_Left[t-1],         np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], d, eps, bias_factor, debug=False)
            Rg_Left[t]    = lrp_linear(self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g], np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], d, eps, bias_factor, debug=False)
            Rx[t]         = lrp_linear(self.x[t],        self.Wxh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Whh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
                   
        return Rx, Rh_Left[-1].sum()+Rc_Left[-1].sum()