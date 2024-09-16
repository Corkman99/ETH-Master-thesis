import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    
    def __init__(self,
                 input_size,out_dim,
                 hidden_size,init_zero,
                 forecast_steps):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.init_zero = init_zero
        self.forecast_steps = int(forecast_steps)
        self.out_dim = out_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dtype=torch.float,
            batch_first=False)
        
        self.linear = nn.Linear(hidden_size,out_dim*self.forecast_steps)

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        if self.init_zero:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float)
        else:
            h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float)
            c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float)

        return (h_0,c_0)
    
    def forward(self,seq):
        init = self.init_hidden(seq.shape[1])
        hidden, (_, _) = self.lstm(seq,init) # hidden has shape (seq_length,batch_size,hidden_size), hidden[-1].shape = (batch_size,hidden_size)
        out = self.linear(hidden[-1]) # has shape (batch_size,out_dim*forecast_steps)
        out = out.view(self.forecast_steps,seq.shape[1],-1) # formatting
        return out