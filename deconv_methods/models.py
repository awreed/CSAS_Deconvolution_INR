import torch
import torch.nn as nn
import numpy as np


class swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)


class MLP_FF2D_MLP(nn.Module):
  def __init__(self, nf=128, out_ch=1, act='none'):
    super(MLP_FF2D_MLP, self).__init__()
    self.main = nn.Sequential(
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, nf*2, bias=False),
        nn.ReLU(),
        nn.Linear(nf*2, out_ch, bias=True)
    )
    self.act_fun = None

    assert act in ['none', 'relu', 'sigmoid', 'tanh'], "Activation should be 'none',\
    'relu' or 'sigmoid'"

    if act == 'none':
      self.act_fun = lambda x: x
    if act == 'relu':
      self.act_fun = lambda x: torch.nn.functional.relu(x)
    if act == 'sigmoid':
      self.act_fun = lambda x: torch.sigmoid(x)
    if act == 'tanh':
      self.act_fun = lambda x: torch.nn.functional.tanh(x)

  def forward(self, x):
    x = self.main(x)
    return self.act_fun(x)


class FourierFeaturesVector(torch.nn.Module):                                                                                 
    def __init__(self, num_input_channels, mapping_size=256, scale=10):         
        super().__init__()                                                      
                                                                                
        self._num_input_channels = num_input_channels                           
        self._mapping_size = mapping_size                                       
        self._B = torch.randn((num_input_channels, mapping_size), dtype=torch.float64) * scale
                                                                                
    def forward(self, x):                                                       
        assert x.dim() == 2

        l, c = x.shape

        assert c == self._num_input_channels, "number channels wrong"
       
        # From [B, C, W, H] to [(B*W*H), C] 
        x = x @ self._B.to(x.device)                                            
                                                                                
        x = 2 * np.pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


