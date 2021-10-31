import math
from typing import ForwardRef

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self, dim: int, 
        n_layers : int, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.n_layers = n_layers
    
    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(
        self, dim: int, 
        n_layers: int, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.n_layers = n_layers
    
    def forward(self, x: torch.Tensor):
        pass


class Transformer(nn.Module):
    def __init__(
        self, vocab_len: int, 
        model_dim: int, 
        n: int = 6, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n = n
        self.model_dim = model_dim
        self.root_dim = math.sqrt(model_dim)

        self.embedding = nn.Embedding(vocab_len, model_dim)

        # initialise encoder
        
        # initialize decoder

        self.out_ffnn = nn.Sequential(
            nn.Linear(512, vocab_len),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor):
        # get input embedding and scale
        x = self.embedding(x)
        x.mul_(self.root_dim)


        pass