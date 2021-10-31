import argparse

import torch
from torch.cuda import is_available
from torch.utils.data import DataLoader
import torchtext
from torchtext.datasets import Multi30k

from model.model import Transformer

# argument handling

batch_size = 256
using_gpu = True
# if gpu flag is enabled
device = torch.device('cuda' if (torch.cuda.is_available() and using_gpu) else 'cpu')

# dataset

dataset = Multi30k(
    root='./data', split=('train', 'valid', 'test'), 
    language_pair=('en','de')
)

dataloader_args = {
    'batch_size': batch_size,
    'pin_memory': True
}

train_loader, valid_loader, test_loader = [DataLoader(data, **dataloader_args) for data in dataset]

# initialize model

# training parameters

# training loop

