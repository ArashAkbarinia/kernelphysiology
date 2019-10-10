"""
Collection of architectures for GEETUP in PyTorch.
"""

import os
import sys

import torch
from torch import nn

from .salema import Salema
from .tased import Tased


def which_network(network_name):
    mean_std = None
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        architecture = checkpoint['arch']
        network = which_architecture(architecture)
        network.load_state_dict(checkpoint['state_dict'])
        if 'mean_std' in checkpoint:
            mean_std = checkpoint['mean_std']
    else:
        network = which_architecture(network_name)
        architecture = network_name
    return network, architecture, mean_std


def which_architecture(architecture):
    if architecture.lower() == 'tased':
        return Tased()
    elif architecture.lower() == 'salema':
        return Salema()
    elif architecture.lower() == 'centre':
        return CentreModel()
    else:
        sys.exit('Architecture %s not supported.' % architecture)


class CentreModel(nn.Module):

    def __init__(self):
        super(CentreModel, self).__init__()

    def forward(self, x):
        x = torch.zeros(
            (x.shape[0], x.shape[3], x.shape[4]), device=x.device
        )
        centre_row = round(x.shape[1] / 2)
        centre_col = round(x.shape[2] / 2)
        x[:, centre_row, centre_col] = 1
        return x
