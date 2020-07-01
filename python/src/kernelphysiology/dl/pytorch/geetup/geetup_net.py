"""
Collection of architectures for GEETUP in PyTorch.
"""

import os
import sys

import torch
from torch import nn

from .salema import Salema
from .tased import Tased
from . import resnet3d
from . import wavenet3d

from kernelphysiology.dl.pytorch.models.model_utils import get_preprocessing_function


def which_network(network_name, **kwargs):
    mean_std = None
    # defining meand ans std
    if kwargs['in_chns'] == 3:
        mean, std = get_preprocessing_function('rgb')
        mean_std = [mean, std]
    elif kwargs['in_chns'] == 1:
        mean, std = get_preprocessing_function('grey')
        mean_std = [mean, std]
    else:
        mean, std = get_preprocessing_function(kwargs['in_chns'])
        mean_std = [mean, std]
    # checking whether it's an architecture or a network
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        architecture = checkpoint['arch']
        if 'kwargs' in checkpoint:
            kwargs = checkpoint['kwargs']
        network = which_architecture(architecture, **kwargs)
        network.load_state_dict(checkpoint['state_dict'])
        if 'mean_std' in checkpoint:
            mean_std = checkpoint['mean_std']
    else:
        network = which_architecture(network_name, **kwargs)
        architecture = network_name
    return network, architecture, mean_std


def which_architecture(architecture, **kwargs):
    if architecture.lower() == 'tased':
        return Tased(**kwargs)
    elif architecture.lower() == 'salema':
        return Salema()
    elif architecture.lower() == 'centre':
        return CentreModel()
    elif 'resnet' in architecture.lower():
        return resnet3d.__dict__[architecture](**kwargs)
    elif 'wavenet' in architecture.lower():
        return wavenet3d.__dict__[architecture](**kwargs)
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
