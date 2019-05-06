"""
Helpers functions for models in Pytorch.
"""

import os

import torch
import torch.nn as nn
import torchvision.models as pmodels
import torchvision.transforms as transforms

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def which_network_classification(network_name, dataset):
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        model = which_architecture(checkpoint['arch'])
        # TODO: for each dataset a class of network should be defined
        if dataset == 'leaf':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 30)
        elif dataset == 'fruits':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 23)
        model.load_state_dict(checkpoint['state_dict'])
        target_size = checkpoint['target_size']
    elif network_name == 'inception_v3':
        target_size = 299
        model = pmodels.__dict__[network_name](
            pretrained=True, aux_logits=False)
    else:
        model = pmodels.__dict__[network_name](pretrained=True)
        target_size = 224
    return model, target_size


def which_network(network_name, task_type, dataset):
    # FIXME: network should be acosiated to dataset
    if task_type == 'classification':
        (model, target_size) = which_network_classification(
            network_name, dataset)
    return model, target_size


def which_architecture(network_name):
    if network_name == 'inception_v3':
        model = pmodels.__dict__[network_name](
            pretrained=False, aux_logits=False)
    else:
        model = pmodels.__dict__[network_name](pretrained=False)
    return model


# TODO: use different values fo preprocessing
def get_preprocessing_function(preprocessing):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    return normalize
