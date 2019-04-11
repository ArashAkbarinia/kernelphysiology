'''
Utility functoins for models in Pytorch.
'''


import os

import torch
import torchvision.models as pmodels
import torchvision.transforms as transforms


def which_network_classification(network_name):
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu0')
        model = which_architecture(checkpoint['arch'])
        model.load_state_dict(checkpoint['state_dict'])
        target_size = checkpoint['target_size']
    elif network_name == 'inception_v3':
        target_size = 299
        model = pmodels.__dict__[network_name](
            pretrained=True, aux_logits=False)
    else:
        model = pmodels.__dict__[network_name](pretrained=True)
        target_size = 224
    return (model, target_size)


def which_network(network_name, task_type):
    # FIXME: network should be acosiated to dataset
    if task_type == 'classification':
        (model, target_size) = which_network_classification(network_name)
    return (model, target_size)


def which_architecture(network_name):
    if network_name == 'inception_v3':
        model = pmodels.__dict__[network_name](
            pretrained=False, aux_logits=False)
    else:
        model = pmodels.__dict__[network_name](pretrained=False)
    return model


def get_preprocessing_function(preprocessing):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    return normalize
