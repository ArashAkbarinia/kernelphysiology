"""
Helpers functions for models in Pytorch.
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as pmodels
import torchvision.models.segmentation as seg_models

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from kernelphysiology.dl.pytorch import models as custom_models
from kernelphysiology.dl.pytorch.models.lesion_utility import lesion_kernels


def _get_conv(module, layer_num, conv_num):
    sub_layer = [*module[:layer_num]]
    sub_layer = nn.Sequential(*sub_layer)

    module_list = list(module[layer_num].children())
    if (isinstance(module[layer_num], pmodels.resnet.BasicBlock) or
            isinstance(module[layer_num], custom_models.resnet.BasicBlock)):
        conv_ind = (conv_num - 1) * 3 + 2
    else:
        conv_ind = (conv_num - 1) * 2 + 2
    sub_conv = nn.Sequential(*module_list[:conv_ind])
    return sub_layer, sub_conv


class LayerActivation(nn.Module):
    def __init__(self, model, layer_name):
        super(LayerActivation, self).__init__()

        # FIXME: only for resnet at this point
        self.sub_layer = None
        self.sub_conv = None
        if layer_name == 'fc':
            self.features = model
        elif layer_name == 'avgpool':
            self.features = nn.Sequential(*list(model.children()))
        else:
            name_split = layer_name.split('.')
            last_layer = 4
            sub_layer = None
            sub_conv = None
            if 'layer' in name_split[0]:
                layer_num = int(name_split[1])
                conv_num = int(name_split[2][-1])
                if name_split[0] == 'layer1':
                    layerx = model.layer1
                    last_layer = 4
                elif name_split[0] == 'layer2':
                    layerx = model.layer2
                    last_layer = 5
                elif name_split[0] == 'layer3':
                    layerx = model.layer3
                    last_layer = 6
                elif name_split[0] == 'layer4':
                    layerx = model.layer4
                    last_layer = 7
                sub_layer, sub_conv = _get_conv(layerx, layer_num, conv_num)
            self.features = nn.Sequential(*list(model.children())[:last_layer])
            self.sub_layer = sub_layer
            self.sub_conv = sub_conv

    def forward(self, x):
        x = self.features(x)
        if self.sub_layer is not None:
            x = self.sub_layer(x)
        if self.sub_conv is not None:
            x = self.sub_conv(x)
        return x


class IntermediateModel(nn.Module):
    def __init__(self, original_model, num_categories, dr_rate, model_name):
        super(IntermediateModel, self).__init__()
        if 'densenet' in model_name:
            layer_number = 1
        else:
            layer_number = 2

        if 'resnet' in model_name:
            num_ftrs = original_model.fc.in_features
        elif model_name == "alexnet":
            num_ftrs = original_model.classifier[6].in_features
        elif 'vgg' in model_name:
            num_ftrs = 512 * 7 * 7
        elif 'densenet' in model_name:
            num_ftrs = original_model.classifier.in_features

        self.features = nn.Sequential(
            *list(original_model.children())[:-layer_number])
        if 'vgg' in model_name:
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc = nn.Linear(num_ftrs, num_categories)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class NewClassificationModel(nn.Module):
    def __init__(self, original_model, layer, num_classes):
        super(NewClassificationModel, self).__init__()

        # FIXME: this only works for custom ResNets
        if type(layer) is str:
            if layer == 'layer1':
                layer = 4
            elif layer == 'layer2':
                layer = 5
            elif layer == 'layer3':
                layer = 6
            elif layer == 'layer4':
                layer = 7
            org_classes = list(original_model.children())[layer][
                0].conv1.in_channels
        else:
            org_classes = original_model.fc.in_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(org_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def which_network_classification(network_name, num_classes, **kwargs):
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        customs = None
        if 'customs' in checkpoint:
            customs = checkpoint['customs']
            # TODO: num_classes is just for backward compatibility
            if 'num_classes' not in customs:
                customs['num_classes'] = num_classes
        model = which_architecture(checkpoint['arch'], customs=customs)

        # TODO: for each dataset a class of network should be defined
        # if dataset == 'leaf':
        #     num_ftrs = model.fc.in_features
        #     model.fc = nn.Linear(num_ftrs, 30)
        # elif dataset == 'fruits':
        #     num_ftrs = model.fc.in_features
        #     model.fc = nn.Linear(num_ftrs, 23)
        # FIXME: this is for transfer learning or adding a dropout
        # elif 'wcs' in dataset:
        #     if '_330' in dataset:
        #         model = IntermediateModel(model, 330, 0, checkpoint['arch'])
        #     elif '_1600' in dataset:
        #         model = IntermediateModel(model, 1600, 0, checkpoint['arch'])

        model.load_state_dict(checkpoint['state_dict'])
        target_size = checkpoint['target_size']
    elif network_name == 'inception_v3':
        target_size = 299
        model = pmodels.__dict__[network_name](
            pretrained=True, aux_logits=False
        )
    else:
        model = pmodels.__dict__[network_name](pretrained=True)
        target_size = 224

    model = lesion_kernels(model, **kwargs)
    return model, target_size


def which_network_segmentation(network_name, num_classes, **kwargs):
    if os.path.isfile(network_name):
        checkpoint = torch.load(network_name, map_location='cpu')
        customs = None
        aux_loss = None
        if 'customs' in checkpoint:
            customs = checkpoint['customs']
            # TODO: num_classes is just for backward compatibility
            if 'num_classes' not in customs:
                customs['num_classes'] = num_classes
            if 'aux_loss' in customs:
                aux_loss = customs['aux_loss']
            backbone = customs['backbone']
        # TODO: for now only predefined models
        # model = which_architecture(checkpoint['arch'], customs=customs)
        model = custom_models.__dict__[checkpoint['arch']](
            backbone, num_classes=num_classes, pretrained=False,
            aux_loss=aux_loss
        )

        model.load_state_dict(checkpoint['state_dict'])
        target_size = checkpoint['target_size']
    else:
        model = seg_models.__dict__[network_name](
            num_classes=num_classes, pretrained=True, aux_loss=True
        )
        target_size = 480

    model = lesion_kernels(model, **kwargs)
    return model, target_size


def which_network(network_name, task_type, **kwargs):
    # FIXME: network should be acosiated to dataset
    if task_type == 'classification':
        (model, target_size) = which_network_classification(
            network_name, **kwargs
        )
    elif task_type == 'segmentation':
        (model, target_size) = which_network_segmentation(
            network_name, **kwargs
        )
    else:
        sys.exit('Task type %s is not supported.' % task_type)
    return model, target_size


def create_custom_resnet(network_name, customs=None):
    # TODO: make this nicer!!!!
    cus_res = ['resnet_basic_custom_', 'resnet_bottleneck_custom_']
    if cus_res[0] not in network_name and cus_res[1] not in network_name:
        return network_name, customs
    if customs is None:
        customs = dict()
    net_parts = network_name.split('_')

    network_name = '%s_%s_%s' % (net_parts[0], net_parts[1], net_parts[2])
    if 'pooling_type' not in customs:
        customs['pooling_type'] = 'max_pooling'
    if 'num_classes' not in customs:
        customs['num_classes'] = 1000
    customs['blocks'] = [int(net_parts[i]) for i in range(3, 7)]
    customs['num_kernels'] = int(net_parts[7])
    return network_name, customs


def which_architecture(network_name, customs=None):
    if customs is None:
        if network_name == 'inception_v3':
            model = pmodels.__dict__[network_name](
                pretrained=False, aux_logits=False
            )
        else:
            model = pmodels.__dict__[network_name](pretrained=False)
    else:
        pooling_type = customs['pooling_type']
        num_classes = customs['num_classes']
        if 'in_chns' in customs:
            in_chns = customs['in_chns']
        else:
            # assuming if it doesn't exist, it's 3
            in_chns = 3
        # differentiating between custom models and nominal one
        if 'blocks' in customs and customs['blocks'] is not None:
            num_kernels = customs['num_kernels']
            model = custom_models.__dict__[network_name](
                customs['blocks'], pretrained=False, pooling_type=pooling_type,
                in_chns=in_chns, num_classes=num_classes, inplanes=num_kernels
            )
        else:
            model = custom_models.__dict__[network_name](
                pretrained=False, pooling_type=pooling_type,
                in_chns=in_chns, num_classes=num_classes
            )
    return model


# TODO: use different values fo preprocessing
def get_preprocessing_function(colour_space, colour_vision=None):
    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]
    if colour_space in ['rgb', 'red', 'green', 'blue', 'grey3']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif colour_space == 'grey':
        mean = [0.5]
        std = [0.25]
    elif colour_space == 'lab' or colour_space == 'lms':
        if 'dichromat' in colour_vision or 'anopia' in colour_vision:
            mean = [0.5, 0.5]
            std = [0.25, 0.25]
        elif colour_vision == 'monochromat':
            mean = [0.5]
            std = [0.25]
    else:
        # just create mean and std based on number of channels
        mean = []
        std = []
        for i in range(colour_space):
            mean.append(0.5)
            std.append(0.25)
    return mean, std


def info_conv_parameters(state_dict):
    num_kernels = 0
    num_parameters = 0
    for layer in state_dict.keys():
        if 'conv' in layer:
            current_layer = state_dict[layer].numpy()
            num_kernels += current_layer.shape[0]
            num_parameters += current_layer.size
    return num_kernels, num_parameters
