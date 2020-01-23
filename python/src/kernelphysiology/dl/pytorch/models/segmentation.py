"""
Collection of segmentation networks.
"""

import os

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead

from kernelphysiology.dl.pytorch.models import utils as model_utils

__all__ = [
    'fcn_resnet', 'deeplabv3_resnet'
]


def _segm_resnet(name, backbone_name, num_classes, aux, **kwargs):
    # FIXME: 1000 and _
    if os.path.isfile(backbone_name):
        backbone, _ = model_utils.which_network_classification(
            backbone_name, 1000, **kwargs
        )
    else:
        backbone = model_utils.which_architecture(backbone_name, **kwargs)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, num_classes, aux_loss, **kwargs):
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    return model


def fcn_resnet(backbone, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet backbone.
    """
    return _load_model('fcn', backbone, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet(backbone, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet backbone.
    """
    return _load_model('deeplabv3', backbone, num_classes, aux_loss, **kwargs)
