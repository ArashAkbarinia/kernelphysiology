"""
Collection of segmentation networks.
"""

import os

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead

from torchvision.models import resnet as presnet
from kernelphysiology.dl.pytorch.models import resnet as cresnet
from kernelphysiology.dl.pytorch.models import model_utils as model_utils

__all__ = [
    'fcn_resnet', 'deeplabv3_resnet'
]


def _segm_resnet(name, backbone_name, num_classes, aux, **kwargs):
    # FIXME: 1000 and _
    if isinstance(backbone_name, dict):
        backbone = model_utils.which_architecture(
            backbone_name['arch'], backbone_name['customs']
        )
    elif os.path.isfile(backbone_name):
        if 'backbone' in kwargs:
            del kwargs['backbone']
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
        layer3 = list(backbone.layer3)[-1]
        # Depending on Bottleneck or basic choose the num_features
        if isinstance(layer3, (cresnet.Bottleneck, presnet.Bottleneck)):
            inplanes = list(layer3.children())[-2].num_features
        else:
            inplanes = list(layer3.children())[-1].num_features
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    layer4 = list(backbone.layer4)[-1]
    if isinstance(layer4, (cresnet.Bottleneck, presnet.Bottleneck)):
        inplanes = list(layer4.children())[-2].num_features
    else:
        inplanes = list(layer4.children())[-1].num_features
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
