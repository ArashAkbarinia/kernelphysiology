"""

"""

import torch
import torch.nn as nn

from torchvision.models import segmentation

from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.pytorch.models import resnet_simclr
from kernelphysiology.dl.experiments.contrast.transparency_model import \
    get_transparency_model


def _resnet_features(model, network_name, layer, grey_width):
    if type(layer) is str:
        if layer == 'layer1':
            layer = 4
            if grey_width:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 849408
                else:
                    org_classes = 849408
            else:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 524288
                else:
                    org_classes = 524288
        elif layer == 'layer2':
            layer = 5
            if grey_width:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 849408
                else:
                    org_classes = 3397632
            else:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 524288
                else:
                    org_classes = 2097152
        elif layer == 'layer3':
            layer = 6
            if grey_width:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 424704
                else:
                    org_classes = 1698816
            else:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 262144
                else:
                    org_classes = 1048576
        elif layer == 'layer4':
            layer = 7
            if grey_width:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 215040
                elif 'deeplabv3_' in network_name or 'fcn_' in network_name:
                    org_classes = 3397632
                else:
                    org_classes = 860160
            else:
                if network_name in ['resnet18', 'resnet34']:
                    org_classes = 131072
                elif 'deeplabv3_' in network_name or 'fcn_' in network_name:
                    org_classes = 2097152
                else:
                    org_classes = 524288
    else:
        org_classes = 512
    features = nn.Sequential(*list(model.children())[:layer])
    return features, org_classes


def _mobilenet_v2_features(model, network_name, layer, grey_width):
    layer = int(layer[1:])
    org_classes = [
        1698816, 849408, 318528, 318528, 106176, 106176, 106176, 53760, 53760,
        53760, 53760, 80640, 80640, 80640, 35200, 35200, 35200, 70400, 281600,
    ]
    features = nn.Sequential(*list(model.features.children())[:layer + 1])
    return features, org_classes[layer]


class VGG(nn.Module):

    def __init__(self, model, network_name, layer, grey_width):
        super(VGG, self).__init__()
        self.classifier = None
        if type(layer) is str:
            layer_parts = layer.split('_')
            block = layer_parts[0]
            bind = int(layer_parts[1])
            if block == 'classifier':
                selected_features = []
                for l in list(model.children())[:-1]:
                    selected_features.append((l))
                self.features = nn.Sequential(*selected_features)

                selected_classifier = []
                for i, l in enumerate(list(list(model.children())[-1])):
                    selected_classifier.append((l))
                    if i == bind:
                        break
                self.classifier = nn.Sequential(*selected_classifier)
                self.org_classes = 4096
            elif network_name == 'vgg11_bn':
                selected_features = []
                for i, l in enumerate(list(list(model.children())[-3])):
                    selected_features.append((l))
                    if i == bind:
                        break
                self.features = nn.Sequential(*selected_features)
                all_org_classes = [
                    0, 0, 0, 3397632,
                    0, 0, 0, 1698816,
                    0, 0, 0, 0, 0, 0, 849408,
                    0, 0, 0, 0, 0, 0, 419328,
                    0, 0, 0, 0, 0, 0, 97280
                ]
                self.org_classes = all_org_classes[i]
        else:
            self.org_classes = 25088
            self.features = nn.Sequential(*list(model.children())[:layer])

    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x


def _vgg_features(model, network_name, layer, grey_width):
    features = VGG(model, network_name, layer)
    org_classes = features.org_classes
    return features, org_classes


class NewClassificationModel(nn.Module):
    def __init__(self, network_name, transfer_weights=None, grey_width=True):
        super(NewClassificationModel, self).__init__()
        num_classes = 2

        checkpoint = None
        # assuming network_name is path
        if transfer_weights is None:
            checkpoint = torch.load(network_name, map_location='cpu')
            network_name = checkpoint['arch']
            transfer_weights = checkpoint['transfer_weights']

        if 'deeplabv3_' in network_name or 'fcn_' in network_name:
            model = segmentation.__dict__[network_name](pretrained=True)
        elif network_name == 'transparency':
            model = get_transparency_model()
        elif network_name == 'simclr':
            model = resnet_simclr.ResNetSimCLR('resnet50', 128)
            dpath = '/home/arash/Software/repositories/kernelphysiology/data/simclr_resnet50.pth'
            simclr_pretrained = torch.load(dpath, map_location='cpu')
            model.load_state_dict(simclr_pretrained)
        else:
            (model, _) = model_utils.which_network(
                transfer_weights[0], 'classification', num_classes=1000
            )
        # print(model)
        layer = -1
        if len(transfer_weights) == 2:
            layer = transfer_weights[1]

        if 'deeplabv3_' in network_name or 'fcn_' in network_name:
            features, org_classes = _resnet_features(
                model.backbone, network_name, layer, grey_width
            )
        elif network_name == 'transparency':
            features, org_classes = _resnet_features(
                model.encoder, network_name, layer, grey_width
            )
        elif network_name == 'simclr':
            features, org_classes = _resnet_features(
                model.features, network_name, layer, grey_width
            )
        elif 'resnet' in network_name:
            features, org_classes = _resnet_features(
                model, network_name, layer, grey_width
            )
        elif 'vgg' in network_name:
            features, org_classes = _vgg_features(
                model, network_name, layer, grey_width
            )
        elif 'mobilenet_v2' in network_name:
            features, org_classes = _mobilenet_v2_features(
                model, network_name, layer, grey_width
            )
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(org_classes, num_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x
