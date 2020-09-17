"""

"""

import torch
import torch.nn as nn

from kernelphysiology.dl.pytorch.models import model_utils


def _resnet_features(model, network_name, layer):
    if type(layer) is str:
        if layer == 'layer1':
            layer = 4
            if network_name in ['resnet18', 'resnet34']:
                org_classes = 849408
            else:
                org_classes = 849408
        elif layer == 'layer2':
            layer = 5
            if network_name in ['resnet18', 'resnet34']:
                org_classes = 849408
            else:
                org_classes = 3397632
        elif layer == 'layer3':
            layer = 6
            if network_name in ['resnet18', 'resnet34']:
                org_classes = 424704
            else:
                org_classes = 1698816
        elif layer == 'layer4':
            layer = 7
            if network_name in ['resnet18', 'resnet34']:
                org_classes = 215040
            else:
                org_classes = 860160
    else:
        org_classes = 512
    features = nn.Sequential(*list(model.children())[:layer])
    return features, org_classes


class VGG(nn.Module):

    def __init__(self, model, network_name, layer):
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


def _vgg_features(model, network_name, layer):
    features = VGG(model, network_name, layer)
    org_classes = features.org_classes
    return features, org_classes


class NewClassificationModel(nn.Module):
    def __init__(self, network_name, transfer_weights=None):
        super(NewClassificationModel, self).__init__()
        num_classes = 2

        checkpoint = None
        # assuming network_name is path
        if transfer_weights is None:
            checkpoint = torch.load(network_name, map_location='cpu')
            network_name = checkpoint['arch']
            transfer_weights = checkpoint['transfer_weights']

        (model, _) = model_utils.which_network(
            transfer_weights[0], 'classification', num_classes=1000
        )
        # print(model)
        layer = -1
        if len(transfer_weights) == 2:
            layer = transfer_weights[1]

        if 'resnet' in network_name:
            features, org_classes = _resnet_features(model, network_name, layer)
        elif 'vgg' in network_name:
            features, org_classes = _vgg_features(model, network_name, layer)
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
