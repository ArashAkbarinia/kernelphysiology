"""

"""

import sys
import torch
import torch.nn as nn
from torch.nn import functional as t_functional

from . import pretrained_models


class ColourDiscrimination(nn.Module):
    def __init__(self, architecture, target_size, transfer_weights=None):
        super(ColourDiscrimination, self).__init__()

        num_classes = 3

        checkpoint = None
        # assuming architecture is path
        if transfer_weights is None:
            print('Loading model from %s!' % architecture)
            checkpoint = torch.load(architecture, map_location='cpu')
            architecture = checkpoint['arch']
            transfer_weights = checkpoint['transfer_weights']

        model = pretrained_models.get_pretrained_model(architecture, transfer_weights)
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        model = pretrained_models.get_backbone(architecture, model)

        layer = -1
        if len(transfer_weights) >= 2:
            layer = transfer_weights[1]

        if (
                'deeplabv3_' in architecture or 'fcn_' in architecture or 'deeplab' in architecture
                or 'resnet' in architecture or 'resnext' in architecture
        ):
            features, org_classes = pretrained_models._resnet_features(model, architecture, layer)
        else:
            sys.exit('Unsupported network %s' % architecture)
        self.features = features

        # the numbers for fc layers are hard-coded according to larger image size.
        scale_factor = (target_size / 224) * num_classes
        self.fc = nn.Linear(int(org_classes * scale_factor), 1)
        # self.fc = nn.Conv2d(512 * 4, num_classes, kernel_size=1, bias=False)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if checkpoint is not None:
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x0, x1, x2, x3):
        x0 = self.features(x0)
        x0 = x0.view(x0.size(0), -1)
        x1 = self.features(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.features(x2)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.features(x3)
        x3 = x3.view(x3.size(0), -1)
        # x = torch.cat([x0, x1, x2, x3], dim=1)

        # comp3 = self.fc(torch.cat([x0, x1, x2], dim=1))
        # comp2 = self.fc(torch.cat([x0, x1, x3], dim=1))
        # comp1 = self.fc(torch.cat([x0, x2, x3], dim=1))
        # comp0 = self.fc(torch.cat([x1, x2, x3], dim=1))
        comp3 = self.fc(torch.abs(torch.cat([x3-x0, x3-x1, x3-x2], dim=1)))
        comp2 = self.fc(torch.abs(torch.cat([x2-x0, x2-x1, x2-x3], dim=1)))
        comp1 = self.fc(torch.abs(torch.cat([x1-x0, x1-x2, x1-x3], dim=1)))
        comp0 = self.fc(torch.abs(torch.cat([x0-x1, x0-x2, x0-x3], dim=1)))


        # x = self.avgpool(x)
        # x = torch.tanh(x)
        # x = x.view(x.size(0), -1)
        return torch.cat([comp0, comp1, comp2, comp3], dim=1)

    def loss_function(self, output, target):
        loss = 0
        for i in range(4):
            loss += t_functional.binary_cross_entropy_with_logits(output[:, i], target[:, i])
        return loss / (4 * output.shape[0])
