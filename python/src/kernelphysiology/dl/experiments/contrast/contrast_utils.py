import torch
import torch.nn as nn

from torchvision.models import resnet as presnet
from kernelphysiology.dl.pytorch.models import resnet as cresnet
from kernelphysiology.dl.pytorch import models as custom_models


class NewClassificationModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(NewClassificationModel, self).__init__()

        last_layer = list(original_model.children())[-1][0]
        if isinstance(last_layer, (cresnet.Bottleneck, presnet.Bottleneck)):
            org_classes = last_layer.conv3.out_channels
        else:
            org_classes = last_layer.conv2.out_channels
        self.features = nn.Sequential(*list(original_model.children()))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(org_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FaceModel(nn.Module):
    def __init__(self, network_name, transfer_weights):
        super(FaceModel, self).__init__()

        if 'resnet18' in network_name:
            backbone = 'resnet18'
        elif 'resnet50' in network_name:
            backbone = 'resnet50'
        model = custom_models.__dict__['deeplabv3_resnet'](
            backbone, num_classes=21, aux_loss=False
        )
        model = model.backbone

        model = NewClassificationModel(model, 500)
        face_net = torch.load(transfer_weights[0], map_location='cpu')
        model.load_state_dict(face_net['state_dict'])
        self.backbone = model.features
