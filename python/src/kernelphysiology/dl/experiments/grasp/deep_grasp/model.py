"""
The network for grasping.
"""

import os
import sys
import torch
import torch.nn as nn

__all__ = [
    'ResNet', 'resnet_basic', 'resnet_bottleneck'
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    stride = (stride, 1)
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=bias, dilation=dilation, padding_mode='reflect'
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    stride = (stride, 1)
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_chns, num_classes=2,
                 zero_init_residual=False, norm_layer=None,
                 inplanes=4, kernel_size=(3, 3), bias=False, stride=2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_chns = in_chns
        self.inplanes = inplanes
        padding = [k - 2 for k in kernel_size]
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                self.in_chns, self.inplanes, kernel_size=kernel_size, bias=bias,
                groups=self.in_chns, padding=padding, padding_mode='reflect'
            ),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(
            block, self.inplanes, layers[0], stride=stride
        )
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=stride
        )
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=stride,
        )
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=stride,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fcm = 8
        if layers[0] == 0:
            fcm = 1
            block.expansion = 1
        elif layers[1] == 0:
            fcm = 1
        elif layers[2] == 0:
            fcm = 2
        elif layers[3] == 0:
            fcm = 4
        self.fc = nn.Linear(
            inplanes * fcm * block.expansion + 1,
            num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        # if None, nothing in this layer
        if blocks > 0:
            norm_layer = self._norm_layer
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes, planes, norm_layer=norm_layer
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, kinematic, intensity):
        kinematic = self.layer0(kinematic)
        kinematic = self.layer1(kinematic)
        kinematic = self.layer2(kinematic)
        kinematic = self.layer3(kinematic)
        kinematic = self.layer4(kinematic)

        kinematic = self.avgpool(kinematic)
        kinematic = kinematic.view(kinematic.size(0), -1)

        x = torch.cat([kinematic, intensity], dim=1)
        x = self.fc(x)

        return x


def _resnet(block_type, planes, pretrained, **kwargs):
    model = ResNet(block_type, planes, **kwargs)
    if pretrained:
        if os.path.exists(pretrained):
            state_dict = torch.load(pretrained, map_location='cpu')
        else:
            sys.exit('Provided weights path (%s) does not exist.' % pretrained)
        model.load_state_dict(state_dict)
    return model


def resnet_basic(planes, pretrained=False, **kwargs):
    return _resnet(BasicBlock, planes, pretrained, **kwargs)


def resnet_bottleneck(planes, pretrained=False, **kwargs):
    return _resnet(Bottleneck, planes, pretrained, **kwargs)


def load_pretrained(path):
    net_data = torch.load(path, map_location='cpu')
    planes = net_data['net_info']['blocks']
    kwargs = net_data['net_info']['kwargs']
    if net_data['arch'] == 'resnet_bottleneck':
        network = resnet_bottleneck(planes, **kwargs)
    elif net_data['arch'] == 'resnet_basic':
        network = resnet_basic(planes, **kwargs)
    else:
        sys.exit('Unsupported architecture!')
    network.load_state_dict(net_data['state_dict'])
    return network
