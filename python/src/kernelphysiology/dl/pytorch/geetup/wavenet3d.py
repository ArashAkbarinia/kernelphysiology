"""
WaveNet3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'WaveNet', 'wavenet_basic', 'wavenet_bottleneck'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv1x3x3_transpose(in_planes, out_planes, padding=(0, 0, 0), stride=1,
                        output_padding=(0, 0, 0)):
    # 1x3x3 convolution transpose with padding
    return nn.ConvTranspose3d(
        in_planes, out_planes, kernel_size=(1, 3, 3), padding=padding,
        stride=(1, stride, stride), output_padding=output_padding, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    )
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


def upsample_basic_block(x, planes, stride):
    out = F.upsample(x, scale_factor=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    )
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockTranspose(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, padding=0, stride=1, upsample=None,
                 output_padding=(0, 1, 1)):
        super(BasicBlockTranspose, self).__init__()
        self.conv1 = conv1x3x3_transpose(
            inplanes, planes, padding, stride, output_padding=output_padding
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3_transpose(planes, planes, (0, 1, 1), 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckTranspose(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BottleneckTranspose, self).__init__()
        self.conv1 = nn.ConvTranspose3d(
            inplanes, planes, kernel_size=(1, 1, 1),
            padding=(0, 0, 0), bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.ConvTranspose3d(
            planes, planes, kernel_size=(1, 3, 3),
            padding=(0, 1, 1), stride=(1, stride, stride),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.ConvTranspose3d(
            planes, round(planes / 4),
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0), bias=False
        )
        self.bn3 = nn.BatchNorm3d(round(planes / 4))
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


# TODO: support other numbers except division by 4
class WaveNet(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', in_chns=3,
                 inplanes=64):
        super(WaveNet, self).__init__()
        self.in_chns = in_chns
        self.inplanes = inplanes
        self.conv1 = nn.Conv3d(
            self.in_chns, self.inplanes, kernel_size=7, stride=(1, 2, 2),
            padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, inplanes, layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], shortcut_type, stride=2
        )
        # conv transpose layers
        self.layer1t = self._make_layer_transpose(
            BasicBlockTranspose, inplanes * 8, layers[3], shortcut_type,
            stride=2
        )
        self.layer2t = self._make_layer_transpose(
            BasicBlockTranspose, inplanes * 4, layers[2], shortcut_type,
            stride=2
        )
        self.layer3t = self._make_layer_transpose(
            BasicBlockTranspose, inplanes * 2, layers[1], shortcut_type,
            stride=2
        )
        self.layer4t = self._make_layer_transpose(
            BasicBlockTranspose, inplanes, layers[0], shortcut_type, stride=2
        )
        self.saliency = conv1x3x3_transpose(
            self.inplanes, 1, padding=(0, 1, 1), stride=1,
            output_padding=(0, 0, 0)
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_transpose(self, block, planes, blocks, shortcut_type,
                              stride=1):
        layers = []
        if blocks > 0:
            upsample = None
            if stride != 1 or self.inplanes != round(planes / block.expansion):
                if shortcut_type == 'A':
                    upsample = partial(upsample_basic_block, stride=stride)
                else:
                    upsample = nn.Sequential(
                        nn.ConvTranspose3d(
                            self.inplanes, round(planes / block.expansion),
                            kernel_size=(1, 3, 3), stride=(1, stride, stride),
                            padding=(0, 1, 1), bias=False,
                            output_padding=(0, 1, 1)
                        ),
                        nn.BatchNorm3d(round(planes / block.expansion))
                    )

            layers.append(
                block(self.inplanes, planes, (0, 1, 1), stride, upsample)
            )
            self.inplanes = round(planes / block.expansion)
            for _ in range(1, blocks):
                layers.append(block(
                    self.inplanes, planes, padding=(0, 1, 1), output_padding=0
                ))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        layers = []
        if blocks > 0:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(
                        downsample_basic_block, planes=planes * block.expansion,
                        stride=stride
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(
                            self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False
                        ),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # in their architecture, channel is first then frame.
        x = x.permute(0, 2, 1, 3, 4)
        input_size = [1, *x.size()[-2:]]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer1t(x)
        x = self.layer2t(x)
        x = self.layer3t(x)
        x = self.layer4t(x)

        x = F.interpolate(x, size=input_size)
        x = self.saliency(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(3), x.size(4))

        return x


def wavenet_basic(planes, **kwargs):
    """Constructs a ResNet-Basic-Custom model.
    """
    model = WaveNet(BasicBlock, planes, **kwargs)
    return model


def wavenet_bottleneck(planes, **kwargs):
    """Constructs a ResNet-Bottleneck-Custom model.
    """
    model = WaveNet(Bottleneck, planes, **kwargs)
    return model
