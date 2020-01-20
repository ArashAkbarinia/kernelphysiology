"""
WaveNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_funcs
from torch.autograd import Variable
import math

__all__ = [
    'WaveNet', 'wavenet_basic', 'wavenet_bottleneck'
]


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv1x1_transpose(in_channels, out_channels, stride=1):
    """1x1 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x3_transpose(in_channels, out_channels, padding=1, stride=1,
                      output_padding=0):
    """3x3 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, padding=padding,
        stride=stride, output_padding=output_padding, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
                 output_padding=(1, 1)):
        super(BasicBlockTranspose, self).__init__()
        self.conv1 = conv3x3_transpose(
            inplanes, planes, padding, stride, output_padding=output_padding
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_transpose(planes, planes, 1, 1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
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


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        out_planes_expanded = out_planes * self.expansion

        self.conv1c = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2c = conv3x3(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3c = conv1x1(out_planes, out_planes_expanded)
        self.bn3 = nn.BatchNorm2d(out_planes_expanded)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1c(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2c(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3c(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlockTranspose(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, upsample=None):
        super(BottleneckBlockTranspose, self).__init__()
        out_planes_expanded = out_planes * self.expansion
        # TODO: not a nice solution!
        if upsample is not None:
            out_planes_expanded = round(out_planes_expanded / 2)

        self.conv1t = conv1x1_transpose(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2t = conv3x3_transpose(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3t = conv1x1_transpose(out_planes, out_planes_expanded)
        self.bn3 = nn.BatchNorm2d(out_planes_expanded)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1t(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2t(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3t(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class WaveNet(nn.Module):

    def __init__(self, block_type, num_blocks, in_channels=3,
                 num_kernels=64, num_classes=1000):
        super(WaveNet, self).__init__()
        self.in_channels = in_channels
        self.current_planes = num_kernels

        # prior to the systematic layers
        self.preprocess = self._preprocess_layer()

        # start of the systematic convolutional layers
        self.layer1c = self._make_layer(
            block_type, num_kernels, 1 * num_blocks[0], stride=1
        )
        self.layer2c = self._make_layer(
            block_type, num_kernels * 2, num_blocks[1], stride=2
        )
        self.layer3c = self._make_layer(
            block_type, num_kernels * 4, num_blocks[2], stride=2
        )
        self.layer4c = self._make_layer(
            block_type, num_kernels * 8, num_blocks[3], stride=2
        )

        if isinstance(block_type, BasicBlock):
            transpose_block_type = BasicBlockTranspose
        else:
            transpose_block_type = BottleneckBlockTranspose

        # start of the systematic transpose layers
        self.layer1t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 8, num_blocks[3], stride=2
        )
        self.layer2t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 4, num_blocks[2], stride=2
        )
        self.layer3t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 2, num_blocks[1], stride=2
        )
        self.layer4t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 1, num_blocks[0], stride=2
        )

        self.saliency = conv3x3_transpose(
            self.current_planes, 1, padding=1, stride=1, output_padding=0
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _preprocess_layer(self):
        layers = [
            nn.Conv2d(
                self.in_channels, self.current_planes, kernel_size=7, stride=2,
                padding=3, bias=False
            ),
            nn.BatchNorm2d(self.current_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        return nn.Sequential(*layers)

    def _make_layer_transpose(self, block_type, nkernels, num_blocks, stride=1):
        layers = []
        if num_blocks > 0:
            out_channels = round(nkernels * block_type.expansion / 2)
            for _ in range(1, num_blocks):
                layers.append(block_type(self.current_planes, nkernels))

            upsample = None
            if stride != 1 or self.current_planes != out_channels:
                upsample = nn.Sequential(
                    conv1x1_transpose(
                        self.current_planes, out_channels, stride=stride
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(
                block_type(self.current_planes, nkernels, stride, upsample)
            )
            self.current_planes = out_channels

        return nn.Sequential(*layers)

    def _make_layer(self, block_type, nkernels, num_blocks, stride=1):
        layers = []
        if num_blocks > 0:
            out_channels = nkernels * block_type.expansion
            downsample = None
            if stride != 1 or self.current_planes != out_channels:
                downsample = nn.Sequential(
                    conv1x1(
                        self.current_planes, out_channels, stride
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(
                block_type(self.current_planes, nkernels, stride, downsample)
            )

            self.current_planes = out_channels
            for _ in range(1, num_blocks):
                layers.append(block_type(self.current_planes, nkernels))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.preprocess(x)

        x = self.layer1c(x)
        x = self.layer2c(x)
        x = self.layer3c(x)
        x = self.layer4c(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.layer1t(x)
        x = self.layer2t(x)
        x = self.layer3t(x)
        x = self.layer4t(x)

        x = torch_funcs.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )
        x = self.saliency(x)
        x = self.sigmoid(x)
        # x = x.view(x.size(0), x.size(3), x.size(4))

        return x


def wavenet_basic(num_blocks=None, **kwargs):
    """Constructs a WaveNet-Basic-Custom model."""
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    model = WaveNet(BasicBlock, num_blocks, **kwargs)
    return model


def wavenet_bottleneck(num_blocks=None, **kwargs):
    """Constructs a WaveNet-Bottleneck-Custom model."""
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    model = WaveNet(BottleneckBlock, num_blocks, **kwargs)
    return model
