"""
https://github.com/MichiganCOG/TASED-Net
original target size 224x384
"""

import torch
from torch import nn


class Tased(nn.Module):
    eps = 1e-5
    momentum = 0.1

    def __init__(self, kf=1, in_chns=3):
        super(Tased, self).__init__()
        n64 = int(64 * kf)
        n192 = int(192 * kf)
        n480 = int(480 * kf)
        n832 = int(832 * kf)
        n1024 = int(1024 * kf)

        self.base1 = nn.Sequential(
            SepConv3d(in_chns, n64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                         padding=(0, 1, 1)),
            BasicConv3d(n64, n64, kernel_size=1, stride=1),
            SepConv3d(n64, n192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1))
        self.maxm2 = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1),
                                  padding=(0, 0, 0))
        self.maxt2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1), return_indices=True)
        self.base2 = nn.Sequential(
            Mixed_3b(kf),
            Mixed_3c(kf),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1))
        self.maxm3 = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1),
                                  padding=(0, 0, 0))
        self.maxt3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1), return_indices=True)
        self.base3 = nn.Sequential(
            Mixed_4b(kf),
            Mixed_4c(kf),
            Mixed_4d(kf),
            Mixed_4e(kf),
            Mixed_4f(kf),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1),
                                  padding=(0, 0, 0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2),
                                  padding=(0, 0, 0), return_indices=True)
        self.base4 = nn.Sequential(
            Mixed_5b(kf),
            Mixed_5c(kf),
        )
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(n1024, n1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(n1024, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(n1024, n832, kernel_size=(1, 3, 3), stride=1,
                               padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(n832, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),
        )
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2),
                                      padding=(0, 0, 0))
        self.convtsp2 = nn.Sequential(
            nn.ConvTranspose3d(n832, n480, kernel_size=(1, 3, 3), stride=1,
                               padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(n480, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),
        )
        self.unpool2 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                      padding=(0, 1, 1))
        self.convtsp3 = nn.Sequential(
            nn.ConvTranspose3d(n480, n192, kernel_size=(1, 3, 3), stride=1,
                               padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(n192, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),
        )
        self.unpool3 = nn.MaxUnpool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                      padding=(0, 1, 1))
        self.convtsp4 = nn.Sequential(
            nn.ConvTranspose3d(n192, n64, kernel_size=(1, 4, 4),
                               stride=(1, 2, 2),
                               padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(n64, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),

            nn.Conv3d(n64, n64, kernel_size=(2, 1, 1), stride=(2, 1, 1),
                      bias=False),
            nn.BatchNorm3d(n64, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(n64, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),

            nn.Conv3d(4, 4, kernel_size=(2, 1, 1), stride=(2, 1, 1),
                      bias=False),
            nn.BatchNorm3d(4, eps=Tased.eps, momentum=Tased.momentum,
                           affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(4, 4, kernel_size=(1, 4, 4), stride=(1, 2, 2),
                               padding=(0, 1, 1), bias=False),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # in their architecture, channel is first then frame.
        x = x.permute(0, 2, 1, 3, 4)
        y3 = self.base1(x)
        y = self.maxp2(y3)
        y3 = self.maxm2(y3)
        _, i2 = self.maxt2(y3)
        y2 = self.base2(y)
        y = self.maxp3(y2)
        y2 = self.maxm3(y2)
        _, i1 = self.maxt3(y2)
        y1 = self.base3(y)
        y = self.maxt4(y1)
        y, i0 = self.maxp4(y)
        y0 = self.base4(y)

        z = self.convtsp1(y0)
        z = self.unpool1(z, i0)
        z = self.convtsp2(z)
        z = self.unpool2(z, i1, y2.size())
        z = self.convtsp3(z)
        z = self.unpool3(z, i2, y3.size())
        z = self.convtsp4(z)
        z = z.view(z.size(0), z.size(3), z.size(4))

        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=Tased.eps,
                                 momentum=Tased.momentum,
                                 affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes,
                                kernel_size=(1, kernel_size, kernel_size),
                                stride=(1, stride, stride),
                                padding=(0, padding, padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=Tased.eps,
                                   momentum=Tased.momentum,
                                   affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes,
                                kernel_size=(kernel_size, 1, 1),
                                stride=(stride, 1, 1), padding=(padding, 0, 0),
                                bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=Tased.eps,
                                   momentum=Tased.momentum,
                                   affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self, kf):
        n16 = int(16 * kf)
        n32 = int(32 * kf)
        n64 = int(64 * kf)
        n96 = int(96 * kf)
        n128 = int(128 * kf)
        n192 = int(192 * kf)
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n192, n64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n192, n96, kernel_size=1, stride=1),
            SepConv3d(n96, n128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n192, n16, kernel_size=1, stride=1),
            SepConv3d(n16, n32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n192, n32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self, kf):
        n32 = int(32 * kf)
        n64 = int(64 * kf)
        n96 = int(96 * kf)
        n128 = int(128 * kf)
        n192 = int(192 * kf)
        n256 = int(256 * kf)

        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(n256, n128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n256, n128, kernel_size=1, stride=1),
            SepConv3d(n128, n192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n256, n32, kernel_size=1, stride=1),
            SepConv3d(n32, n96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n256, n64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self, kf):
        n16 = int(16 * kf)
        n48 = int(48 * kf)
        n64 = int(64 * kf)
        n96 = int(96 * kf)
        n192 = int(192 * kf)
        n208 = int(208 * kf)
        n480 = int(480 * kf)
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n480, n192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n480, n96, kernel_size=1, stride=1),
            SepConv3d(n96, n208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n480, n16, kernel_size=1, stride=1),
            SepConv3d(n16, n48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n480, n64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self, kf):
        n24 = int(24 * kf)
        n64 = int(64 * kf)
        n112 = int(112 * kf)
        n160 = int(160 * kf)
        n224 = int(224 * kf)
        n512 = int(512 * kf)
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n512, n160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n512, n112, kernel_size=1, stride=1),
            SepConv3d(n112, n224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n512, n24, kernel_size=1, stride=1),
            SepConv3d(n24, n64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n512, n64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self, kf):
        n24 = int(24 * kf)
        n64 = int(64 * kf)
        n128 = int(128 * kf)
        n256 = int(256 * kf)
        n512 = int(512 * kf)
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n512, n128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n512, n128, kernel_size=1, stride=1),
            SepConv3d(n128, n256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n512, n24, kernel_size=1, stride=1),
            SepConv3d(n24, n64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n512, n64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self, kf):
        n32 = int(32 * kf)
        n64 = int(64 * kf)
        n112 = int(112 * kf)
        n144 = int(144 * kf)
        n288 = int(288 * kf)
        n512 = int(512 * kf)
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n512, n112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n512, n144, kernel_size=1, stride=1),
            SepConv3d(n144, n288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n512, n32, kernel_size=1, stride=1),
            SepConv3d(n32, n64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n512, n64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self, kf):
        n32 = int(32 * kf)
        n128 = int(128 * kf)
        n160 = int(160 * kf)
        n256 = int(256 * kf)
        n320 = int(320 * kf)
        n528 = int(528 * kf)
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n528, n256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n528, n160, kernel_size=1, stride=1),
            SepConv3d(n160, n320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n528, n32, kernel_size=1, stride=1),
            SepConv3d(n32, n128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n528, n128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self, kf):
        n32 = int(32 * kf)
        n128 = int(128 * kf)
        n160 = int(160 * kf)
        n256 = int(256 * kf)
        n320 = int(320 * kf)
        n832 = int(832 * kf)
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n832, n256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n832, n160, kernel_size=1, stride=1),
            SepConv3d(n160, n320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n832, n32, kernel_size=1, stride=1),
            SepConv3d(n32, n128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n832, n128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self, kf):
        n48 = int(48 * kf)
        n128 = int(128 * kf)
        n192 = int(192 * kf)
        n384 = int(384 * kf)
        n832 = int(832 * kf)
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(n832, n384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(n832, n192, kernel_size=1, stride=1),
            SepConv3d(n192, n384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(n832, n48, kernel_size=1, stride=1),
            SepConv3d(n48, n128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(n832, n128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
