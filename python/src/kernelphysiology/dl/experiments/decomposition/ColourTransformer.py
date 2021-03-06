"""
A model to learn parameters of a LAB-like transformation.
"""

import torch
from torch import nn
import torch.utils.data

from kernelphysiology.dl.pytorch.optimisations import losses


class LabTransformer(nn.Module):
    def __init__(self, trans_mat=None, ref_white=None, distortion=None,
                 trans_mat_gr=True, ref_white_gr=True, distortion_gr=True,
                 linear=True):
        super(LabTransformer, self).__init__()

        if trans_mat is None:
            trans_mat = torch.rand((3, 3))
        else:
            trans_mat = torch.tensor(trans_mat)
        self.trans_mat = nn.Parameter(trans_mat, requires_grad=trans_mat_gr)

        if ref_white is None:
            ref_white = torch.rand(3)
        else:
            ref_white = torch.tensor(ref_white)
        self.ref_white = nn.Parameter(ref_white, requires_grad=ref_white_gr)

        self.linear = linear
        if not self.linear:
            if distortion is None:
                distortion = torch.rand(5)
            else:
                distortion = torch.tensor(distortion)
            self.distortion = nn.Parameter(
                distortion, requires_grad=distortion_gr
            )

        self.rec_mse = 0
        self.inv_mse = 0
        self.out_mse = 0
        # vals = [116.0, 16.0, 500.0, 200.0, 0.2068966]
        # vals = [e / 500 for e in vals]

    def forward(self, x):
        x = self.rgb2rnd(x)
        x = torch.tanh(x)
        return x

    def rgb2rnd(self, rgb):
        rgb_arr = torch.zeros(rgb.shape, device=rgb.get_device())
        for i in range(3):
            x_r = rgb[:, 0:1, ] * self.trans_mat[i, 0]
            y_g = rgb[:, 1:2, ] * self.trans_mat[i, 1]
            z_b = rgb[:, 2:3, ] * self.trans_mat[i, 2]
            rgb_arr[:, i:i + 1, ] = x_r + y_g + z_b

        # scale by tristimulus values of the reference white point
        ref_white = self.ref_white + 1e-4
        white_arr = torch.zeros(rgb.shape, device=rgb.get_device())
        for i in range(3):
            white_arr[:, i:i + 1, ] = rgb_arr[:, i:i + 1, ] / ref_white[i]

        if not self.linear:
            vals = self.distortion + 1e-4
            eta = vals[4]
            m = (1 / 3) * (eta ** -2)
            t0 = eta ** 3

            # Nonlinear distortion and linear transformation
            mask = rgb_arr > t0
            rgb_arr[mask] = rgb_arr[mask].pow(1 / 3)
            rgb_arr[~mask] = m * rgb_arr[~mask] + (vals[1] / vals[0])

            x = rgb_arr[:, 0:1, ]
            y = rgb_arr[:, 1:2, ]
            z = rgb_arr[:, 2:3, ]

            # Vector scaling
            L = (vals[0] * y) - vals[1]
            a = vals[2] * (x - y)
            b = vals[3] * (y - z)

            nonlin_arr = torch.zeros(rgb.shape, device=rgb.get_device())
            nonlin_arr[:, 0:1, ] = L
            nonlin_arr[:, 1:2, ] = a
            nonlin_arr[:, 2:3, ] = b

            output = nonlin_arr
        else:
            output = white_arr
        return output

    def rnd2rgb(self, rnd, clip=False):
        rnd = torch.atanh(rnd)
        rnd_arr = torch.zeros(rnd.shape, device=rnd.get_device())

        if not self.linear:
            vals = self.distortion + 1e-4
            eta = vals[4]

            L = rnd_arr[:, 0:1, ]
            a = rnd_arr[:, 1:2, ]
            b = rnd_arr[:, 2:3, ]
            y = (L + vals[1]) / vals[0]
            x = (a / vals[2]) + y
            z = y - (b / vals[3])

            rnd_arr[:, 0:1, ] = x
            rnd_arr[:, 1:2, ] = y
            rnd_arr[:, 2:3, ] = z

            mask = rnd_arr > eta
            rnd_arr[mask] = rnd_arr[mask].pow(3.)
            rnd_arr[~mask] = (rnd_arr[~mask] - (vals[1] / vals[0])) * 3 * (
                    eta ** 2)
        else:
            for i in range(3):
                rnd_arr[:, i:i + 1, ] = rnd[:, i:i + 1, ]

        # rescale to the reference white (illuminant)
        ref_white = self.ref_white + 1e-4
        white_arr = torch.zeros(rnd.shape, device=rnd.get_device())
        for i in range(3):
            white_arr[:, i:i + 1, ] = rnd_arr[:, i:i + 1, ] * ref_white[i]

        rgb = torch.zeros(rnd.shape, device=rnd.get_device())
        rgb_transform = torch.inverse(self.trans_mat)
        for i in range(3):
            x_r = white_arr[:, 0:1, ] * rgb_transform[i, 0]
            y_g = white_arr[:, 1:2, ] * rgb_transform[i, 1]
            z_b = white_arr[:, 2:3, ] * rgb_transform[i, 2]
            rgb[:, i:i + 1, ] = x_r + y_g + z_b

        if clip:
            rgb[torch.isnan(rgb)] = 0
            rgb = (rgb * 0.5) + 0.5
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        return rgb

    def loss_function(self, out_space, in_space, model_rec):
        self.rec_mse = losses.decomposition_loss(model_rec, out_space)

        target_rgb = self.rnd2rgb(out_space, clip=True)
        self.inv_mse = losses.decomposition_loss(target_rgb, in_space)

        model_rgb = self.rnd2rgb(model_rec, clip=True)
        self.out_mse = losses.decomposition_loss(model_rgb, in_space)

        return self.rec_mse + self.inv_mse + self.out_mse

    def latest_losses(self):
        return {
            'rec_mse': self.rec_mse, 'inv_mse': self.inv_mse,
            'out_mse': self.out_mse
        }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inchns, outchns, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(outchns * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inchns, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, outchns * self.expansion)
        self.bn3 = norm_layer(outchns * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNetTransformer(nn.Module):
    def __init__(self, layers=1):
        super(ResNetTransformer, self).__init__()

        self.layers = layers

        for i in range(layers):
            setattr(
                self, 'e%.3d' % i, Bottleneck(3, 3, norm_layer=nn.BatchNorm2d)
            )
        for i in range(layers):
            setattr(
                self, 'd%.3d' % i, Bottleneck(3, 3, norm_layer=nn.BatchNorm2d)
            )

        self.rec_mse = 0
        self.inv_mse = 0
        self.out_mse = 0

    def forward(self, x):
        rnd = self.e000(x)
        for i in range(1, self.layers):
            rnd = getattr(self, 'e%.3d' % i)(rnd)
        rnd = torch.tanh(rnd)

        rgb = self.d000(rnd)
        for i in range(1, self.layers):
            rgb = getattr(self, 'd%.3d' % i)(rgb)
        rgb = torch.tanh(rgb)
        return rnd, rgb

    def rnd2rgb(self, x):
        for i in range(self.layers):
            x = getattr(self, 'd%.3d' % i)(x)
        rgb = torch.tanh(x)
        return rgb

    def loss_function(self, y, x_inv, y_rec, x):
        self.rec_mse = losses.decomposition_loss(y_rec, y)
        self.inv_mse = losses.decomposition_loss(x_inv, x)

        return self.rec_mse + self.inv_mse + self.out_mse

    def latest_losses(self):
        return {
            'rec_mse': self.rec_mse, 'inv_mse': self.inv_mse,
            'out_mse': self.out_mse
        }


class ConvTransformer(nn.Module):
    def __init__(self, layers=1, bias=False, tmat=None, bvec=None):
        super(ConvTransformer, self).__init__()

        self.layers = layers

        if bvec is not None:
            bias = True
        self.bias = bias

        for i in range(layers):
            setattr(
                self, 't%.3d' % i, nn.Sequential(
                    nn.Conv2d(3, 3, 1, 1, groups=1, bias=self.bias),
                    nn.Tanh()
                )
            )
        if tmat is not None:
            for i in range(self.layers):
                getattr(self, 't%.3d' % i)[0].weight = nn.Parameter(
                    torch.tensor(tmat).float()
                )
        if bvec is not None:
            for i in range(self.layers):
                getattr(self, 't%.3d' % i)[0].bias = nn.Parameter(
                    torch.tensor(bvec).float()
                )

        self.rec_mse = 0
        self.inv_mse = 0
        self.out_mse = 0

    def forward(self, x):
        for i in range(self.layers):
            x = getattr(self, 't%.3d' % i)(x)
        return x

    def rnd2rgb(self, y):
        for i in range(self.layers - 1, -1, -1):
            trans_mat = getattr(self, 't%.3d' % i)[0].weight.detach().squeeze()
            trans_mat = torch.inverse(trans_mat)

            y = torch.atanh(y)
            if self.bias:
                bias_vec = getattr(self, 't%.3d' % i)[0].bias.detach().squeeze()
                for i in range(3):
                    y[:, i, ] -= bias_vec[i]
            device = y.get_device()
            device = 'cpu' if device == -1 else device
            x = torch.zeros(y.shape, device=device)
            for i in range(3):
                x_r = y[:, 0:1, ] * trans_mat[i, 0]
                y_g = y[:, 1:2, ] * trans_mat[i, 1]
                z_b = y[:, 2:3, ] * trans_mat[i, 2]
                x[:, i:i + 1, ] = x_r + y_g + z_b
            y = x.clone()
        return x

    def loss_function(self, y, x, y_rec):
        self.rec_mse = losses.decomposition_loss(y_rec, y)

        x_inv = self.rnd2rgb(y)
        self.inv_mse = losses.decomposition_loss(x_inv, x)

        # y_rec_inv = self.rnd2rgb(y_rec)
        # self.out_mse = losses.decomposition_loss(y_rec_inv, x)

        return self.rec_mse + self.inv_mse + self.out_mse

    def latest_losses(self):
        return {
            'rec_mse': self.rec_mse, 'inv_mse': self.inv_mse,
            'out_mse': self.out_mse
        }
