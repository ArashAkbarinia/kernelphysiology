"""
A model to learn parameters of a LAB-like transformation.
"""

import abc
import numpy as np
import logging

import torch
from torch import nn
import torch.utils.data

from kernelphysiology.dl.experiments.decomposition import nearest_embed
from kernelphysiology.dl.pytorch.optimisations import losses


class LabTransformer(nn.Module):
    def __init__(self, trans_mat=None, ref_white=None, distortion=None,
                 linear=True):
        super(LabTransformer, self).__init__()

        if trans_mat is None:
            trans_mat = torch.rand((3, 3))
        else:
            trans_mat = torch.tensor(trans_mat)
        self.trans_mat = nn.Parameter(trans_mat, requires_grad=True)

        if ref_white is None:
            ref_white = torch.rand(3)
        else:
            ref_white = torch.tensor(ref_white)
        self.ref_white = nn.Parameter(ref_white, requires_grad=True)

        self.linear = linear
        if not self.linear:
            if distortion is None:
                distortion = torch.rand(5)
            else:
                distortion = torch.tensor(distortion)
            self.distortion = nn.Parameter(distortion, requires_grad=True)

        self.rec_mse = 0
        # vals = [116.0, 16.0, 500.0, 200.0, 0.2068966]
        # vals = [e / 500 for e in vals]

    def forward(self, x):
        return self.rgb2rnd(x)

    def rgb2rnd(self, rgb):
        arr = rgb.clone()
        for i in range(3):
            x_r = rgb[:, 0:1, ] * self.trans_mat[i, 0]
            y_g = rgb[:, 1:2, ] * self.trans_mat[i, 1]
            z_b = rgb[:, 2:3, ] * self.trans_mat[i, 2]
            arr[:, i:i + 1, ] = x_r + y_g + z_b

        # scale by tristimulus values of the reference white point
        for i in range(3):
            arr[:, i:i + 1, ] /= (self.ref_white[i] + 1e-4)

        if not self.linear:
            vals = self.distortion
            vals += 1e-4
            eta = vals[4]
            m = (1 / 3) * (eta ** -2)
            t0 = eta ** 3

            # Nonlinear distortion and linear transformation
            mask = arr > t0
            arr[mask] = arr[mask].pow(1 / 3)
            arr[~mask] = m * arr[~mask] + (vals[1] / vals[0])

            x = arr[:, 0:1, ]
            y = arr[:, 1:2, ]
            z = arr[:, 2:3, ]

            # Vector scaling
            L = (vals[0] * y) - vals[1]
            a = vals[2] * (x - y)
            b = vals[3] * (y - z)

            arr[:, 0:1, ] = L
            arr[:, 1:2, ] = a
            arr[:, 2:3, ] = b
        return arr

    def rnd2rgb(self, rnd, clip=False):
        arr = rnd.clone()

        if not self.linear:
            vals = self.distortion
            vals += 1e-4
            eta = vals[4]

            L = arr[:, 0:1, ]
            a = arr[:, 1:2, ]
            b = arr[:, 2:3, ]
            y = (L + vals[1]) / vals[0]
            x = (a / vals[2]) + y
            z = y - (b / vals[3])

            # invalid = torch.nonzero(z < 0)
            # if invalid.shape[0] > 0:
            #     z[invalid] = 0

            arr[:, 0:1, ] = x
            arr[:, 1:2, ] = y
            arr[:, 2:3, ] = z

            mask = arr > eta
            arr[mask] = arr[mask].pow(3.)
            arr[~mask] = (arr[~mask] - (vals[1] / vals[0])) * 3 * (eta ** 2)

        # rescale to the reference white (illuminant)
        for i in range(3):
            arr[:, i:i + 1, ] *= (self.ref_white[i] + 1e-4)

        rgb = arr.clone()
        rgb_transform = np.linalg.inv(self.trans_mat.detach().cpu())
        for i in range(3):
            x_r = arr[:, 0:1, ] * rgb_transform[i, 0]
            y_g = arr[:, 1:2, ] * rgb_transform[i, 1]
            z_b = arr[:, 2:3, ] * rgb_transform[i, 2]
            rgb[:, i:i + 1, ] = x_r + y_g + z_b

        if clip:
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        return rgb

    def loss_function(self, out_space, in_space, model_rec):
        self.rec_mse = losses.decomposition_loss(model_rec, out_space)

        return self.rec_mse

    def latest_losses(self):
        return {
            'rec_mse': self.rec_mse
        }
