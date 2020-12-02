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
    def __init__(self):
        super(LabTransformer, self).__init__()
        self.trans_mat = torch.rand((3, 3))
        self.ref_white = torch.rand(3)

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
            arr[:, i:i + 1, ] /= self.ref_white[i]

        # Nonlinear distortion and linear transformation
        mask = arr > 0.008856
        arr[mask] = arr[mask].pow(1 / 3)
        arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

        x = arr[:, 0:1, ]
        y = arr[:, 1:2, ]
        z = arr[:, 2:3, ]

        # Vector scaling
        vals = [116.0, 16.0, 500.0, 200.0]
        L = (vals[0] * y) - vals[1]
        a = vals[2] * (x - y)
        b = vals[3] * (y - z)

        arr[:, 0:1, ] = L
        arr[:, 1:2, ] = a
        arr[:, 2:3, ] = b
        return arr
