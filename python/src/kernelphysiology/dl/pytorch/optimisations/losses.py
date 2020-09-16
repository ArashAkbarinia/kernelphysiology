"""
A collection of loss functions.
"""

import torch
from torch.nn import functional as nnf


def decomposition_loss(x_pred, x_true):
    mse = 0
    for key in x_true.keys():
        if key == 'hsv':
            mse += hue_loss(x_pred[key], x_true[key])
        else:
            mse += nnf.mse_loss(x_pred[key], x_true[key])
    return mse


def hue_loss(x_pred, x_true):
    ret = x_pred - x_true
    ret[ret > 1] -= 2
    ret[ret < -1] += 2
    ret = ret ** 2
    return torch.mean(ret)
