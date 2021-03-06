"""
A collection of loss functions.
"""

import torch
from torch.nn import functional as nnf


def decomposition_loss(x_pred, x_true, out_space=None):
    if out_space == 'hsv':
        mse = hue_loss(x_pred, x_true)
    else:
        mse = nnf.mse_loss(x_pred, x_true)
    return mse


def decomposition_loss_dict(x_pred, x_true):
    mse = 0
    for key in x_true.keys():
        mse += decomposition_loss(x_pred[key], x_true[key], key)
    return mse


def hue_loss(x_pred, x_true):
    ret = x_pred - x_true
    ret[ret > 1] -= 2
    ret[ret < -1] += 2
    ret = ret ** 2
    return torch.mean(ret)
