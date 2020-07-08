"""
Collection of different normalisation of a signal.
"""

import numpy as np


def min_max_normalise(x, low=0, high=1, minv=None, maxv=None):
    if minv is None:
        minv = x.min()
    if maxv is None:
        maxv = x.max()
    output = low + (x - minv) * (high - low) / (maxv - minv)
    return output


def im2double_max(image):
    if image.dtype == 'uint8':
        image = image.astype('float32')
        return image / 255, 255
    else:
        image = image.astype('float32')
        max_pixel = np.max(image)
        image /= max_pixel
        return image, max_pixel


def img_midvals(image):
    if image.dtype == 'uint8':
        return (128, 128, 128)
    else:
        image = image.astype('float32')
        max_pixel = np.max(image)
        if max_pixel > 1.0:
            # TODO: maybe considering channelwise maxs
            midvals = max_pixel / 2
            midvals = (midvals, midvals, midvals)
        else:
            midvals = (0.5, 0.5, 0.5)
        return midvals


def im2double(image):
    image, _ = im2double_max(image)
    return image


def rgb2double(x):
    if x.dtype == 'uint8':
        x = im2double(x)
    else:
        assert x.max() <= 1, 'rgb must be either uint8 or in the range of [0 1]'
    return x


def clip01(x):
    x = np.maximum(x, 0)
    x = np.minimum(x, 1)
    return x


def uint8im(image):
    image = clip01(image)
    image *= 255
    return image.astype('uint8')
