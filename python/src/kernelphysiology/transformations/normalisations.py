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
        # FIXME: check if this has consequences some where
        # if 1 < max_pixel <= 255:
        #     return image / 255, 255
        # else:
        image /= max_pixel
        return image, max_pixel


def im2double(image):
    image, _ = im2double_max(image)
    return image
