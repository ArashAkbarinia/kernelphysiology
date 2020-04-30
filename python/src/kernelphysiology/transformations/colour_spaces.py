"""
Functions related to change or manipulation of colour spaces.
"""

import numpy as np
import sys

import cv2

from kernelphysiology.transformations import normalisations

dkl_from_rgb = np.array(
    [[+0.49995000, +0.50001495, +0.49999914],
     [+0.99998394, -0.29898596, +0.01714922],
     [-0.17577361, +0.15319546, -0.99994349]]
)

rgb_from_dkl = np.array(
    [[+0.4252, +1.4304, +0.1444],
     [+0.8273, -0.5912, -0.2360],
     [+0.2268, +0.7051, -0.9319]]
).T


def rgb012dkl(x):
    return np.dot(x, rgb_from_dkl)


def rgb2dkl(x):
    return rgb012dkl(normalisations.rgb2double(x))


def rgb2dkl01(x):
    x = rgb2dkl(x)
    x /= 2
    x[:, :, 1] += 0.5
    x[:, :, 2] += 0.5
    return x


def dkl2rgb(x):
    return normalisations.uint8im(dkl2rgb01(x))


def dkl2rgb01(x):
    return np.dot(x, dkl_from_rgb)


def dkl012rgb(x):
    return normalisations.uint8im(dkl012rgb01(x))


def dkl012rgb01(x):
    x[:, :, 1] -= 0.5
    x[:, :, 2] -= 0.5
    x *= 2
    return dkl2rgb01(x)


def rgb2hsv01(x):
    assert x.dtype == 'uint8'
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x = x.astype('float')
    x[:, :, 0] /= 180
    x[:, :, 1:] /= 255
    return x


def hsv012rgb(x):
    x[:, :, 0] *= (180 / 255)
    x = normalisations.uint8im(x)
    return cv2.cvtColor(x, cv2.COLOR_HSV2RGB)


def rgb2opponency(image_rgb, opponent_space='lab'):
    image_rgb = normalisations.rgb2double(image_rgb)
    if opponent_space is None:
        # it's already in opponency
        image_opponent = image_rgb
    elif opponent_space == 'lab':
        image_opponent = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    elif opponent_space == 'dkl':
        image_opponent = rgb012dkl(image_rgb)
    else:
        sys.exit('Not supported colour space %s' % opponent_space)
    return image_opponent


def opponency2rgb(image_opponent, opponent_space='lab'):
    if opponent_space is None:
        # it's already in rgb
        image_rgb = image_opponent
    elif opponent_space == 'lab':
        image_rgb = cv2.cvtColor(image_opponent, cv2.COLOR_LAB2RGB)
        image_rgb = normalisations.uint8im(image_rgb)
    elif opponent_space == 'dkl':
        image_rgb = dkl2rgb(image_opponent)
    else:
        sys.exit('Not supported colour space %s' % opponent_space)
    return image_rgb


def get_max_lightness(opponent_space='lab'):
    if opponent_space is None:
        # it's already in rgb
        max_lightness = 255
    elif opponent_space == 'lab':
        max_lightness = 100
    elif opponent_space == 'dkl':
        max_lightness = 2
    else:
        sys.exit('Not supported colour space %s' % opponent_space)
    return max_lightness
