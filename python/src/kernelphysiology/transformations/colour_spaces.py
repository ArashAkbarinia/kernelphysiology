"""
Functions related to change or manipulation of colour spaces.
"""

import numpy as np
import sys

from skimage.color import rgb2lab, lab2rgb

from kernelphysiology.transformations.normalisations import min_max_normalise


def rgb2dkl(x):
    transform_mat = np.array(
        [[0.4252, 1.4304, 0.1444],
         [0.8273, -0.5912, -0.2360],
         [0.2268, 0.7051, -0.9319]]
    )
    return np.dot(x, transform_mat.T)


def dkl2rgb(x):
    transform_mat = np.array(
        [[0.49995, 0.50001495, 0.49999914],
         [0.99998394, -0.29898596, 0.01714922],
         [-0.17577361, 0.15319546, -0.99994349]]
    )
    return np.dot(x, transform_mat)


def rgb2opponency(image_rgb, colour_space='lab'):
    if colour_space == 'lab':
        image_opponent = rgb2lab(image_rgb)
    elif colour_space == 'dkl':
        image_opponent = rgb2dkl(image_rgb)
    else:
        sys.exit('Not supported colour space %s' % colour_space)
    return image_opponent


def opponency2rgb(image_opponent, colour_space='lab'):
    if colour_space == 'lab':
        image_rgb = lab2rgb(image_opponent)
    elif colour_space == 'dkl':
        image_rgb = dkl2rgb(image_opponent)
        image_rgb = min_max_normalise(image_rgb)
    else:
        sys.exit('Not supported colour space %s' % colour_space)
    return image_rgb


def get_max_lightness(colour_space='lab'):
    if colour_space == 'lab':
        max_lightness = 100
    elif colour_space == 'dkl':
        max_lightness = 2
    else:
        sys.exit('Not supported colour space %s' % colour_space)
    return max_lightness
