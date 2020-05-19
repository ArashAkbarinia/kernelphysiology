"""
Preparing the input image to be inputted to a network.
"""

import numpy as np

import cv2

from kernelphysiology.utils import imutils
from kernelphysiology.transformations import colour_spaces
from kernelphysiology.transformations import normalisations


class ColourTransformation(object):

    def __init__(self, colour_inds, colour_space='rgb'):
        self.colour_inds = colour_inds
        self.colour_space = colour_space

    def __call__(self, img):
        if self.colour_space != 'rgb' or self.colour_inds is not None:
            img = np.asarray(img).copy()
            if self.colour_space == 'lab':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            elif self.colour_space == 'labhue':
                img_hue = colour_spaces.lab2lch01(
                    colour_spaces.rgb2opponency(img.copy(), 'lab')
                )
                img_hue = normalisations.uint8im(img_hue)
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                img = np.concatenate([img_lab, img_hue[:, :, 2:3]], axis=2)
            elif self.colour_space == 'dkl':
                img = colour_spaces.rgb2dkl01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'hsv':
                img = colour_spaces.rgb2hsv01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'lms':
                img = colour_spaces.rgb2lms01(img)
                img = normalisations.uint8im(img)
            elif self.colour_space == 'yog':
                img = colour_spaces.rgb2yog01(img)
                img = normalisations.uint8im(img)

            # if colour_inds is None, we consider it as trichromat
            if self.colour_inds is not None:
                # TODO: 0 contrast lightness means all to be 50
                if self.colour_inds == 0:
                    img[:, :, self.colour_inds] = 50
                else:
                    img[:, :, self.colour_inds] = 0
            # FIXME: right now it's only for lab
            if self.colour_space == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return img


class ChannelTransformation(object):

    def __init__(self, colour_inds, colour_space='rgb'):
        self.colour_inds = [0, 1, 2]
        self.colour_inds = [e for e in self.colour_inds if e not in colour_inds]
        self.colour_space = colour_space

    def __call__(self, img):
        if self.colour_space == 'rgb':
            return img
        else:
            img = img[self.colour_inds]
            return img


class MosaicTransformation(object):

    def __init__(self, mosaic_pattern):
        self.mosaic_pattern = mosaic_pattern

    def __call__(self, img):
        img = np.asarray(img).copy()
        img = imutils.im2mosaic(img, self.mosaic_pattern)
        return img


class UniqueTransformation(object):

    def __init__(self, manipulation_function, **kwargs):
        self.manipulation_function = manipulation_function
        self.kwargs = kwargs

    def __call__(self, x):
        x = self.manipulation_function(x, **self.kwargs)
        return x
