"""
Preparing the input image to be inputted to a network.
"""

import sys
import numpy as np
import random

import cv2

from kernelphysiology.dl.pytorch.utils.cv2_transforms import _call_recursive
from kernelphysiology.utils import imutils
from kernelphysiology.transformations import colour_spaces
from kernelphysiology.transformations import normalisations


class ColourSpaceTransformation(object):

    def __init__(self, colour_space='rgb'):
        self.colour_space = colour_space

    def __call__(self, img):
        # TODO: move the if statmenets to a separate function to speed up.
        if self.colour_space != 'rgb':
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
            elif self.colour_space == 'gry':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # For convenience we represent a grey-scale image in 3 channels.
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                sys.exit(
                    'ColourSpaceTransformation does not support %s.' %
                    self.colour_space
                )
        return img


class VisionTypeTransformation(object):

    def __init__(self, colour_inds, colour_space='rgb'):
        self.colour_inds = colour_inds
        self.colour_space = colour_space

    def __call__(self, img):
        if self.colour_space != 'rgb' or self.colour_inds is not None:
            img = np.asarray(img).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

            # if colour_inds is None, we consider it as trichromat
            if self.colour_inds is not None:
                # TODO: 0 contrast lightness means all to be 50
                if self.colour_inds == 0:
                    img[:, :, self.colour_inds] = 50
                else:
                    img[:, :, self.colour_inds] = 128
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

    def __call__(self, x):
        if type(x) is list:
            kwargs = {'mosaic_type': self.mosaic_pattern}
            return _call_recursive(x, imutils.im2mosaic, **kwargs)
        x = imutils.im2mosaic(x, self.mosaic_pattern)
        return x


class UniqueTransformation(object):

    def __init__(self, manipulation_function, **kwargs):
        self.manipulation_function = manipulation_function
        self.kwargs = kwargs

    def __call__(self, x):
        if type(x) is list:
            return _call_recursive(x, self.manipulation_function, **self.kwargs)
        x = self.manipulation_function(x, **self.kwargs)
        return x


class RandomAugmentationTransformation(object):

    def __init__(self, augmentation_settings, num_augmentations=None):
        self.augmentation_settings = augmentation_settings
        self.num_augmentations = num_augmentations
        self.all_inds = [*range(len(self.augmentation_settings))]

    def __call__(self, x):
        # finding the manipulations to be applied
        if self.num_augmentations is None:
            augmentation_inds = self.all_inds
        else:
            augmentation_inds = random.sample(
                self.all_inds, self.num_augmentations
            )
            # sorting it according to the order provided by user
            augmentation_inds.sort()

        for i in augmentation_inds:
            current_augmentation = self.augmentation_settings[i].copy()
            manipulation_function = current_augmentation['function']
            kwargs = current_augmentation['kwargs']

            # if the value is in the form of a list, we select a random value
            # in between those numbers
            for key, val in kwargs.items():
                if isinstance(val, list):
                    kwargs[key] = random.uniform(*val)

            x = manipulation_function(x, **kwargs)
        return x


class PredictionTransformation(object):

    def __init__(self, parameters, colour_space='rgb', tmp_c=False):
        # FIXME: tmp_c
        self.parameters = parameters
        self.colour_space = colour_space.upper()
        self.tmpc = tmp_c

    def __call__(self, x):
        manipulation_function = self.parameters['function']
        kwargs = self.parameters['kwargs']
        if self.colour_space.lower() != 'rgb' and self.tmpc:
            kwargs['colour_space'] = None

        x = manipulation_function(x, **kwargs)
        return x
