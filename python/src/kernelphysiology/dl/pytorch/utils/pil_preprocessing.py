"""
Transformations in PIL.
"""

import numpy as np
import random

from PIL import Image as PilImage
from PIL import ImageCms

from kernelphysiology.utils import imutils

rgb_p = ImageCms.createProfile('sRGB')
lab_p = ImageCms.createProfile('LAB')

rgb2lab = ImageCms.buildTransformFromOpenProfiles(rgb_p, lab_p, 'RGB', 'LAB')
lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, rgb_p, 'LAB', 'RGB')


class ColourTransformation(object):

    def __init__(self, colour_inds, colour_space='rgb'):
        self.colour_inds = colour_inds
        self.colour_space = colour_space

    def __call__(self, img):
        if self.colour_space == 'lab' or self.colour_inds is not None:
            img = ImageCms.applyTransform(img, rgb2lab)
            if self.colour_inds is not None:
                img = np.asarray(img).copy()
                # TODO: 0 contrast lightness means all to be 50
                if self.colour_inds == 0:
                    img[:, :, self.colour_inds] = 50
                else:
                    img[:, :, self.colour_inds] = 0
                img = PilImage.fromarray(img, 'LAB')
            if self.colour_space == 'rgb':
                img = ImageCms.applyTransform(img, lab2rgb)
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
        img = PilImage.fromarray(img)
        return img


class PredictionTransformation(object):

    def __init__(self, parameters, is_pill_img=True, colour_space='rgb',
                 tmp_c=False):
        self.parameters = parameters
        self.is_pill_img = is_pill_img
        self.colour_space = colour_space.upper()
        self.tmpc = tmp_c

    def __call__(self, x):
        if self.is_pill_img:
            x = np.asarray(x, dtype='uint8')

        manipulation_function = self.parameters['function']
        kwargs = self.parameters['kwargs']
        if self.colour_space.lower() != 'rgb' and self.tmpc:
            kwargs['colour_space'] = None

        x = manipulation_function(x, **kwargs)

        # converting it back to pil image
        if self.is_pill_img:
            x = PilImage.fromarray(np.uint8(x), self.colour_space)
        return x


class RandomAugmentationTransformation(object):

    def __init__(self, augmentation_settings, num_augmentations=None,
                 is_pill_img=True):
        self.augmentation_settings = augmentation_settings
        self.num_augmentations = num_augmentations
        self.all_inds = [*range(len(self.augmentation_settings))]
        self.is_pill_img = is_pill_img

    def __call__(self, x):
        if self.is_pill_img:
            x = np.asarray(x, dtype='uint8')

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

        # converting it back to pil image
        if self.is_pill_img:
            x = PilImage.fromarray(np.uint8(x), 'RGB')
        return x
