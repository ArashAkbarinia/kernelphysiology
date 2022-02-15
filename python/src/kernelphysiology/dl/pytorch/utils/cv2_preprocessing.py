"""
Preparing the input image to be inputted to a network.
"""

import sys
import numpy as np
import random

from PIL import Image as PilImage
import cv2

from kernelphysiology.dl.pytorch.utils.cv2_transforms import _call_recursive
from kernelphysiology.utils import imutils
from kernelphysiology.transformations import colour_spaces
from kernelphysiology.transformations import frequency_domains
from kernelphysiology.transformations import normalisations

colour_conversions = [
    'dkl', 'lab', 'lch', 'lms', 'hsv',
]


class MultipleOutputTransformation(object):

    def __init__(self, outputs):
        self.out_funs = dict()
        for out_type in outputs:
            if out_type == 'input':
                self.out_funs[out_type] = None
            else:
                self.out_funs[out_type] = {
                    'fun': DecompositionTransformation(out_type),
                    'kwargs': dict()
                }

    def __call__(self, imgs):
        org_img, processed_img = imgs
        output_imgs = dict()
        for key, val in self.out_funs.items():
            if key == 'input':
                output_imgs[key] = processed_img
            else:
                output_imgs[key] = val['fun'](org_img, **val['kwargs'])
        return output_imgs


class DecompositionTransformation(object):
    def __init__(self, deompose_type):
        self.decompose_type = deompose_type.lower()
        self.decompose_fun = None
        for tmp in colour_spaces.SUPPORTED_COLOUR_SPACES:
            if tmp in self.decompose_type:
                self.decompose_fun = colour_spaces.rgb2all
                break
        if self.decompose_fun is None:
            for tmp in frequency_domains.SUPPORTED_WAVELETS:
                if tmp in self.decompose_type:
                    self.decompose_fun = frequency_domains.rgb2all
                    break
        if self.decompose_fun is None:
            sys.exit('Unsupported decomposition %s.' % self.decompose_type)

    def __call__(self, img):
        img = self.decompose_fun(img, self.decompose_type)
        return img


class ColourSpaceTransformation(object):

    def __init__(self, colour_space='rgb'):
        self.colour_space = colour_space

    def __call__(self, img):
        # TODO: move the if statmenets to a separate function to speed up.
        if self.colour_space != 'rgb':
            img = np.asarray(img).copy()
            if self.colour_space == 'lab':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
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


class SpatialFrequencyTransformation(object):

    def __init__(self, sf_filter, sf_filter_chn):
        self.hsf_cut, self.lsf_cut = sf_filter
        if sf_filter_chn is None:
            self.chn_info = None
        else:
            w_colour_space, w_chn = sf_filter_chn.split('_')
            if w_colour_space == 'dkl':
                self.chn_info = dict()
                self.chn_info['f_convert'] = colour_spaces.rgb2dkl01
                self.chn_info['b_convert'] = colour_spaces.dkl012rgb
                self.chn_info['chn_ind'] = 0
                if w_chn == 'rg':
                    self.chn_info['chn_ind'] = 1
                elif w_chn == 'yb':
                    self.chn_info['chn_ind'] = 2

    def __call__(self, x):
        kwargs = {
            'hsf_cut': self.hsf_cut, 'lsf_cut': self.lsf_cut,
            'chn_info': self.chn_info
        }
        if type(x) is list:
            return _call_recursive(x, imutils.filter_img_sf, **kwargs)
        x = imutils.filter_img_sf(x, **kwargs)
        return x


class RotateHueTransformation(object):

    def __init__(self, hue_angle):
        self.hue_angle = hue_angle

    def __call__(self, x):
        kwargs = {'hue_angle': self.hue_angle}
        if type(x) is list:
            return _call_recursive(x, imutils.rotate_hue, **kwargs)
        x = imutils.rotate_hue(x, **kwargs)
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
            augmentation_inds = random.sample(self.all_inds, self.num_augmentations)
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


class PredictionTransformationPilSeg(object):

    def __init__(self, parameters, colour_space='rgb', tmp_c=False):
        # FIXME: tmp_c
        self.parameters = parameters
        self.colour_space = colour_space.upper()
        self.tmpc = tmp_c

    def __call__(self, x, target):
        x = np.asarray(x).copy()

        manipulation_function = self.parameters['function']
        kwargs = self.parameters['kwargs']
        if self.colour_space.lower() != 'rgb' and self.tmpc:
            kwargs['colour_space'] = None

        x = manipulation_function(x, **kwargs)
        x = PilImage.fromarray(x)
        return x, target
