"""
Preparing the input image to be inputted to a network.
"""

import numpy as np
import random
import warnings

from PIL import Image as PilImage
from PIL import ImageCms

from kernelphysiology.utils.imutils import get_colour_inds

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
            # if colour_inds is None, we consider it as trichromat
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


def colour_transformation(transformation_type, colour_space='rgb'):
    ct = []
    if colour_space != 'lms':
        colour_inds = get_colour_inds(transformation_type)
        ct.append(ColourTransformation(colour_inds, colour_space))
    return ct


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


def channel_transformation(transformation_type, colour_space='rgb'):
    ct = []
    if transformation_type != 'trichromat':
        colour_inds = get_colour_inds(transformation_type)
        # check if it's a valid colour index
        if colour_inds is not None:
            ct.append(ChannelTransformation(colour_inds, colour_space))
        else:
            warnings.warn('Unsupported colour transformation %s' % colour_space)
    return ct


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    # inverting the normalisation for each channel
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] * std[i]) + mean[i]
    tensor = tensor.clamp(0, 1)
    return tensor


def normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    # normalising the channels
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] - mean[i]) / std[i]
    return tensor


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
        if self.colour_space != 'rgb' and self.tmpc:
            kwargs['colour_space'] = None

        x = manipulation_function(x, **kwargs)

        # converting it back to pil image
        if self.is_pill_img:
            x = PilImage.fromarray(x.astype('uint8'), self.colour_space)
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
                    kwargs[key] = np.random.uniform(*val)

            x = manipulation_function(x, **kwargs)

        # converting it back to pil image
        if self.is_pill_img:
            x = PilImage.fromarray(x.astype('uint8'), 'RGB')
        return x
