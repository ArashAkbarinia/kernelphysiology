"""
Preparing the input image to be inputted to a network.
"""

import numpy as np
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
            # TODO: only for LAB that 0 contrast lightness means all to be 50
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
            img = img[self.colour_inds,]
            return img


def channel_transformation(transformation_type, colour_space='rgb'):
    ct = []
    if transformation_type != 'trichromat':
        colour_inds = get_colour_inds(transformation_type)
        # check if it's a valid colour index
        if colour_inds is not None:
            ct.append(ChannelTransformation(colour_inds, colour_space))
        else:
            warnings.warn('Unsupported colour transformation' % type)
    return ct


class ImageTransformation(object):

    def __init__(
            self,
            manipulation_function,
            manipulation_value,
            manipulation_radius):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.manipulation_radius = manipulation_radius

    def __call__(self, img):
        img = np.asarray(img, dtype='uint8')
        manipulation_value = np.random.uniform(*self.manipulation_value)
        img = self.manipulation_function(img,
                                         manipulation_value,
                                         mask_radius=self.manipulation_radius)
        img *= 255
        img = PilImage.fromarray(img.astype('uint8'), 'RGB')
        return img


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


class PreprocessingTransformation(object):

    def __init__(
            self,
            manipulation_function,
            manipulation_value,
            manipulation_radius):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.manipulation_radius = manipulation_radius

    def __call__(self, x):
        x = np.asarray(x, dtype='uint8')
        x = self.manipulation_function(x, self.manipulation_value,
                                       mask_radius=self.manipulation_radius,
                                       preprocessing_function=None)
        x = PilImage.fromarray(x.astype('uint8'), 'RGB')
        return x


class RandomPreprocessingTransformation(object):

    def __init__(
            self,
            manipulation_function,
            manipulation_value,
            manipulation_radius):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.manipulation_radius = manipulation_radius

    def __call__(self, x):
        x = np.asarray(x, dtype='uint8')
        manipulation_value = np.random.uniform(*self.manipulation_value)
        x = self.manipulation_function(x, manipulation_value,
                                       mask_radius=self.manipulation_radius,
                                       preprocessing_function=None)
        x = PilImage.fromarray(x.astype('uint8'), 'RGB')
        return x
