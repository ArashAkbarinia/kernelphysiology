"""
Preparing the input image to be inputted to a network.
"""

import warnings

from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.utils import imutils


def colour_transformation(vision_type, colour_space='rgb'):
    ct = []
    if colour_space != 'lms':
        colour_inds = imutils.get_colour_inds(vision_type)
        ct.append(cv2_preprocessing.VisionTypeTransformation(
            colour_inds, colour_space
        ))
    return ct


def channel_transformation(vision_type, colour_space='rgb'):
    ct = []
    if vision_type != 'trichromat':
        colour_inds = imutils.get_colour_inds(vision_type)
        # check if it's a valid colour index
        if colour_inds is not None:
            ct.append(cv2_preprocessing.ChannelTransformation(
                colour_inds, colour_space
            ))
        else:
            warnings.warn('Unsupported colour transformation %s' % colour_space)
    return ct


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    if type(mean) not in [tuple, list]:
        mean = tuple([mean for _ in range(tensor.shape[1])])
    if type(std) not in [tuple, list]:
        std = tuple([std for _ in range(tensor.shape[1])])
    # inverting the normalisation for each channel
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] * std[i]) + mean[i]
    tensor = tensor.clamp(0, 1)
    return tensor


def normalise_tensor(tensor, mean, std):
    tensor = tensor.clone()
    if type(mean) not in [tuple, list]:
        mean = tuple([mean for _ in range(tensor.shape[1])])
    if type(std) not in [tuple, list]:
        std = tuple([std for _ in range(tensor.shape[1])])
    # normalising the channels
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] - mean[i]) / std[i]
    return tensor


def mosaic_transformation(mosaic_pattern):
    return cv2_preprocessing.MosaicTransformation(mosaic_pattern)


def sf_transformation(sf_filter):
    return cv2_preprocessing.SpatialFrequencyTransformation(sf_filter)


def prediction_transformation(parameters, colour_space='rgb', tmp_c=False):
    return cv2_preprocessing.PredictionTransformation(
        parameters, colour_space=colour_space, tmp_c=tmp_c
    )


def prediction_transformation_seg(parameters, colour_space='rgb', tmp_c=False):
    return cv2_preprocessing.PredictionTransformationPilSeg(
        parameters, colour_space=colour_space, tmp_c=tmp_c
    )


def random_augmentation(augmentation_settings, num_augmentations=None):
    return cv2_preprocessing.RandomAugmentationTransformation(
        augmentation_settings, num_augmentations
    )
