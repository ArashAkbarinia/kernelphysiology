"""
Settings related to data augmentation.
"""

from kernelphysiology.utils import imutils


def do_nothing(x, _nothing):
    return x


supported_training_manipulations = {
    'contrast': imutils.adjust_contrast,
    'gamma': imutils.adjust_gamma
}

supported_testing_manipulations = {
    'contrast': imutils.adjust_contrast,
    'gamma': imutils.adjust_gamma,
    'blur': imutils.gaussian_blur,
    's_p_noise': imutils.s_p_noise,
    'speckle_noise': imutils.speckle_noise,
    'gaussian_noise': imutils.gaussian_noise,
    'poisson_noise': imutils.poisson_noise,
    'keep_red': imutils.keep_red_channel,
    'keep_green': imutils.keep_green_channel,
    'keep_blue': imutils.keep_blue_channel,
    'chromaticity': imutils.reduce_chromaticity,
    'red_green': imutils.reduce_red_green,
    'yellow_blue': imutils.reduce_yellow_blue,
    'lightness': imutils.reduce_lightness,
    'invert_chromaticity': imutils.invert_chromaticity,
    'invert_opponency': imutils.invert_colour_opponency,
    'invert_lightness': imutils.invert_lightness,
    'original': do_nothing
}


def get_training_augmentations():
    return supported_training_manipulations


def get_testing_augmentations():
    return supported_testing_manipulations
