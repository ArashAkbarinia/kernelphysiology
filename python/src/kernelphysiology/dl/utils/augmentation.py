"""
Settings related to data augmentation.
"""

from kernelphysiology.utils import imutils

supported_image_manipulations = {
    'contrast': imutils.adjust_contrast,
    'gamma': imutils.adjust_gamma
}


def get_supported_image_manipulations():
    return supported_image_manipulations
