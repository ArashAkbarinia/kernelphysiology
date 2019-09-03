"""
Settings related to data augmentation.
"""

from kernelphysiology.utils import imutils

supported_training_manipulations = {
    'contrast': imutils.adjust_contrast,
    'gamma': imutils.adjust_gamma
}


def get_training_augmentations():
    return supported_training_manipulations
