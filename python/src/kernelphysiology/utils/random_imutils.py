"""
Utility functions for image processing with random parameters.
"""

import random

from kernelphysiology.utils import imutils


def adjust_illuminant(image, illuminant, **kwargs):
    random_illuminant = []
    for i in range(image.shape[2]):
        random_illuminant.append(random.uniform(*illuminant))
    return imutils.adjust_illuminant(image, random_illuminant, **kwargs)


def adjust_contrast(image, amount, channel_wise=False, **kwargs):
    if channel_wise and len(image.shape) > 2:
        random_amounts = []
        for i in range(image.shape[2]):
            random_amounts.append(random.uniform(*amount))
    else:
        random_amounts = random.uniform(*amount)
    return imutils.adjust_contrast(image, random_amounts, **kwargs)


def adjust_gamma(image, amount, channel_wise=False, **kwargs):
    if channel_wise and len(image.shape) > 2:
        random_amounts = []
        for i in range(image.shape[2]):
            random_amounts.append(random.uniform(*amount))
    else:
        random_amounts = random.uniform(*amount)
    return imutils.adjust_gamma(image, random_amounts, **kwargs)
