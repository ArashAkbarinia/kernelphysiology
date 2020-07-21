"""
A collection of frequency related transformations e.g. Fourier and Wavelengths.
"""

import numpy as np
import sys

import pywt
import cv2

from kernelphysiology.transformations import normalisations

SUPPORTED_WAVELETS = ['db1']  # pywt.wavelist(kind='discrete')


def rgb2all(img, freq_type):
    """Converts the RGB to a grey image and applies frequency decomposition.

    :param img: input image in RGB colour space.
    :param freq_type: type of a wavelength wavelet or fourier decomposition.
    :return: decomposed image with a size according to the decomposition type.
    """
    img = img.copy()
    img = normalisations.rgb2double(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if freq_type in SUPPORTED_WAVELETS:
        ll, (lh, hl, hh) = pywt.dwt2(img, freq_type)
        # NOTE: we convert them to a range of 0 to 1
        img = np.zeros((*ll.shape, 4))
        img[:, :, 0] = ll
        img[:, :, 1] = (lh + 1)
        img[:, :, 2] = (hl + 1)
        img[:, :, 3] = (hh + 1)
        img /= 2
    else:
        sys.exit('frequency_domain.rgb2all does not support %s.' % freq_type)
    return img
