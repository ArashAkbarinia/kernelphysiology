"""
A collection of frequency related transformations e.g. Fourier and Wavelengths.
"""

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
        img = pywt.dwt2(img, freq_type)
    else:
        sys.exit('frequency_domain.rgb2all does not support %s.' % freq_type)
    return img
