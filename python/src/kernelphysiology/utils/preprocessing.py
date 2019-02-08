'''
Preprocessing functions.
'''


import commons

import sys
import numpy as np
import time
import datetime

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, test_arg_parser
from kernelphysiology.dl.keras.utils import get_top_k_accuracy
from kernelphysiology.dl.keras.models.utils import which_network, get_preprocessing_function
from kernelphysiology.dl.keras.datasets.utils import which_dataset
from kernelphysiology.utils.imutils import gaussian_blur, gaussian_noise
from kernelphysiology.utils.imutils import s_p_noise, speckle_noise, poisson_noise
from kernelphysiology.utils.imutils import adjust_gamma, adjust_contrast, adjust_illuminant
from kernelphysiology.utils.imutils import random_occlusion


def occlusion_preprocessing(img, var, mask_radius=None, preprocessing_function=None):
    img = random_occlusion(img, object_instances=1, object_ratio=var) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def speckle_noise_preprocessing(img, var, mask_radius=None, preprocessing_function=None):
    img = speckle_noise(img, var, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def s_p_noise_preprocessing(img, amount, mask_radius=None, preprocessing_function=None):
    img = s_p_noise(img, amount, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def poisson_noise_preprocessing(img, _, mask_radius=None, preprocessing_function=None):
    img = poisson_noise(img, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def gaussian_noise_preprocessing(img, var, mask_radius=None, preprocessing_function=None):
    img = gaussian_noise(img, var, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def colour_constancy_preprocessing(img, illuminant, mask_radius=None, preprocessing_function=None):
    # FIXME: for now it's only one channel, make it a loop for all channels
    illuminant = (illuminant, 1, 1)
    img = adjust_illuminant(img, illuminant, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def gamma_preprocessing(img, amount, mask_radius=None, preprocessing_function=None):
    img = adjust_gamma(img, amount, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def gaussian_preprocessing(img, sigma, mask_radius=None, preprocessing_function=None):
    img = gaussian_blur(img, sigma, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def contrast_preprocessing(img, contrast, mask_radius=None, preprocessing_function=None):
    img = adjust_contrast(img, contrast, mask_radius=mask_radius) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img
