"""
Creating mask images or filters.
"""

import numpy as np
import math
import sys

from skimage import feature
from skimage import morphology
from skimage.transform import resize
import cv2

from kernelphysiology.transformations.normalisations import im2double


def create_mask_image_canny(image, sigma=1.0, low_threshold=0.9,
                            high_threshold=0.9, use_quantiles=True,
                            dialation=None):
    image_mask = np.zeros(image.shape, np.uint8)
    if sigma is not None:
        image = im2double(image)

        if len(image.shape) > 2:
            chns = image.shape[2]
            # convert the image to one channel
            image = image.sum(axis=2)
        else:
            chns = 1

        sigma_sign = np.sign(sigma)
        if sigma_sign == -1:
            sigma = np.abs(sigma)

        image_mask = feature.canny(
            image, sigma, low_threshold, high_threshold,
            use_quantiles=use_quantiles
        )
        if dialation is not None:
            if type(dialation) is float:
                dialation = int(dialation)
            if type(dialation) is int:
                dialation = morphology.square(dialation)
            image_mask = morphology.dilation(image_mask, dialation)

        # repeating this for number of channels in input image
        if chns != 1:
            image_mask = np.expand_dims(image_mask, axis=2)
            image_mask = np.repeat(image_mask, chns, axis=2)

        image_mask = np.uint8(image_mask)
        if sigma_sign == 1:
            image_mask = 1 - image_mask
    return image_mask


def create_mask_image_shape(image, is_circle=True, mask_length=None):
    """Creating a mask image with given radius or given side"""
    image_mask = np.zeros(image.shape, np.uint8)
    if mask_length is not None:
        radius_sign = np.sign(mask_length)
        if radius_sign == -1:
            mask_length = np.abs(mask_length)
        rows = image.shape[0]
        cols = image.shape[1]
        smaller_side = np.minimum(rows, cols)
        mask_length = int(math.floor(mask_length * smaller_side * 0.5))
        if mask_length >= 3:
            centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))
            if is_circle:
                image_mask = cv2.circle(
                    image_mask, centre, mask_length, (1, 1, 1), -1
                )
            else:
                rect = (
                    centre[0] - mask_length, centre[1] - mask_length,
                    2 * mask_length, 2 * mask_length
                )
                image_mask = cv2.rectangle(
                    image_mask, rect, (1, 1, 1), -1
                )
            if radius_sign == 1:
                image_mask = 1 - image_mask
    return image_mask


def create_mask_image_texture(image, texture_type, inverse=False):
    image_mask = np.load(texture_type)
    if (image_mask.shape[0] != image.shape[0]) or (
            image_mask.shape[1] != image.shape[1]):
        image_mask = resize(image_mask, (image.shape[0], image.shape[1]))
        image_mask = image_mask > 0
    image_mask = np.uint8(image_mask)
    if inverse:
        image_mask = 1 - image_mask
    if len(image.shape) > 2:
        chns = image.shape[2]
        if chns != 1:
            image_mask = np.expand_dims(image_mask, axis=2)
            image_mask = np.repeat(image_mask, chns, axis=2)
    return image_mask


def bayer_filter_array(image, colour_channel):
    image_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    if colour_channel == 'red':
        image_mask[0::2, 1::2] = 1
    elif colour_channel == 'green':
        image_mask[0::2, 0::2] = 1
        image_mask[1::2, 1::2] = 1
    elif colour_channel == 'blue':
        image_mask[1::2, 0::2] = 1
    return image_mask


def retina_filter_array(image, colour_channel):
    image_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    if colour_channel == 'red':
        image_mask[0::2, 0::3] = 1
        image_mask[0::2, 2::3] = 1
        image_mask[1::3, 1::3] = 1
    elif colour_channel == 'green':
        image_mask[1::3, 0::3] = 1
        image_mask[1::3, 2::3] = 1
        image_mask[2::3, 1::3] = 1
    elif colour_channel == 'blue':
        image_mask[0::3, 1::3] = 1
    return image_mask


def random_filter_array(image):
    """Create a random filter array. If input image is a MxN matrix, the output
    pixels are True or False randomly. If input image is a MxNxC, for each pixel
    one channel is True while the others are False.

    :param image: numpy array matrix.
    :return: a binary matrix.
    """
    if len(image.shape) < 3:
        return np.random.choice([True, False], size=image.shape)
    image_mask = np.zeros(image.shape, np.uint8)
    rand_inds = np.random.randint(image.shape[2], size=image.shape[:2])
    for i in range(image.shape[2]):
        image_mask[:, :, i] = rand_inds == i
    return image_mask


def colour_filter_array(image, mosaic_type, **kwargs):
    if mosaic_type == 'bayer':
        image_mask = bayer_filter_array(image, **kwargs)
    elif mosaic_type == 'retina':
        image_mask = retina_filter_array(image, **kwargs)
    else:
        sys.exit('Unsupported mosaic type %s' % mosaic_type)
    return image_mask


def create_mask_image(image, mask_type, **kwargs):
    if mask_type is None:
        image_mask = np.zeros(image.shape, np.uint8)
    elif mask_type == 'circle':
        image_mask = create_mask_image_shape(image, True, **kwargs)
    elif mask_type == 'square':
        image_mask = create_mask_image_shape(image, False, **kwargs)
    elif mask_type == 'canny':
        image_mask = create_mask_image_canny(image, **kwargs)
    elif mask_type == 'texture':
        image_mask = create_mask_image_texture(image, **kwargs)
    elif mask_type == 'mosaic':
        image_mask = colour_filter_array(image, **kwargs)
    else:
        sys.exit('Unsupported mask type %s' % mask_type)
    return image_mask


def ring_mask(img, inner, outer):
    rows = img.shape[0]
    cols = img.shape[1]
    smaller_side = np.minimum(rows, cols)
    centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))

    if inner == 0:
        mask_in = np.ones(img.shape, np.uint8)
        in_length = 0
    else:
        if type(inner) is float and inner < 1.0:
            in_length = int(math.floor(inner * smaller_side * 0.5))
        else:
            in_length = int(inner - 1)
        mask_in = np.zeros(img.shape, np.uint8)
        mask_in = 1 - cv2.circle(mask_in, centre, in_length, (1, 1, 1), -1)

    if outer == 0:
        mask_out = np.ones(img.shape, np.uint8)
    else:
        if type(outer) is float and abs(outer) < 1.0:
            outer = 1 - outer
            out_length = int(math.floor(outer * smaller_side * 0.5))
        elif outer < 0:
            out_length = centre[0] - int(abs(outer))
        else:
            out_length = in_length + int(outer)
        mask_out = np.zeros(img.shape, np.uint8)
        mask_out = cv2.circle(mask_out, centre, out_length, (1, 1, 1), -1)

    mask_img = np.logical_and(mask_in, mask_out)
    return mask_img
