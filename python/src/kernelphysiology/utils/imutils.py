"""
Utility functions for image processing.
"""

from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.draw import rectangle

import numpy as np
import math

import cv2

from kernelphysiology.filterfactory.gaussian import gaussian_kernel2
from kernelphysiology.filterfactory.mask import create_mask_image_canny
from kernelphysiology.filterfactory.mask import create_mask_image
from kernelphysiology.transformations.colour_spaces import rgb2opponency
from kernelphysiology.transformations.colour_spaces import opponency2rgb


def im2double_max(image):
    if image.dtype == 'uint8':
        image = image.astype('float32')
        return image / 255, 255
    else:
        image = image.astype('float32')
        max_pixel = np.max(image)
        if 1 < max_pixel <= 255:
            return image / 255, 255
        else:
            image /= max_pixel
            return image, max_pixel


def im2double(image):
    if image.dtype == 'uint8':
        image = image.astype('float32')
        return image / 255
    else:
        image = image.astype('float32')
        max_pixel = np.max(image)
        if 1 < max_pixel <= 255:
            return image / 255
        else:
            image /= max_pixel
            return image


# TODO: merge it with Keras image manipulation class
def get_random_crop_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    """Get parameters for ``crop`` for a random sized crop.
    Args:
        img (numpy array): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = img.shape[0] * img.shape[1]

    for attempt in range(10):
        target_area = np.random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(np.random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[0] and h < img.shape[1]:
            i = np.random.randint(0, img.shape[1] - h)
            j = np.random.randint(0, img.shape[0] - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = img.shape[0] / img.shape[1]
    if in_ratio < min(ratio):
        w = img.shape[0]
        h = w / min(ratio)
    elif in_ratio > max(ratio):
        h = img.shape[1]
        w = h * max(ratio)
    else:  # whole image
        w = img.shape[0]
        h = img.shape[1]
    i = (img.shape[1] - h) // 2
    j = (img.shape[0] - w) // 2
    return int(i), int(j), int(h), int(w)


def resize_to_min_side(img, target_size):
    # NOTE: assuming only square images
    min_side = target_size[0]
    # resize
    height = img.shape[0]
    width = img.shape[1]
    new_height = height * min_side // min(img.shape[:2])
    new_width = width * min_side // min(img.shape[:2])
    img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    # FIXME: only for 8 bit images
    img = np.minimum(img, 255)
    return img


def centre_crop(img, target_size):
    # crop
    height = img.shape[0]
    width = img.shape[1]
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    right = (width + target_size[0]) // 2
    bottom = (height + target_size[1]) // 2
    img = img[top:bottom, left:right]
    return img


# NOTE: image_data_format is 'channel_last'
def crop_image_random(img, target_size):
    (i, j, h, w) = get_random_crop_params(img)
    img = img[j:(j + w), i:(i + h)]

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    # FIXME: only for 8 bit images
    img = np.minimum(img, 255)

    return img


# TODO: support other interpolation methods
def crop_image_centre(img, target_size, extended_crop=None):
    (dx, dy) = target_size
    # FIXME: by default it should be 0 rather than 32
    if extended_crop is None:
        extended_crop = (dx + 32, dy + 32)
    img = resize_to_min_side(img, extended_crop)
    return centre_crop(img, target_size)


def invert_colour_opponency(image, mask_radius=None, colour_space='lab'):
    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    rg = image_opponent[:, :, 1].copy()
    image_opponent[:, :, 1] = image_opponent[:, :, 2].copy()
    image_opponent[:, :, 2] = rg

    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def invert_chromaticity(image, mask_radius=None, colour_space='lab'):
    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1:3] *= -1
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def invert_lightness(image, mask_radius=None, colour_space='lab'):
    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    max_lightness = get_max_lightness(colour_space=colour_space)
    image_opponent[:, :, 0] = max_lightness - image_opponent[:, :, 0]
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


# FIXME: move to colour spacs
def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = np.arctan(z / np.sqrt(XsqPlusYsq))  # theta
    az = np.arctan2(y, x)  # phi
    return np.array([r, az, elev])


## Rotation matrix along the x axis (luminance axis)
def rotation(X,
             teta):  # rotation of a an image coded in color one opponent space around achromatic axis
    RM = np.array([[1, 0, 0], [0, np.cos(teta), -np.sin(teta)],
                   [0, np.sin(teta), np.cos(teta)]])
    return np.dot(X, RM.T)


def rgb2pca(x):
    M = np.array([[0.66666, 1, -0.5], [0.66666, 0, 1], [0.66666, -1, -0.5]])
    return np.dot(x, M)


def pca2rgb(x):
    M = np.array([[0.66666, 1, -0.5], [0.66666, 0, 1], [0.66666, -1, -0.5]])
    return np.dot(x, np.linalg.inv(M))


def get_max_lightness(colour_space='lab'):
    if colour_space == 'lab':
        max_lightness = 100
    elif colour_space == 'dkl':
        max_lightness = 2
    return max_lightness


def rotate_hue(image, hue_angle, mask_radius=None, norm_fact=0.4):
    hue_angle = math.radians(hue_angle)
    image = im2double(image) - 0.5
    im_pca = rgb2pca(image)

    # FIXME: this is a trick to avoid getting out of gamma
    norm = norm_fact / np.amax(np.absolute(im_pca))

    output = pca2rgb(rotation(im_pca * norm, hue_angle)) + 0.5
    return output


def keep_red_channel(image, amount, mask_radius=None):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 1] *= amount
    output[:, :, 2] *= amount
    output *= max_pixel
    return output


def keep_green_channel(image, amount, mask_radius=None):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 0] *= amount
    output[:, :, 2] *= amount
    output *= max_pixel
    return output


def keep_blue_channel(image, amount, mask_radius=None):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 0] *= amount
    output[:, :, 1] *= amount
    output *= max_pixel
    return output


def reduce_red_green(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def reduce_yellow_blue(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 2] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def reduce_chromaticity(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1:3] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def reduce_lightness(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image, max_pixel = im2double_max(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    max_lightness = get_max_lightness(colour_space=colour_space)
    image_opponent[:, :, 0] = ((1 - amount) / 2 + np.multiply(
        image_opponent[:, :, 0] / max_lightness, amount)) * max_lightness
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    output *= max_pixel
    return output


def adjust_contrast(image, contrast_level, pixel_variatoin=0, mask_radius=None,
                    mask_type='circle', **kwargs):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert (contrast_level >= 0.0), 'contrast_level too low.'
    assert (contrast_level <= 1.0), 'contrast_level too high.'

    image, max_pixel = im2double_max(image)

    image_org = image.copy()
    if mask_type == 'circle':
        image_mask = create_mask_image(image, mask_radius, True)
    elif mask_type == 'square':
        image_mask = create_mask_image(image, mask_radius, False)
    else:
        image_mask = create_mask_image_canny(image, sigma=mask_radius, **kwargs)

    min_contrast = contrast_level - pixel_variatoin
    max_contrast = contrast_level + pixel_variatoin

    contrast_mat = np.random.uniform(
        low=min_contrast, high=max_contrast, size=image.shape
    )

    image_contrast = (1 - contrast_mat) / 2.0 + np.multiply(image, contrast_mat)
    output = image_org * image_mask + image_contrast * (1 - image_mask)
    output *= max_pixel
    return output


def grayscale_contrast(image, contrast_level, mask_radius=None):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), contrast_level,
                           mask_radius=mask_radius)


def adjust_gamma(image, gamma, mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_gamma = image ** gamma
    output = image_org * image_mask + image_gamma * (1 - image_mask)
    output *= max_pixel
    return output


def gaussian_blur(image, sigmax, sigmay=None, meanx=0, meany=0, theta=0,
                  mask_radius=None):
    """
    Blurring the image with a Gaussian kernel.
    """
    image, max_pixel = im2double_max(image)
    image_org = image.copy()

    image_mask = create_mask_image(image, mask_radius)

    g_kernel = gaussian_kernel2(
        sigmax=sigmax, sigmay=sigmay, meanx=meanx, meany=meany, theta=theta
    )
    image_blur = cv2.filter2D(image, -1, g_kernel)
    output = image_org * image_mask + image_blur * (1 - image_mask)

    output *= max_pixel
    return output


def adjust_illuminant(image, illuminant, pixel_variatoin=0, mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    for i in range(image.shape[2]):
        min_illuminant = illuminant[i] - pixel_variatoin
        max_illuminant = illuminant[i] + pixel_variatoin
        illuminant_i = np.random.uniform(
            low=min_illuminant, high=max_illuminant, size=image[:, :, i].shape
        )
        image[:, :, i] = image[:, :, i] * illuminant_i

    output = image_org * image_mask + image * (1 - image_mask)
    output *= max_pixel
    return output


def s_p_noise(image, amount, salt_vs_pepper=0.5, seed=None, clip=True,
              mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(
        image, mode='s&p', seed=seed, clip=clip,
        salt_vs_pepper=salt_vs_pepper, amount=amount
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def speckle_noise(image, var, seed=None, clip=True, mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(
        image, mode='speckle', seed=seed, clip=clip, var=var
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def gaussian_noise(image, var, seed=None, clip=True, mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(
        image, mode='gaussian', seed=seed, clip=clip, var=var
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def poisson_noise(image, seed=None, clip=True, mask_radius=None):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='poisson', seed=seed, clip=clip)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def simulate_distance(image, distance_factor, mask_radius=None):
    if distance_factor <= 1:
        return image
    rows = image.shape[0]
    cols = image.shape[1]
    distance_rows = int(np.round(rows / distance_factor))
    distance_cols = int(np.round(cols / distance_factor))
    output = cv2.resize(image, (distance_rows, distance_cols),
                        interpolation=cv2.INTER_NEAREST)
    output = cv2.resize(output, (rows, cols), interpolation=cv2.INTER_NEAREST)
    return output


# TODO: add other shapes as well
def random_occlusion(image, object_instances=1, object_ratio=0.05):
    output, max_pixel = im2double_max(image)
    (rows, cols, chns) = output.shape
    extent = (round(rows * object_ratio), round(cols * object_ratio))
    for i in range(object_instances):
        rand_row = np.random.randint(0 + extent[0], rows - extent[0])
        rand_col = np.random.randint(0 + extent[1], cols - extent[1])
        start = (rand_row, rand_col)
        # FIXME: if backend shape is different
        (rr, cc) = rectangle(start, extent=extent, shape=output.shape[0:2])
        output[rr.astype('int64'), cc.astype('int64'), :] = 1
    output *= max_pixel
    return output


def local_std(image, window_size=(5, 5)):
    """
    Computing the local standard deviation of an image.
    """
    image, max_pixel = im2double_max(image)

    npixels = window_size[0] * window_size[1]
    kernel = np.ones(window_size, np.float32) / npixels
    # TODO: consider different border treatment
    avg_image = cv2.filter2D(image, -1, kernel)
    std_image = cv2.filter2D((image - avg_image) ** 2, -1, kernel) ** 0.5

    std_image *= max_pixel
    avg_image *= max_pixel
    return std_image, avg_image


def normalise_channel(x, low=0, high=1, minv=None, maxv=None):
    if minv is None:
        minv = x.min()
    if maxv is None:
        maxv = x.max()
    output = low + (x - minv) * (high - low) / (maxv - minv)
    return output


def get_colour_inds(chromaticity_type):
    # FIXME: according to colour space
    colour_inds = None
    if chromaticity_type == 'dichromat_rg':
        colour_inds = [1]
    elif chromaticity_type == 'dichromat_yb':
        colour_inds = [2]
    elif chromaticity_type == 'monochromat':
        colour_inds = [1, 2]
    elif chromaticity_type == 'lightness':
        colour_inds = [0]
    elif chromaticity_type == 'protanopia':
        colour_inds = [0]
    elif chromaticity_type == 'deuteranopia':
        colour_inds = [1]
    elif chromaticity_type == 'tritanopia':
        colour_inds = [2]
    return colour_inds


def max_pixel_ind(im):
    return np.unravel_index(np.argmax(im, axis=None), im.shape)
