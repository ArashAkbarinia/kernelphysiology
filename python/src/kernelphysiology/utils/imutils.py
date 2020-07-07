"""
Utility functions for image processing.
"""

from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.draw import rectangle

import numpy as np
import random
import math

import cv2

from kernelphysiology.filterfactory.gaussian import gaussian_kernel2
from kernelphysiology.filterfactory.mask import create_mask_image
from kernelphysiology.filterfactory.mask import colour_filter_array
from kernelphysiology.transformations.colour_spaces import rgb2opponency
from kernelphysiology.transformations.colour_spaces import opponency2rgb
from kernelphysiology.transformations.colour_spaces import get_max_lightness
from kernelphysiology.transformations.normalisations import im2double_max
from kernelphysiology.transformations.normalisations import im2double
from kernelphysiology.transformations.normalisations import img_midvals


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
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[0] and h < img.shape[1]:
            i = random.randint(0, img.shape[1] - h)
            j = random.randint(0, img.shape[0] - w)
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


def invert_colour_opponency(image, colour_space='lab'):
    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    rg = image_opponent[:, :, 1].copy()
    image_opponent[:, :, 1] = image_opponent[:, :, 2].copy()
    image_opponent[:, :, 2] = rg
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def invert_chromaticity(image, colour_space='lab'):
    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    image_opponent[:, :, 1:3] *= -1
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def invert_lightness(image, colour_space='lab'):
    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    max_lightness = get_max_lightness(opponent_space=colour_space)
    image_opponent[:, :, 0] = max_lightness - image_opponent[:, :, 0]
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


# FIXME: move to colour spacs
def cart2sph(x, y, z):
    xsq_plus_ysq = x ** 2 + y ** 2
    r = np.sqrt(xsq_plus_ysq + z ** 2)  # r
    elev = np.arctan(z / np.sqrt(xsq_plus_ysq))  # theta
    az = np.arctan2(y, x)  # phi
    return np.array([r, az, elev])


# Rotation matrix along the x axis (luminance axis)
def rotation(x, teta):
    # rotation of a an image coded in color one opponent space around
    # achromatic axis
    rm = np.array([[1, 0, 0], [0, np.cos(teta), -np.sin(teta)],
                   [0, np.sin(teta), np.cos(teta)]])
    return np.dot(x, rm.T)


def rgb2pca(x):
    m = np.array([[0.66666, 1, -0.5], [0.66666, 0, 1], [0.66666, -1, -0.5]])
    return np.dot(x, m)


def pca2rgb(x):
    m = np.array([[0.66666, 1, -0.5], [0.66666, 0, 1], [0.66666, -1, -0.5]])
    return np.dot(x, np.linalg.inv(m))


def rotate_hue(image, hue_angle, norm_fact=0.4):
    hue_angle = math.radians(hue_angle)
    image = im2double(image) - 0.5
    im_pca = rgb2pca(image)

    # FIXME: this is a trick to avoid getting out of gamma
    norm = norm_fact / np.amax(np.absolute(im_pca))

    output = pca2rgb(rotation(im_pca * norm, hue_angle)) + 0.5
    return output


def keep_red_channel(image, amount):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 1] *= amount
    output[:, :, 2] *= amount
    output *= max_pixel
    return output


def keep_green_channel(image, amount):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 0] *= amount
    output[:, :, 2] *= amount
    output *= max_pixel
    return output


def keep_blue_channel(image, amount):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output, max_pixel = im2double_max(image)
    output[:, :, 0] *= amount
    output[:, :, 1] *= amount
    output *= max_pixel
    return output


def reduce_red_green(image, amount, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    image_opponent[:, :, 1] *= amount
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def reduce_yellow_blue(image, amount, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    image_opponent[:, :, 2] *= amount
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def reduce_chromaticity(image, amount, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    image_opponent[:, :, 1:3] *= amount
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def reduce_lightness(image, amount, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image_opponent = rgb2opponency(image, opponent_space=colour_space)
    max_lightness = get_max_lightness(opponent_space=colour_space)
    image_opponent[:, :, 0] = ((1 - amount) / 2 + np.multiply(
        image_opponent[:, :, 0] / max_lightness, amount)) * max_lightness
    output = opponency2rgb(image_opponent, opponent_space=colour_space)
    return output


def adjust_contrast(image, amount, pixel_variatoin=0, mask_type=None,
                    **kwargs):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert (amount >= 0.0), 'contrast_level too low.'
    assert (amount <= 1.0), 'contrast_level too high.'

    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    min_contrast = amount - pixel_variatoin
    max_contrast = amount + pixel_variatoin

    contrast_mat = np.random.uniform(
        low=min_contrast, high=max_contrast, size=image.shape
    )

    image_contrast = (1 - contrast_mat) / 2.0 + np.multiply(image, contrast_mat)
    output = image_org * image_mask + image_contrast * (1 - image_mask)
    output *= max_pixel
    return output


def grayscale_contrast(image, amount, mask_radius=None):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), amount,
                           mask_radius=mask_radius)


def adjust_gamma(image, amount, pixel_variatoin=0, mask_type=None, **kwargs):
    # TODO: for all other manipulations
    if isinstance(amount, list):
        amount = random.uniform(*amount)
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    min_gamma = amount - pixel_variatoin
    max_gamma = amount + pixel_variatoin

    gamma_mat = np.random.uniform(
        low=min_gamma, high=max_gamma, size=image.shape
    )

    image_gamma = image ** gamma_mat
    output = image_org * image_mask + image_gamma * (1 - image_mask)
    output *= max_pixel
    return output


def gaussian_blur(image, sigmax, sigmay=None, meanx=0, meany=0, theta=0,
                  mask_type=None, **kwargs):
    """
    Blurring the image with a Gaussian kernel.
    """
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    g_kernel = gaussian_kernel2(
        sigmax=sigmax, sigmay=sigmay, meanx=meanx, meany=meany, theta=theta
    )
    image_blur = cv2.filter2D(image, -1, g_kernel)
    output = image_org * image_mask + image_blur * (1 - image_mask)

    output *= max_pixel
    return output


def adjust_illuminant(image, illuminant, pixel_variatoin=0, mask_type=None,
                      **kwargs):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

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
              mask_type=None, **kwargs):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    image_noise = random_noise(
        image, mode='s&p', seed=seed, clip=clip,
        salt_vs_pepper=salt_vs_pepper, amount=amount
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def speckle_noise(image, amount, seed=None, clip=True, mask_type=None,
                  **kwargs):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    image_noise = random_noise(
        image, mode='speckle', seed=seed, clip=clip, var=amount
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def gaussian_noise(image, amount, seed=None, clip=True, mask_type=None,
                   **kwargs):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    image_noise = random_noise(
        image, mode='gaussian', seed=seed, clip=clip, var=amount
    )
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def poisson_noise(image, seed=None, clip=True, mask_type=None, **kwargs):
    image, max_pixel = im2double_max(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_type, **kwargs)

    image_noise = random_noise(image, mode='poisson', seed=seed, clip=clip)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    output *= max_pixel
    return output


def im2mosaic(image, mosaic_type=None, masks=None):
    if mosaic_type is None or len(image.shape) == 2:
        return image
    image = image.copy()

    midvals = img_midvals(image)
    if masks is None:
        masks = [
            colour_filter_array(image, mosaic_type, colour_channel='red'),
            colour_filter_array(image, mosaic_type, colour_channel='green'),
            colour_filter_array(image, mosaic_type, colour_channel='blue')
        ]
    for i in range(3):
        img_ch = image[:, :, i]
        img_ch[masks[i] == 0] = midvals[i]
        image[:, :, i] = img_ch
    return image


# TODO: add other shapes as well
def random_occlusion(image, object_instances=1, object_ratio=0.05):
    output, max_pixel = im2double_max(image)
    (rows, cols, chns) = output.shape
    extent = (round(rows * object_ratio), round(cols * object_ratio))
    for i in range(object_instances):
        rand_row = random.randint(0 + extent[0], rows - extent[0])
        rand_col = random.randint(0 + extent[1], cols - extent[1])
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


def get_colour_inds(vision_type):
    # FIXME: according to colour space
    colour_inds = None
    if vision_type == 'dichromat_rg':
        colour_inds = [1]
    elif vision_type == 'dichromat_yb':
        colour_inds = [2]
    elif vision_type == 'monochromat':
        colour_inds = [1, 2]
    elif vision_type == 'lightness':
        colour_inds = [0]
    elif vision_type == 'protanopia':
        colour_inds = [0]
    elif vision_type == 'deuteranopia':
        colour_inds = [1]
    elif vision_type == 'tritanopia':
        colour_inds = [2]
    return colour_inds


def max_pixel_ind(im):
    return np.unravel_index(np.argmax(im, axis=None), im.shape)


def top_pixeld_ind(im, percentage):
    assert (percentage >= 0.0), 'percentage too low.'
    assert (percentage <= 1.0), 'percentage too high.'
    sorted_inds = np.unravel_index(np.argsort(im, axis=None), im.shape)
    num_elements = im.size
    selected_pixels = int(np.floor(num_elements * percentage) + 1)
    rows = sorted_inds[0][-1:-selected_pixels:-1]
    cols = sorted_inds[1][-1:-selected_pixels:-1]
    return rows, cols


def heat_map_from_point(point, target_size, g_kernel=None, sigma=1.5):
    rows = target_size[0]
    cols = target_size[1]
    heat_map = np.zeros((rows, cols, 1), dtype=np.float32)

    pr = point[0]
    pc = point[1]
    if sigma == 0:
        heat_map[pr, pc] = 1
        return heat_map

    if g_kernel is None:
        g_kernel = gaussian_kernel2(sigma)

    if pr >= 0 and pc >= 0:
        sr = pr - (g_kernel.shape[0] // 2)
        sc = pc - (g_kernel.shape[1] // 2)
        # making sure they're within the range of image
        gsr = np.maximum(0, -sr)
        gsc = np.maximum(0, -sc)

        er = sr + g_kernel.shape[0]
        ec = sc + g_kernel.shape[1]
        # making sure they're within the range of image
        sr = np.maximum(0, sr)
        sc = np.maximum(0, sc)

        ger = np.minimum(g_kernel.shape[0], g_kernel.shape[0] - (er - rows))
        gec = np.minimum(g_kernel.shape[1], g_kernel.shape[1] - (ec - cols))

        er = np.minimum(er, rows)
        ec = np.minimum(ec, cols)
        g_max = g_kernel[gsr:ger, gsc:gec].max()
        heat_map[sr:er, sc:ec, 0] = g_kernel[gsr:ger, gsc:gec] / g_max
    return heat_map


def roll_image(img, rows, cols):
    img = np.roll(img, rows, axis=0)
    img = np.roll(img, cols, axis=1)
    return img
