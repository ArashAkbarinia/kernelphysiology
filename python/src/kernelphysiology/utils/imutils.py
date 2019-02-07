'''
Utility functoins for image processing.
'''


from skimage.util import random_noise
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.draw import rectangle

import numpy as np
import math

import cv2

from kernelphysiology.filterfactory.gaussian import gaussian_kernel2


def im2double(image):
    if image.dtype == 'uint8':
        image = image.astype('float32')
        return image / 255
    else:
        image = image.astype('float32')
        max_pixel = np.max(image)
        if max_pixel > 1 and max_pixel <= 255:
            return image / 255
        else:
            # FIXME: not handling these cases
            return image


def create_mask_image(image, mask_radius=None):
    image_mask = np.zeros(image.shape, np.uint8)
    if mask_radius is not None:
        image_mask = np.zeros(image.shape, np.uint8)
        radius_sign = np.sign(mask_radius)
        if radius_sign == -1:
            mask_radius = 1 + mask_radius
        (rows, cols, chns) = image.shape
        smaller_side = np.minimum(rows, cols)
        mask_radius = int(math.floor(mask_radius * smaller_side))
        if mask_radius > 3:
            centre = (int(math.floor(rows / 2)), int(math.floor(cols / 2)))
            image_mask = cv2.circle(image_mask, centre, mask_radius, (1, 1, 1), -1)
            if radius_sign == 1:
                image_mask = 1 - image_mask
    return image_mask


def reduce_red_green(image, amount):
    assert(amount >= 0.0), 'amount too low.
    assert(amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_lab = rgb2lab(image)
    image_lab[:, :, 1] *= amount
    output = lab2rgb(image_lab)
    return output


def reduce_yellow_blue(image, amount):
    assert(amount >= 0.0), 'amount too low.
    assert(amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_lab = rgb2lab(image)
    image_lab[:, :, 2] *= amount
    output = lab2rgb(image_lab)
    return output


def reduce_chromacity(image, amount):
    assert(amount >= 0.0), 'amount too low.
    assert(amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_lab = rgb2lab(image)
    image_lab[:, :, 1:3] *= amount
    output = lab2rgb(image_lab)
    return output


def reduce_lightness(image, amount):
    assert(amount >= 0.0), 'amount too low.
    assert(amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_lab = rgb2lab(image)
    image_lab[:, :, 0] = ((1 - amount) / 2 + np.multiply(im_lab[:, :, 0] / 100, amount)) * 100
    output = lab2rgb(image_lab)
    return output


def adjust_contrast(image, contrast_level, pixel_variatoin=0, mask_radius=None):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."

    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    min_contrast = contrast_level - pixel_variatoin
    max_contrast = contrast_level + pixel_variatoin

    contrast_level_mat = np.random.uniform(low=min_contrast, high=max_contrast, size=image.shape)

    image_contrast = (1 - contrast_level_mat) / 2.0 + np.multiply(image, contrast_level_mat)
    output = image_org * image_mask + image_contrast * (1 - image_mask)
    return output


def grayscale_contrast(image, contrast_level, mask_radius=None):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), contrast_level, mask_radius=mask_radius)


def adjust_gamma(image, gamma, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_gamma = image ** gamma
    output = image_org * image_mask + image_gamma * (1 - image_mask)
    return output


def gaussian_blur(image, sigmax, sigmay=None, meanx=0, meany=0, theta=0, mask_radius=None):
    '''
    Blurring the image with a Gaussian kernel.
    '''
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    g_kernel = gaussian_kernel2(sigmax=sigmax, sigmay=sigmay, meanx=meanx,
                                meany=meany, theta=theta)
    image_blur = cv2.filter2D(image, -1, g_kernel)
    output = image_org * image_mask + image_blur * (1 - image_mask)
    return output


def adjust_illuminant(image, illuminant, pixel_variatoin=0, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    for i in range(image.shape[2]):
        min_illuminant = illuminant[i] - pixel_variatoin
        max_illuminant = illuminant[i] + pixel_variatoin
        illuminant_i = np.random.uniform(low=min_illuminant, high=max_illuminant, size=image[:,:,i].shape)
        image[:, :, i] = image[:, :, i] * illuminant_i

    output = image_org * image_mask + image * (1 - image_mask)
    return output


def s_p_noise(image, amount, salt_vs_pepper=0.5, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='s&p', seed=seed, clip=clip, salt_vs_pepper=salt_vs_pepper, amount=amount)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def speckle_noise(image, var, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='speckle', seed=seed, clip=clip, var=var)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def gaussian_noise(image, var, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='gaussian', seed=seed, clip=clip, var=var)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def poisson_noise(image, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='poisson', seed=seed, clip=clip)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


# TODO: add other shapes as well
def random_occlusion(image, object_instances=1, object_ratio=0.05):
    out = im2double(image)
    (rows, cols, chns) = out.shape
    extent = (round(rows * object_ratio), round(cols * object_ratio))
    for i in range(object_instances):
        rand_row = np.random.randint(0+extent[0], rows-extent[0])
        rand_col = np.random.randint(0+extent[1], cols-extent[1])
        start = (rand_row, rand_col)
        # FIXME: if backend shape is different
        (rr, cc) = rectangle(start, extent=extent, shape=out.shape[0:2])
        out[rr.astype('int64'), cc.astype('int64'), :] = 1
    return out


# TODO: we're returning everything in 0 to 1, should convert it back to the
# original range
def local_std(image, window_size=(5, 5)):
    '''
    Computing the local standard deviation of an image.
    '''
    image = im2double(image)

    npixels = window_size[0] * window_size[1]
    kernel = np.ones(window_size, np.float32) / npixels
    # TODO: consider different border treatment
    avg_image = cv2.filter2D(image, -1, kernel)
    std_image =  cv2.filter2D((image - avg_image) ** 2, -1, kernel) ** 0.5
    return (std_image, avg_image)
