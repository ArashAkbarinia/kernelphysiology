'''
Utility functoins for image processing.
'''


from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.draw import rectangle

import numpy as np

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


def adjust_contrast(image, contrast_level, pixel_variatoin=0):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."

    image = im2double(image)

    min_contrast = contrast_level - pixel_variatoin
    max_contrast = contrast_level + pixel_variatoin

    contrast_level_mat = np.random.uniform(low=min_contrast, high=max_contrast, size=image.shape)

    return (1 - contrast_level_mat) / 2.0 + np.multiply(image, contrast_level_mat)


def adjust_gamma(image, gamma):
    image = im2double(image)
    image = image ** gamma
    return image


def gaussian_blur(image, sigmax, sigmay=None, meanx=0, meany=0, theta=0):
    '''
    Blurring the image with a Gaussian kernel.
    '''
    image = im2double(image)
    g_kernel = gaussian_kernel2(sigmax=sigmax, sigmay=sigmay, meanx=meanx,
                                meany=meany, theta=theta)
    image = cv2.filter2D(image, -1, g_kernel)
    return image


def grayscale_contrast(image, contrast_level):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), contrast_level)


def adjust_illuminant(image, illuminant, pixel_variatoin=0):
    image = im2double(image)

    for i in range(image.shape[2]):
        min_illuminant = illuminant[i] - pixel_variatoin
        max_illuminant = illuminant[i] + pixel_variatoin
        illuminant_i = np.random.uniform(low=min_illuminant, high=max_illuminant, size=image[:,:,i].shape)
        image[:, :, i] = image[:, :, i] * illuminant_i

    return image


def s_p_noise(image, amount, salt_vs_pepper=0.5, seed=None, clip=True):
    out = im2double(image)
    out = random_noise(out, mode='s&p', seed=seed, clip=clip, salt_vs_pepper=salt_vs_pepper, amount=amount)
    return out


def speckle_noise(image, var, seed=None, clip=True):
    out = im2double(image)
    out = random_noise(out, mode='speckle', seed=seed, clip=clip, var=var)
    return out


def gaussian_noise(image, var, seed=None, clip=True):
    out = im2double(image)
    out = random_noise(out, mode='gaussian', seed=seed, clip=clip, var=var)
    return out


def poisson_noise(image, seed=None, clip=True):
    out = im2double(image)
    out = random_noise(out, mode='poisson', seed=seed, clip=clip)
    return out


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