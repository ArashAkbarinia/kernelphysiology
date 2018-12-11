from skimage.color import rgb2gray
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
    
    pixel_variatoin = np.random.uniform(low=min_contrast, high=max_contrast, size=image.shape)
    contrast_level = np.ones(image.shape) * pixel_variatoin

    return (1 - contrast_level) / 2.0 + np.multiply(image, contrast_level)


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


def adjust_illuminant(image, illuminant):
    image = im2double(image)
    for i in range(image.shape[2]):
        image[:, :, i] = image[:, :, i] * illuminant[i]

    return image


# FIXME: uniform noise and s_p noise should be identical with respect to percentage
def uniform_noise(image, width, rng=np.random.RandomState(seed=1)):
    '''
    Convert to grayscale. Adjust contrast. Apply uniform noise.
    parameters:
    - image: a numpy.ndarray 
    - width: a scalar indicating width of additive uniform noise
             -> then noise will be in range [-width, width]
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    '''
    image = im2double(image)
    for i in range(image.shape[2]):
        image[:, :, i] = apply_uniform_noise(image[:, :, i], -width, width, rng)

    return image


def s_p_noise(image, amount, s_vs_p=0.5):
    out = im2double(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
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


def apply_uniform_noise(image, low, high, rng=None):
    """Apply uniform noise to an image, clip outside values to 0 and 1.

    parameters:
    - image: a numpy.ndarray 
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    nrow = image.shape[0]
    ncol = image.shape[1]

    image = image + get_uniform_noise(low, high, nrow, ncol, rng)

    #clip values
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def get_uniform_noise(low, high, nrow, ncol, rng=None):
    """Return uniform noise within [low, high) of size (nrow, ncol).

    parameters:
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - nrow: number of rows of desired noise
    - ncol: number of columns of desired noise
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    if rng is None:
        return np.random.uniform(low=low, high=high, size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high, size=(nrow, ncol))


def is_in_bounds(mat, low, high):
    """Return wether all values in 'mat' fall between low and high.

    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all(np.logical_and(mat >= 0, mat <= 1))