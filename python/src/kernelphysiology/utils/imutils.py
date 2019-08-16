"""
Utility functoins for image processing.
"""

from skimage.util import random_noise
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.draw import rectangle
from skimage import feature

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


def create_mask_image_canny(image, sigma=1.0, low_threshold=0.9,
                            high_threshold=0.9, use_quantiles=True):
    image_mask = np.zeros(image.shape, np.uint8)
    if sigma is not None:
        if len(image.shape) > 2:
            chns = image.shape[2]
        else:
            chns = 1

        # convert the image to one channel
        image = image.astype('float32')
        max_pixel = image.max()
        image /= max_pixel
        image = image.sum(axis=2)

        sigma_sign = np.sign(sigma)
        if sigma_sign == -1:
            sigma = np.abs(sigma)

        image_mask = feature.canny(
            image, sigma, low_threshold, high_threshold,
            use_quantiles=use_quantiles
        )

        # repeating this for number of channels in input image
        if chns != 1:
            image_mask = np.expand_dims(image_mask, axis=2)
            image_mask = np.repeat(image_mask, chns, axis=2)

        image_mask = image_mask.astype('uint8')
        if sigma_sign == 1:
            image_mask = 1 - image_mask
    return image_mask


def create_mask_image(image, mask_radius=None, is_circle=True):
    """Creating a mask image with given radius or given side"""
    image_mask = np.zeros(image.shape, np.uint8)
    if mask_radius is not None:
        radius_sign = np.sign(mask_radius)
        if radius_sign == -1:
            mask_radius = np.abs(mask_radius)
        (rows, cols, chns) = image.shape
        smaller_side = np.minimum(rows, cols)
        mask_radius = int(math.floor(mask_radius * smaller_side * 0.5))
        if mask_radius >= 3:
            centre = (int(math.floor(cols / 2)), int(math.floor(rows / 2)))
            if is_circle:
                image_mask = cv2.circle(
                    image_mask, centre, mask_radius, (1, 1, 1), -1
                )
            else:
                rect = (centre[0] - mask_radius, centre[1] - mask_radius,
                        2 * mask_radius, 2 * mask_radius)
                image_mask = cv2.rectangle(
                    image_mask, rect, (1, 1, 1), -1
                )
            if radius_sign == 1:
                image_mask = 1 - image_mask
    return image_mask


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
    if (in_ratio < min(ratio)):
        w = img.shape[0]
        h = w / min(ratio)
    elif (in_ratio > max(ratio)):
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
    (height, width, _) = img.shape
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
    (height, width, _) = img.shape
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
    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    rg = image_opponent[:, :, 1].copy()
    image_opponent[:, :, 1] = image_opponent[:, :, 2].copy()
    image_opponent[:, :, 2] = rg

    output = opponency2rgb(image_opponent, colour_space=colour_space)
    return output


def invert_chromaticity(image, mask_radius=None, colour_space='lab'):
    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1:3] *= -1
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    return output


def invert_lightness(image, mask_radius=None, colour_space='lab'):
    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    max_lightness = get_max_lightness(colour_space=colour_space)
    image_opponent[:, :, 0] = max_lightness - image_opponent[:, :, 0]
    output = opponency2rgb(image_opponent, colour_space=colour_space)
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


# TODO: move to a colour space file
def rgb2dkl(x):
    M = np.array([[0.4252, 1.4304, 0.1444], [0.8273, -0.5912, -0.2360],
                  [0.2268, 0.7051, -0.9319]])
    return np.dot(x, M.T)


def dkl2rgb(x):
    M = np.array([[0.49995, 0.50001495, 0.49999914],
                  [0.99998394, -0.29898596, 0.01714922],
                  [-0.17577361, 0.15319546, -0.99994349]])
    return np.dot(x, M)


def get_max_lightness(colour_space='lab'):
    if colour_space == 'lab':
        max_lightness = 100
    elif colour_space == 'dkl':
        max_lightness = 2
    return max_lightness


def rgb2opponency(image_rgb, colour_space='lab'):
    if colour_space == 'lab':
        image_opponent = rgb2lab(image_rgb)
    elif colour_space == 'dkl':
        image_opponent = rgb2dkl(image_rgb)
    return image_opponent


def opponency2rgb(image_opponent, colour_space='lab'):
    if colour_space == 'lab':
        image_rgb = lab2rgb(image_opponent)
    elif colour_space == 'dkl':
        image_rgb = dkl2rgb(image_opponent)
        image_rgb = normalise_channel(image_rgb)
    return image_rgb


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

    output = im2double(image)
    output[:, :, 1] *= amount
    output[:, :, 2] *= amount
    return output


def keep_green_channel(image, amount, mask_radius=None):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output = im2double(image)
    output[:, :, 0] *= amount
    output[:, :, 2] *= amount
    return output


def keep_blue_channel(image, amount, mask_radius=None):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    output = im2double(image)
    output[:, :, 0] *= amount
    output[:, :, 1] *= amount
    return output


def reduce_red_green(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    return output


def reduce_yellow_blue(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 2] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    return output


def reduce_chromaticity(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    image_opponent[:, :, 1:3] *= amount
    output = opponency2rgb(image_opponent, colour_space=colour_space)
    return output


def reduce_lightness(image, amount, mask_radius=None, colour_space='lab'):
    assert (amount >= 0.0), 'amount too low.'
    assert (amount <= 1.0), 'amount too high.'

    image = im2double(image)
    image_opponent = rgb2opponency(image, colour_space=colour_space)
    max_lightness = get_max_lightness(colour_space=colour_space)
    image_opponent[:, :, 0] = ((1 - amount) / 2 + np.multiply(
        image_opponent[:, :, 0] / max_lightness, amount)) * max_lightness
    output = opponency2rgb(image_opponent, colour_space=colour_space)
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

    # image = im2double(image)
    image = image.astype('float32')
    max_pixel = image.max()
    image /= max_pixel

    image_org = image.copy()
    if mask_type == 'circle':
        image_mask = create_mask_image(image, mask_radius, True)
    elif mask_type == 'square':
        image_mask = create_mask_image(image, mask_radius, True)
    else:
        image_mask = create_mask_image_canny(image, sigma=mask_radius, **kwargs)

    min_contrast = contrast_level - pixel_variatoin
    max_contrast = contrast_level + pixel_variatoin

    contrast_level_mat = np.random.uniform(
        low=min_contrast, high=max_contrast, size=image.shape
    )

    image_contrast = (1 - contrast_level_mat) / 2.0 + np.multiply(
        image, contrast_level_mat)
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
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_gamma = image ** gamma
    output = image_org * image_mask + image_gamma * (1 - image_mask)
    return output


def gaussian_blur(image, sigmax, sigmay=None, meanx=0, meany=0, theta=0,
                  mask_radius=None):
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
        illuminant_i = np.random.uniform(low=min_illuminant,
                                         high=max_illuminant,
                                         size=image[:, :, i].shape)
        image[:, :, i] = image[:, :, i] * illuminant_i

    output = image_org * image_mask + image * (1 - image_mask)
    return output


def s_p_noise(image, amount, salt_vs_pepper=0.5, seed=None, clip=True,
              mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='s&p', seed=seed, clip=clip,
                               salt_vs_pepper=salt_vs_pepper, amount=amount)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def speckle_noise(image, var, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='speckle', seed=seed, clip=clip,
                               var=var)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def gaussian_noise(image, var, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='gaussian', seed=seed, clip=clip,
                               var=var)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def poisson_noise(image, seed=None, clip=True, mask_radius=None):
    image = im2double(image)
    image_org = image.copy()
    image_mask = create_mask_image(image, mask_radius)

    image_noise = random_noise(image, mode='poisson', seed=seed, clip=clip)
    output = image_org * image_mask + image_noise * (1 - image_mask)
    return output


def simulate_distance(image, distance_factor, mask_radius=None):
    if distance_factor <= 1:
        return image
    (rows, cols, _) = image.shape
    distance_rows = int(np.round(rows / distance_factor))
    distance_cols = int(np.round(cols / distance_factor))
    output = cv2.resize(image, (distance_rows, distance_cols),
                        interpolation=cv2.INTER_NEAREST)
    output = cv2.resize(output, (rows, cols), interpolation=cv2.INTER_NEAREST)
    return output


# TODO: add other shapes as well
def random_occlusion(image, object_instances=1, object_ratio=0.05):
    out = im2double(image)
    (rows, cols, chns) = out.shape
    extent = (round(rows * object_ratio), round(cols * object_ratio))
    for i in range(object_instances):
        rand_row = np.random.randint(0 + extent[0], rows - extent[0])
        rand_col = np.random.randint(0 + extent[1], cols - extent[1])
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
    std_image = cv2.filter2D((image - avg_image) ** 2, -1, kernel) ** 0.5
    return (std_image, avg_image)


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
