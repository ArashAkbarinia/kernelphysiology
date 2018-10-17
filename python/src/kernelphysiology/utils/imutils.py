from skimage.color import rgb2gray
from skimage.io import imread, imsave
from scipy.misc import toimage
import numpy as np

#import wrapper as wr

###########################################################
#   IMAGE IO
###########################################################

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""

    return imread(path) / 255.0


def save_img(image, imgname, use_JPEG=False):
    """Save image as either .jpeg or .png"""

    if use_JPEG:
        imsave(imgname+".JPEG", image) 
    else:
        toimage(image,
                cmin=0.0, cmax=1.0).save(imgname+".png")


###########################################################
#   IMAGE MANIPULATION
###########################################################


def im2double(image):
    image = image.astype('float32')
    return image / 255


def adjust_contrast(image, contrast_level, pixel_variatoin=0):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert(contrast_level >= 0.0), "contrast_level too low."
    assert(contrast_level <= 1.0), "contrast_level too high."

    if image.dtype == 'uint8':
        image = im2double(image)
    else:
        max_pixel = np.max(image)
        if max_pixel > 1 and max_pixel <= 255:
            image = im2double(image)

    min_contrast = contrast_level - pixel_variatoin
    max_contrast = contrast_level + pixel_variatoin
    
    pixel_variatoin = np.random.uniform(low=min_contrast, high=max_contrast, size=image.shape)
    contrast_level = np.ones(image.shape) * pixel_variatoin

    return (1 - contrast_level) / 2.0 + np.multiply(image, contrast_level)


def grayscale_contrast(image, contrast_level):
    """Convert to grayscale. Adjust contrast.

    parameters:
    - image: a numpy.ndarray 
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    return adjust_contrast(rgb2gray(image), contrast_level)


def uniform_noise(image, width, contrast_level, rng):
    """Convert to grayscale. Adjust contrast. Apply uniform noise.

    parameters:
    - image: a numpy.ndarray 
    - width: a scalar indicating width of additive uniform noise
             -> then noise will be in range [-width, width]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    image = grayscale_contrast(image, contrast_level)

    return apply_uniform_noise(image, -width, width, rng)


###########################################################
#   HELPER FUNCTIONS
###########################################################

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
        return np.random.uniform(low=low, high=high,
                                 size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high,
                           size=(nrow, ncol))


def is_in_bounds(mat, low, high):
    """Return wether all values in 'mat' fall between low and high.

    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all(np.logical_and(mat >= 0, mat <= 1))


#def eidolon_partially_coherent_disarray(image, reach, coherence, grain):
#    """Return parametrically distorted images (produced by Eidolon factory.
#
#    For more information on the effect of different distortions, please
#    have a look at the paper: Koenderink et al., JoV 2017,
#    Eidolons: Novel stimuli for vision research).
#
#    - image: a numpy.ndarray
#    - reach: float, controlling the strength of the manipulation
#    - coherence: a float within [0, 1] with 1 = full coherence
#    - grain: float, controlling how fine-grained the distortion is
#    """
#
#    return wr.partially_coherent_disarray(wr.data_to_pic(image),
#                                          reach, coherence, grain)

