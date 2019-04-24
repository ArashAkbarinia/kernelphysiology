"""
Preparing the input image to be inputted to a network.
"""

import numpy as np
import warnings
from PIL import Image as PilImage
from PIL import ImageCms

rgb_p = ImageCms.createProfile('sRGB')
lab_p = ImageCms.createProfile('LAB')

rgb2lab = ImageCms.buildTransformFromOpenProfiles(rgb_p, lab_p, 'RGB', 'LAB')
lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, rgb_p, 'LAB', 'RGB')


class ColourTransformation(object):

    def __init__(self, colour_inds):
        self.colour_inds = colour_inds

    def __call__(self, img):
        img = ImageCms.applyTransform(img, rgb2lab)
        img = np.asarray(img).copy()
        img[:, :, self.colour_inds] = 0
        img = PilImage.fromarray(img, 'LAB')
        img = ImageCms.applyTransform(img, lab2rgb)
        return img


def colour_transformation(transformation_type):
    ct = []
    if transformation_type != 'trichromat':
        colour_inds = None
        if transformation_type == 'dichromat_rg':
            colour_inds = [1]
        elif transformation_type == 'dichromat_yb':
            colour_inds = [2]
        elif transformation_type == 'monochromat':
            colour_inds = [1, 2]
        # check if it's a valid colour index
        if colour_inds is not None:
            ct.append(ColourTransformation(colour_inds))
        else:
            warnings.warn('Unsupported colour transformation' % type)
    return ct


class ImageTransformation(object):

    def __init__(
            self,
            manipulation_function,
            manipulation_value,
            manipulation_radius):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.manipulation_radius = manipulation_radius

    def __call__(self, img):
        img = np.asarray(img, dtype='uint8')
        manipulation_value = np.random.uniform(*self.manipulation_value)
        img = self.manipulation_function(img,
                                         manipulation_value,
                                         mask_radius=self.manipulation_radius)
        img = PilImage.fromarray(img.astype('uint8'), 'RGB')
        return img
