"""
Preparing the input image to be inputted to a network.
"""


import numpy as np
import warnings
from PIL import Image as PilImage

from kernelphysiology.utils.imutils import reduce_red_green
from kernelphysiology.utils.imutils import reduce_yellow_blue
from kernelphysiology.utils.imutils import reduce_chromaticity


class ColourTransformation(object):

    def __init__(
            self,
            manipulation_function,
            manipulation_value=0,
            colour_space='lab'):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.colour_space = colour_space

    def __call__(self, img):
        img = np.asarray(img, dtype='uint8')
        img = self.manipulation_function(img,
                                         self.manipulation_value,
                                         colour_space=self.colour_space)
        img = PilImage.fromarray(img.astype('uint8'), 'RGB')
        return img


def colour_transformation(transformation_type, colour_space='lab'):
    ct = []
    if transformation_type != 'trichromat':
        manipulation_function = None
        if transformation_type == 'dichromat_rg':
            manipulation_function = reduce_red_green
        elif transformation_type == 'dichromat_yb':
            manipulation_function = reduce_yellow_blue
        elif transformation_type == 'monochromat':
            manipulation_function = reduce_chromaticity
        # check if it's a valid manipulation
        if manipulation_function is not None:
            ct.append(
                ColourTransformation(
                    manipulation_function,
                    colour_space=colour_space))
        else:
            warnings.warn('Unsupported colour transformation' % type)
    return ct
