"""
Converting from and to label matrices.
"""

import numpy as np

from kernelphysiology.utils.visualise import colour_world


def colour_label(label_mat, colours=None, num_labels=None, dataset=None):
    assert colours is not None or num_labels is not None or dataset is not None
    if colours is None:
        if num_labels is not None:
            colours = colour_world.default_colours(num_labels)
        else:
            colours = colour_world.dataset_colourmaps(dataset)
    colour_img = np.zeros((*label_mat.shape, 3))
    nlabels = len(colours)
    for l in range(nlabels):
        colour_img[label_mat == l] = colours[l]
    return colour_img
