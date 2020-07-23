"""
The world of colours!
"""

import sys
import numpy as np
from matplotlib import pyplot as plt


def _get_cmap(n, name='gist_rainbow'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard colormap name."""
    return plt.cm.get_cmap(name, n)


def default_colours(num_colours, alpha_chn=False):
    tmp_colours = _get_cmap(num_colours)
    colours = []
    for i in range(num_colours):
        if alpha_chn:
            colours.append(tmp_colours(i))
        else:
            colours.append(tmp_colours(i)[:-1])
    return colours


def create_pascal_label_colourmap():
    """Creates a label colourmap used in PASCAL VOC segmentation benchmark.
    Borrowed from get_dataset_colormap.py of Tensorflow GitHub

    Returns:
      A colourmap for visualizing segmentation results.
    """
    colourmap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colourmap[:, channel] |= _bit_get(ind, channel) << shift
        ind >>= 3

    colourmap = list(np.float32(colourmap) / 255)
    return colourmap


def dataset_colourmaps(dataset):
    if 'voc' in dataset or 'pascal' in dataset:
        colourmaps = create_pascal_label_colourmap()
    else:
        sys.exit('Default colourmap for dataset is not supported %s' % dataset)
    return colourmaps


def _bit_get(val, idx):
    """Gets the bit value.
    Borrowed from get_dataset_colormap.py of Tensorflow GitHub

    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.

    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1
