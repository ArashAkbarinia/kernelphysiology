"""
A collection of function to help with matrices and arrays.
"""

import numpy as np


def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
    return idx
