'''
Collection of different metrics.
'''


import numpy as np
import math


def reproduction_angular_error_radian(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)

    l1 = l1 / l1.sum()
    l2 = l2 / l2.sum()

    l2l1 = l2 / l1
    w1 = l2l1 / (np.sum(l2l1 ** 2) ** 0.5)
    w2 = 1 / (3 ** 0.5)

    w1w2 = np.minimum(np.sum(w1 * w2), 1)
    w1w2 = np.maximum(w1w2, -1)
    r = math.acos(w1w2)
    return r


def reproduction_angular_error_degree(l1, l2):
    r = reproduction_angular_error_radian(l1, l2)
    r = math.degrees(r)
    return r