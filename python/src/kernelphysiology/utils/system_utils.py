"""
A collection of utility function for interacting with system.
"""

import os


def set_visible_gpus(gpus):
    # this is better done with CUDA_VISIBLE_DEVICES
    # if gpus[0] == -1 or gpus is None:
    #     gpus = []
    # os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in gpus)
    # gpus = [*range(len(gpus))]
    return gpus
