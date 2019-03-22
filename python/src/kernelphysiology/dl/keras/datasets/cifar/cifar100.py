'''
Reading the CIFAR-100 dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import keras
from keras.utils.data_utils import get_file
from keras import backend as K

from kernelphysiology import commons
from kernelphysiology.dl.keras.datasets.cifar.cifar_utils import load_batch


def load_data(label_mode='fine', dirname=os.path.join(commons.python_root, 'data/datasets/cifar/cifar100/'), which_outputs=[]):
    """Loads CIFAR100 dataset.

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    if not os.path.exists(os.path.join(dirname, 'meta')):
        origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        path = get_file(dirname, origin=origin, untar=True)
    else:
        path = dirname

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    # natural versus man made ground truth
    if 'natural_vs_manmade' in which_outputs:
        y_train_nvm = np.zeros((len(y_train), 1), dtype='bool')
        y_test_nvm = np.zeros((len(y_test), 1), dtype='bool')

        # TODO: for coarse labels as well
        man_made_inds = [9, 10, 16, 28, 61,
                         22, 39, 40, 86, 87,
                         5, 20, 25, 84, 94,
                         12, 17, 37, 68, 76,
                         8, 13, 48, 58, 90,
                         41, 69, 81, 85, 89]
        for m in man_made_inds:
            y_train_nvm = y_train_nvm | (y_train == m)
            y_test_nvm = y_test_nvm | (y_test == m)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    if 'natural_vs_manmade' in which_outputs:
        y_train_nvm = keras.utils.to_categorical(y_train_nvm, 2)
        y_test_nvm = keras.utils.to_categorical(y_test_nvm, 2)

    y_train = {'all_classes': y_train}
    y_test = {'all_classes': y_test}

    if 'natural_vs_manmade' in which_outputs:
        y_train['natural_vs_manmade'] = y_train_nvm
        y_test['natural_vs_manmade'] = y_test_nvm
    return (x_train, y_train), (x_test, y_test)
