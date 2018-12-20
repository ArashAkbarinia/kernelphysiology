'''
Reading the CIFAR-100 dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons
import numpy as np
import os
import keras
from kernelphysiology.dl.keras.datasets.cifar.cifar_utils import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K


def load_data(label_mode='fine', dirname=os.path.join(commons.python_root, 'data/datasets/cifar/cifar100/')):
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

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    return (x_train, y_train), (x_test, y_test)
