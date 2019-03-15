'''
Reading the CIFAR-10 dataset.
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


def load_data(dirname=os.path.join(commons.python_root, 'data/datasets/cifar/cifar10/'), which_outputs=[]):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if not os.path.exists(os.path.join(dirname, 'batches.meta')):
        origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        path = get_file(dirname, origin=origin, untar=True)
    else:
        path = dirname

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    # natural versus man made ground truth
    if 'natural_vs_manmade' in which_outputs:
        y_train_nvm = np.zeros((num_train_samples, 1), dtype='bool')
        y_test_nvm = np.zeros((len(y_test), 1), dtype='bool')

        man_made_inds = [0, 1, 8, 9]
        for m in man_made_inds:
            y_train_nvm = y_train_nvm | (y_train == m)
            y_test_nvm = y_test_nvm | (y_test == m)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if 'natural_vs_manmade' in which_outputs:
        y_train_nvm = keras.utils.to_categorical(y_train_nvm, 2)
        y_test_nvm = keras.utils.to_categorical(y_test_nvm, 2)

    y_train = {'all_classes': y_train}
    y_test = {'all_classes': y_test}

    if 'natural_vs_manmade' in which_outputs:
        y_train['natural_vs_manmade'] = y_train_nvm
        y_test['natural_vs_manmade'] = y_test_nvm
    return (x_train, y_train), (x_test, y_test)
