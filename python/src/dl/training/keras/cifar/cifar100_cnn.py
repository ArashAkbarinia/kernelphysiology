'''Train a simple deep CNN on the CIFAR100 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
import cifar
from keras.utils.data_utils import get_file
from keras import backend as K


def load_data(label_mode='fine', dirname='cifar-100-python'):
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
    x_train, y_train = cifar.load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = cifar.load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


confs = cifar.CifarConfs(num_classes=100, args=sys.argv)

# The data, split between train and test sets:
(confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = load_data('fine', os.path.join(confs.project_root, 'data/datasets/cifar/cifar100/'))

cifar.start_training(confs)
