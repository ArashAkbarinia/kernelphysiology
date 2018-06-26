'''Train a simple deep CNN on the CIFAR10 small images dataset.

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


def load_data(dirname='cifar-10-batches-py'):
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
         y_train[(i - 1) * 10000: i * 10000]) = cifar.load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = cifar.load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


confs = cifar.CifarConfs(num_classes=10)
confs.area1_nlayers = sys.argv[1]

# The data, split between train and test sets:
(confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = load_data(os.path.join(confs.project_root, 'data/datasets/cifar/cifar10/'))

cifar.start_training(confs)
