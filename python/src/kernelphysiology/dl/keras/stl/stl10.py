'''
Reading the STL-10 dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons
import os
import sys

import urllib.request as urllib
import tarfile
import numpy as np
import keras


# number of classes in the STL-10 dataset
N_CLASSES = 10

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'


def read_labels(dirname):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(dirname, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def download_and_extract(dirname):
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = dirname
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = 'stl10'
    filepath = os.path.join(dest_directory, 'test_X.bin')
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_data(dirname=os.path.join(commons.python_root, 'data/datasets/stl/stl10/')):
    # download the extract the dataset.
    download_and_extract(dirname)

    # load the train and test data and labels.
    x_train = read_all_images(os.path.join(dirname, 'train_X.bin'))
    y_train = read_labels(os.path.join(dirname, 'train_y.bin'))
    x_test = read_all_images(os.path.join(dirname, 'test_X.bin'))
    y_test = read_labels(os.path.join(dirname, 'test_y.bin'))

    # convert the labels to be zero based.
    y_train -= 1
    y_test -= 1

    # convert labels to hot-one vectors.
    y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    return (x_train, y_train), (x_test, y_test)
