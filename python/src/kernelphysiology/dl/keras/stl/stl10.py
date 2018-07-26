from __future__ import print_function

import commons
import os
import sys

import argparse
import urllib.request as urllib
import tarfile
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from keras.utils import multi_gpu_model

import dl.keras.contrast_net as cnet

# number of classes in the STL-10 dataset.
N_CLASSES = 10

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = os.path.join(commons.python_root, 'data/datasets/stl/')

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'stl10/train_X.bin')

# path to the binary train file with labels
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'stl10/train_y.bin')

# path to the binary test file with image data
TEST_DATA_PATH = os.path.join(DATA_DIR, 'stl10/test_X.bin')

# path to the binary test file with labels
TEST_LABELS_PATH = os.path.join(DATA_DIR, 'stl10/test_y.bin')

# path to class names file
CLASS_NAMES_PATH = os.path.join(DATA_DIR, 'stl10/class_names.txt')


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
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


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
    plt.axis('off')

    plt.imshow(image)
    plt.savefig(name, bbox_inches='tight', dpi=96)


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = 'stl10'
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def start_training(args):
    print('x_train shape:', args.x_train.shape)
    print(args.x_train.shape[0], 'train samples')
    print(args.x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    args.y_train = keras.utils.to_categorical(args.y_train, args.num_classes)
    args.y_test = keras.utils.to_categorical(args.y_test, args.num_classes)
    

    print('Processing with %d layers in area 1' % args.area1_nlayers)
    args.model_name += str(args.area1_nlayers)
    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    check_points = ModelCheckpoint(os.path.join(args.log_dir, 'weights.{epoch:05d}.h5'), period=args.log_period)
    args.callbacks = [check_points, csv_logger]

    args.area1_nlayers = int(args.area1_nlayers)
    
    model = cnet.build_classifier_model(args=args)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    if args.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        args.model = model
        args.parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            args.model = model
        parallel_model = multi_gpu_model(args.model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        args.parallel_model = parallel_model

    args.x_train = cnet.preprocess_input(args.x_train)
    args.x_test = cnet.preprocess_input(args.x_test)
    
    args = cnet.train_model(args)

    # Score trained model.
    scores = args.model.evaluate(args.x_test, args.y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def load_data(dirname=None):
    # download the extract the dataset.
    download_and_extract()

    # load the train and test data and labels.
    x_train = read_all_images(TRAIN_DATA_PATH)
    y_train = read_labels(TRAIN_LABELS_PATH)
    x_test = read_all_images(TEST_DATA_PATH)
    y_test = read_labels(TEST_LABELS_PATH)

    return (x_train, y_train), (x_test, y_test)

    
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()

    # convert all images to floats in the range [0, 1]
    x_train = cnet.preprocess_input(x_train)
    x_test = cnet.preprocess_input(x_test)

    # convert the labels to be zero based.
    y_train -= 1
    y_test -= 1

    # convert labels to hot-one vectors.
    y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, N_CLASSES)


    parser = argparse.ArgumentParser(description='Training STL.')
    parser.add_argument('--a1', dest='area1_nlayers', type=int, default=1, help='The number of layers in area 1 (default: 1)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--a1reduction', dest='area1_reduction', action='store_true', default=False, help='Whether to include a reduction layer in area 1 (default: False)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')
    parser.add_argument('--name', dest='experiment_name', type=str, default='', help='The name of the experiment (default: None)')

    args = parser.parse_args()

    args.x_train = x_train
    args.x_test = x_test
    args.num_classes = N_CLASSES

    model = cnet.build_classifier_model(args)

    cnet.train_model(args=args)
