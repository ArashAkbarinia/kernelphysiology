'''
Common utility funcions for Keras.
'''

import numpy as np

import argparse

import math
import keras
from keras import backend as K
from PIL import Image as pil_image

def get_conv2ds(model, topn=math.inf):
    conv2d_inds = []
    for i in range(0, len(model.layers)):
        if type(model.layers[i]) is keras.layers.convolutional.Conv2D:
            conv2d_inds.append(i)
    return conv2d_inds


def set_area_trainable_false(model, num_areas):
    current_area = 1
    for i in range(0, len(model.layers)):
        if type(model.layers[i]) is keras.layers.pooling.MaxPooling2D:
            if num_areas == current_area:
                break
            current_area += 1
        else:
            model.layers[i].trainable = False
    return model


def common_arg_parser(args):
    parser = argparse.ArgumentParser(description='Training prominent nets of Keras.')
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network', type=str, help='Which network to be used')

    parser.add_argument('--area1layers', dest='area1layers', type=int, default=0, help='The number of layers in area 1 (default: 0)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--a1reduction', dest='area1_reduction', action='store_true', default=False, help='Whether to include a reduction layer in area 1 (default: False)')
    parser.add_argument('--a1dilation', dest='area1_dilation', action='store_true', default=False, help='Whether to include dilation in kernels in area 1 (default: False)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--train_contrast', dest='train_contrast', type=int, default=100, help='The level of contrast to be used at training (default: 100)')

    parser.add_argument('--name', dest='experiment_name', type=str, default='Ex', help='The name of the experiment (default: Ex)')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default=None, help='The path to a previous checkpoint to continue (default: None)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')

    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--target_size', dest='target_size', type=int, default=224, help='Target size (default: 224)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--steps', dest='steps', type=int, default=None, help='Number of steps per epochs (default: number of samples divided by the batch size)')
    parser.add_argument('--preprocessing', dest='preprocessing', type=str, default=None, help='The preprocessing function (default: network preprocessing function)')
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', default=False, help='Whether to augment data (default: False)')

    return parser.parse_args(args)


def keras_resize_img(img, target_size, resample=pil_image.NEAREST):
    img = pil_image.fromarray(img).resize(target_size, resample)

    return img


class ResizeGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_data, y_data, num_classes, batch_size=32, target_size=(224, 224), preprocessing_function=None, shuffle=True):
        'Initialisation'
        self.x_data = x_data
        self.y_data = y_data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle

        if K.image_data_format() == 'channels_last':
            self.out_shape = (*self.target_size, self.x_data.shape[3])
        elif K.image_data_format() == 'channels_first':
            self.out_shape = (self.x_data.shape[1], *self.target_size)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.x_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate indices of the batch
        current_batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # generate data
        (x_batch, y_batch) = self.__data_generation(current_batch)

        return (x_batch, y_batch)

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(self.x_data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, current_batch):
        'Generates data containing batch_size samples'
        # initialisation
        x_batch = np.empty((self.batch_size, *self.out_shape), dtype='float32')
        y_batch = np.empty((self.batch_size, self.num_classes), dtype=int)

        # generate data
        for i, im_id in enumerate(current_batch):
            # store sample
            x_batch[i,] = keras_resize_img(self.x_data[im_id,], self.target_size)

            # store class
            y_batch[i,] = self.y_data[im_id,]

        if self.preprocessing_function:
            x_batch = self.preprocessing_function(x_batch)

        return (x_batch, y_batch)