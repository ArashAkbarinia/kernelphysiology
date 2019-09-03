"""
Common utility functions for Keras.
"""

import numpy as np

import math
import keras
import cv2
from keras import backend as K
from functools import partial
from PIL import Image as pil_image

from kernelphysiology.utils.image import ImageDataGenerator

from kernelphysiology.utils.imutils import adjust_contrast


def get_top_k_accuracy(k):
    top_k_acc = partial(keras.metrics.top_k_categorical_accuracy, k=k)
    top_k_acc.__name__ = 'top_%d_acc' % k
    return top_k_acc


def get_input_shape(target_size):
    # check the input shape
    if K.image_data_format() == 'channels_last':
        input_shape = (*target_size, 3)
    elif K.image_data_format() == 'channels_first':
        input_shape = (3, *target_size)
    return input_shape


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


def keras_resize_img(img, target_size, resample=pil_image.NEAREST):
    img = pil_image.fromarray(img).resize(target_size, resample)

    return img


class ResizeGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, x_data, y_data, num_classes, batch_size=32,
                 target_size=(224, 224), preprocessing_function=None,
                 shuffle=True):
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
        current_batch = self.indices[
                        index * self.batch_size:(index + 1) * self.batch_size]

        # generate data
        (x_batch, y_batch) = self.__data_generation(current_batch)

        return x_batch, y_batch

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
            x_batch[i,] = keras_resize_img(self.x_data[im_id,],
                                           self.target_size)

            # store class
            y_batch[i,] = self.y_data[im_id,]

        if self.preprocessing_function:
            x_batch = self.preprocessing_function(x_batch)

        return x_batch, y_batch


def get_validatoin_generator(args, x_test, y_test,
                             validation_preprocessing_function):
    (args.validation_generator, args.validation_samples) = resize_generator(
        x_test, y_test, batch_size=args.batch_size,
        target_size=args.target_size,
        preprocessing_function=validation_preprocessing_function)

    return args


def get_generators(args, x_train, y_train, x_test, y_test,
                   train_preprocessing_function,
                   validation_preprocessing_function):
    (args.train_generator, args.train_samples) = \
        resize_generator(x_train,
                         y_train,
                         batch_size=args.batch_size,
                         target_size=args.target_size,
                         preprocessing_function=train_preprocessing_function,
                         horizontal_flip=args.horizontal_flip,
                         vertical_flip=args.vertical_flip,
                         zoom_range=args.zoom_range,
                         width_shift_range=args.width_shift_range,
                         height_shift_range=args.height_shift_range)

    (args.validation_generator, args.validation_samples) = resize_generator(
        x_test, y_test, batch_size=args.batch_size,
        target_size=args.target_size,
        preprocessing_function=validation_preprocessing_function)

    return args


def resize_generator(x_data, y_data, target_size, batch_size=32,
                     preprocessing_function=None,
                     horizontal_flip=False, vertical_flip=False,
                     zoom_range=0.0, width_shift_range=0.0,
                     height_shift_range=0.0):
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 zoom_range=zoom_range,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range
                                 )
    data_batches = datagen.flow(x_data, y_data, batch_size=batch_size)
    if target_size[0] != x_data.shape[2]:
        return (resize_iterator(data_batches, target_size), x_data.shape[0])
    else:
        return data_batches, x_data.shape[0]


def resize_iterator(batches, target_size):
    '''
    Take as input a Keras ImageGen (Iterator) and generates resize according to
    target_size from the image batches generated by the original iterator.
    '''
    while True:
        batch_x, data_y = next(batches)
        if K.image_data_format() == 'channels_last':
            data_x = np.zeros((batch_x.shape[0], *target_size, 3))
        elif K.image_data_format() == 'channels_first':
            data_x = np.zeros((batch_x.shape[0], 3, *target_size))
        for i in range(batch_x.shape[0]):
            # TODO: consider different interpolation for resize
            data_x[i,] = cv2.resize(batch_x[i,], target_size)
        yield data_x, data_y


# TODO: add a random crop function


def contrast_generator(batches, contrast_range):
    '''
    Take as input a Keras ImageGen (Iterator) and generates random contrast
    maniputed from the image batches generated by the original iterator.
    '''
    while True:
        batch_x, data_y = next(batches)
        data_x = np.zeros(batch_x.shape)
        for i in range(batch_x.shape[0]):
            data_x[i,] = adjust_contrast(
                batch_x[i,], np.random.uniform(*contrast_range)
            )
        yield data_x, data_y
