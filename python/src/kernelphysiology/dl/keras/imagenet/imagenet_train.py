'''
Training the IMAGENET dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons
import os

import argparse
import keras
import tensorflow as tf
from keras import backend as K, regularizers
from keras.engine.training import Model
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, Input
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from keras.utils import multi_gpu_model

import imagenet


def build_classifier_model(args):
    n_conv_blocks = 5  # number of convolution blocks to have in our model.
    n_filters = 64  # number of filters to use in the first convolution block.
    l2_reg = regularizers.l2(2e-4)  # weight to use for L2 weight decay.
    activation = 'elu'  # the activation function to use after each linear operation.

    if K.image_data_format() == 'channels_first':
        input_shape = (3, imagenet.HEIGHT, imagenet.WIDTH)
    else:
        input_shape = (imagenet.HEIGHT, imagenet.WIDTH, 3)

    x = input_1 = Input(shape=input_shape)

    area1_nlayers = args.area1_nlayers
    area1_batchnormalise = args.area1_batchnormalise
    area1_activation = args.area1_activation
    # each convolution block consists of two sub-blocks of Conv->Batch-Normalization->Activation,
    # followed by a Max-Pooling and a Dropout layer.
    for i in range(n_conv_blocks):
        if i == 0:
            x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
            if area1_nlayers == 1:
                x = BatchNormalization()(x)
                x = Activation(activation=activation)(x)
            else:
                if area1_batchnormalise:
                    x = BatchNormalization()(x)
                if area1_activation:
                    x = Activation(activation=activation)(x)

            if area1_nlayers == 2:
                #
                x = Conv2D(filters=44, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)
                x = Activation(activation=activation)(x)
            if area1_nlayers == 3:
                #
                x = Conv2D(filters=37, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                if area1_batchnormalise:
                    x = BatchNormalization()(x)
                if area1_activation:
                    x = Activation(activation=activation)(x)

                #
                x = Conv2D(filters=37, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)
                x = Activation(activation=activation)(x)
            if area1_nlayers == 4:
                #
                x = Conv2D(filters=27, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                if area1_batchnormalise:
                    x = BatchNormalization()(x)
                if area1_activation:
                    x = Activation(activation=activation)(x)

                #
                x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                if area1_batchnormalise:
                    x = BatchNormalization()(x)
                if area1_activation:
                    x = Activation(activation=activation)(x)

                #
                x = Conv2D(filters=27, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)
                x = Activation(activation=activation)(x)
        else:
            shortcut = Conv2D(filters=n_filters, kernel_size=(1, 1), padding='same', kernel_regularizer=l2_reg)(x)
            x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
            x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)

            x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2_reg)(x)
            x = Add()([shortcut, x])
            x = BatchNormalization()(x)
            x = Activation(activation=activation)(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.25)(x)

        n_filters *= 2

    # finally, we flatten the output of the last convolution block, and add two Fully-Connected layers.
    x = Flatten()(x)
    x = Dense(units=512, kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation(activation=activation)(x)

    x = Dropout(rate=0.5)(x)
    x = Dense(units=imagenet.N_CLASSES, kernel_regularizer=l2_reg)(x)
    output = Activation(activation='softmax')(x)

    return Model(inputs=[input_1], outputs=[output])


def train_classifier(x_train, y_train, x_test, y_test, model_output_path=None, batch_size=64, epochs=100, initial_lr=1e-3, args=None):
    def lr_scheduler(epoch):
        if epoch < 20:
            return initial_lr
        elif epoch < 40:
            return initial_lr / 2
        elif epoch < 50:
            return initial_lr / 4
        elif epoch < 60:
            return initial_lr / 8
        elif epoch < 70:
            return initial_lr / 16
        elif epoch < 80:
            return initial_lr / 32
        elif epoch < 90:
            return initial_lr / 64
        else:
            return initial_lr / 128

    opt = keras.optimizers.Adam(initial_lr)

    if args.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        parallel_model = multi_gpu_model(model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model_name = 'keras_stl10_area_'
    if args.area1_batchnormalise:
        model_name += 'bnr_'
    if args.area1_activation:
        model_name += 'act_'
    if args.add_dog:
        model_name += 'dog_'
    model_name += str(args.area1_nlayers)
    save_dir = os.path.join(commons.python_root, 'data/nets/stl/stl10/')
    log_dir = os.path.join(save_dir, model_name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'), append=False, separator=';')
    check_points = ModelCheckpoint(os.path.join(log_dir, 'weights.{epoch:05d}.h5'), period=25)

    if not args.multi_gpus == None:
        batch_size *= args.multi_gpus
        parallel_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, validation_data=(x_test, y_test),
                           callbacks=[LearningRateScheduler(lr_scheduler), csv_logger, check_points])
    else:
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                  verbose=1, validation_data=(x_test, y_test),
                  callbacks=[LearningRateScheduler(lr_scheduler), csv_logger, check_points])

    model_name += '.h5'
    model_output_path = os.path.join(save_dir, model_name)
    print('saving trained model to:', model_output_path)
    model.save(model_output_path)


def preprocess_input(img):
    img = img.astype('float32')
    img = (img - 127.5) / 127.5
    return img


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = imagenet.load_data()

    # convert all images to floats in the range [0, 1]
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    # convert the labels to be zero based.
    y_train -= 1
    y_test -= 1

    # convert labels to hot-one vectors.
    y_train = keras.utils.to_categorical(y_train, imagenet.N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, imagenet.N_CLASSES)


    parser = argparse.ArgumentParser(description='Training STL.')
    parser.add_argument('--a1', dest='area1_nlayers', type=int, default=1, help='The number of layers in area 1 (default: 1)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')

    args = parser.parse_args()

    model = build_classifier_model(args)
    model.summary()

    train_classifier(x_train, y_train, x_test, y_test, args=args)
