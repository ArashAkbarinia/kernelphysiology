"""
The main script for GEETUP.
"""

import keras
from keras import backend as K
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils import multi_gpu_model

import tensorflow as tf
import cv2

import numpy as np
import sys
import os
import pickle
import logging
import argparse

from functools import partial
from functools import update_wrapper

from kernelphysiology.dl.keras.video import geetup_db
from kernelphysiology.utils.imutils import max_pixel_ind
from kernelphysiology.utils.path_utils import create_dir


def euc_error(y_true, y_pred, target_size):
    y_true_inds = K.argmax(y_true, axis=2)
    y_true_inds = tf.unravel_index(K.reshape(y_true_inds, [-1]), target_size)
    y_pred_inds = K.argmax(y_pred, axis=2)
    y_pred_inds = tf.unravel_index(K.reshape(y_pred_inds, [-1]), target_size)

    true_pred_diff = K.sum((y_true_inds - y_pred_inds) ** 2, axis=0)
    euc_distance = tf.sqrt(
        tf.cast(true_pred_diff, dtype=tf.float32))
    return tf.reduce_mean(euc_distance)


def lr_schedule_resnet(epoch, lr):
    new_lr = lr * (0.1 ** (epoch // (45 / 3)))
    return new_lr


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # inverting the normalisation for each channel
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] * std[i]) + mean[i]
    return tensor


def normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # normalising the channels
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] - mean[i]) / std[i]
    return tensor


def class_net_fcn_2p_lstm(input_shape, image_net=None, mid_layer=None):
    c = 32
    input_img = Input(input_shape, name='input')
    if image_net is not None:
        c0 = TimeDistributed(image_net)(input_img)
    else:
        x = input_img
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(c0)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)
    c1 = BatchNormalization()(c1)

    x = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = TimeDistributed(ZeroPadding2D(padding=((1, 0), (1, 0))))(x)

    if image_net is not None:
        x_mid = TimeDistributed(mid_layer)(input_img)
        x = Concatenate()([x_mid, x])

    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c2 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)
    c2 = BatchNormalization()(c2)

    x = TimeDistributed(ZeroPadding2D(padding=(1, 1)))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)

    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c3 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)
    c3 = BatchNormalization()(c3)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = Concatenate()([c2, x])
    x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    c1 = TimeDistributed(ZeroPadding2D(padding=((1, 0), (1, 0))))(c1)
    x = Concatenate()([c1, x])

    x = TimeDistributed(UpSampling2D((4, 4)))(x)

    output = TimeDistributed(
        Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'),
        name='output')(x)

    output = Reshape((-1, input_shape[1] * input_shape[2], 1))(output)
    model = keras.models.Model(input_img, output)
    return model


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def visualise_results(current_image, gt, pred, file_name):
    rows, cols, _ = current_image.shape
    gt_pixel = max_pixel_ind(
        np.reshape(gt.squeeze(), (rows, cols))
    )
    cv2.circle(current_image, gt_pixel, 15, (0, 255, 0))

    pred_pixel = max_pixel_ind(
        np.reshape(pred.squeeze(), (rows, cols))
    )
    cv2.circle(current_image, pred_pixel, 15, (0, 0, 255))

    euc_dis = np.linalg.norm(
        np.asarray(pred_pixel) - np.asarray(gt_pixel))
    cx = round(rows / 2)
    cy = round(cols / 2)
    cv2.putText(current_image, str(int(euc_dis)), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.imwrite(file_name, current_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GEETUP Train/Test')
    parser.add_argument(
        '--weights',
        dest='weights',
        type=str,
        default=None,
        help='Path to the weights')
    parser.add_argument(
        '--log_dir',
        dest='log_dir',
        type=str,
        default='Ex',
        help='Path to the logging directory (default: Ex')
    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[0],
        help='List of GPUs to be used (default: [0])')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
        help='Only evaluation (default: False)')
    parser.add_argument(
        '--random',
        dest='random',
        type=int,
        default=None,
        help='Number of random images to try (default: None)')

    args = parser.parse_args(sys.argv[1:])

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in args.gpus)
    gpus = [*range(len(args.gpus))]

    create_dir(args.log_dir)

    logging.basicConfig(
        filename=args.log_dir + 'experiment_info.log', filemode='w',
        format='%(levelname)s: %(message)s', level=logging.INFO
    )

    lr_schedule_lambda = partial(lr_schedule_resnet, lr=0.1)

    frames_gap = 10
    sequence_length = 9
    batch_size = 8
    target_size = (224, 224)

    mean = [103.939, 116.779, 123.68]
    std = [1, 1, 1]

    preprocess = partial(normalise_tensor, mean=mean, std=std)

    training_list = []
    if args.evaluate is False:
        pickle_in = open('two_parts_training.pickle', 'rb')
        training_list = pickle.load(pickle_in)

        training_generator = geetup_db.GeetupGenerator(
            training_list,
            batch_size=batch_size,
            target_size=target_size,
            gaussian_sigma=30.5,
            preprocessing_function=preprocess)

    pickle_in = open('two_parts_testing.pickle', 'rb')
    testing_list = pickle.load(pickle_in)

    print('Training %d, Testing %d' % (len(training_list), len(testing_list)))

    testing_generator = geetup_db.GeetupGenerator(
        testing_list,
        batch_size=batch_size,
        target_size=target_size,
        gaussian_sigma=30.5,
        preprocessing_function=preprocess,
        shuffle=not args.evaluate)

    resnet = keras.applications.ResNet50(weights='imagenet')
    for i, layer in enumerate(resnet.layers):
        layer.trainable = False
    resnet_mid = keras.models.Model(inputs=resnet.input,
                                    outputs=resnet.get_layer(
                                        'activation_22').output)
    resnet = keras.models.Model(inputs=resnet.input,
                                outputs=resnet.get_layer(
                                    'activation_10').output)
    # outputs = [resnet.get_layer('activation_10').output,
    #           resnet.get_layer('activation_22').output]
    # resnet = K.function([resnet.input, K.learning_phase()], outputs)
    model = class_net_fcn_2p_lstm((sequence_length, *target_size, 3), resnet,
                                  resnet_mid)
    if args.evaluate:
        model.load_weights(args.weights)

    euc_metric = wrapped_partial(euc_error, target_size=target_size)

    metrics = [euc_metric]
    loss = 'binary_crossentropy'
    opt = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=False)
    if len(gpus) == 1:
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss=loss, optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(model, gpus=gpus)
        # TODO: this compilation probably is not necessary
        parallel_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    if args.random is not None:
        for i in range(args.random):
            ran_ind = np.random.randint(0, testing_generator.__len__())
            x, y = testing_generator.__getitem__(ran_ind)
            pred_fix = model.predict_on_batch(x)
            x = inv_normalise_tensor(x, mean=mean, std=std)
            for b in range(x.shape[0]):
                for f in range(x.shape[1]):
                    file_name = 'random_results/image_%d_%d_%d.jpg' % \
                                (ran_ind, b, f)
                    current_image = x[b, f,].squeeze()
                    current_image = current_image[:, :, [2, 1, 0]].copy()
                    visualise_results(current_image, y[b, f,],
                                      pred_fix[b, f,], file_name)
    elif args.evaluate:
        # TODO: multiprocessing
        [loss_eval, euc_eval] = model.evaluate_generator(
            generator=testing_generator,
            use_multiprocessing=False,
            workers=1,
            verbose=1
        )
        print(loss_eval, euc_eval)
    else:
        last_checkpoint_logger = ModelCheckpoint(
            args.log_dir + '/model_weights_last.h5',
            verbose=1,
            save_weights_only=True,
            save_best_only=False)
        csv_logger = CSVLogger(
            os.path.join(args.log_dir, 'log.csv'),
            append=False, separator=';')

        steps_per_epoch = 10000
        validation_steps = 100
        if parallel_model is not None:
            parallel_model.fit_generator(generator=training_generator,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_data=testing_generator,
                                         validation_steps=validation_steps,
                                         use_multiprocessing=False,
                                         workers=1, epochs=45, verbose=1,
                                         callbacks=[
                                             csv_logger,
                                             LearningRateScheduler(
                                                 lr_schedule_lambda),
                                             last_checkpoint_logger])
        else:
            model.fit_generator(generator=training_generator,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=testing_generator,
                                validation_steps=validation_steps,
                                use_multiprocessing=False,
                                workers=1, epochs=45, verbose=1,
                                callbacks=[
                                    csv_logger,
                                    LearningRateScheduler(lr_schedule_lambda),
                                    last_checkpoint_logger])
