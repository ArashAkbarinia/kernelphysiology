from keras.models import Sequential
from keras.layers import Reshape, Input, UpSampling2D, Concatenate, \
    ZeroPadding2D
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import keras
from keras.layers.core import Lambda
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from functools import partial, update_wrapper

from keras import backend as K
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import MaxPooling3D, MaxPooling2D
import glob
from skimage import io, color, transform
import sys
import os
import pickle
import logging
import tensorflow as tf
from keras.utils import multi_gpu_model

from kernelphysiology.filterfactory import gaussian
from kernelphysiology.dl.keras.video import geetup_db


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


def show_sample(sample):
    sample = sample.squeeze()
    n_frames = sample.shape[0]
    fig = plt.figure(figsize=(14, 2))
    for i in range(sample.shape[0]):
        plt.subplot(2, round(n_frames / 2), i + 1)
        to_be_displayed = sample[i,].squeeze()  # / sample[i, ].max()
        plt.imshow(to_be_displayed, cmap='gray')


def show_results(sample, gt, predict):
    sample = sample.squeeze()
    gt = gt.squeeze()
    predict = predict.squeeze()
    n_frames = sample.shape[0]
    fig = plt.figure(figsize=(14, 2))

    for i in range(sample.shape[0]):
        plt.subplot(2, round(n_frames / 2), i + 1)
        to_be_displayed = sample[i,].squeeze()  # / sample[i, ].max()
        top_layer = gt[i,].squeeze()  # / gt[i, ].max()
        im1 = plt.imshow(to_be_displayed, cmap=plt.cm.gray)
        im2 = plt.imshow(top_layer, cmap=plt.cm.viridis, alpha=.5)

    fig = plt.figure(figsize=(14, 2))

    for i in range(sample.shape[0]):
        plt.subplot(2, round(n_frames / 2), i + 1)
        to_be_displayed = sample[i,].squeeze()  # / sample[i, ].max()
        top_layer = predict[i,].squeeze()  # / predict[i, ].max()
        im1 = plt.imshow(to_be_displayed, cmap=plt.cm.gray)
        im2 = plt.imshow(top_layer, cmap=plt.cm.viridis, alpha=.5)


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # inverting the normalisation for each channel
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] * std[i]) + mean[i]
    tensor = tensor.clip(0, 1)
    return tensor


def normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # normalising the channels
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] - mean[i]) / std[i]
    return tensor


def read_data(dataset_dir, target_size=(40, 40, 3), overwrite=False):
    input_frames = []
    fixation_points = []
    for i, subject_dir in enumerate(glob.glob(dataset_dir + '/*/segments/1/')):
        if i == 20:
            break
        if overwrite is False & os.path.isfile(subject_dir + 'frames.pickle'):
            pickle_in = open(subject_dir + 'frames.pickle', 'rb')
            current_input_frames = pickle.load(pickle_in)
            pickle_in = open(subject_dir + 'fixations.pickle', 'rb')
            current_fixation_points = pickle.load(pickle_in)
        else:
            current_input_frames, current_fixation_points = \
                read_one_directory(subject_dir, target_size=target_size)
            pickle_out = open(subject_dir + 'frames.pickle', 'wb')
            pickle.dump(current_input_frames, pickle_out)
            pickle_out.close()
            pickle_out = open(subject_dir + 'fixations.pickle', 'wb')
            pickle.dump(current_fixation_points, pickle_out)
            pickle_out.close()
        input_frames.extend(current_input_frames)
        fixation_points.extend(current_fixation_points)
    input_frames = np.array(input_frames)
    fixation_points = np.array(fixation_points)
    return input_frames, fixation_points


def read_one_directory(dataset_dir, target_size=(40, 40, 3)):
    frames_gap = 10
    rows = target_size[0]
    cols = target_size[1]
    chns = target_size[2]
    input_frames = []
    fixation_points = []
    g_fix = gaussian.gaussian_kernel2(1.5, 1.5)
    for video_dir in glob.glob(dataset_dir + '/*/'):
        video_ind = video_dir.split('/')[-2].split('_')[-1]
        fixation_array = np.loadtxt(
            dataset_dir + 'SUBSAMP_EYETR_' + video_ind + '.txt')
        current_num_frames = len(glob.glob((video_dir + '/*.jpg')))
        selected_frame_infs = [*range(0, current_num_frames, frames_gap)]
        acceptable_frame_seqs = len(selected_frame_infs) - frames_gap
        if acceptable_frame_seqs < 0:
            continue
        if fixation_array.shape[0] != current_num_frames:
            logging.info('%s' % video_dir)
            continue
        current_input_frames = np.zeros(
            (acceptable_frame_seqs, frames_gap, rows, cols, chns),
            dtype=np.float)
        current_fixation_points = np.zeros(
            (acceptable_frame_seqs, frames_gap, rows, cols, 1),
            dtype=np.float)
        for i in range(acceptable_frame_seqs + frames_gap - 1):
            img_ind = selected_frame_infs[i]
            current_img = io.imread(
                video_dir + '/frames' + str(img_ind + 1) + '.jpg')
            if chns == 1:
                current_img = color.rgb2gray(current_img)
            current_img = current_img.astype('float') / 255
            org_rows = current_img.shape[0]
            org_cols = current_img.shape[1]
            # TODO: use better interpolation
            current_img = transform.resize(current_img, (rows, cols))

            current_fixation = np.zeros((rows, cols, 1))
            if fixation_array[img_ind, 1] > 0 and fixation_array[
                img_ind, 0] > 0:
                fpr = int(fixation_array[img_ind, 1] * (rows / org_rows))
                fpc = int(fixation_array[img_ind, 0] * (cols / org_cols))

                sr = fpr - (g_fix.shape[0] // 2)
                sc = fpc - (g_fix.shape[1] // 2)
                # making sure they're within the range of image
                gsr = np.maximum(0, -sr)
                gsc = np.maximum(0, -sc)

                er = sr + g_fix.shape[0]
                ec = sc + g_fix.shape[1]
                # making sure they're within the range of image
                sr = np.maximum(0, sr)
                sc = np.maximum(0, sc)

                er_diff = er - current_img.shape[0]
                ec_diff = ec - current_img.shape[1]
                ger = np.minimum(g_fix.shape[0], g_fix.shape[0] - er_diff)
                gec = np.minimum(g_fix.shape[1], g_fix.shape[1] - ec_diff)

                er = np.minimum(er, current_img.shape[0])
                ec = np.minimum(ec, current_img.shape[1])

                current_fixation[sr:er, sc:ec, 0] = \
                    g_fix[gsr:ger, gsc:gec] / g_fix[gsr:ger, gsc:gec].max()
            for j in range(i + 1):
                current_ind = j
                if (i - j) < frames_gap and current_ind < acceptable_frame_seqs:
                    current_input_frames[current_ind, i - j, :, :, :] = \
                        current_img.copy()
                    current_fixation_points[current_ind, i - j,] = \
                        current_fixation.copy()
        input_frames.extend(current_input_frames)
        fixation_points.extend(current_fixation_points)

    input_frames = np.array(input_frames)
    fixation_points = np.array(fixation_points)
    input_frames = normalise_tensor(input_frames, mean, std)
    return input_frames, fixation_points


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


os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in sys.argv[2:])
gpus = [*range(len(sys.argv) - 2)]

logging.basicConfig(filename='experiment_info.log', filemode='w',
                    format='%(levelname)s: %(message)s', level=logging.INFO)

lr_schedule_lambda = partial(lr_schedule_resnet, lr=0.1)
# input_frames, fixation_points = read_data(sys.argv[1])

frames_gap = 10
sequence_length = 9
batch_size = 8
target_size = (224, 224)
overwrite = False

pickle_in = open('two_parts_training.pickle', 'rb')
training_list = pickle.load(pickle_in)
pickle_in = open('two_parts_testing.pickle', 'rb')
testing_list = pickle.load(pickle_in)

print('Training %d, Testing %d' % (len(training_list), len(testing_list)))

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
mean = [103.939, 116.779, 123.68]
std = [1, 1, 1]

preprocess = partial(normalise_tensor, mean=mean, std=std)
training_generator = geetup_db.GeetupGenerator(training_list,
                                               batch_size=batch_size,
                                               target_size=target_size,
                                               gaussian_sigma=30.5,
                                               preprocessing_function=preprocess)
testing_generator = geetup_db.GeetupGenerator(testing_list,
                                              batch_size=batch_size,
                                              target_size=target_size,
                                              gaussian_sigma=30.5,
                                              preprocessing_function=preprocess,
                                              shuffle=True)

resnet = keras.applications.ResNet50(weights='imagenet')
for i, layer in enumerate(resnet.layers):
    layer.trainable = False
resnet_mid = keras.models.Model(inputs=resnet.input,
                                outputs=resnet.get_layer(
                                    'activation_22').output)
resnet = keras.models.Model(inputs=resnet.input,
                            outputs=resnet.get_layer('activation_10').output)
# outputs = [resnet.get_layer('activation_10').output,
#           resnet.get_layer('activation_22').output]
# resnet = K.function([resnet.input, K.learning_phase()], outputs)
model = class_net_fcn_2p_lstm((sequence_length, *target_size, 3), resnet,
                              resnet_mid)


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


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

last_checkpoint_logger = ModelCheckpoint('model_weights_last.h5', verbose=1,
                                         save_weights_only=True,
                                         save_best_only=False)

steps_per_epoch = 10000
validation_steps = 100
if not parallel_model == None:
    parallel_model.fit_generator(generator=training_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=testing_generator,
                                 validation_steps=validation_steps,
                                 use_multiprocessing=False,
                                 workers=8, epochs=45,
                                 callbacks=[
                                     LearningRateScheduler(lr_schedule_lambda),
                                     last_checkpoint_logger])
else:
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=testing_generator,
                        validation_steps=validation_steps,
                        use_multiprocessing=False,
                        workers=8, epochs=45,
                        callbacks=[
                            LearningRateScheduler(lr_schedule_lambda),
                            last_checkpoint_logger])
