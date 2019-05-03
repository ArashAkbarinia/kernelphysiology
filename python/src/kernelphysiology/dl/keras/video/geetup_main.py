from keras.models import Sequential
from keras.layers import Reshape, Input, UpSampling2D, Concatenate
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import keras
from keras.callbacks import LearningRateScheduler
from functools import partial

from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import MaxPooling3D, MaxPooling2D
import glob
from skimage import io, color, transform
import sys

from kernelphysiology.filterfactory import gaussian


def lr_schedule_resnet(epoch, lr):
    new_lr = lr * (0.1 ** (epoch // (90 / 3)))
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
    print(tensor.shape)
    # normalising the channels
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] - mean[i]) / std[i]
    return tensor


def read_data(dataset_dir):
    n_frames = 15
    n_samples = len(glob.glob(dataset_dir + '/*/'))
    rows = 40
    cols = 40
    chns = 1
    input_frames = np.zeros((n_samples * n_frames, n_frames, rows, cols, chns),
                            dtype=np.float)
    fixation_points = np.zeros((n_samples * n_frames, n_frames, rows, cols, 1),
                               dtype=np.float)
    g_fix = gaussian.gaussian_kernel2(1.5, 1.5)
    for s, video_dir in enumerate(glob.glob(dataset_dir + '/*/')):
        fixation_array = np.loadtxt(
            dataset_dir + 'SUBSAMP_EYETR_' + str(s + 1) + '.txt')
        for i in range(n_frames * 2 - 1):
            current_img = io.imread(video_dir + '/frames' + str(i + 1) + '.jpg')
            current_img = color.rgb2gray(current_img).astype('float') / 255
            org_rows = current_img.shape[0]
            org_cols = current_img.shape[1]
            current_img = transform.resize(current_img, (rows, cols))

            current_fixation = np.zeros((rows, cols, 1))
            if fixation_array[i, 1] > 0 and fixation_array[i, 0] > 0:
                fpr = int(fixation_array[i, 1] * (rows / org_rows))
                fpc = int(fixation_array[i, 0] * (cols / org_cols))

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

                current_fixation[sr:er, sc:ec, 0] = g_fix[gsr:ger,
                                                    gsc:gec] / g_fix[gsr:ger,
                                                               gsc:gec].max()

            for j in range(n_frames):
                current_ind = s * n_frames + j
                if i - j < n_frames:
                    input_frames[current_ind, i - j, :, :, 0] = current_img
                    fixation_points[current_ind, i - j,] = current_fixation

    input_frames = normalise_tensor(input_frames, [0.5], [0.25])
    return input_frames, fixation_points


def class_net_fcn_2p_lstm(input_shape):
    c = 16
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(input_img)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c2 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    c3 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                    return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = Concatenate()([c2, x])
    x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = Concatenate()([c1, x])
    # x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    #     x = TimeDistributed(Conv2D(3, kernel_size=(3, 3), padding='same'))(x)
    #     x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #     x = TimeDistributed(Conv2D(3, kernel_size=(3, 3), padding='same'))(x)
    #     x = TimeDistributed(UpSampling2D((2, 2)))(x)

    output = TimeDistributed(
        Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'),
        name='output')(x)

    model = keras.models.Model(input_img, output)
    #     loss = categorical_crossentropy_3d_w(10, class_dim=-1)
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model


lr_schedule_lambda = partial(lr_schedule_resnet, lr=0.1)
input_frames, fixation_points = read_data(sys.argv[1])

seq = class_net_fcn_2p_lstm(input_frames.shape[1:])
seq.fit(input_frames[:1000],
        fixation_points[:1000,],
#         np.reshape(fixation_points[:1000,], (-1, n_frames, rows * cols, 1)),
        batch_size=10, epochs=5,
        validation_split=0.25,
        callbacks=[LearningRateScheduler(lr_schedule_lambda)])