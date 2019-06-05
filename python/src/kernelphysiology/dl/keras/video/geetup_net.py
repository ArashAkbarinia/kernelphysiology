"""
Collection of networks for GEETUP.
"""

import keras
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


def class_net_fcn_2p_lstm(input_shape, image_net=None, mid_layer=None,
                          frame_based=False):
    c = 32
    input_img = Input(input_shape, name='input')
    if image_net is not None:
        c0 = TimeDistributed(image_net)(input_img)
    else:
        c0 = TimeDistributed(
            Conv2D(c, kernel_size=(3, 3), strides=4, padding='same'))(input_img)
    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(c0)
    else:
        x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(c0)
    x = BatchNormalization()(x)
    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(x)
    x = BatchNormalization()(x)
    if frame_based:
        c1 = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        c1 = ConvLSTM2D(filters=c, kernel_size=(3, 3), padding='same',
                        return_sequences=True)(x)
    c1 = BatchNormalization()(c1)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    #    x = TimeDistributed(ZeroPadding2D(padding=((1, 0), (1, 0))))(x)

    if mid_layer is not None:
        x_mid = TimeDistributed(mid_layer)(input_img)
        x = Concatenate()([x_mid, x])

    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(x)
    x = BatchNormalization()(x)
    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(x)
    x = BatchNormalization()(x)
    if frame_based:
        c2 = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        c2 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                        return_sequences=True)(x)
    c2 = BatchNormalization()(c2)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)

    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(x)
    x = BatchNormalization()(x)
    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                       return_sequences=True)(x)
    x = BatchNormalization()(x)
    if frame_based:
        c3 = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    else:
        c3 = ConvLSTM2D(filters=2 * c, kernel_size=(3, 3), padding='same',
                        return_sequences=True)(x)
    c3 = BatchNormalization()(c3)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = Concatenate()([c2, x])
    x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #    c1 = TimeDistributed(ZeroPadding2D(padding=((1, 0), (1, 0))))(c1)
    x = Concatenate()([c1, x])

    x = TimeDistributed(UpSampling2D((4, 4)))(x)

    output = TimeDistributed(
        Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'),
        name='output')(x)

    output = Reshape((-1, input_shape[1] * input_shape[2], 1))(output)
    model = keras.models.Model(input_img, output)
    return model
