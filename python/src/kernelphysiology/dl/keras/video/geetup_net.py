"""
Collection of networks for GEETUP.
"""

import keras
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed


def get_network(architecture_name, input_shape, frame_based, weights=None,
                all_frames=False):
    model = which_architecture(architecture_name, input_shape, frame_based,
                               all_frames)
    if weights is not None:
        model.load_weights(weights)
    return model


def _object_detection_net():
    resnet = keras.applications.ResNet50(weights='imagenet')
    for i, layer in enumerate(resnet.layers):
        layer.trainable = False
    return resnet


def which_architecture(architecture_name, input_shape, frame_based, all_frames):
    if architecture_name == 'draft':
        return draft_architecture(
            input_shape, None, None, frame_based, all_frames
        )
    elif architecture_name == 'draft_lm':
        resnet = _object_detection_net()
        resnet_mid = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_22').output
        )
        resnet = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_10').output
        )
        model = draft_architecture(
            input_shape, resnet, resnet_mid, frame_based, all_frames
        )
        return model
    elif architecture_name == 'draft_l':
        resnet = _object_detection_net()
        resnet = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_10').output
        )
        model = draft_architecture(
            input_shape, resnet, None, frame_based, all_frames
        )
        return model
    elif architecture_name == 'resnet':
        return resnet_architecture(input_shape, None, None, frame_based)
    elif architecture_name == 'resnet_lm':
        resnet = _object_detection_net()
        resnet_mid = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_22').output
        )
        resnet = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_10').output
        )
        model = resnet_architecture(
            input_shape, resnet, resnet_mid, frame_based
        )
        return model
    elif architecture_name == 'resnet_l':
        resnet = _object_detection_net()
        resnet = keras.models.Model(
            inputs=resnet.input,
            outputs=resnet.get_layer('activation_10').output
        )
        model = resnet_architecture(
            input_shape, resnet, None, frame_based
        )
        return model


def draft_architecture(input_shape, image_net=None, mid_layer=None,
                       frame_based=False, all_frames=False):
    c = 32
    input_img = Input(input_shape, name='input')
    if image_net is not None:
        c0 = TimeDistributed(image_net)(input_img)
    else:
        c0 = TimeDistributed(
            Conv2D(c, kernel_size=(3, 3), strides=4, padding='same')
        )(input_img)
    if frame_based:
        x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(c0)
    else:
        x = ConvLSTM2D(
            filters=c, kernel_size=(3, 3), padding='same',
            return_sequences=True
        )(c0)
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
    # TODO: make it according to frame based and time integration
    x = TimeDistributed(Conv2D(c, kernel_size=(3, 3), padding='same'))(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    #    c1 = TimeDistributed(ZeroPadding2D(padding=((1, 0), (1, 0))))(c1)
    x = Concatenate()([c1, x])

    x = TimeDistributed(UpSampling2D((4, 4)))(x)

    if all_frames:
        output = TimeDistributed(
            Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'),
            name='output')(x)
    else:
        output = ConvLSTM2D(1, kernel_size=(3, 3), padding='same',
                            activation='sigmoid', return_sequences=False,
                            name='output')(x)

    if all_frames:
        output = Reshape((-1, input_shape[1] * input_shape[2], 1))(output)
    else:
        output = Reshape((1, input_shape[1] * input_shape[2], 1))(output)
    model = keras.models.Model(input_img, output)
    return model


def resnet_layer(inputs, num_kernels, rf_size, conv_first=True,
                 activation='relu', batch_normalization=True, strides=(1, 1),
                 padding='same', frame_based=False):
    if frame_based:
        conv = TimeDistributed(Conv2D(
            num_kernels, kernel_size=rf_size, strides=strides, padding=padding))
    else:
        conv = ConvLSTM2D(num_kernels, rf_size, strides=strides,
                          padding=padding, return_sequences=True)
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_architecture(input_shape, image_net=None, mid_layer=None,
                        frame_based=False, depth=20):
    num_filters_in = 16
    input_img = Input(input_shape, name='input')

    if image_net is not None:
        x = TimeDistributed(image_net)(input_img)
    else:
        x = TimeDistributed(
            Conv2D(
                num_filters_in, kernel_size=(3, 3), strides=4, padding='same'
            )
        )(input_img)
    if frame_based:
        x = TimeDistributed(
            Conv2D(num_filters_in, kernel_size=(7, 7), padding='same')
        )(x)
    else:
        x = ConvLSTM2D(
            filters=num_filters_in, kernel_size=(7, 7),
            padding='same', return_sequences=True
        )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_res_blocks = int((depth - 2) / 9)
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_kernels=num_filters_in,
                             rf_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             frame_based=frame_based)
            y = resnet_layer(inputs=y,
                             num_kernels=num_filters_in,
                             rf_size=3,
                             conv_first=False,
                             frame_based=frame_based)
            y = resnet_layer(inputs=y,
                             num_kernels=num_filters_out,
                             rf_size=1,
                             conv_first=False,
                             frame_based=frame_based)
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_kernels=num_filters_out,
                                 rf_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 frame_based=frame_based)
            x = keras.layers.add([x, y])
            if strides == 2 and stage == 1:
                if mid_layer is not None:
                    x_mid = TimeDistributed(mid_layer)(input_img)
                    x = keras.layers.concatenate([x, x_mid])
                    x = resnet_layer(inputs=x,
                                     num_kernels=num_filters_out,
                                     rf_size=1,
                                     strides=1,
                                     activation=None,
                                     batch_normalization=False,
                                     frame_based=frame_based)

        num_filters_in = num_filters_out

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    if frame_based:
        x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(
            filters=16, kernel_size=(3, 3),
            padding='same', return_sequences=True
        )(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    if frame_based:
        x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), padding='same'))(x)
    else:
        x = ConvLSTM2D(
            filters=16, kernel_size=(3, 3),
            padding='same', return_sequences=True
        )(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = TimeDistributed(UpSampling2D((4, 4)))(x)
    output = TimeDistributed(
        Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'),
        name='output'
    )(x)

    output = Reshape((-1, input_shape[1] * input_shape[2], 1))(output)
    model = keras.models.Model(input_img, output)
    return model
