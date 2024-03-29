"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model

from kernelphysiology.dl.keras.models import blocks


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 kernel_initializer='he_normal',
                 pyramid_levels=1,
                 name_base=None):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = blocks.pyramid_block(inputs, num_filters, kernel_size, name_base,
                             strides=strides,
                             activation_type=activation,
                             batch_normalization=batch_normalization,
                             conv_first=conv_first,
                             kernel_initializer=kernel_initializer,
                             num_levels=pyramid_levels)
    return x


def resnet_v1(input_shape, depth, num_classes=10, kernel_initializer='he_normal',
              pyramid_levels=1, num_kernels=16, output_types=[]):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    if kernel_initializer is None:
        kernel_initializer='he_normal'
    # Start model definition.
    # FIXME: only works with 16
    num_filters = num_kernels
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, kernel_initializer=kernel_initializer, pyramid_levels=pyramid_levels)

    other_outputs = []
    if 'natural_vs_manmade' in output_types:
        x_nvm = AveragePooling2D(pool_size=8)(x)
        y_nvm = Flatten()(x_nvm)
        natural_vs_manmade_outout = Dense(2, activation='softmax',
                                          kernel_initializer=kernel_initializer,
                                          name='natural_vs_manmade')(y_nvm)
        other_outputs.append(natural_vs_manmade_outout)
    if 'illuminant' in output_types:
        x_ilum = AveragePooling2D(pool_size=8)(x)
        y_ilum = Flatten()(x_ilum)
        illuminant_outout = Dense(3, activation='softmax',
                                  kernel_initializer=kernel_initializer,
                                  name='illuminant')(y_ilum)
        other_outputs.append(illuminant_outout)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             kernel_initializer=kernel_initializer,
                             pyramid_levels=pyramid_levels)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             kernel_initializer=kernel_initializer,
                             pyramid_levels=pyramid_levels)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 kernel_initializer=kernel_initializer,
                                 pyramid_levels=pyramid_levels)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    all_classes_output = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer=kernel_initializer,
                    name='all_classes')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[all_classes_output, *other_outputs])
    return model


def resnet_v2(input_shape, depth, num_classes=10, kernel_initializer='he_normal',
              pyramid_levels=1, num_kernels=16):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    if kernel_initializer is None:
        kernel_initializer='he_normal'
    # Start model definition.
    num_filters_in = num_kernels
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                     kernel_initializer=kernel_initializer,
                     pyramid_levels=pyramid_levels)

    # Instantiate the stack of residual units
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
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             kernel_initializer=kernel_initializer,
                             pyramid_levels=pyramid_levels)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             kernel_initializer=kernel_initializer,
                             pyramid_levels=pyramid_levels)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             kernel_initializer=kernel_initializer,
                             pyramid_levels=pyramid_levels)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 kernel_initializer=kernel_initializer,
                                 pyramid_levels=pyramid_levels)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer=kernel_initializer)(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
