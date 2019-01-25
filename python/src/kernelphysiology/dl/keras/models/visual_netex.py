'''
A deep neuronal network inspired by visual cortex of human brain.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import math
import numpy as np

import keras
from keras.layers import Input
from keras import layers
from keras.layers.core import Lambda
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate, Average
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = ''
WEIGHTS_PATH_NO_TOP = ''

RGC_POPULATION = 32
LGN_POPULATION = 32

# TODO: As much as 95% of input in the LGN comes from the visual cortex


def koniocellular(s_cone, rf_size=(3, 3), num_konio=math.ceil(0.8 * LGN_POPULATION),
                  kernel_initializer='he_normal'):
    # little known about them, a lot of speculation
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = s_cone
    kernel_constraint = None
    name = 'koniocellular_cells'
    activation_type = 'relu'

    x = Conv2D(num_konio, rf_size, strides=(1, 1), kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    # output of koniocellular contains 6 layers
    name = 'lgn_k'
    x = Conv2D(6, (1, 1), strides=(1, 1), name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    return x


def parvocellular(lm_cones, rf_size=(3, 3), num_midget=math.ceil(0.8 * RGC_POPULATION),
                  num_parvo=math.ceil(0.8 * LGN_POPULATION), kernel_initializer='he_normal'):
    # about 80% of RGCs (at least 20 types) are midget cells
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = lm_cones
    kernel_constraint = None
    name = 'midget_cells'
    activation_type = 'relu'

    x = Conv2D(num_midget, rf_size, strides=(1, 1), kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    # parvocellular cells in lgn
    name = 'parvocellular_cells'
    x = Conv2D(num_parvo, rf_size, strides=(1, 1), kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    # output of parvocellular contains 4 layers
    name = 'lgn_p'
    x = Conv2D(4, (1, 1), strides=(1, 1), name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    return x


def magnocellular(lms_cones, rf_size=(7, 7), num_parasol=math.ceil(0.1 * RGC_POPULATION),
                  num_magno=math.ceil(0.1 * LGN_POPULATION), kernel_initializer='he_normal'):
    # about 10% of RGCs (at least 20 types) are parasol cells
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = lms_cones
    kernel_constraint = keras.constraints.NonNeg()
    name = 'parasol_cells'
    activation_type = 'relu'

    x = Conv2D(num_parasol, rf_size, strides=(1, 1), kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    # NOTE: given weights are none negative, activatoin might not be necessary
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    # magnocellular cells in lgn
    name = 'magnocellular_cells'
    kernel_constraint = None
    x = Conv2D(num_magno, rf_size, strides=(1, 1), kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    # output of magnocellular contains 2 layers
    name = 'lgn_m'
    x = Conv2D(2, (1, 1), strides=(1, 1), name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    return x


def v1_layer(lgn, name, rf_size=(3, 3), strides=(1, 1), kernel_constraint=None,
             activation_type='relu', num_kernels=16, kernel_initializer='he_normal'):
    # little known about them, a lot of speculation
    x = lgn
    name = 'v1_' + name

    x = rf_pattern(x, num_kernels, rf_size, name, strides=strides, activation_type=activation_type)

    return x


def v2_layer(v1, lgn_output, name, rf_size=(3, 3), strides=(1, 1), kernel_constraint=None,
             activation_type='relu', num_kernels=16, kernel_initializer='he_normal'):
    # little known about them, a lot of speculation
    x_lgn = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='v2_lgn_'+name)(lgn_output)
    x_v1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='v2_v1_'+name)(v1)
    x = Concatenate()([x_lgn, x_v1])
    name = 'v2_' + name

    x = rf_pattern(x, num_kernels, rf_size, name, strides=strides, activation_type=activation_type)

    return x


def v4_layer(v2, lgn_output, rf_size=(3, 3), strides=(1, 1), kernel_constraint=None,
             activation_type='relu', num_kernels=16, kernel_initializer='he_normal'):
    # little known about them, a lot of speculation
    x_lgn = MaxPooling2D((9, 9), strides=(4, 4), padding='same', name='v4_lgn')(lgn_output)
    x_v2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='v4_v2')(v2)
    x = Concatenate()([x_lgn, x_v2])
    name = 'v4'

    x = rf_pattern(x, num_kernels, rf_size, name, strides=strides, activation_type=activation_type)

    return x


def rf_pattern(x, num_kernels, rf_size, name, strides=(1, 1), activation_type='relu',
               kernel_constraint=None, kernel_initializer='he_normal'):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = Conv2D(num_kernels, rf_size, strides=strides, kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer, padding='same', name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    name = name + '_depth'
    x = Conv2D(4, (1, 1), strides=(1, 1), name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    return x


def VisualNetex(include_top=True, weights=None,
                input_tensor=None, input_shape=None,
                pooling=None,
                classes=1000):
    # TODO: add CIFAR-100, and STL
    if not (weights in {'imagenet', 'cifar10', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         '`imagenet`, `cifar10`, '
                         '(pre-training on each dataset), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet':
        if include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
        default_size = 224
    elif weights == 'cifar10':
        if include_top and classes != 10:
            raise ValueError('If using `weights` as cifar10 with `include_top`'
                             ' as true, `classes` should be 10')
        default_size = 32

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x_lms = img_input
    #  FIXME: is it okay that input image has been shifted to mean 0?
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        x_lm = Lambda(lambda x : x[:, :, :, 0:2], name='l_m_cones')(img_input)
        x_s = Lambda(lambda x : x[:, :, :, 2:3], name='s_cones')(img_input)
    else:
        bn_axis = 1
        x_lm = Lambda(lambda x : x[0:2, :, :, :], name='l_m_cones')(img_input)
        x_s = Lambda(lambda x : x[2:3, :, :, :], name='s_cones')(img_input)

    p_stream = parvocellular(x_lm)
    m_stream = magnocellular(x_lms)
    k_stream = koniocellular(x_s)

    k_v1_layers = []
    m_v1_layers = []
    p_v1_layers = []
    lgn_output = Concatenate()([p_stream, m_stream, k_stream])
    for l in range(6):
        k_l = Lambda(lambda x : x[:, :, :, l:l+1], name='k' + str(l))(k_stream)
        if l < 2:
            m_l = Lambda(lambda x : x[:, :, :, l:l+1], name='m' + str(l))(m_stream)
            m_v1_layers.append(v1_layer(m_l, 'm' + str(l)))
            k_pm = Concatenate(name='k' + str(l) + 'm')([k_l, m_l])
        else:
            p_l = Lambda(lambda x : x[:, :, :, l-2:l-1], name='p' + str(l-2))(p_stream)
            p_v1_layers.append(v1_layer(p_l, 'p' + str(l-2)))
            k_pm = Concatenate(name='k' + str(l) + 'p')([k_l, p_l])
        k_v1_layers.append(v1_layer(k_pm, 'k' + str(l)))

    # v1 outputs
    k_v1_out = Concatenate()([*k_v1_layers])
    m_v1_out = Concatenate()([*m_v1_layers])
    p_v1_out = Concatenate()([*p_v1_layers])

    # v2 outputs
    k_v2_out = v2_layer(k_v1_out, lgn_output, 'k')
    m_v2_out = v2_layer(m_v1_out, lgn_output, 'm')
    p_v2_out = v2_layer(p_v1_out, lgn_output, 'p')

    v2 = Concatenate(name='v2_kmp')([k_v2_out, m_v2_out, p_v2_out])

    # v4 outputs
    v4 = v4_layer(v2, lgn_output)
    x = v4

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='visual_netex')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    model.summary()
    keras.utils.vis_utils.plot_model(model, to_file='/home/arash/Desktop/visual_netx.png')
    return model