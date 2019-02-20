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
import string

import keras
from keras.layers import Input
from keras import layers
from keras.layers.core import Lambda
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D, DepthwiseConv2D
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


class LayerContainer:
    num_kernels = 0
    rf_size = (3, 3)
    strides = (1, 1)
    activation_type = 'relu'
    kernel_constraint = None
    kernel_initializer = None
    kernel_function = Conv2D
    name = None

    def __init__(self, name, kernel_initializer='he_normal'):
        self.name = name
        self.kernel_initializer = kernel_initializer


def koniocellular(s_cone, rf_size=(3, 3), num_konio=math.ceil(0.8 * LGN_POPULATION),
                  kernel_initializer='he_normal'):
    # little known about them, a lot of speculation

    l_konio = LayerContainer('koniocellular_cells', kernel_initializer)
    l_konio.rf_size = rf_size
    l_konio.num_kernels = num_konio
    x = conv_norm_rect(s_cone, l_konio)

    # output of koniocellular contains 6 layers
    l_lgn_k = LayerContainer('lgn_k', kernel_initializer)
    l_lgn_k.rf_size = (1, 1)
    l_lgn_k.num_kernels = 6
    x = conv_norm_rect(x, l_lgn_k)

    return x


def parvocellular(lm_cones, rf_size=(3, 3), num_midget=math.ceil(0.8 * RGC_POPULATION),
                  num_parvo=math.ceil(0.8 * LGN_POPULATION), kernel_initializer='he_normal'):
    # about 80% of RGCs (at least 20 types) are midget cells

    l_midget = LayerContainer('midget_cells', kernel_initializer)
    l_midget.rf_size = rf_size
    l_midget.num_kernels = num_midget
    x = conv_norm_rect(lm_cones, l_midget)

    # parvocellular cells in lgn
    l_parvo = LayerContainer('parvocellular_cells', kernel_initializer)
    l_parvo.num_kernels = num_parvo
    x = conv_norm_rect(x, l_parvo)

    # output of parvocellular contains 4 layers
    l_lgn_p = LayerContainer('lgn_p', kernel_initializer)
    l_lgn_p.rf_size = (1, 1)
    l_lgn_p.num_kernels = 4
    x = conv_norm_rect(x, l_lgn_p)

    return x


def magnocellular(lms_cones, rf_size=(7, 7), num_parasol=math.ceil(0.1 * RGC_POPULATION),
                  num_magno=math.ceil(0.1 * LGN_POPULATION), kernel_initializer='he_normal'):
    # about 10% of RGCs (at least 20 types) are parasol cells

    l_parasol = LayerContainer('parasol_cells', kernel_initializer)
    l_parasol.rf_size = rf_size
    l_parasol.num_kernels = num_parasol
    l_parasol.kernel_constraint = keras.constraints.NonNeg()
    x = conv_norm_rect(lms_cones, l_parasol)

    # magnocellular cells in lgn
    l_magno = LayerContainer('magnocellular_cells', kernel_initializer)
    l_magno.rf_size = rf_size
    l_magno.num_kernels = num_magno
    x = conv_norm_rect(x, l_magno)

    # output of magnocellular contains 2 layers
    l_lgn_m = LayerContainer('lgn_m', kernel_initializer)
    l_lgn_m.rf_size = (1, 1)
    l_lgn_m.num_kernels = 2
    x = conv_norm_rect(x, l_lgn_m)

    return x


def visual_areas(x, area_number, num_neurons_low=4, rf_size_low=3):
    # The superficial layer 1 has very few neurons but many axons, dendrites
    # and synapses
    l1 = LayerContainer('l01_a%02d' % (area_number))
    l1.num_kernels = num_neurons_low
    l1.rf_size = (rf_size_low, rf_size_low)
#    l1.kernel_function = DepthwiseConv2D
    x1 = conv_norm_rect(x[0], l1)

    # Layers 2 and 3 consists of a dense array of cell bodies and many local
    # dendritic interconnections. These layers appear to receive a direct input
    # from the intercalated layers of the lateral geniculate as well
    # (Fitzpatrick et al., 1983; Hendry and Yoshioka, 1994), and the outputs
    # from layers 2 and 3 are sent to other cortical areas
    l2 = LayerContainer('l02_a%02d' % (area_number))
    l2.num_kernels = num_neurons_low * 2
    if x[1] is not None:
        x2i = Concatenate(name='l02i_a%02d' % (area_number))([x1, x[1]])
    else:
        x2i = x1
    x2 = conv_norm_rect(x2i, l2)

    l3 = LayerContainer('l03_a%02d' % (area_number))
    l3.num_kernels = num_neurons_low * 2
    if x[2] is not None:
        x3i = Concatenate(name='l03i_a%02d' % (area_number))([x2, x[2]])
    else:
        x3i = x1
    x3 = conv_norm_rect(x3i, l3)

    # Layer 4 has been subdivided into several parts.It contains small,
    # irregularily shaped nerve cells
    if x[3] is not None:
        x4i = Concatenate(name='l04i_a%02d' % (area_number))([x3, x[3]])
    else:
        x4i = x1
    x4s = []
    size_var = [0, 2]
    k = 0
    for i in size_var:
        for j in size_var:
            l4 = LayerContainer('l04%s_a%02d' % (string.ascii_lowercase[k], area_number))
            l4.num_kernels = num_neurons_low
            l4.rf_size = (rf_size_low + i, rf_size_low + j)
            x4s.append(conv_norm_rect(x4i, l4))
            k += 1
    x4 = Concatenate(name='l04s_a%02d' % (area_number))(x4s)

    # Layer 5 contains relatively few cell bodies compared to the surrounding
    # layers.
    l5 = LayerContainer('l05_a%02d' % (area_number))
    l5.num_kernels = num_neurons_low * 2
    if x[4] is not None:
        x5i = Concatenate(name='l05i_a%02d' % (area_number))([x4, x[4]])
    else:
        x5i = x1
    x5 = conv_norm_rect(x5i, l5)

    # Layer 6 is dense with cells and sends a large output back to the lateral
    # geniculate nucleus (Toyoma, 1969).
    l6 = LayerContainer('l06_a%02d' % (area_number))
    l6.num_kernels = num_neurons_low * 2
    if x[5] is not None:
        x6i = Concatenate(name='l06i_a%02d' % (area_number))([x5, x[5]])
    else:
        x6i = x1
    x6 = conv_norm_rect(x6i, l6)

    x_all = Concatenate(name='column_a%02d' % (area_number))([x1, x2, x3, x4, x5, x6])
    l_all = LayerContainer('area%02d' % (area_number))
    l_all.num_kernels = num_neurons_low * 2
    l_all.rf_size = (1, 1)
    x = conv_norm_rect(x_all, l_all)
    return (x, x1, x2, x3, x4, x5, x6)


def conv_norm_rect(x, layer_info):
    num_kernels = layer_info.num_kernels
    rf_size = layer_info.rf_size
    strides = layer_info.strides
    kernel_constraint = layer_info.kernel_constraint
    kernel_initializer = layer_info.kernel_initializer
    name = layer_info.name
    activation_type = layer_info.activation_type

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = layer_info.kernel_function(num_kernels, rf_size, strides=strides,
                                   kernel_constraint=kernel_constraint,
                                   kernel_initializer=kernel_initializer,
                                   padding='same', name=name+'_conv')(x)
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
    elif weights == 'cifar10':
        if include_top and classes != 10:
            raise ValueError('If using `weights` as cifar10 with `include_top`'
                             ' as true, `classes` should be 10')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
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

    lgn_output = Concatenate()([p_stream, m_stream, k_stream])
    stream = [lgn_output, None, None, lgn_output, None, None]
    a0s = a1s = a2s = a3s = a4s = a5s = a6s = []
    for area_number in [1, 2, 4]:
        stream = visual_areas(stream, area_number, num_neurons_low=4, rf_size_low=3)
#        k_l = Lambda(lambda x : x[:, :, :, l:l+1], name='k' + str(l))(k_stream)
#        if l < 2:
#            m_l = Lambda(lambda x : x[:, :, :, l:l+1], name='m' + str(l))(m_stream)
#            m_v1_layers.append(v1_layer(m_l, 'm' + str(l)))
#            k_pm = Concatenate(name='k' + str(l) + 'm')([k_l, m_l])
#        else:
#            p_l = Lambda(lambda x : x[:, :, :, l-2:l-1], name='p' + str(l-2))(p_stream)
#            p_v1_layers.append(v1_layer(p_l, 'p' + str(l-2)))
#            k_pm = Concatenate(name='k' + str(l) + 'p')([k_l, p_l])
#        k_v1_layers.append(v1_layer(k_pm, 'k' + str(l)))

    x = stream[0]

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)
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

    keras.utils.vis_utils.plot_model(model, to_file='/home/arash/Software/repositories/visual_netx.png')
    return model