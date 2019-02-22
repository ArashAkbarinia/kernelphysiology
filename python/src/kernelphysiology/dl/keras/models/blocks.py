'''
Common blocks for various architecture.
'''


from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras import backend as K


def pyramid_block(x, num_filters, kernel_size, name_base, padding='same', num_levels=4):
    filters_level = round(num_filters / num_levels)
    f_0 = num_filters - (filters_level * num_levels)

    xis = []
    for i in range(num_levels):
        num_kernels = filters_level
        # to compensate for rounding
        if i == 0:
            num_kernels += f_0
        rf_size = kernel_size + (i * 2)
        name = '%s_%02d' % (name_base, i)
        xi = conv_norm_rect(x, num_kernels, rf_size, name=name, padding=padding)
        xis.append(xi)
    x = layers.Concatenate()(xis)
    return 


def conv_norm_rect(x, num_kernels, rf_size, name, strides=(1, 1),
                   kernel_constraint=None, kernel_initializer='he_normal',
                   padding='same', activation_type='relu'):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = Conv2D(num_kernels, rf_size, strides=strides,
               kernel_constraint=kernel_constraint,
               kernel_initializer=kernel_initializer,
               padding=padding, name=name+'_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name+'_bn')(x)
    x = Activation(activation_type, name=name+'_'+activation_type)(x)

    return x