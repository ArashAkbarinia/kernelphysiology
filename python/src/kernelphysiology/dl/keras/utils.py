'''
Common utility funcions for Keras.
'''


import math
import keras


def get_conv2ds(model, topn=math.inf):
    conv2d_inds = []
    for i in range(0, len(model.layers)):
        if type(model.layers[i]) is keras.layers.convolutional.Conv2D:
            conv2d_inds.append(i)
    return conv2d_inds


def set_area_trainable_false(model, num_areas):
    current_area = 1
    for i in range(0, len(model.layers)):
        if type(model.layers[i]) is keras.layers.pooling.MaxPooling2D:
            if num_areas == current_area:
                break
            current_area += 1
        else:
            model.layers[i].trainable = False
    return model
