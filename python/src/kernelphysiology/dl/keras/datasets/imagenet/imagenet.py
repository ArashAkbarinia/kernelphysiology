'''
Reading the IMAGENET dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kernelphysiology.dl.keras.utils.image import ImageDataGenerator


def train_generator(dirname, batch_size=32, target_size=(224, 224), preprocessing_function=None,
                    crop_type='random', shuffle=True,
                    horizontal_flip=False, vertical_flip=False,
                    zoom_range=0.0, width_shift_range=0.0, height_shift_range=0.0):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function,
                                       horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                       zoom_range=zoom_range,
                                       width_shift_range=width_shift_range, height_shift_range=height_shift_range)

    train_generator = train_datagen.flow_from_directory(dirname, target_size=target_size, batch_size=batch_size,
                                                        crop_type=crop_type, shuffle=shuffle)

    return train_generator


def validation_generator(dirname, batch_size=32, target_size=(224, 224), preprocessing_function=None, crop_type='centre'):
    validarion_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    validatoin_generator = validarion_datagen.flow_from_directory(dirname, target_size=target_size, batch_size=batch_size,
                                                                  crop_type=crop_type, shuffle=False)

    return validatoin_generator