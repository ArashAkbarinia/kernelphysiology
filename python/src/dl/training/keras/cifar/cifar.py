'''
Utilities common to CIFAR10 and CIFAR100 datasets.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys


# finding the root of the project
current_path = os.getcwd()
python_root = 'kernelphysiology/python/'
project_dir = current_path.split(python_root, 1)[0]
python_root = os.path.join(project_dir, python_root)
sys.path += [os.path.join(python_root, 'src/')]


from six.moves import cPickle
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model
    
    
from filterfactory.gaussian import gauss


class CifarConfs:
    project_root = python_root
    
    batch_size = 32
    num_classes = None
    epochs = 100
    log_period = round(epochs / 4)
    data_augmentation = False
    area1_nlayers = 1
    add_dog = True
    multi_gpus = None
    
    model_name = None
    save_dir = None
    log_dir = None
    dog_path = None
    
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    
    def __init__(self, num_classes, args):
        self.num_classes = num_classes
        
        self.model_name = 'keras_cifar%d_area_' % self.num_classes
        self.save_dir = os.path.join(self.project_root, 'data/nets/cifar/cifar%d/' % self.num_classes)
        self.dog_path = os.path.join(self.save_dir, 'dog.h5')
        
        argc = len(args)
        if argc > 1:
            self.area1_nlayers = args[1]
        if argc > 2:
            self.add_dog = int(args[2]) == 1
        if argc > 3:
            self.multi_gpus = int(args[3])

def start_training(confs):
    print('x_train shape:', confs.x_train.shape)
    print(confs.x_train.shape[0], 'train samples')
    print(confs.x_test.shape[0], 'test samples')
    
    # Convert class vectors to binary class matrices.
    confs.y_train = keras.utils.to_categorical(confs.y_train, confs.num_classes)
    confs.y_test = keras.utils.to_categorical(confs.y_test, confs.num_classes)
    
    
    print('Processing with %s layers in area 1' % confs.area1_nlayers)
    confs.model_name += confs.area1_nlayers
    confs.log_dir = os.path.join(confs.save_dir, confs.model_name)
    if not os.path.isdir(confs.log_dir):
        os.mkdir(confs.log_dir)
    csv_logger = CSVLogger(os.path.join(confs.log_dir, 'log.csv'), append=False, separator=';')
    check_points = ModelCheckpoint(os.path.join(confs.log_dir, 'weights.{epoch:05d}.h5'), period=confs.log_period)
    confs.callbacks = [check_points, csv_logger]
    
    confs.area1_nlayers = int(confs.area1_nlayers)
    
    model = generate_model(confs=confs)
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    if confs.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        confs.model = model
        confs.parallel_model = None
    else:
        with tf.device('/cpu:0'):
            confs.model = model
        parallel_model = multi_gpu_model(confs.model, gpus=confs.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        confs.parallel_model = parallel_model
    
    confs.x_train = confs.x_train.astype('float32')
    confs.x_test = confs.x_test.astype('float32')
    confs.x_train /= 255
    confs.x_test /= 255
    
    confs = train_model(confs)
    
    
    # Score trained model.
    scores = confs.model.evaluate(confs.x_test, confs.y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def train_model(confs):
    x_train = confs.x_train
    y_train = confs.y_train
    x_test = confs.x_test
    y_test = confs.y_test
    callbacks = confs.callbacks
    batch_size = confs.batch_size
    epochs = confs.epochs
    
    if not confs.data_augmentation:
        print('Not using data augmentation.')
        if not confs.parallel_model == None:
            batch_size *= confs.multi_gpus
            confs.parallel_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)
        else:
            confs.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        if not confs.parallel_model == None:
            batch_size *= confs.multi_gpus
            confs.parallel_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                               epochs=epochs,
                                               validation_data=(x_test, y_test),
                                               workers=4)
        else:
            confs.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      workers=4)
    
    # Save model and weights
    if not os.path.isdir(confs.save_dir):
        os.makedirs(confs.save_dir)
    model_name = confs.model_name + '.h5'
    model_path = os.path.join(confs.save_dir, model_name)
    confs.model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    return confs


def dog_layer(confs, nkernels, kernel_size, nchannels=3):
    dogs = np.zeros((kernel_size, kernel_size, nchannels, nkernels))
    for i in range(0, nkernels):
        for j in range(0, nchannels):
            sigma1 = np.random.uniform(0, 1)
            g1 = gauss.gkern(kernel_size, sigma1)
            sigma2 = np.random.uniform(0, 1)
            g2 = gauss.gkern(kernel_size, sigma2)
            dg = -g1 + g2
            dogs[:, :, j, i] = dg
    return dogs


def generate_model(confs):
    area1_nlayers = confs.area1_nlayers
    
    model = Sequential()
    
    kernel_size = 3
    nkernels = 64
    model.add(Conv2D(nkernels, (kernel_size, kernel_size), padding='same', input_shape=confs.x_train.shape[1:]))
    
    if confs.add_dog:
        if confs.dog_path == None or not os.path.exists(confs.dog_path):
            weights = model.layers[0].get_weights()
            dogs = dog_layer(confs, nkernels, kernel_size, nchannels=np.size(weights[0], 2))
            weights[0] = dogs
            
            model.layers[0].set_weights(weights)
            model.layers[0].trainable = False
            model.layers[0].name = 'dog'
            
            model.add(Conv2D(64, (3, 3), padding='same', name='afterdog'))
            
            model.save(confs.dog_path)
        else:
            print('Reading the DoG file')
            model = load_model(confs.dog_path)
    
    model.add(Activation('relu'))
    
    if area1_nlayers == 2:
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 40:
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 41:
        model.add(Conv2D(20, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(12, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 42:
        model.add(Conv2D(12, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(20, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 50:
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 51:
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
    elif area1_nlayers == 52:
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(confs.num_classes))
    model.add(Activation('softmax'))
    
    return model
