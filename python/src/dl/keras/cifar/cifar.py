'''
Utilities common to CIFAR10 and CIFAR100 datasets.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons
import os
import sys

from six.moves import cPickle
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint

import dl.keras.contrast_net as cnet


class CifarConfs:
    python_root = commons.python_root

    batch_size = 64
    num_classes = None
    epochs = 100
    log_period = round(epochs / 4)
    data_augmentation = False
    area1_nlayers = 1
    area1_batchnormalise = True
    area1_activation = True
    area1_reduction = False
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
    
    def __init__(self, args):
        self.num_classes = args.num_classes
        
        self.model_name = 'keras_cifar%d_area_%s_' % (self.num_classes, args.experiment_name)
        self.save_dir = os.path.join(self.python_root, 'data/nets/cifar/cifar%d/' % self.num_classes)
        self.dog_path = os.path.join(self.save_dir, 'dog.h5')

        self.area1_nlayers = args.area1_nlayers

        self.area1_batchnormalise = args.area1_batchnormalise
        if self.area1_batchnormalise:
            self.model_name += 'bnr_'
        self.area1_activation = args.area1_activation
        if self.area1_activation:
            self.model_name += 'act_'
        self.area1_reduction = args.area1_reduction
        if self.area1_activation:
            self.model_name += 'red_'
        self.add_dog = args.add_dog
        if self.add_dog:
            self.model_name += 'dog_'
        self.multi_gpus = args.multi_gpus


def start_training(confs):
    print('x_train shape:', confs.x_train.shape)
    print(confs.x_train.shape[0], 'train samples')
    print(confs.x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    confs.y_train = keras.utils.to_categorical(confs.y_train, confs.num_classes)
    confs.y_test = keras.utils.to_categorical(confs.y_test, confs.num_classes)
    

    print('Processing with %d layers in area 1' % confs.area1_nlayers)
    confs.model_name += str(confs.area1_nlayers)
    confs.log_dir = os.path.join(confs.save_dir, confs.model_name)
    if not os.path.isdir(confs.log_dir):
        os.mkdir(confs.log_dir)
    csv_logger = CSVLogger(os.path.join(confs.log_dir, 'log.csv'), append=False, separator=';')
    check_points = ModelCheckpoint(os.path.join(confs.log_dir, 'weights.{epoch:05d}.h5'), period=confs.log_period)
    confs.callbacks = [check_points, csv_logger]

    confs.area1_nlayers = int(confs.area1_nlayers)
    
    model = cnet.build_classifier_model(confs=confs)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    if confs.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        confs.model = model
        confs.parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            confs.model = model
        parallel_model = multi_gpu_model(confs.model, gpus=confs.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        confs.parallel_model = parallel_model

    confs.x_train = cnet.preprocess_input(confs.x_train)
    confs.x_test = cnet.preprocess_input(confs.x_test)
    
    confs = cnet.train_model(confs)

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
