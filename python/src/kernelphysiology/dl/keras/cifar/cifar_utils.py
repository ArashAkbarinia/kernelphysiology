'''
Utilities common to CIFAR10 and CIFAR100 datasets.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import cPickle
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint

import kernelphysiology.dl.keras.contrast_net as cnet


def start_training(args):
    print('x_train shape:', args.x_train.shape)
    print(args.x_train.shape[0], 'train samples')
    print(args.x_test.shape[0], 'test samples')

    print('Processing with %d layers in area 1' % args.area1_nlayers)
    args.model_name += str(args.area1_nlayers)
    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    check_points = ModelCheckpoint(os.path.join(args.log_dir, 'weights.{epoch:05d}.h5'), period=args.log_period)
    args.callbacks = [check_points, csv_logger]

    args.area1_nlayers = int(args.area1_nlayers)
    
    model = cnet.build_classifier_model(confs=args)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    if args.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        args.model = model
        args.parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            args.model = model
        parallel_model = multi_gpu_model(args.model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        args.parallel_model = parallel_model

    args.x_train = cnet.preprocess_input(args.x_train)
    args.x_test = cnet.preprocess_input(args.x_test)
    
    args = cnet.train_model(args)

    # Score trained model.
    scores = args.model.evaluate(args.x_test, args.y_test, verbose=1)
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
