'''
Train prominent DNN architectures on various different datasets.
'''


import os
import commons
import time
import datetime
import sys

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from kernelphysiology.dl.keras.prominent_utils import train_arg_parser, train_prominent_prepares


def start_training_generator(args):

    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights.h5'), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    # TODO: put a proper plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-3)
    args.callbacks = [csv_logger, checkpoint_logger, reduce_lr]

    # TODO: put a switch case according to each network
#    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
    opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)

    model = args.model
    if args.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        parallel_model = multi_gpu_model(model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)
    else:
        model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)

    # save model and weights
    model_name = args.model_name + '.h5'
    model_path = os.path.join(args.save_dir, model_name)
    model.save(model_path, include_optimizer=False)


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = train_arg_parser(sys.argv[1:])
    args = train_prominent_prepares(args)

    dataset_name = args.dataset.lower()
    network_name = args.network.lower()

    # preparing arguments
    network_dir = os.path.join(commons.python_root, 'data/nets/%s/%s/%s/' % (''.join([i for i in dataset_name if not i.isdigit()]), dataset_name, network_name))
    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)
    args.save_dir = os.path.join(network_dir, args.experiment_name)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # preparing the name of the model
    args.model_name = 'keras_%s_%s_area_%d_contrast_%d' % (dataset_name, network_name, args.area1layers, args.train_contrast)
    if args.area1_batchnormalise:
        args.model_name += '_bnr'
    if args.area1_activation:
        args.model_name += '_act'
    if args.area1_reduction:
        args.model_name += '_red'
    if args.area1_dilation:
        args.model_name += '_dil'
    if args.add_dog:
        args.model_name += '_dog'
        args.dog_path = os.path.join(args.save_dir, 'dog.h5')

    start_training_generator(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))