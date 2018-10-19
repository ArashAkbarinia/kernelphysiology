'''
Train prominent DNN architectures on various different datasets.
'''


import os
import commons
import time
import datetime
import sys
import logging
import numpy as np

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from kernelphysiology.dl.keras.prominent_utils import train_arg_parser, train_prominent_prepares
from kernelphysiology.dl.keras.prominent_utils import get_top_k_accuracy


def lr_metric_call_back(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def start_training_generator(args):
    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    logging.basicConfig(filename=os.path.join(args.log_dir, 'experiment_info.log'),
                        filemode='w',
                        format='%(levelname)s: %(message)s', level=logging.INFO)
    # dumping the entire argument list
    logging.info('%s' % args)
    logging.info('Preprocessing %s' % args.preprocessing)
    logging.info('Horizontal flip is %s' % args.horizontal_flip)
    logging.info('Vertical flip is %s' % args.vertical_flip)
    logging.info('Contrast augmentation %s' % args.contrast_range)
    logging.info('Width shift range augmentation %f' % args.width_shift_range)
    logging.info('Height shift range augmentation %f' % args.height_shift_range)
    logging.info('Zoom range augmentation %f' % args.zoom_range)

    best_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_best.h5'), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    last_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_last.h5'), verbose=1, save_weights_only=True, save_best_only=False)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    # TODO: put a proper plateau as of now I guess is never called
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=0.01, min_lr=1e-7)
    logging.info('ReduceLROnPlateau monitor=%s factor=%f, patience=%d, min_delta=%f, min_lr=%f' % (reduce_lr.monitor, reduce_lr.factor, reduce_lr.patience, reduce_lr.min_delta, reduce_lr.min_lr))
    args.callbacks = [csv_logger, best_checkpoint_logger, last_checkpoint_logger, reduce_lr]

    # TODO: add more optimisers and parametrise from argument line
    if args.optimiser.lower() == 'adam':
        if args.lr is None:
            lr = 1e-3
        else:
            lr = args.lr
        if args.decay is None:
            decay = 1e-6
        else:
            decay = args.decay
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = None
        amsgrad = False
        opt = keras.optimizers.Adam(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
        logging.info('Optimiser Adam lr=%f decay=%f beta_1=%f beta_2=%f epsilon=%s amsgrad=%s' % (lr, decay, beta_1, beta_2, epsilon, amsgrad))
    elif args.optimiser.lower() == 'sgd':
        if args.lr is None:
            lr = 1e-1
        else:
            lr = args.lr
        if args.decay is None:
            decay = 1e-4
        else:
            decay = args.decay
        momentum = 0.9
        nesterov = False
        opt = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        logging.info('Optimiser SGD lr=%f decay=%f momentum=%f nesterov=%s' % (lr, decay, momentum, nesterov))
    elif args.optimiser.lower() == 'rmsprop':
        if args.lr is None:
            lr = 0.045
        else:
            lr = args.lr
        if args.decay is None:
            decay = 1e-4
        else:
            decay = args.decay
        rho = 0.9
        epsilon = 1.0
        opt = keras.optimizers.RMSprop(lr=lr, decay=decay, rho=rho, epsilon=epsilon)
        logging.info('Optimiser RMSprop lr=%f decay=%f rho=%f epsilon=%f' % (lr, decay, rho, epsilon))

    if args.exp_decay is not None:
        def exp_decay(epoch):
           new_lr = lr * np.exp(-args.exp_decay * epoch)
           return new_lr
        args.callbacks.append(LearningRateScheduler(exp_decay))
        logging.info('Exponential decay=%f' % (args.exp_decay))

    top_k_acc = get_top_k_accuracy(args.top_k)
    lr_metric = lr_metric_call_back(opt)
    metrics = ['accuracy', top_k_acc, lr_metric]

    model = args.model
    if len(args.gpus) == 1:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(model, gpus=args.gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                                     validation_data=args.validation_generator, validation_steps=args.validation_steps,
                                     callbacks=args.callbacks, initial_epoch=args.initial_epoch,
                                     workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                            validation_data=args.validation_generator, validation_steps=args.validation_steps,
                            callbacks=args.callbacks, initial_epoch=args.initial_epoch,
                            workers=args.workers, use_multiprocessing=args.use_multiprocessing)

    # save model and weights
    model_name = args.model_name + '.h5'
    model_path = os.path.join(args.save_dir, model_name)
    model.save(model_path, include_optimizer=False)


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = train_arg_parser(sys.argv[1:])
    args = train_prominent_prepares(args)

    dataset_name = args.dataset.lower()
    network_name = args.network_name.lower()
    optimiser = args.optimiser.lower()

    # preparing directories
    dataset_parent_path = os.path.join(commons.python_root, 'data/nets/%s/' % (''.join([i for i in dataset_name if not i.isdigit()])))
    create_dir(dataset_parent_path)
    dataset_child_path = os.path.join(dataset_parent_path, dataset_name)
    create_dir(dataset_child_path)
    network_parent_path = os.path.join(dataset_child_path, network_name)
    create_dir(network_parent_path)
    network_dir = os.path.join(network_parent_path, optimiser)
    create_dir(network_dir)
    args.save_dir = os.path.join(network_dir, args.experiment_name)
    create_dir(args.save_dir)

    # preparing the name of the model
    args.model_name = 'model_area_%s' % (args.area1layers)

    start_training_generator(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))