'''
Train prominent DNN architectures on various different datasets.
'''


import os
import time
import datetime
import sys
import logging
from functools import partial

import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from kernelphysiology.dl.keras.prominent_utils import train_prominent_prepares
from kernelphysiology.dl.keras.utils import get_top_k_accuracy

from kernelphysiology.dl.keras.initialisations.initialise import initialse_weights
from kernelphysiology.dl.keras.optimisations.optimise import set_optimisation, get_default_lrs, get_default_decays
from kernelphysiology.dl.keras.optimisations.optimise import exp_decay, lr_schedule_resnet, lr_schedule_arash, lr_schedule_file, lr_schedule_nepochs
from kernelphysiology.dl.keras.optimisations.metrics import reproduction_angular_error, mean_absolute_error

from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.dl.utils import argument_handler


class TimeHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs):
        logs['epoch_time'] = time.time() - self.epoch_time_start


class LrHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay)))))
        logs['lr'] = lr


def lr_metric_call_back(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay))))
    return lr


def read_trainability(layer_arg):
    if os.path.isfile(layer_arg):
        layers = []
        with open(layer_arg) as f:
            lines = f.readlines()
            for line in lines:
                layers.append(line.strip())
        return layers
    else:
        return layer_arg


def handle_trainability(model, args):
    if args.trainable_layers is not None:
        layers = read_trainability(args.trainable_layers)
        trainable_bool = True
    elif args.untrainable_layers is not None:
        layers = read_trainability(args.untrainable_layers)
        trainable_bool = False
    else:
        return model
    for layer in model.layers:
        if layer.name in layers:
            layer.trainable = trainable_bool
        else:
            layer.trainable = not trainable_bool
    return model


def start_training_generator(args):
    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    logging.basicConfig(filename=os.path.join(args.log_dir, 'experiment_info.log'),
                        filemode='w',
                        format='%(levelname)s: %(message)s', level=logging.INFO)
    # TODO: better logging
    # dumping the entire argument list
    logging.info('%s' % args)
    logging.info('Preprocessing %s' % args.preprocessing)
    logging.info('Horizontal flip is %s' % args.horizontal_flip)
    logging.info('Vertical flip is %s' % args.vertical_flip)
    logging.info('Contrast augmentation %s (%s)' % (args.contrast_range, args.local_contrast_variation))
    logging.info('Width shift range augmentation %f' % args.width_shift_range)
    logging.info('Height shift range augmentation %f' % args.height_shift_range)
    logging.info('Zoom range augmentation %f' % args.zoom_range)

    # loggers
    best_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_best.h5'), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    last_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_last.h5'), verbose=1, save_weights_only=True, save_best_only=False)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    reduce_lr = ReduceLROnPlateau(monitor=args.plateau_monitor, factor=args.plateau_factor,
                                  patience=args.plateau_patience, min_delta=args.plateau_min_delta,
                                  min_lr=args.plateau_min_lr, cooldown=0)
    logging.info('ReduceLROnPlateau monitor=%s factor=%f, patience=%d, min_delta=%f, min_lr=%f' % (reduce_lr.monitor, reduce_lr.factor, reduce_lr.patience, reduce_lr.min_delta, reduce_lr.min_lr))
    time_callback = TimeHistory()
    lr_callback = LrHistory()
    callbacks = [time_callback, lr_callback, csv_logger, best_checkpoint_logger, last_checkpoint_logger, reduce_lr]

    if args.log_period > 0:
        period_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_{epoch:03d}.h5'), save_weights_only=True, period=args.log_period)
        callbacks.append(period_checkpoint_logger)

    # optimisation
    if args.lr is None:
        args.lr = get_default_lrs(optimiser_name=args.optimiser)
    if args.decay is None:
        args.decay = get_default_decays(optimiser_name=args.optimiser)
    opt = set_optimisation(args)

    logging.info('Optimiser %s: %s' % (args.optimiser, opt.get_config()))

    if args.exp_decay is not None:
        exp_decay_lambda = partial(exp_decay, lr=args.lr, exp_decay=args.exp_decay)
        callbacks.append(LearningRateScheduler(exp_decay_lambda))
        logging.info('Exponential decay=%f' % (args.exp_decay))
    if args.lr_schedule is not None:
        if args.lr_schedule.isdigit():
            lr_schedule_lambda = partial(lr_schedule_nepochs, n=int(args.lr_schedule), lr=args.lr)
        elif os.path.isfile(args.lr_schedule):
            lr_schedule_lambda = partial(lr_schedule_file, file_path=args.lr_schedule)
        elif args.lr_schedule == 'resnet':
            lr_schedule_lambda = partial(lr_schedule_resnet, lr=args.lr)
        elif args.lr_schedule == 'arash':
            lr_schedule_lambda = partial(lr_schedule_arash, lr=args.lr)
        callbacks.append(LearningRateScheduler(lr_schedule_lambda))

    # metrics
#    lr_metric = lr_metric_call_back(opt)
    all_classes_metrics = ['accuracy']
    if args.top_k is not None:
        all_classes_metrics.append(get_top_k_accuracy(args.top_k))
    metrics = {'all_classes': all_classes_metrics}

    model = args.model
    # initialising the network with specific weights
    model = initialse_weights(model, args)
    # set which layers are trainable or untrainable
    model = handle_trainability(model, args)
    model.summary(print_fn=logging.info)

    # TODO: extra losses should be passed directly from the dataset
    # TODO: unequal types should be taken into consideration
    losses = {'all_classes': 'categorical_crossentropy'}
    class_weight = {'all_classes': None}
    loss_weights = {"all_classes": 1.0}
    if 'natural_vs_manmade' in args.output_types:
        losses['natural_vs_manmade'] = 'binary_crossentropy'
        if args.dataset == 'cifar10':
            class_weight['natural_vs_manmade'] = {0: 0.4, 1: 0.6}
        elif args.dataset == 'cifar100':
            class_weight['natural_vs_manmade'] = {0: 0.3, 1: 0.7}
        loss_weights['natural_vs_manmade'] = 1
    if 'illuminant' in args.output_types:
        losses['illuminant'] = mean_absolute_error()
        loss_weights['illuminant'] = 1
        metrics['illuminant'] = [reproduction_angular_error()]

    if len(args.gpus) == 1:
        model.compile(loss=losses, optimizer=opt, metrics=metrics, loss_weights=loss_weights)
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss=losses, optimizer=opt, metrics=metrics, loss_weights=loss_weights)
        parallel_model = multi_gpu_model(model, gpus=args.gpus)
        # TODO: this compilation probably is not necessary
        parallel_model.compile(loss=losses, optimizer=opt, metrics=metrics, loss_weights=loss_weights)

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                                     validation_data=args.validation_generator, validation_steps=args.validation_steps,
                                     callbacks=callbacks, initial_epoch=args.initial_epoch,
                                     workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                            validation_data=args.validation_generator, validation_steps=args.validation_steps,
                            callbacks=callbacks, initial_epoch=args.initial_epoch,
                            workers=args.workers, use_multiprocessing=args.use_multiprocessing,
                            class_weight=class_weight)

    # save model and weights
    model_name = args.model_name + '.h5'
    model_path = os.path.join(args.save_dir, model_name)
    model.save(model_path, include_optimizer=False)


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = argument_handler.train_arg_parser(sys.argv[1:])
    args = train_prominent_prepares(args)

    dataset_name = args.dataset.lower()
    network_name = args.network_name.lower()
    optimiser = args.optimiser.lower()

    # preparing directories
    args.save_dir = prepare_training.prepare_output_directories(dataset_name=dataset_name,
                                                               network_name=network_name,
                                                               optimiser=optimiser,
                                                               load_weights=args.load_weights,
                                                               experiment_name=args.experiment_name,
                                                               framework='keras')

    # preparing the name of the model
    args.model_name = 'model_area_%s' % (args.area1layers)

    start_training_generator(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))
