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
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from kernelphysiology.dl.keras.prominent_utils import train_arg_parser, train_prominent_prepares
from kernelphysiology.dl.keras.prominent_utils import get_top_k_accuracy

from kernelphysiology.filterfactory.gaussian import gaussian_kernel2

from kernelphysiology.utils.path_utils import create_dir


def lr_metric_call_back(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay))))
    return lr


def initialise_with_gaussian(model, sigmax, sigmay=None, meanx=0, meany=0, theta=0,
                             which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with doG', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        sigmax_dc = np.random.uniform(0, sigmax)
                        if sigmay is not None:
                            sigmay_dc = np.random.uniform(0, sigmay)
                        else:
                            sigmay_dc = None
                        meanx_dc = np.random.uniform(-meanx, meanx)
                        meany_dc = np.random.uniform(-meany, meany)
                        theta_dc = np.random.uniform(-theta, theta)
                        g_kernel = gaussian_kernel2(sigmax=sigmax_dc, sigmay=sigmay_dc, meanx=meanx_dc,
                                              meany=meany_dc, theta=theta_dc, width=rows, threshold=1e-4)
                        weights[0][:, :, c, d] = g_kernel
                model.layers[i].set_weights(weights)
    return model


def initialise_with_dog(model, dog_sigma, dog_surround, which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with doG', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        sigmax1 = np.random.uniform(0, dog_sigma)
                        g1 = gaussian_kernel2(sigmax=sigmax1, sigmay=None, meanx=0,
                                              meany=0, theta=0, width=rows, threshold=1e-4)
                        sigmax2 = np.random.uniform(0, dog_sigma * dog_surround)
                        g2 = gaussian_kernel2(sigmax=sigmax2, sigmay=None, meanx=0,
                                              meany=0, theta=0, width=rows, threshold=1e-4)
                        weights[0][:, :, c, d] = g1 - g2
                model.layers[i].set_weights(weights)
    return model


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
    logging.info('Contrast augmentation %s (%s)' % (args.contrast_range, args.local_contrast_variation))
    logging.info('Width shift range augmentation %f' % args.width_shift_range)
    logging.info('Height shift range augmentation %f' % args.height_shift_range)
    logging.info('Zoom range augmentation %f' % args.zoom_range)

    best_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_best.h5'), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    last_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_last.h5'), verbose=1, save_weights_only=True, save_best_only=False)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    # TODO: put a proper plateau as of now I guess is never called
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=0.01, min_lr=1e-10)
    logging.info('ReduceLROnPlateau monitor=%s factor=%f, patience=%d, min_delta=%f, min_lr=%f' % (reduce_lr.monitor, reduce_lr.factor, reduce_lr.patience, reduce_lr.min_delta, reduce_lr.min_lr))
    callbacks = [csv_logger, best_checkpoint_logger, last_checkpoint_logger, reduce_lr]

    if args.log_period > 0:
        period_checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights_{epoch:03d}.h5'), save_weights_only=True, period=args.log_period)
        callbacks.append(period_checkpoint_logger)

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
    elif args.optimiser.lower() == 'adagrad':
        if args.lr is None:
            lr = 1e-2
        else:
            lr = args.lr
        if args.decay is None:
            decay = 1e-5
        else:
            decay = args.decay
        opt = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)

    logging.info('Optimiser %s: %s' % (args.optimiser, opt.get_config()))

    if args.exp_decay is not None:
        def exp_decay(epoch):
           new_lr = lr * np.exp(-args.exp_decay * epoch)
           return new_lr
        callbacks.append(LearningRateScheduler(exp_decay))
        logging.info('Exponential decay=%f' % (args.exp_decay))

    top_k_acc = get_top_k_accuracy(args.top_k)
    lr_metric = lr_metric_call_back(opt)
    metrics = ['accuracy', top_k_acc, lr_metric]

    model = args.model
    # initialising the network with specific weights
    if args.initialise is not None:
        if args.initialise.lower() == 'dog':
            model = initialise_with_dog(model, dog_sigma=args.dog_sigma, dog_surround=args.dog_surround)
        if args.initialise.lower() == 'gaussian':
            model = initialise_with_gaussian(model, sigmax=args.g_sigmax, sigmay=args.g_sigmay,
                                             meanx=args.g_meanx, meany=args.g_meany, theta=args.g_theta)

    if len(args.gpus) == 1:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(model, gpus=args.gpus)
        # TODO: this compilation probably is not necessary
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                                     validation_data=args.validation_generator, validation_steps=args.validation_steps,
                                     callbacks=callbacks, initial_epoch=args.initial_epoch,
                                     workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1,
                            validation_data=args.validation_generator, validation_steps=args.validation_steps,
                            callbacks=callbacks, initial_epoch=args.initial_epoch,
                            workers=args.workers, use_multiprocessing=args.use_multiprocessing)

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
    if args.load_weights is not None:
        f_s_dir = os.path.join(network_dir, 'fine_tune')
    else:
        f_s_dir = os.path.join(network_dir, 'scratch')
    create_dir(f_s_dir)
    args.save_dir = os.path.join(f_s_dir, args.experiment_name)
    create_dir(args.save_dir)

    # preparing the name of the model
    args.model_name = 'model_area_%s' % (args.area1layers)

    start_training_generator(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))