"""
The main script for GEETUP.
"""

import keras
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils import multi_gpu_model

import tensorflow as tf
import cv2

import numpy as np
import sys
import os
import pickle
import logging

from functools import partial
from functools import update_wrapper

from kernelphysiology.utils.imutils import max_pixel_ind
from kernelphysiology.utils.path_utils import create_dir
from kernelphysiology.dl.keras.video import geetup_net
from kernelphysiology.dl.keras.video import geetup_db
from kernelphysiology.dl.keras.video import geetup_opts


def euc_error(y_true, y_pred, target_size):
    y_true_inds = K.argmax(y_true, axis=2)
    y_true_inds = tf.unravel_index(K.reshape(y_true_inds, [-1]), target_size)
    y_pred_inds = K.argmax(y_pred, axis=2)
    y_pred_inds = tf.unravel_index(K.reshape(y_pred_inds, [-1]), target_size)

    true_pred_diff = K.sum((y_true_inds - y_pred_inds) ** 2, axis=0)
    euc_distance = tf.sqrt(
        tf.cast(true_pred_diff, dtype=tf.float32))
    return tf.reduce_mean(euc_distance)


def lr_schedule_resnet(epoch, lr):
    new_lr = lr * (0.1 ** (epoch // (45 / 3)))
    return new_lr


def inv_normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # inverting the normalisation for each channel
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] * std[i]) + mean[i]
    return tensor


def normalise_tensor(tensor, mean, std):
    tensor = tensor.copy()
    # normalising the channels
    for i in range(tensor.shape[4]):
        tensor[:, :, :, :, i] = (tensor[:, :, :, :, i] - mean[i]) / std[i]
    return tensor


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def visualise_results(current_image, gt, pred, file_name):
    rows, cols, _ = current_image.shape
    gt_pixel = max_pixel_ind(
        np.reshape(gt.squeeze(), (rows, cols))
    )
    cv2.circle(current_image, gt_pixel, 15, (0, 255, 0))

    pred_pixel = max_pixel_ind(
        np.reshape(pred.squeeze(), (rows, cols))
    )
    cv2.circle(current_image, pred_pixel, 15, (0, 0, 255))

    euc_dis = np.linalg.norm(
        np.asarray(pred_pixel) - np.asarray(gt_pixel))
    cx = round(rows / 2)
    cy = round(cols / 2)
    cv2.putText(current_image, str(int(euc_dis)), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    cv2.imwrite(file_name, current_image)


def euc_error_image(pred, gt):
    pred_ind = np.asarray(max_pixel_ind(pred))
    gt_ind = np.asarray(max_pixel_ind(gt))
    return np.linalg.norm(pred_ind - gt_ind)


def evaluate(model, args, validation_name):
    pickle_in = open(args.validation_file, 'rb')
    testing_list = pickle.load(pickle_in)
    testing_generator = geetup_db.GeetupGenerator(
        testing_list,
        batch_size=args.batch_size,
        target_size=args.target_size,
        gaussian_sigma=30.5,
        preprocessing_function=preprocess,
        shuffle=False)

    all_results = np.zeros(
        (testing_generator.num_sequences, args.sequence_length)
    )
    j = 0
    for i in range(testing_generator.__len__()):
        if i % 100 == 0:
            print('Processing %s %d of %d (Euc avg %.2f med %.2f)' %
                  (validation_name, j, testing_generator.num_sequences,
                   all_results[:j, ].mean(), np.median(all_results[:j, ]))
                  )
        x, y = testing_generator.__getitem__(i)
        y = np.reshape(y,
                       (y.shape[0], y.shape[1],
                        args.target_size[0], args.target_size[1])
                       )
        pred_fix = model.predict_on_batch(x)
        pred_fix = np.reshape(pred_fix, y.shape)
        for b in range(x.shape[0]):
            for f in range(x.shape[1]):
                all_results[j, f] = euc_error_image(
                    pred_fix[b, f,].squeeze(), y[b, f,].squeeze()
                )
            j += 1
    result_file = '%s/%s_pred.pickle' % (args.log_dir, validation_name)
    pickle_out = open(result_file, 'wb')
    pickle.dump(all_results, pickle_out)
    pickle_out.close()


if __name__ == "__main__":

    parser = geetup_opts.argument_parser()
    args = geetup_opts.check_args(parser, sys.argv[1:])

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in args.gpus)
    gpus = [*range(len(args.gpus))]

    create_dir(args.log_dir)

    logging.basicConfig(
        filename=args.log_dir + '/experiment_info.log', filemode='w',
        format='%(levelname)s: %(message)s', level=logging.INFO
    )

    lr_schedule_lambda = partial(lr_schedule_resnet, lr=0.1)

    frames_gap = 10
    args.sequence_length = 9
    args.target_size = (224, 224)

    mean = [103.939, 116.779, 123.68]
    std = [1, 1, 1]

    preprocess = partial(normalise_tensor, mean=mean, std=std)

    training_list = []
    if args.evaluate is False:
        pickle_in = open(args.train_file, 'rb')
        training_list = pickle.load(pickle_in)

        training_generator = geetup_db.GeetupGenerator(
            training_list,
            batch_size=args.batch_size,
            target_size=args.target_size,
            gaussian_sigma=30.5,
            preprocessing_function=preprocess)

        pickle_in = open(args.validation_file, 'rb')
        testing_list = pickle.load(pickle_in)

        testing_generator = geetup_db.GeetupGenerator(
            testing_list,
            batch_size=args.batch_size,
            target_size=args.target_size,
            gaussian_sigma=30.5,
            preprocessing_function=preprocess,
            shuffle=not args.evaluate)

    print('Training %d, Testing %d' % (len(training_list), len(testing_list)))

    model = geetup_net.get_network(
        args.architecture,
        input_shape=(args.sequence_length, *args.target_size, 3),
        frame_based=args.frame_based,
        weights=args.weights
    )

    euc_metric = wrapped_partial(euc_error, target_size=args.target_size)

    metrics = [euc_metric]
    loss = 'binary_crossentropy'
    opt = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=False)
    if len(gpus) == 1:
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss=loss, optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(model, gpus=gpus)
        # TODO: this compilation probably is not necessary
        parallel_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    if args.random is not None:
        if len(args.random) == 1:
            which_images = range(args.random)
        else:
            which_images = args.random
        for i in which_images:
            if len(args.random) == 1:
                ran_ind = np.random.randint(0, testing_generator.__len__())
            else:
                ran_ind = i
            x, y = testing_generator.__getitem__(ran_ind)
            pred_fix = model.predict_on_batch(x)
            x = inv_normalise_tensor(x, mean=mean, std=std)
            for b in range(x.shape[0]):
                for f in range(x.shape[1]):
                    file_name = 'random_results/image_%d_%d_%d.jpg' % \
                                (ran_ind, b, f)
                    current_image = x[b, f,].squeeze()
                    current_image = current_image[:, :, [2, 1, 0]].copy()
                    visualise_results(current_image, y[b, f,],
                                      pred_fix[b, f,], file_name)
    elif args.evaluate:
        evaluate(model, args, 'validation_name')
    else:
        last_checkpoint_logger = ModelCheckpoint(
            args.log_dir + '/model_weights_last.h5',
            verbose=1,
            save_weights_only=True,
            save_best_only=False)
        csv_logger = CSVLogger(
            os.path.join(args.log_dir, 'log.csv'),
            append=False, separator=';')

        steps_per_epoch = round(10000 / args.batch_size)
        validation_steps = round(100 / args.batch_size)
        if parallel_model is not None:
            parallel_model.fit_generator(
                generator=training_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=testing_generator,
                validation_steps=validation_steps,
                use_multiprocessing=False,
                workers=1, epochs=args.epochs, verbose=1,
                callbacks=[
                    csv_logger,
                    LearningRateScheduler(
                        lr_schedule_lambda),
                    last_checkpoint_logger]
            )
        else:
            model.fit_generator(
                generator=training_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=testing_generator,
                validation_steps=validation_steps,
                use_multiprocessing=False,
                workers=1, epochs=args.epochs, verbose=1,
                callbacks=[
                    csv_logger,
                    LearningRateScheduler(lr_schedule_lambda),
                    last_checkpoint_logger]
            )
        args.validation_file = '/home/arash/Software/repositories/kernelphysiology/python/data/datasets/geetup/testing_all_subjects.pickle'
        evaluate(model, args, 'all_subjects')
        args.validation_file = '/home/arash/Software/repositories/kernelphysiology/python/data/datasets/geetup/testing_inter_subjects.pickle'
        evaluate(model, args, 'inter_subjects')