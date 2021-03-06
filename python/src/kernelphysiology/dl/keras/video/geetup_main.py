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

import numpy as np
import random
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
from kernelphysiology.dl.geetup import geetup_opts
from kernelphysiology.dl.geetup import geetup_visualise


def euc_error(y_true, y_pred, target_size, axis=2):
    y_true_inds = K.argmax(y_true, axis=axis)
    y_true_inds = tf.unravel_index(K.reshape(y_true_inds, [-1]), target_size)
    y_pred_inds = K.argmax(y_pred, axis=axis)
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


def evaluate(model, args, validation_name, only_name_and_gt=False):
    testing_list, args.sequence_length, args.frames_gap = read_pickle(
        args.validation_file, args.frames_gap)

    testing_generator = geetup_db.GeetupGenerator(
        testing_list,
        batch_size=args.batch_size,
        target_size=args.target_size,
        frames_gap=args.frames_gap,
        sequence_length=args.sequence_length,
        gaussian_sigma=30.5,
        preprocessing_function=preprocess,
        shuffle=False,
        all_frames=args.all_frames,
        only_name_and_gt=only_name_and_gt
    )

    if args.all_frames:
        num_frames = args.sequence_length
    else:
        num_frames = 1

    all_results = np.zeros(
        (testing_generator.num_sequences, num_frames)
    )
    all_preds = np.zeros(
        (testing_generator.num_sequences, num_frames, 2)
    )
    if only_name_and_gt:
        video_names = np.empty(
            (testing_generator.num_sequences, num_frames), dtype='<U180'
        )

    j = 0
    for i in range(testing_generator.__len__()):
        if i % 100 == 0:
            print('Processing %s %d of %d (Euc avg %.2f med %.2f)' %
                  (validation_name, j, testing_generator.num_sequences,
                   all_results[:j, ].mean(), np.median(all_results[:j, ]))
                  )
        x, y = testing_generator.__getitem__(i)
        if only_name_and_gt:
            pred_ind = (args.target_size[0] / 2, args.target_size[1] / 2)
        else:
            y = np.reshape(y,
                           (y.shape[0], y.shape[1],
                            args.target_size[0], args.target_size[1])
                           )
            pred_fix = model.predict_on_batch(x)
            pred_fix = np.reshape(pred_fix, y.shape)
        # looping through batches
        for b in range(y.shape[0]):
            # looping through frames
            for f in range(y.shape[1]):
                if only_name_and_gt is False:
                    pred_ind = np.asarray(
                        max_pixel_ind(pred_fix[b, f,].squeeze())
                    )
                    gt_ind = np.asarray(max_pixel_ind(y[b, f,].squeeze()))
                else:
                    video_names[j, f] = x[b, f].squeeze()
                    gt_ind = y[b, f,].squeeze()
                all_results[j, f] = np.linalg.norm(pred_ind - gt_ind)
                all_preds[j, f,] = pred_ind
            j += 1
    # saving Euc distances
    result_file = '%s/%s_pred.pickle' % (args.log_dir, validation_name)
    pickle_out = open(result_file, 'wb')
    pickle.dump(all_results, pickle_out)
    pickle_out.close()
    # saving predictions distances
    result_file = '%s/%s_model_out.pickle' % (args.log_dir, validation_name)
    pickle_out = open(result_file, 'wb')
    pickle.dump(all_preds, pickle_out)
    pickle_out.close()
    if only_name_and_gt:
        result_file = '%s/%s_frames.pickle' % (args.log_dir, validation_name)
        pickle_out = open(result_file, 'wb')
        pickle.dump(video_names, pickle_out)
        pickle_out.close()


def random_image(model, args):
    testing_list, args.sequence_length, args.frames_gap = read_pickle(
        args.validation_file, args.frames_gap)

    testing_generator = geetup_db.GeetupGenerator(
        testing_list,
        batch_size=args.batch_size,
        target_size=args.target_size,
        frames_gap=args.frames_gap,
        sequence_length=args.sequence_length,
        gaussian_sigma=30.5,
        preprocessing_function=preprocess,
        shuffle=False,
        all_frames=args.all_frames
    )

    if len(args.random) == 1:
        which_images = range(args.random[0])
    else:
        which_images = args.random
    for i in which_images:
        if len(args.random) == 1:
            ran_ind = random.randint(0, testing_generator.__len__())
        else:
            ran_ind = i
        x, y = testing_generator.__getitem__(ran_ind)
        pred_fix = model.predict_on_batch(x)
        x = inv_normalise_tensor(x, mean=mean, std=std)
        for b in range(y.shape[0]):
            for f in range(y.shape[1]):
                file_name = '%s/image_%d_%d_%d.jpg' % \
                            (args.log_dir, ran_ind, b, f)
                current_image = x[b, f,].squeeze()
                current_image = current_image[:, :, [2, 1, 0]].copy()
                _ = geetup_visualise.draw_circle_results(
                    current_image, y[b, f,], pred_fix[b, f,], file_name
                )
                # only saving the first batch, since the others are very similar
                break


def read_pickle(pickle_path, frames_gap=None):
    pickle_in = open(pickle_path, 'rb')
    pickle_info = pickle.load(pickle_in)
    video_list = pickle_info['video_list']
    sequence_length = pickle_info['sequence_length']
    if frames_gap is None:
        frames_gap = pickle_info['frames_gap']
    return video_list, sequence_length, frames_gap


if __name__ == "__main__":

    parser = geetup_opts.argument_parser()
    args = geetup_opts.check_args(parser, sys.argv[1:])

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in args.gpus)
    gpus = [*range(len(args.gpus))]

    create_dir(args.log_dir)
    # for training organise the output file
    if args.evaluate is False:
        # add architecture to directory
        args.log_dir = os.path.join(args.log_dir, args.architecture)
        create_dir(args.log_dir)
        # add frame based or time integration to directory
        if args.frame_based:
            time_or_frame = 'frame_based'
        else:
            time_or_frame = 'time_integration'
        args.log_dir = os.path.join(args.log_dir, time_or_frame)
        create_dir(args.log_dir)
        # add scratch or fine tune to directory
        if args.weights is None:
            new_or_tune = 'scratch'
        else:
            new_or_tune = 'fine_tune'
        args.log_dir = os.path.join(args.log_dir, new_or_tune)
        create_dir(args.log_dir)
    # add experiment name to directory
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    create_dir(args.log_dir)

    logging.basicConfig(
        filename=args.log_dir + '/experiment_info.log', filemode='w',
        format='%(levelname)s: %(message)s', level=logging.INFO
    )

    lr_schedule_lambda = partial(lr_schedule_resnet, lr=0.1)

    args.target_size = (224, 224)

    mean = [103.939, 116.779, 123.68]
    std = [1, 1, 1]

    preprocess = partial(normalise_tensor, mean=mean, std=std)

    training_list = []
    if args.evaluate is False:
        training_list, args.sequence_length, args.frames_gap = read_pickle(
            os.path.join(args.data_dir, args.train_file), args.frames_gap)

        training_generator = geetup_db.GeetupGenerator(
            training_list,
            batch_size=args.batch_size,
            target_size=args.target_size,
            frames_gap=args.frames_gap,
            sequence_length=args.sequence_length,
            gaussian_sigma=30.5,
            preprocessing_function=preprocess,
            all_frames=args.all_frames
        )

        # during training the validation is only for sanity check
        testing_list = training_list

        testing_generator = geetup_db.GeetupGenerator(
            testing_list,
            batch_size=args.batch_size,
            target_size=args.target_size,
            frames_gap=args.frames_gap,
            sequence_length=args.sequence_length,
            gaussian_sigma=30.5,
            preprocessing_function=preprocess,
            shuffle=not args.evaluate,
            all_frames=args.all_frames
        )

        print('Training %d, Testing %d' %
              (len(training_list), len(testing_list)))

    if args.architecture == 'centre':
        evaluate(None, args, 'validation_name', True)
    else:
        model = geetup_net.get_network(
            args.architecture,
            input_shape=(args.sequence_length, *args.target_size, 3),
            frame_based=args.frame_based,
            weights=args.weights,
            all_frames=args.all_frames
        )

        euc_metric = wrapped_partial(euc_error, target_size=args.target_size)

        metrics = [euc_metric]
        loss = 'binary_crossentropy'
        opt = keras.optimizers.SGD(lr=0.1, decay=0, momentum=0.9,
                                   nesterov=False)
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
            random_image(model, args)
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
                        LearningRateScheduler(lr_schedule_lambda),
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
            args.validation_file = os.path.join(args.data_dir,
                                                'testing_all_subjects.pickle')
            evaluate(model, args, 'all_subjects')
            args.validation_file = os.path.join(args.data_dir,
                                                'testing_inter_subjects.pickle')
            evaluate(model, args, 'inter_subjects')
