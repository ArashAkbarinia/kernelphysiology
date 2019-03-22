'''
Testing a Keras model of vairous datasets for different image manipulations.
'''


import sys
import numpy as np
import time
import datetime

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, test_arg_parser
from kernelphysiology.dl.keras.utils import get_top_k_accuracy
from kernelphysiology.dl.keras.models.utils import which_network, get_preprocessing_function
from kernelphysiology.dl.keras.datasets.utils import which_dataset
from kernelphysiology.dl.keras.datasets.coco.evaluation import evaluate_coco
from kernelphysiology.utils.preprocessing import which_preprocessing


def evaluate_classification(args):
    if args.validation_steps is None:
        args.validation_steps = args.validation_samples / args.batch_size

    top_k_acc = get_top_k_accuracy(args.top_k)
    metrics = ['accuracy', top_k_acc]
    opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)
    if len(args.gpus) == 1:
        # the compilation being necessary is a bug of keras
        args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        current_results = args.model.evaluate_generator(generator=args.validation_generator, verbose=1,
                                                        steps=args.validation_steps,
                                                        workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        with tf.device('/cpu:0'):
            args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        current_results = parallel_model.evaluate_generator(generator=args.validation_generator, verbose=1,
                                                            steps=args.validation_steps,
                                                            workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    return current_results


def evaluate_detection(args):
    # FIXME move it
    # FIXME apecify which model is for which dataset
    current_results = evaluate_coco(args.model, args.validation_set, args.coco, ['bbox', 'segm'], limit=args.image_limit, preprocessing_function=args.validation_preprocessing_function)
    return current_results


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = test_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    (image_manipulation_type, image_manipulation_values, image_manipulation_function) = which_preprocessing(args)

    if args.task_type == 'classification':
        results_top1 = np.zeros((image_manipulation_values.shape[0], len(args.networks)))
        results_topk = np.zeros((image_manipulation_values.shape[0], len(args.networks)))
    elif args.task_type == 'detection':
        num_results_report = 12
        results_bbox = np.zeros((image_manipulation_values.shape[0], len(args.networks), num_results_report))
        results_segm = np.zeros((image_manipulation_values.shape[0], len(args.networks), num_results_report))

    # maybe if only one preprocessing is used, the generators can be called only once
    for j, network_name in enumerate(args.networks):
        # FIXME: reading the place of dataset
        if args.task_type == 'detection':
            args = which_dataset(args, dataset_name)
        # which network
        args = which_network(args, network_name, args.task_type)
        for i, manipulation_value in enumerate(image_manipulation_values):
            preprocessing = args.preprocessings[j]
            current_manipulation_preprocessing = lambda img : image_manipulation_function(img, manipulation_value, mask_radius=args.mask_radius,
                                                                                          preprocessing_function=get_preprocessing_function(preprocessing))
            args.validation_preprocessing_function = current_manipulation_preprocessing

            print('Processing network %s and %s %f' % (network_name, image_manipulation_type, manipulation_value))

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            if args.task_type == 'classification':
                args = which_dataset(args, dataset_name)
                current_results = evaluate_classification(args)
                results_top1[i, j] = current_results[1]
                results_topk[i, j] = current_results[2]

                # saving the results in a CSV format
                # it's redundant to store the results as each loop, but this is
                # good for when it crashes
                np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
                np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')
            elif args.task_type == 'detection':
                current_results = evaluate_detection(args)
                results_bbox[i, j, :] = current_results[0]
                results_segm[i, j, :] = current_results[1]
                for k in range(num_results_report):
                    np.savetxt('%s_bbox%02d.csv' % (args.output_file, k), results_bbox[:, :, k], delimiter=',')
                    np.savetxt('%s_segm%02d.csv' % (args.output_file, k), results_segm[:, :, k], delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
