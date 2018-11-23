'''
Testing a Keras model of CIFAR or STL against different levels of contrast.
'''


import commons

import sys
import numpy as np
import time
import datetime

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, test_arg_parser
from kernelphysiology.dl.keras.prominent_utils import get_preprocessing_function, get_top_k_accuracy
from kernelphysiology.dl.keras.prominent_utils import which_network, which_dataset
from kernelphysiology.utils.imutils import adjust_contrast, gaussian_blur, s_p_noise, adjust_gamma, uniform_noise_colour, adjust_illuminant


def uniform_noise_preprocessing(img, width, preprocessing_function=None):
    img = uniform_noise_colour(img, width, 1, np.random.RandomState(seed=1)) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def colour_constancy_preprocessing(img, illuminant, preprocessing_function=None):
    img = adjust_illuminant(img, illuminant) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def gamma_preprocessing(img, amount, preprocessing_function=None):
    img = adjust_gamma(img, amount) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def s_p_preprocessing(img, amount, preprocessing_function=None):
    img = s_p_noise(img, amount) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def gaussian_preprocessing(img, win_size, preprocessing_function=None):
    img = gaussian_blur(img, win_size) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def contrast_preprocessing(img, contrast, preprocessing_function=None):
    img = adjust_contrast(img, contrast) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = test_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    # FIXME: all types of noise
    contrasts = np.array(args.contrasts) # / 100
    results_top1 = np.zeros((contrasts.shape[0], len(args.networks)))
    results_topk = np.zeros((contrasts.shape[0], len(args.networks)))
    # maybe if only one preprocessing is used, the generators can be called only once
    for j, network_name in enumerate(args.networks):
        # w1hich architecture
        args = which_network(args, network_name)
        for i, contrast in enumerate(contrasts):
            preprocessing = args.preprocessings[j]
#            current_contrast_preprocessing = lambda img : gaussian_preprocessing(img, win_size=(contrast, contrast),
#                                                                                 preprocessing_function=get_preprocessing_function(preprocessing))
#            current_contrast_preprocessing = lambda img : s_p_preprocessing(img, amount=contrast, 
#                                                                            preprocessing_function=get_preprocessing_function(preprocessing))
#            current_contrast_preprocessing = lambda img : gamma_preprocessing(img, amount=contrast, 
#                                                                              preprocessing_function=get_preprocessing_function(preprocessing))
            current_contrast_preprocessing = lambda img : colour_constancy_preprocessing(img, illuminant=(1, contrast, contrast), 
                                                                                         preprocessing_function=get_preprocessing_function(preprocessing))
#            current_contrast_preprocessing = lambda img : uniform_noise_preprocessing(img, width=contrast, 
#                                                                                      preprocessing_function=get_preprocessing_function(preprocessing))
            args.validation_preprocessing_function = current_contrast_preprocessing

            print('Processing network %s and contrast %f' % (network_name, contrast))

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            args = which_dataset(args, dataset_name)

            top_k_acc = get_top_k_accuracy(args.top_k)
            metrics = ['accuracy', top_k_acc]
            opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)
            if len(args.gpus) == 1:
                # the compilation being necessary is a bug of keras
                args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                current_results = args.model.evaluate_generator(generator=args.validation_generator, verbose=1,
                                                                workers=args.workers, use_multiprocessing=args.use_multiprocessing)
            else:
                with tf.device('/cpu:0'):
                    args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
                parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                current_results = parallel_model.evaluate_generator(generator=args.validation_generator, verbose=1,
                                                                    workers=args.workers, use_multiprocessing=args.use_multiprocessing)
            results_top1[i, j] = current_results[1]
            results_topk[i, j] = current_results[2]

            # saving the results in a CSV format
            # it's redundant to store the results as each loop, but this is
            # good for when it crashes
            np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
            np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
