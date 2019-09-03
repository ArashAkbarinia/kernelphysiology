'''
Analysing the activation of kernels.
'''


import commons

import sys
import numpy as np
import time
import datetime

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, activation_arg_parser
from kernelphysiology.dl.keras.prominent_utils import get_preprocessing_function, get_top_k_accuracy
from kernelphysiology.dl.keras.prominent_utils import which_network, which_dataset
from kernelphysiology.utils.imutils import adjust_contrast


def predict_network(args):
    top_k_acc = get_top_k_accuracy(args.top_k)
    metrics = ['accuracy', top_k_acc]
    opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)
    if len(args.gpus) == 1:
        # the compilation being necessary is a bug of keras
#        args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        current_results = args.model.predict_generator(generator=args.validation_generator, verbose=1,
                                                       workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
#        with tf.device('/cpu:0'):
#            args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
#        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
        current_results = parallel_model.predict_generator(generator=args.validation_generator, verbose=1,
                                                           workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    return current_results


def contrast_preprocessing(img, contrast, preprocessing_function=None):
    img = adjust_contrast(img, contrast)
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def get_activations(network, layer):
    layer_model = keras.Model(inputs=network.input, outputs=layer.output)
    return layer_model


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = activation_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    contrasts = np.array(args.contrasts) / 100
    results_top1 = np.zeros((contrasts.shape[0], len(args.networks)))
    results_topk = np.zeros((contrasts.shape[0], len(args.networks)))
    # maybe if only one preprocessing is used, the generators can be called only once
    for j, network_name in enumerate(args.networks):
        # w1hich architecture
        args = which_network(args, network_name)
        for i, contrast in enumerate(contrasts):
            preprocessing = args.preprocessings[j]
            current_contrast_preprocessing = lambda img : contrast_preprocessing(img, contrast=contrast,
                                                                                 preprocessing_function=get_preprocessing_function(preprocessing))
            args.validation_preprocessing_function = current_contrast_preprocessing

            print('Processing network %s and contrast %f' % (network_name, contrast))

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            args = which_dataset(args, dataset_name)

            # FIXME: the model is getting overwritten
#            args.model = get_activations(args.model, args.model.layers[2])
            current_results = predict_network(args)
            print(current_results.shape)
            current_results = current_results.flatten()

            results_top1[i, j] = np.mean(current_results)
            results_topk[i, j] = np.median(current_results)

    # saving the results in a CSV format
    np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
    np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
