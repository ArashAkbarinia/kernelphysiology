'''
Analysing the activation of kernels.
'''


import commons

import sys
import numpy as np
import time
import datetime
import copy

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, activation_arg_parser
from kernelphysiology.dl.keras.prominent_utils import get_preprocessing_function
from kernelphysiology.dl.keras.prominent_utils import which_network, which_dataset
from kernelphysiology.dl.keras.analysis.analysis_generator import multiple_models_generator
from kernelphysiology.utils.imutils import adjust_contrast


def predict_network(model1, model2, args):
    if len(args.gpus) == 1:
        current_results = multiple_models_generator(model1, model2, generator=args.validation_generator,
                                                    verbose=1, workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        print('not supported')
#        parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
#        current_results = parallel_model.predict_generator(generator=args.validation_generator, verbose=1,
#                                                           workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    return current_results


def contrast_preprocessing(img, contrast, preprocessing_function=None):
    img = adjust_contrast(img, contrast)
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def get_activations(network, layer):
    layer_model = keras.Model(inputs=network.input, outputs=layer.output)
    return layer_model


def get_network_preprocessing(contrast, preprocessing):
    current_contrast_preprocessing = lambda img : contrast_preprocessing(img, contrast=contrast,
                                                                         preprocessing_function=get_preprocessing_function(preprocessing))
    return current_contrast_preprocessing


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = activation_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    contrasts = np.array(args.contrasts) / 100

    args1 = copy.deepcopy(args)
    args2 = copy.deepcopy(args)

    # TODO: maybe limit it to two networks, as of now the assumption is that anyway
    network_name1 = args.networks[0]
    args1 = which_network(args1, network_name1)
    network_name2 = args.networks[1]
    args2 = which_network(args2, network_name2)

    # the first and last layer are not of our interest
    nlayers = len(args1.model.layers) - 2
    results_top1 = np.zeros((contrasts.shape[0], nlayers))
    results_topk = np.zeros((contrasts.shape[0], nlayers))
    for i, contrast in enumerate(contrasts):
        # TODO: consider different preprocessing functoins
        args.validation_preprocessing_function = get_network_preprocessing(contrast, args.preprocessings[0])
#        args1.validation_preprocessing_function = get_network_preprocessing(contrast, args.preprocessings[0])
#        args2.validation_preprocessing_function = get_network_preprocessing(contrast, args.preprocessings[1])

        print('Processing networks %s and %s with contrast %f' % (network_name1, network_name2, contrast))

        # which dataset
        # reading it after the model, because each might have their own
        # specific size
        args = which_dataset(args, dataset_name)

        for j in range(nlayers):
            # the first layer is input layer
            # TODO: pass the vonvolutional layers as a parameter
            if type(args1.model.layers[j + 1]) is keras.layers.convolutional.Conv2D and args1.model.layers[j + 1].name == 'res3c_branch2c':
                print('layer', args1.model.layers[j + 1].name)
                model1 = get_activations(args1.model, args1.model.layers[j + 1])
                model2 = get_activations(args2.model, args2.model.layers[j + 1])
                current_results = predict_network(model1, model2, args)
                np.savetxt('%s_layer_%03d_%s.csv' % (args.output_file, j,  args1.model.layers[j + 1].name), current_results, delimiter=',')
    
#                results_top1[i, j] = np.mean(current_results[:, 0])
#                results_topk[i, j] = np.median(current_results[:, 0])
#
#    # saving the results in a CSV format
#    np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
#    np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
