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

from kernelphysiology.dl.keras.analysis.analysis_generator import predict_generator
from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, test_arg_parser
from kernelphysiology.dl.keras.utils import get_top_k_accuracy
from kernelphysiology.dl.keras.models.utils import which_network, get_preprocessing_function
from kernelphysiology.dl.keras.datasets.utils import which_dataset
from kernelphysiology.utils.imutils import gaussian_blur, gaussian_noise
from kernelphysiology.utils.imutils import s_p_noise, speckle_noise, poisson_noise
from kernelphysiology.utils.imutils import adjust_gamma, adjust_contrast, adjust_illuminant
from kernelphysiology.utils.imutils import random_occlusion
from kernelphysiology.utils.preprocessing import which_preprocessing


def predict_network(args):
    if len(args.gpus) == 1:
        current_results = predict_generator(args.model, generator=args.validation_generator,
                                            verbose=1, workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    else:
        print('not supported')
#        parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
#        current_results = parallel_model.predict_generator(generator=args.validation_generator, verbose=1,
#                                                           workers=args.workers, use_multiprocessing=args.use_multiprocessing)
    return current_results


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = test_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    (image_manipulation_type, image_manipulation_values, image_manipulation_function) = which_preprocessing(args)

    results_top1 = np.zeros((image_manipulation_values.shape[0], len(args.networks)))
    results_topk = np.zeros((image_manipulation_values.shape[0], len(args.networks)))

    # maybe if only one preprocessing is used, the generators can be called only once
    for j, network_name in enumerate(args.networks):
        # w1hich architecture
        args = which_network(args, network_name)
        for i, manipulation_value in enumerate(image_manipulation_values):
            preprocessing = args.preprocessings[j]
            current_manipulation_preprocessing = lambda img : image_manipulation_function(img, manipulation_value, mask_radius=args.mask_radius,
                                                                                          preprocessing_function=get_preprocessing_function(preprocessing))
            args.validation_preprocessing_function = current_manipulation_preprocessing

            print('Processing network %s and %s %f' % (network_name, image_manipulation_type, manipulation_value))

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            args = which_dataset(args, dataset_name)
            if args.validation_steps is None:
                args.validation_steps = args.validation_samples / args.batch_size

            current_results = predict_network(args)
            current_results = np.array(current_results)
            np.savetxt('%s_%s_%s_%s.csv' % (args.output_file, network_name, image_manipulation_type, str(manipulation_value)), current_results, delimiter=',')

            results_top1[i, j] = np.mean(current_results[:, 0])
            results_topk[i, j] = np.median(current_results[:, 0])

            # saving the results in a CSV format
            # it's redundant to store the results as each loop, but this is
            # good for when it crashes
#            np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
#            np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
