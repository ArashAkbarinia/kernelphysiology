'''
Testing a Keras model of CIFAR or STL against different levels of contrast.
'''


import sys
import numpy as np
import time
import datetime

from kernelphysiology.dl.keras.analysis.analysis_generator import predict_generator
from kernelphysiology.dl.keras.models.utils import which_network, get_preprocessing_function
from kernelphysiology.dl.keras.datasets.utils import which_dataset
from kernelphysiology.utils.preprocessing import which_preprocessing

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares
from kernelphysiology.dl.utils import argument_handler


def predict_network(args):
    if len(args.gpus) == 1:
        current_results = predict_generator(args.model, generator=args.validation_generator, metrics=args.top_k,
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

    args = argument_handler.test_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    (image_manipulation_type, image_manipulation_values, image_manipulation_function) = which_preprocessing(args)

    # maybe if only one preprocessing is used, the generators can be called only once
    for j, network_name in enumerate(args.networks):
        # w1hich architecture
        args = which_network(args, network_name, args.task_type)
        # FIXME: for now it only supprts classiication
        # TODO: merge code with evaluation
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
            np.savetxt('%s_%s_%s_%s.csv' % (args.output_file, args.network_names[j], image_manipulation_type, str(manipulation_value)), current_results, delimiter=',', fmt='%i')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
