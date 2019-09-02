"""
The utility functoins for datasets.
"""

import numpy as np

from kernelphysiology.dl.keras.datasets.cifar import cifar_train
from kernelphysiology.dl.keras.datasets.stl import stl_train
from kernelphysiology.dl.keras.datasets.imagenet import imagenet_train
from kernelphysiology.dl.keras.datasets.coco import coco


def which_dataset(args, dataset_name):
    if args.dynamic_gt is not None and len(args.dynamic_gt) > 0:
        train_preprocessing_function = None
        validation_preprocessing_function = None
    else:
        if args.script_type == 'training':
            train_preprocessing_function = args.train_preprocessing_function
        validation_preprocessing_function = args.validation_preprocessing_function

    if dataset_name == 'cifar10':
        if args.script_type == 'training':
            args = cifar_train.prepare_cifar10_generators(
                args,
                train_preprocessing_function,
                validation_preprocessing_function
            )
        else:
            args = cifar_train.cifar10_validatoin_generator(
                args,
                validation_preprocessing_function
            )
    elif dataset_name == 'cifar100':
        if args.script_type == 'training':
            args = cifar_train.prepare_cifar100_generators(
                args,
                train_preprocessing_function,
                validation_preprocessing_function
            )
        else:
            args = cifar_train.cifar100_validatoin_generator(
                args,
                validation_preprocessing_function
            )
    elif dataset_name == 'stl10':
        if args.script_type == 'training':
            args = stl_train.prepare_stl10_generators(
                args,
                train_preprocessing_function,
                validation_preprocessing_function
            )
        else:
            args = stl_train.stl10_validation_generator(
                args,
                validation_preprocessing_function
            )
    elif dataset_name == 'imagenet':
        if args.script_type == 'training':
            args = imagenet_train.prepare_imagenet(
                args,
                train_preprocessing_function,
                validation_preprocessing_function
            )
        else:
            args = imagenet_train.validation_generator(
                args,
                validation_preprocessing_function
            )
    elif dataset_name == 'coco':
        if args.script_type == 'training':
            args = coco.train_config(args)
        else:
            args = coco.validation_config(args)

    if args.dynamic_gt is not None and len(args.dynamic_gt) > 0:
        args.train_generator = dynamic_multiple_gt_generator(
            args.train_generator, args.train_preprocessing_function)
        args.validation_generator = dynamic_multiple_gt_generator(
            args.validation_generator, args.validation_preprocessing_function)
    return args


def dynamic_multiple_gt_generator(batches, preprocessing_function):
    """Take as input a Keras ImageGen (Iterator) and generate gt according to
    applied transformations
    """
    while True:
        x_batch, y_batch = next(batches)
        illuminant_y_batch = np.zeros((x_batch.shape[0], 3))
        for i in range(x_batch.shape[0]):
            (x_batch[i,], transformation_params) = preprocessing_function(
                x_batch[i,])
            illuminant_y_batch[i, :] = transformation_params['illuminant']
        y_batch['illuminant'] = illuminant_y_batch
        yield (x_batch, y_batch)
