"""
Argument handler functions for PyTorch.
"""

import sys
import warnings

from kernelphysiology.dl.utils import argument_groups
from kernelphysiology.dl.utils import augmentation
from kernelphysiology.dl.utils import arguments as ah
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.dl.utils import prepapre_testing


def parse_train_segmentation_arguments(argv):
    description = 'Training a network for the task of image segmentation.'
    parser = ah.common_arg_parser(description=description)

    argument_groups.get_optimisation_group(parser)
    argument_groups.get_parallelisation_group(parser)
    argument_groups.get_augmentation_group(parser)

    args = _check_training_args(parser, argv)

    args.test_only = False
    args.target_size = args.target_size[0]
    if args.lr is None:
        args.lr = 0.02
    if args.weight_decay is None:
        args.weight_decay = 1e-4
    if '-e' not in argv and '--epochs' not in argv:
        args.epochs = 30
    # FIXME: cant take more than one GPU
    args.gpus = args.gpus[0]
    # TODO: why load weights is False?
    args.out_dir = prepare_training.prepare_output_directories(
        dataset_name=args.dataset, network_name=args.network_name,
        optimiser='sgd', load_weights=False,
        experiment_name=args.experiment_name, framework='pytorch'
    )

    return args


def parse_predict_segmentation_arguments(argv):
    description = 'Prediction of a network for the task of image segmentation.'
    parser = ah.common_test_arg_parser(description=description)

    argument_groups.get_network_manipulation_group(parser)
    argument_groups.get_parallelisation_group(parser)
    argument_groups.get_segmentation_group(parser)

    args = _check_test_args(parser, argv)

    args.validation_dir = args.data_dir
    args.target_size = args.target_size[0]
    # TODO: right now only batch size 1 is supported in evaluation
    args.batch_size = 1
    # FIXME: cant take more than one GPU
    args.gpus = args.gpus[0]
    # TODO: why load weights is False?
    (args.network_files,
     args.network_names,
     args.network_chromaticities) = prepapre_testing.prepare_networks_testting(
        args.network_name, args.vision_type
    )

    return args


def test_arg_parser(argvs, extra_args_fun=None):
    parser = ah.common_test_arg_parser()

    argument_groups.get_network_manipulation_group(parser)

    if extra_args_fun is not None:
        extra_args_fun(parser)

    return _check_test_args(parser, argvs)


def _check_test_args(parser, argvs):
    args = ah.check_common_args(parser, argvs, 'testing')

    # checking augmentation parameters
    args.manipulation, args.parameters = ah.create_manipulation_list(
        args.manipulation, args.parameters,
        augmentation.get_testing_augmentations()
    )

    return args


def train_arg_parser(argvs, extra_args_fun=None):
    parser = ah.common_train_arg_parser()

    argument_groups.get_parallelisation_group(parser)
    argument_groups.get_augmentation_group(parser)

    if extra_args_fun is not None:
        extra_args_fun(parser)

    return _check_training_args(parser, argvs)


def _check_training_args(parser, argvs):
    args = ah.check_common_args(parser, argvs, 'training')

    # checking augmentation parameters
    args.augmentation_settings = ah.prepare_augmentations(
        args.augmentation_settings, augmentation.get_training_augmentations()
    )
    if len(args.augmentation_settings) == 0:
        args.num_augmentations = 0
    elif args.num_augmentations is not None:
        if args.num_augmentations == 0:
            sys.exit(
                'When augmentation_settings flag is used, '
                'num_augmentations should be bigger than 0.'
            )
        elif args.num_augmentations > len(args.augmentation_settings):
            warnings.warn(
                'num_augmentations larger than augmentation_settings, '
                'it will be set to the maximum of augmentation_settings.'
            )
        args.num_augmentations = len(args.augmentation_settings)
    return args
