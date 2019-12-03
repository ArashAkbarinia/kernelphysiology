"""
Argument handler functions for PyTorch.
"""

from kernelphysiology.dl.utils import argument_handler as ah
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.dl.utils import prepapre_testing


def parse_train_segmentation_arguments(argv):
    description = 'Training a network for the task of image segmentation.'
    parser = ah.common_arg_parser(description=description)

    ah.get_architecture_group(parser)
    ah.get_optimisation_group(parser)
    ah.get_parallelisation_group(parser)
    ah.get_augmentation_group(parser)

    args = ah.pytorch_check_training_args(parser, argv)

    args.test_only = False
    args.target_size = args.target_size[0]
    if args.lr is None:
        args.lr = 0.01
    if args.weight_decay is None:
        args.weight_decay = 1e-4
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

    ah.get_network_manipulation_group(parser)
    ah.get_parallelisation_group(parser)

    args = ah.pytorch_check_test_args(parser, argv)

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
        args.network_name, args.colour_transformation
    )

    return args
