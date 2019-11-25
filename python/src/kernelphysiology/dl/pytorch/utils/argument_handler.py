"""
Argument handler functions for PyTorch.
"""

from kernelphysiology.dl.utils import argument_handler as ah
from kernelphysiology.dl.utils import prepare_training


def parse_segmentation_arguments(argv):
    description = 'Training a network for the task of image segmentation.'
    parser = ah.common_arg_parser(description=description)

    ah.get_architecture_group(parser)
    ah.get_optimisation_group(parser)
    ah.get_parallelisation_group(parser)

    args = ah.pytorch_check_training_args(parser, argv)

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
