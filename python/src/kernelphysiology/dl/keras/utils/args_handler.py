"""
Keras argument hanlder.
"""

import math

from kernelphysiology.dl.utils import arguments as ah
from kernelphysiology.dl.keras.utils import get_input_shape


def get_plateau_group(parser):
    plateau_group = parser.add_argument_group('plateau')
    plateau_group.add_argument(
        '--plateau_monitor',
        type=str,
        default='val_loss',
        help='The monitor metric (default: val_loss)'
    )
    plateau_group.add_argument(
        '--plateau_factor',
        type=float,
        default=0.1,
        help='The reduction factor (default: 0.1)'
    )
    plateau_group.add_argument(
        '--plateau_patience',
        type=float,
        default=5,
        help='The patience (default: 5)'
    )
    plateau_group.add_argument(
        '--plateau_min_delta',
        type=float,
        default=0.001,
        help='The min_delta (default: 0.001)'
    )
    plateau_group.add_argument(
        '--plateau_min_lr',
        type=float,
        default=0.5e-6,
        help='The min_lr (default: 0.5e-6)'
    )


def get_logging_group(parser):
    logging_group = parser.add_argument_group('logging')

    logging_group.add_argument(
        '--log_period',
        type=int,
        default=0,
        help='The period of logging the epochs weights (default: 0)'
    )
    logging_group.add_argument(
        '--steps_per_epoch',
        type=int,
        default=None,
        help='Training steps per epochs (default: all samples)'
    )
    logging_group.add_argument(
        '--validation_steps',
        type=int,
        default=None,
        help='Validation steps for validations (default: all samples)'
    )


def get_architecture_group(parser):
    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument(
        '--area1layers',
        type=int,
        default=None,
        help='The number of layers in area 1 (default: None)'
    )
    architecture_group.add_argument(
        '--pyramid_conv',
        type=int,
        default=1,
        help='The number of pyramids for convolutions (default: 1)'
    )
    trainable_group = architecture_group.add_argument_group('layers')
    trainable_group = trainable_group.add_mutually_exclusive_group()
    trainable_group.add_argument(
        '--trainable_layers',
        type=str,
        default=None,
        help='Which layerst to train (default: all layers)'
    )
    trainable_group.add_argument(
        '--untrainable_layers',
        type=str,
        default=None,
        help='Which layerst not to train (default: None)'
    )


def get_inisialisation_group(parser):
    initialisation_group = parser.add_argument_group('initialisation')
    weights_group = initialisation_group.add_mutually_exclusive_group()
    weights_group.add_argument(
        '--load_weights',
        type=str,
        default=None,
        help='Whether loading weights from a model (default: None)'
    )
    initialisation_choices = [
        'dog',
        'randdog',
        'sog',
        'randsog',
        'dogsog',
        'g1',
        'g2',
        'gaussian',
        'all'
    ]
    weights_group.add_argument(
        '--initialise',
        type=str,
        default=None,
        choices=initialisation_choices,
        help='Using a specific initialisation of weights (default: None)'
    )
    initialisation_group.add_argument(
        '--same_channels',
        action='store_true',
        default=False,
        help='Identical weights for channels of a kernel (default: False)'
    )
    initialisation_group.add_argument(
        '--tog_sigma',
        type=float,
        default=1,
        help='Sigma of ToG (default: 1)'
    )
    initialisation_group.add_argument(
        '--tog_surround',
        type=float,
        default=5,
        help='Surround enlargement in ToG (default: 5)'
    )
    initialisation_group.add_argument(
        '--g_sigmax',
        type=float,
        default=1,
        help='Sigma-x of Gaussian (default: 1)'
    )
    initialisation_group.add_argument(
        '--g_sigmay',
        type=float,
        default=None,
        help='Sigma-y of Gaussian (default: None)'
    )
    initialisation_group.add_argument(
        '--g_meanx',
        type=float,
        default=0,
        help='Mean-x of Gaussian (default: 0)'
    )
    initialisation_group.add_argument(
        '--g_meany',
        type=float,
        default=0,
        help='Mean-y of Gaussian (default: 0)'
    )
    initialisation_group.add_argument(
        '--g_theta',
        type=float,
        default=0,
        help='Theta of Gaussian (default: 0)'
    )
    initialisation_group.add_argument(
        '--gg_sigma',
        type=float,
        default=1,
        help='Sigma of Gaussian gradient (default: 1)'
    )
    initialisation_group.add_argument(
        '--gg_theta',
        type=float,
        default=math.pi / 2,
        help='Theta of Gaussian gradient (default: pi/2)'
    )
    initialisation_group.add_argument(
        '--gg_seta',
        type=float,
        default=0.5,
        help='Seta of Gaussian gradient (default: 0.5)'
    )


def keras_test_arg_parser(argvs):
    parser = ah.common_test_arg_parser()

    parser.add_argument(
        '--validation_crop_type',
        type=str,
        default='centre',
        choices=[
            'random',
            'centre',
            'none'
        ],
        help='What type of crop (default: centre)'
    )

    args = ah.check_common_args(parser, argvs, 'testing')
    # check the input shape
    args.input_shape = get_input_shape(args.target_size)
    return args


def keras_train_arg_parser(argvs):
    parser = ah.common_train_arg_parser()

    parser.add_argument(
        '--crop_type',
        type=str,
        default='random',
        choices=[
            'random',
            'centre',
            'none'
        ],
        help='What type of crop (default: random)'
    )
    parser.add_argument(
        '--validation_crop_type',
        type=str,
        default='centre',
        choices=[
            'random',
            'centre',
            'none'
        ],
        help='What type of crop (default: centre)'
    )
    parser.add_argument(
        '--output_types',
        type=str,
        nargs='+',
        default=[],
        help='What type of outputs to consider in model (default: None)'
    )

    get_inisialisation_group(parser)
    get_plateau_group(parser)
    get_keras_augmentation_group(parser)
    get_logging_group(parser)

    return keras_check_training_args(parser, argvs)


def keras_check_training_args(parser, argvs):
    args = ah.check_common_args(parser, argvs, 'training')
    # check the input shape
    args.input_shape = get_input_shape(args.target_size)
    return args


def get_keras_augmentation_group(parser):
    keras_augmentation_group = parser.add_argument_group('keras augmentation')

    keras_augmentation_group.add_argument(
        '--noshuffle',
        dest='shuffle',
        action='store_false',
        default=True,
        help='Stop shuffling data (default: False)'
    )
    keras_augmentation_group.add_argument(
        '--horizontal_flip',
        action='store_true',
        default=False,
        help='Perform horizontal flip data (default: False)'
    )
    keras_augmentation_group.add_argument(
        '--vertical_flip',
        action='store_true',
        default=False,
        help='Perform vertical flip (default: False)'
    )
    keras_augmentation_group.add_argument(
        '--zoom_range',
        type=float,
        default=0,
        help='Range of zoom agumentation (default: 0)'
    )
    keras_augmentation_group.add_argument(
        '--width_shift_range',
        type=float,
        default=0,
        help='Range of width shift (default: 0)'
    )
    keras_augmentation_group.add_argument(
        '--height_shift_range',
        type=float,
        default=0,
        help='Range of height shift (default: 0)'
    )


# # TODO: this is not implemented in Pytorch
# parser.add_argument(
#     '--preprocessing',
#     type=str,
#     default=None,
#     help='The preprocessing function (default: according to network)'
# )
# # TODO: this is not implemented in Pytorch
# # FIXME: could cause errors if names mismatch and it should be merged with
# # output parameters
# parser.add_argument(
#     '--dynamic_gt',
#     nargs='+',
#     type=str,
#     default=None,
#     help='Generating dynamically ground-truth (default: None)'
# )

# # workers
# if args.workers > 1:
#     args.use_multiprocessing = True
# else:
#     args.use_multiprocessing = False