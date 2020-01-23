"""
Keras argument hanlder.
"""

from kernelphysiology.dl.utils import arguments as ah
from kernelphysiology.dl.keras.utils import get_input_shape


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

    ah.get_inisialisation_group(parser)
    ah.get_plateau_group(parser)
    get_keras_augmentation_group(parser)
    ah.get_logging_group(parser)

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
