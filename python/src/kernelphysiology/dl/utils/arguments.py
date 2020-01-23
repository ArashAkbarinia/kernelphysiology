"""
Handling input arguments for training/testing a network.
"""

import os
import sys
import argparse
import warnings
import math

from kernelphysiology.utils.controls import isfloat
from kernelphysiology.dl.utils import default_configs
from kernelphysiology.dl.utils import augmentation
from kernelphysiology.dl.keras.utils import get_input_shape


def get_segmentation_group(parser):
    segmentation_group = parser.add_argument_group('segmentation')

    segmentation_group.add_argument(
        '--save_pred',
        action='store_true',
        default=False,
        help='Saving the prediction to an image (default: False)'
    )


def get_colour_space_group(parser):
    colour_space_group = parser.add_argument_group('colour space')

    colour_space_group.add_argument(
        '--colour_space',
        type=str,
        default='rgb',
        choices=[
            'rgb',
            'lab',
            'lms'
        ],
        help='The colour space of network (default: RGB)'
    )

    # TODO: at the moment this is not used, and lab is used
    colour_space_group.add_argument(
        '--opponent_space',
        type=str,
        default='lab',
        choices=[
            'lab',
            'dkl'
        ],
        help='The default colour opponent space (default: lab)'
    )

    # TODO: Keras part is not implemented
    colour_space_group.add_argument(
        '--colour_transformation',
        type=str,
        default='trichromat',
        # TODO: add luminance blindness
        choices=[
            'trichromat',
            'monochromat',
            'dichromat_rg',
            'dichromat_yb',
            'protanopia',
            'deuteranopia',
            'tritanopia'
        ],
        help='The preprocessing colour transformation (default: trichromat)'
    )


def get_architecture_group(parser):
    # better handling the parameters, e.g. pretrained ones are only for
    # imagenet
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
    architecture_group.add_argument(
        '--num_kernels',
        type=int,
        default=64,
        help='The number of convolutional kernels (default: 64)'
    )
    architecture_group.add_argument(
        '-ca', '--custom_arch',
        dest='custom_arch',
        action='store_true',
        help='Custom architectures instead of those defined in libraries'
    )
    architecture_group.add_argument(
        '--pooling_type',
        type=str,
        default='max',
        choices=[
            'max',
            'avg',
            'mix',
            'contrast_avg',
            'contrast_max',
            'contrast'
        ],
        help='The pooling type (default: max)'
    )
    architecture_group.add_argument(
        '--pretrained',
        dest='pretrained',
        action='store_true',
        help='Use pre-trained model'
    )
    architecture_group.add_argument(
        '--blocks',
        nargs='+',
        type=int,
        default=None,
        help='Number of layers in every block (default: None)'
    )
    # TODO: this only makes sense for segmentation
    architecture_group.add_argument(
        '--backbone',
        type=str,
        default=None,
        help='The backbone of segmentation (default: None)'
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


def get_augmentation_group(parser):
    augmentation_group = parser.add_argument_group('augmentation')
    augmentation_group.add_argument(
        '-na', '--num_augmentations',
        type=int,
        default=None,
        help='Number of augmentations applied to each image (default: None)'
    )
    augmentation_group.add_argument(
        '-as', '--augmentation_settings',
        nargs='+',
        type=str,
        default=None,
        help='List of augmentations to be conducted (default: None)'
    )


def get_network_manipulation_group(parser):
    network_manipulation_group = parser.add_argument_group()
    network_manipulation_group.add_argument(
        '--kill_kernels',
        nargs='+',
        type=str,
        default=None,
        help='First layer name followed by kernel indices (default: None)'
    )
    network_manipulation_group.add_argument(
        '--kill_planes',
        nargs='+',
        type=str,
        default=None,
        help='Axis number followed by plane indices ax_<P1> (default: None)'
    )
    network_manipulation_group.add_argument(
        '--kill_lines',
        nargs='+',
        type=str,
        default=None,
        help='Intersection of two planes, <P1>_<L1>_<P2>_<L2> (default: None)'
    )


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


def get_parallelisation_group(parser):
    parallelisation_group = parser.add_argument_group('parallelisation')

    parallelisation_group.add_argument(
        '--world-size',
        default=-1,
        type=int,
        help='Number of nodes for distributed training'
    )
    parallelisation_group.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='Node rank for distributed training'
    )
    parallelisation_group.add_argument(
        '--dist-url',
        default='tcp://224.66.41.62:23456',
        type=str,
        help='URL used to set up distributed training'
    )
    parallelisation_group.add_argument(
        '--dist-backend',
        default='nccl',
        type=str,
        help='Distributed backend'
    )
    parallelisation_group.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Seed for initializing training. '
    )
    parallelisation_group.add_argument(
        '--multiprocessing_distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training'
    )


def get_optimisation_group(parser):
    optimisation_group = parser.add_argument_group('optimisation')

    optimisation_group.add_argument(
        '--optimiser',
        type=str,
        default='sgd',
        help='The optimiser to be used (default: sgd)'
    )
    optimisation_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=None,
        help='The learning rate parameter (default: None)'
    )
    optimisation_group.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='The momentum for optimisation (default 0.9)'
    )
    optimisation_group.add_argument(
        '-wd', '--weight_decay',
        type=float,
        default=None,
        help='The decay weight parameter (default: None)'
    )
    optimisation_group.add_argument(
        '--exp_decay',
        type=float,
        default=None,
        help='The exponential decay (default: None)'
    )
    optimisation_group.add_argument(
        '--lr_schedule',
        type=str,
        default=None,
        help='The custom learning rate scheduler (default: None)'
    )
    optimisation_group.add_argument(
        '-e', '--epochs',
        type=int,
        default=90,
        help='Number of epochs (default: 90)'
    )
    optimisation_group.add_argument(
        '--initial_epoch',
        type=int,
        default=0,
        help='The initial epoch number (default: 0)'
    )
    # TODO: whether it should belong to optimisation group
    optimisation_group.add_argument(
        '--resume',
        default=None,
        type=str,
        help='Path to latest checkpoint (default: None)'
    )
    optimisation_group.add_argument(
        '--aux_loss',
        action='store_true',
        help='auxiliar loss'
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


def common_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        dest='dataset',
        type=str,
        help='Name of the dataset'
    )
    # TODO: add choices
    # TODO: distinguish between architecture and network
    parser.add_argument(
        dest='network_name',
        type=str,
        help='Name of the architecture or network'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Number of classes for unknown datasets (default: None)'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Ex',
        help='The name of the experiment (default: Ex)'
    )
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='Frequency of reporting (default: 100)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Accuracy of top K elements (default: 5)'
    )
    parser.add_argument(
        '--random_images',
        nargs='+',
        type=int,
        default=None,
        help='Number of random images to try (default: None)'
    )

    data_dir_group = parser.add_argument_group('data path')
    data_dir_group.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='The path to the data directory (default: None)'
    )
    data_dir_group.add_argument(
        '--train_dir',
        type=str,
        default=None,
        help='The path to the train directory (default: None)'
    )
    data_dir_group.add_argument(
        '--validation_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )

    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[0],
        help='List of GPUs to be used (default: [0])'
    )
    # TODO: change the default according to training or testing
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=1,
        help='Number of workers for image generator (default: 1)'
    )

    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: according to dataset)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=None,
        help='Target size (default: according to dataset)'
    )
    # TODO: this is not implemented in Pytorch
    parser.add_argument(
        '--preprocessing',
        type=str,
        default=None,
        help='The preprocessing function (default: according to network)'
    )
    # TODO: this is not implemented in Pytorch
    # FIXME: could cause errors if names mismatch and it should be merged with
    # output parameters
    parser.add_argument(
        '--dynamic_gt',
        nargs='+',
        type=str,
        default=None,
        help='Generating dynamically ground-truth (default: None)'
    )
    parser.add_argument(
        '--task_type',
        type=str,
        choices=[
            'classification',
            'segmentation',
            'detection'
        ],
        default=None,
        help='The task to perform by network (default: None)'
    )

    get_colour_space_group(parser)
    return parser


def activation_arg_parser(argvs):
    # FIXME: update activation pipeline
    parser = common_arg_parser(
        'Analysing activation of prominent nets of Keras.'
    )

    parser.add_argument(
        '--contrasts',
        nargs='+',
        type=float,
        default=[1],
        help='List of contrasts to be evaluated (default: [1])')
    return check_common_args(parser, argvs, 'activation')


def keras_test_arg_parser(argvs):
    parser = common_test_arg_parser()

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

    return check_common_args(parser, argvs, 'testing')


def pytorch_test_arg_parser(argvs):
    parser = common_test_arg_parser()

    get_network_manipulation_group(parser)

    return pytorch_check_test_args(parser, argvs)


def pytorch_check_test_args(parser, argvs):
    args = check_common_args(parser, argvs, 'testing')

    # checking augmentation parameters
    args.manipulation, args.parameters = create_manipulation_list(
        args.manipulation, args.parameters,
        augmentation.get_testing_augmentations()
    )

    return args


def common_test_arg_parser(description='Testing a network!'):
    parser = common_arg_parser(description)

    parser.add_argument(
        '--activation_map',
        type=str,
        default=None,
        help='Saving the activation maps (default: None)'
    )

    parser.add_argument(
        '--image_limit',
        type=int,
        default=None,
        help='Number of images to be evaluated (default: None)'
    )

    parser.add_argument(
        '--manipulation',
        type=str,
        default=None,
        help='Image manipulation type to be evaluated (default: None)'
    )

    parser.add_argument(
        '--parameters',
        nargs='+',
        type=str,
        default=None,
        help='Parameters passed to the evaluation function (default: None)'
    )

    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument(
        '--validation_steps',
        type=int,
        default=None,
        help='Validation steps per epoch (default: all samples)'
    )

    return parser


def keras_train_arg_parser(argvs):
    parser = common_train_arg_parser()

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


def pytorch_train_arg_parser(argvs):
    parser = common_train_arg_parser()

    get_parallelisation_group(parser)
    get_augmentation_group(parser)

    # TODO: other than doubleing labels?
    # FIXME: implement for CIFAr and others
    parser.add_argument(
        '--augment_labels',
        dest='augment_labels',
        action='store_true',
        help='Augmenting labels of ground-truth (False)'
    )

    # TODO: this is not supported by all
    parser.add_argument(
        '--neg_params',
        nargs='+',
        type=str,
        default=None,
        help='Negative sample parameters (default: None)'
    )

    # TODO: num_classes is just for backward compatibility
    parser.add_argument(
        '--old_classes',
        default=None,
        type=int,
        help='Number of new classes (default: None)'
    )

    parser.add_argument(
        '--transfer_weights',
        type=str,
        default=None,
        help='Whether transferring weights from a model (default: None)'
    )

    return pytorch_check_training_args(parser, argvs)


def common_train_arg_parser(description='Training a network!'):
    parser = common_arg_parser(description)

    get_architecture_group(parser)
    get_optimisation_group(parser)

    return parser


def set_visible_gpus(gpus):
    if gpus[0] == -1 or gpus is None:
        gpus = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in gpus)
    return gpus


def check_common_args(parser, argvs, script_type):
    # HINT: this is just in order to get rid of EXIF warnings
    warnings.filterwarnings(
        'ignore',
        '.*(Possibly )?corrupt EXIF data.*',
        UserWarning
    )
    warnings.filterwarnings(
        'ignore',
        '.*is a low contrast image.*',
        UserWarning
    )

    args = parser.parse_args(argvs)
    args.script_type = script_type

    # checking whether number of classes is provided
    args.num_classes = default_configs.get_num_classes(
        args.dataset, args.num_classes
    )

    # checking whether traindir and valdir are provided
    if args.data_dir is not None:
        if 'cifar' in args.dataset:
            args.train_dir = args.data_dir
            args.validation_dir = args.data_dir
        else:
            args.train_dir = os.path.join(args.data_dir, 'train')
            args.validation_dir = os.path.join(args.data_dir, 'validation')
    else:
        args.train_dir = args.train_dir
        args.validation_dir = args.validation_dir

    # setting task type
    args.task_type = check_task_type(args.dataset, args.task_type)

    # setting the target size
    args.target_size = default_configs.get_default_target_size(
        args.dataset, args.target_size
    )
    args.target_size = (args.target_size, args.target_size)

    # check the input shape
    args.input_shape = get_input_shape(args.target_size)

    # setting the batch size
    if args.batch_size is None:
        if args.dataset == 'imagenet':
            if args.script_type == 'training':
                args.batch_size = 32
            if args.script_type == 'testing':
                args.batch_size = 64
            if args.script_type == 'activation':
                args.batch_size = 32
        elif 'cifar' in args.dataset or 'stl' in args.dataset:
            if args.script_type == 'training':
                args.batch_size = 256
            if args.script_type == 'testing':
                args.batch_size = 512
            if args.script_type == 'activation':
                args.batch_size = 256
        else:
            if args.script_type == 'training':
                args.batch_size = 32
            if args.script_type == 'testing':
                args.batch_size = 64
            if args.script_type == 'activation':
                args.batch_size = 32
            warnings.warn(
                'default batch_size are used for dataset %s' % args.dataset
            )

    args.gpus = set_visible_gpus(args.gpus)
    args.gpus = [*range(len(args.gpus))]

    # workers
    if args.workers > 1:
        args.use_multiprocessing = True
    else:
        args.use_multiprocessing = False

    # handling the paths
    (args.train_dir,
     args.validation_dir,
     args.data_dir) = default_configs.get_default_dataset_paths(
        args.dataset,
        args.train_dir,
        args.validation_dir,
        args.data_dir,
        args.script_type
    )

    return args


def pytorch_check_training_args(parser, argvs):
    args = check_common_args(parser, argvs, 'training')

    if 'augment_labels' in args and args.augment_labels:
        args.num_classes *= 2
        args.custom_arch = True

    # checking augmentation parameters
    args.augmentation_settings = prepare_augmentations(
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


def keras_check_training_args(parser, argvs):
    args = check_common_args(parser, argvs, 'training')
    return args


def prepare_augmentations(augmentation_settings, supported_functions):
    augmentation_settings = parse_image_modifications(
        augmentation_settings, supported_functions=supported_functions
    )
    for i in range(len(augmentation_settings)):
        for key in augmentation_settings[i]['kwargs'].keys():
            if len(augmentation_settings[i]['kwargs'][key]) == 1:
                elm0 = augmentation_settings[i]['kwargs'][key][0]
                augmentation_settings[i]['kwargs'][key] = elm0
    return augmentation_settings


def create_manipulation_list(manipulation, parameters, supported_functions):
    parameters = parse_image_modifications(
        parameters, supported_functions=supported_functions
    )
    if len(parameters) > 1:
        sys.exit(
            'Currently only one manipulation at a time is supported.'
        )
    elif len(parameters) == 0:
        if manipulation is None:
            manipulation = '_nothing'
            parameters = {
                'function': supported_functions['original'],
                'kwargs': {manipulation: [0]},
                'f_name': 'original'
            }
        else:
            sys.exit(
                'Manipulation %s requires parameters.' % manipulation
            )
    else:
        parameters = parameters[0]

        manipulation_exist = False
        for key in parameters['kwargs'].keys():
            # if key is manipulation we keep it as list to iterate over it
            if key == manipulation or manipulation is None:
                manipulation = key
                manipulation_exist = True
            elif len(parameters['kwargs'][key]) == 1:
                elm0 = parameters['kwargs'][key][0]
                parameters['kwargs'][key] = elm0
        if manipulation_exist is False:
            sys.exit(
                'Manipulation %s not found in parameters.' % manipulation
            )
    return manipulation, parameters


def parse_image_modifications(str_command, supported_functions):
    if str_command is None:
        return []
    functions_settings = []

    i = -1
    param = None
    for key in str_command:
        if key[0:2] == 'f_' and key[2:] in supported_functions:
            key = key[2:]
            i += 1
            functions_settings.append(dict())
            functions_settings[i]['function'] = supported_functions[key]
            functions_settings[i]['kwargs'] = dict()
            functions_settings[i]['f_name'] = key
            param = None
        elif i != -1 and 'function' in functions_settings[i]:
            # if starts with k_, consider it as key
            if key[0:2] == 'k_':
                param = key[2:]
                functions_settings[i]['kwargs'][param] = []
            else:
                val = key
                if isfloat(val):
                    val = float(val)
                functions_settings[i]['kwargs'][param].append(val)
        else:
            warnings.warn('Ignoring argument %s' % key)

    return functions_settings


def check_task_type(dataset, task_type=None):
    if ('cifar' in dataset or 'stl' in dataset or 'wcs' in dataset or
            dataset in ['imagenet', 'leaf', 'fruits']):
        if task_type is not None and task_type != 'classification':
            warnings.warn(
                'Invalid task_type %s: %s only supports classification' %
                (task_type, dataset)
            )
        task_type = 'classification'
    elif 'voc' in dataset:
        task_type = 'segmentation'
    return task_type
