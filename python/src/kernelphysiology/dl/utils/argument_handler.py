"""
Handling input arguments for training/testing a network.
"""

import os
import sys
import argparse
import numpy as np
import warnings
import math

from kernelphysiology.dl.utils import default_configs
from kernelphysiology.dl.keras.datasets.utils import get_default_target_size
from kernelphysiology.dl.keras.utils import get_input_shape


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
        '--experiment_name',
        type=str,
        default='Ex',
        help='The name of the experiment (default: Ex)'
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
        help='Generating dynamically ground-truth (default: None)')
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Accuracy of top K elements (default: None)')
    parser.add_argument(
        '--task_type',
        type=str,
        choices=[
            'classification',
            'detection'
        ],
        default=None,
        help='The task to prform by network (default: None)')
    parser.add_argument(
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
    return check_args(parser, argvs, 'activation')


def test_arg_parser(argvs):
    parser = common_arg_parser(
        'Testing different image classification networks.'
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
        '--activation_map',
        type=str,
        default=None,
        help='Saving the activation maps (default: None)'
    )
    parser.add_argument(
        '--mask_radius',
        type=float,
        default=None,
        help='The radius of image distortion (default: None)'
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='circle',
        help='The type of mask image (default: circle)'
    )
    parser.add_argument(
        '--image_limit',
        type=int,
        default=None,
        help='Number of images to be evaluated (default: None)'
    )
    # TODO: Keras part is not implemented
    parser.add_argument(
        '--distance',
        type=float,
        default=1,
        help='Simulating the viewing distance (default: 1)'
    )

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

    colour_space_group = parser.add_argument_group('colour space')
    # TODO: merge this with colour space
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

    image_degradation_group = parser.add_mutually_exclusive_group()
    image_degradation_group.add_argument(
        '--contrasts',
        nargs='+',
        type=float,
        default=None,
        help='List of contrasts to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--gaussian_sigma',
        nargs='+',
        type=float,
        default=None,
        help='List of Gaussian sigmas to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--s_p_noise',
        nargs='+',
        type=float,
        default=None,
        help='List of salt and pepper noise to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--speckle_noise',
        nargs='+',
        type=float,
        default=None,
        help='List of speckle noise to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--gaussian_noise',
        nargs='+',
        type=float,
        default=None,
        help='List of Gaussian noise to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--poisson_noise',
        action='store_true',
        default=False,
        help='Poisson noise to be evaluated (default: False)')
    image_degradation_group.add_argument(
        '--gammas',
        nargs='+',
        type=float,
        default=None,
        help='List of gammas to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--illuminants',
        nargs='+',
        type=float,
        default=None,
        help='List of illuminations to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--occlusion',
        nargs='+',
        type=float,
        default=None,
        help='List of occlusions to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--chromaticity',
        nargs='+',
        type=float,
        default=None,
        help='List of chromaticity to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--red_green',
        nargs='+',
        type=float,
        default=None,
        help='List of red-green to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--yellow_blue',
        nargs='+',
        type=float,
        default=None,
        help='List of yellow-blue to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--lightness',
        nargs='+',
        type=float,
        default=None,
        help='List of lightness to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--invert_chromaticity',
        action='store_true',
        default=False,
        help='Inverting chromaticity to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--invert_opponency',
        action='store_true',
        default=False,
        help='Inverting colour opponency to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--invert_lightness',
        action='store_true',
        default=False,
        help='Inverting lightness to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--rotate_hue',
        nargs='+',
        type=float,
        default=None,
        help='Rotating hues to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--keep_red',
        nargs='+',
        type=float,
        default=None,
        help='List of keeping red to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--keep_blue',
        nargs='+',
        type=float,
        default=None,
        help='List of keeping blue to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--keep_green',
        nargs='+',
        type=float,
        default=None,
        help='List of keeping green to be evaluated (default: None)')
    image_degradation_group.add_argument(
        '--original_rgb',
        action='store_true',
        default=False,
        help='Testing with the original RGB values (default: False)')

    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument(
        '--validation_steps',
        type=int,
        default=None,
        help='Validation steps per epoch (default: all samples)')

    return check_args(parser, argvs, 'testing')


def train_arg_parser(argvs):
    parser = common_arg_parser('Training prominent nets of Keras.')
    parser.add_argument(
        '--crop_type',
        type=str,
        default='random',
        choices=[
            'random',
            'centre',
            'none'
        ],
        help='What type of crop (default: random)')
    parser.add_argument(
        '--validation_crop_type',
        type=str,
        default='centre',
        choices=[
            'random',
            'centre',
            'none'
        ],
        help='What type of crop (default: centre)')
    parser.add_argument(
        '--output_types',
        type=str,
        nargs='+',
        default=[],
        help='What type of outputs to consider in model (default: None)')

    # better handling the parameters, e.g. pretrained ones are only for
    # imagenet
    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument(
        '--area1layers',
        type=int,
        default=None,
        help='The number of layers in area 1 (default: None)')
    architecture_group.add_argument(
        '--pyramid_conv',
        type=int,
        default=1,
        help='The number of pyramids for convolutions (default: 1)')
    architecture_group.add_argument(
        '--num_kernels',
        type=int,
        default=16,
        help='The number of convolutional kernels (default: 16)')
    architecture_group.add_argument(
        'ca', '--custom_arch',
        dest='custom_arch',
        action='store_true',
        help='Custom models rather the library models'
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

    trainable_group = architecture_group.add_argument_group('layers')
    trainable_group = trainable_group.add_mutually_exclusive_group()
    trainable_group.add_argument(
        '--trainable_layers',
        type=str,
        default=None,
        help='Which layerst to train (default: all layers)')
    trainable_group.add_argument(
        '--untrainable_layers',
        type=str,
        default=None,
        help='Which layerst not to train (default: None)')

    initialisation_group = parser.add_argument_group('initialisation')
    weights_group = initialisation_group.add_mutually_exclusive_group()
    weights_group.add_argument(
        '--load_weights',
        type=str,
        default=None,
        help='Whether loading weights from a model (default: None)')
    initialisation_choices = [
        'dog',
        'randdog',
        'sog',
        'randsog',
        'dogsog',
        'g1',
        'g2',
        'gaussian',
        'all']
    weights_group.add_argument(
        '--initialise',
        type=str,
        default=None,
        choices=initialisation_choices,
        help='Using a specific initialisation of weights (default: None)')
    initialisation_group.add_argument(
        '--same_channels',
        action='store_true',
        default=False,
        help='Identical weights for channels of a kernel (default: False)')
    initialisation_group.add_argument(
        '--tog_sigma',
        type=float,
        default=1,
        help='Sigma of ToG (default: 1)')
    initialisation_group.add_argument(
        '--tog_surround',
        type=float,
        default=5,
        help='Surround enlargement in ToG (default: 5)')
    initialisation_group.add_argument(
        '--g_sigmax',
        type=float,
        default=1,
        help='Sigma-x of Gaussian (default: 1)')
    initialisation_group.add_argument(
        '--g_sigmay',
        type=float,
        default=None,
        help='Sigma-y of Gaussian (default: None)')
    initialisation_group.add_argument(
        '--g_meanx',
        type=float,
        default=0,
        help='Mean-x of Gaussian (default: 0)')
    initialisation_group.add_argument(
        '--g_meany',
        type=float,
        default=0,
        help='Mean-y of Gaussian (default: 0)')
    initialisation_group.add_argument(
        '--g_theta',
        type=float,
        default=0,
        help='Theta of Gaussian (default: 0)')
    initialisation_group.add_argument(
        '--gg_sigma',
        type=float,
        default=1,
        help='Sigma of Gaussian gradient (default: 1)')
    initialisation_group.add_argument(
        '--gg_theta',
        type=float,
        default=math.pi / 2,
        help='Theta of Gaussian gradient (default: pi/2)')
    initialisation_group.add_argument(
        '--gg_seta',
        type=float,
        default=0.5,
        help='Seta of Gaussian gradient (default: 0.5)')

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
    # TODO: change the name to weight_decay
    optimisation_group.add_argument(
        '--wd', '--decay',
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
        '--epochs',
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
        default='',
        type=str,
        help='Path to latest checkpoint (default: none)'
    )

    plateau_group = parser.add_argument_group('plateau')
    plateau_group.add_argument(
        '--plateau_monitor',
        type=str,
        default='val_loss',
        help='The monitor metric (default: val_loss)')
    plateau_group.add_argument(
        '--plateau_factor',
        type=float,
        default=0.1,
        help='The reduction factor (default: 0.1)')
    plateau_group.add_argument(
        '--plateau_patience',
        type=float,
        default=5,
        help='The patience (default: 5)')
    plateau_group.add_argument(
        '--plateau_min_delta',
        type=float,
        default=0.001,
        help='The min_delta (default: 0.001)')
    plateau_group.add_argument(
        '--plateau_min_lr',
        type=float,
        default=0.5e-6,
        help='The min_lr (default: 0.5e-6)')

    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument(
        '--log_period',
        type=int,
        default=0,
        help='The period of logging the epochs weights (default: 0)')
    logging_group.add_argument(
        '--steps_per_epoch',
        type=int,
        default=None,
        help='Training steps per epochs (default: all samples)')
    logging_group.add_argument(
        '--validation_steps',
        type=int,
        default=None,
        help='Validation steps for validations (default: all samples)')

    keras_augmentation_group = parser.add_argument_group('keras augmentation')
    keras_augmentation_group.add_argument(
        '--noshuffle',
        dest='shuffle',
        action='store_false',
        default=True,
        help='Stop shuffling data (default: False)')
    keras_augmentation_group.add_argument(
        '--horizontal_flip',
        action='store_true',
        default=False,
        help='Perform horizontal flip data (default: False)')
    keras_augmentation_group.add_argument(
        '--vertical_flip',
        action='store_true',
        default=False,
        help='Perform vertical flip (default: False)')
    keras_augmentation_group.add_argument(
        '--zoom_range',
        type=float,
        default=0,
        help='Range of zoom agumentation (default: 0)')
    keras_augmentation_group.add_argument(
        '--width_shift_range',
        type=float,
        default=0,
        help='Range of width shift (default: 0)')
    keras_augmentation_group.add_argument(
        '--height_shift_range',
        type=float,
        default=0,
        help='Range of height shift (default: 0)')

    our_augmentation_group = parser.add_argument_group('our augmentation')
    our_augmentation_group.add_argument(
        '--num_augmentation',
        type=int,
        default=None,
        help='Number of types at each instance (default: None)')
    our_augmentation_group.add_argument(
        '--contrast_range',
        nargs='+',
        type=float,
        default=None,
        help='Contrast lower limit (default: None)')
    our_augmentation_group.add_argument(
        '--local_contrast_variation',
        type=float,
        default=0,
        help='Contrast local variation (default: 0)')
    our_augmentation_group.add_argument(
        '--illuminant_range',
        nargs='+',
        type=float,
        default=None,
        help='Lower illuminant limit (default: None)')
    our_augmentation_group.add_argument(
        '--local_illuminant_variation',
        type=float,
        default=0,
        help='Illuminant local variation (default: 0)')
    our_augmentation_group.add_argument(
        '--gaussian_sigma',
        nargs='+',
        type=float,
        default=None,
        help='Gaussian blurring upper limit (default: None)')
    our_augmentation_group.add_argument(
        '--s_p_amount',
        nargs='+',
        type=float,
        default=None,
        help='Salt&pepper upper limit (default: None)')
    our_augmentation_group.add_argument(
        '--gaussian_amount',
        nargs='+',
        type=float,
        default=None,
        help='Gaussian noise upper limit (default: None)')
    our_augmentation_group.add_argument(
        '--speckle_amount',
        nargs='+',
        type=float,
        default=None,
        help='Speckle noise upper limit (default: None)')
    our_augmentation_group.add_argument(
        '--gamma_range',
        nargs='+',
        type=float,
        default=None,
        help='Gamma lower and upper limits (default: None)')
    our_augmentation_group.add_argument(
        '--poisson_noise',
        action='store_true',
        default=False,
        help='Poisson noise (default: False)')
    our_augmentation_group.add_argument(
        '--mask_radius',
        nargs='+',
        type=float,
        default=None,
        help='Augmentation within this radius (default: None)')
    our_augmentation_group.add_argument(
        '--chromatic_contrast',
        nargs='+',
        type=float,
        default=None,
        help='Chromatic contrast lower limit (default: None)')
    our_augmentation_group.add_argument(
        '--luminance_contrast',
        nargs='+',
        type=float,
        default=None,
        help='Luminance contrast lower limit (default: None)')
    our_augmentation_group.add_argument(
        '--red_green',
        nargs='+',
        type=float,
        default=None,
        help='List of red-green to be evaluated (default: None)')
    our_augmentation_group.add_argument(
        '--yellow_blue',
        nargs='+',
        type=float,
        default=None,
        help='List of yellow-blue to be evaluated (default: None)')

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

    return check_training_args(parser, argvs)


def check_args(parser, argvs, script_type):
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

    # setting task type
    args.task_type = check_task_type(args.dataset, args.task_type)

    # setting the target size
    if args.target_size is None:
        args.target_size = get_default_target_size(args.dataset)
    else:
        args.target_size = (args.target_size, args.target_size)
    # check the input shape
    args.input_shape = get_input_shape(args.target_size)

    # setting the default top_k
    if args.top_k is None:
        if args.dataset == 'imagenet':
            args.top_k = 5

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

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in args.gpus)
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
        args.data_dir
    )

    return args


def check_training_args(parser, argvs):
    args = check_args(parser, argvs, 'training')

    # checking augmentation parameters
    augmentation_types = get_augmentation_types(args)
    if args.num_augmentation is not None:
        # there should be at least one sort of augmentation in this case
        if not augmentation_types:
            sys.exit(
                'When num_augmentation flag is used, '
                'at least one sort of augmentation should be specified')
        else:
            args.augmentation_types = np.array(augmentation_types)
    elif len(augmentation_types) > 0:
        sys.exit(
            'When one sort of augmentation is used '
            'num_augmentation flag must be specified')
    return args


def get_augmentation_types(args):
    # TODO make them one variable with name and make sure they're two
    # elements
    augmentation_types = []
    if args.illuminant_range is not None:
        args.illuminant_range = np.array(args.illuminant_range)
        augmentation_types.append('illuminant')
    if args.contrast_range is not None:
        args.contrast_range = np.array(args.contrast_range)
        augmentation_types.append('contrast')
    if args.gaussian_sigma is not None:
        args.gaussian_sigma = np.array(args.gaussian_sigma)
        augmentation_types.append('blur')
    if args.s_p_amount is not None:
        args.s_p_amount = np.array(args.s_p_amount)
        augmentation_types.append('s_p')
    if args.gaussian_amount is not None:
        args.gaussian_amount = np.array(args.gaussian_amount)
        augmentation_types.append('gaussian')
    if args.speckle_amount is not None:
        args.speckle_amount = np.array(args.speckle_amount)
        augmentation_types.append('speckle')
    if args.gamma_range is not None:
        args.gamma_range = np.array(args.gamma_range)
        augmentation_types.append('gamma')
    if args.poisson_noise is True:
        augmentation_types.append('poisson')
    if args.chromatic_contrast is not None:
        args.chromatic_contrast = np.array(args.chromatic_contrast)
        augmentation_types.append('chromatic_contrast')
    if args.luminance_contrast is not None:
        args.luminance_contrast = np.array(args.luminance_contrast)
        augmentation_types.append('luminance_contrast')
    if args.yellow_blue is not None:
        args.yellow_blue = np.array(args.yellow_blue)
        augmentation_types.append('yellow_blue')
    if args.red_green is not None:
        args.red_green = np.array(args.red_green)
        augmentation_types.append('red_green')
    return augmentation_types


def check_task_type(dataset, task_type=None):
    if ('cifar' in dataset or 'stl' in dataset or 'wcs' in dataset or
            dataset in ['imagenet', 'leaf', 'fruits']):
        if task_type is not None and task_type != 'classification':
            warnings.warn(
                'Invalid task_type %s: %s only supports classification' %
                (task_type, dataset)
            )
        task_type = 'classification'
    elif 'coco' in dataset:
        # TODO: add other tasks as well
        task_type = 'detection'
    return task_type
