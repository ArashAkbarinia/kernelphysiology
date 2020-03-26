"""
Handling input arguments for training/testing a network.
"""

import os
import sys
import argparse
import warnings

from kernelphysiology.dl.utils import argument_groups
from kernelphysiology.utils.controls import isfloat
from kernelphysiology.dl.utils import default_configs


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

    argument_groups.get_colour_space_group(parser)
    return parser


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


def common_train_arg_parser(description='Training a network!'):
    parser = common_arg_parser(description)

    argument_groups.get_architecture_group(parser)
    argument_groups.get_optimisation_group(parser)

    return parser


def set_visible_gpus(gpus):
    if gpus[0] == -1 or gpus is None:
        gpus = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in gpus)
    return gpus


def parse_args_nested(parser, argvs):
    """parsing the arguments according to their groups"""
    args = parser.parse_args(argvs)

    arg_groups = dict()
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    args = argparse.Namespace(**arg_groups)
    return args


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
