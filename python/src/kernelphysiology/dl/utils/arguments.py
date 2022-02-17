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
from kernelphysiology.utils import system_utils


class ArgumentParser(argparse.ArgumentParser):
    def add_argument_group(self, *args, **kwargs):
        ignore = ['positional arguments', 'optional arguments']
        if args[0] in ignore or ('title' in kwargs.keys() and kwargs['title'] in ignore):
            return super().add_argument_group(*args, **kwargs)
        for group in self._action_groups:
            if group.title == args[0] or ('title' in kwargs and group.title == kwargs['title']):
                return group
        return super().add_argument_group(*args, **kwargs)


def common_arg_parser(description):
    parser = ArgumentParser(description=description)

    argument_groups.get_dataset_group(parser)
    argument_groups.get_network_group(parser)
    argument_groups.get_output_group(parser)
    argument_groups.get_input_group(parser)
    argument_groups.get_logging_group(parser)
    argument_groups.get_routine_group(parser)

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

    argument_groups.get_optimisation_group(parser)

    return parser


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

    args.gpus = system_utils.set_visible_gpus(args.gpus)

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
    if len(parameters) == 0:
        if manipulation is None:
            manipulation = '_nothing'
            parameters = {
                'function': supported_functions['original'],
                'kwargs': {manipulation: [0]},
                'f_name': 'original'
            }
        else:
            sys.exit('Manipulation %s requires parameters.' % manipulation)
    else:
        if len(parameters) > 1:
            for i in range(1, len(parameters)):
                param_i = parameters[i]
                for kwarg_key, kwarg_val in param_i['kwargs'].items():
                    if len(kwarg_val) > 1:
                        f_name = param_i['f_name']
                        sys.exit(
                            'Only one manipulation is supported [%s %s].' % (f_name, kwarg_key)
                        )
                    parameters[i]['kwargs'][kwarg_key] = kwarg_val[0]
            other_params = parameters[1:]
            parameters = parameters[0]
            parameters['others'] = other_params
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
            sys.exit('Manipulation %s not found in parameters.' % manipulation)
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
            dataset in ['imagenet', 'leaves', 'fruits']):
        if task_type is not None and task_type != 'classification':
            warnings.warn(
                'Invalid task_type %s: %s only supports classification' %
                (task_type, dataset)
            )
        task_type = 'classification'
    elif 'voc' in dataset:
        task_type = 'segmentation'
    return task_type
