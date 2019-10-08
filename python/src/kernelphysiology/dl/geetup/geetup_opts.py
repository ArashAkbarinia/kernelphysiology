"""
Options of the GEETUP.
"""

import os
import sys
import argparse


def check_args(parser, argv):
    args = parser.parse_args(argv)
    if args.random is not None:
        args.evaluate = True
    if args.evaluate:
        if args.validation_file is None:
            sys.exit('The validation file should be specified')
        if (args.architecture != 'centre' and
                not os.path.isfile(args.architecture)):
            sys.exit('Models weights most be specified.')
    else:
        if args.data_dir is None:
            sys.exit('The data dir should be specified')
        elif args.train_file is None:
            sys.exit('The training file should be specified')
    return args


def argument_parser():
    parser = argparse.ArgumentParser(description='GEETUP Train/Test')
    parser.add_argument(
        dest='architecture',
        type=str,
        help='Type of architecture to be used'
    )

    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--data_dir',
        type=str,
        help='Path to the data folder'
    )
    data_group.add_argument(
        '--train_file',
        type=str,
        help='Path to the training file'
    )
    data_group.add_argument(
        '--validation_file',
        type=str,
        help='Path to the validation file'
    )
    # This parameter is only for testing generalisation across frame gaps
    data_group.add_argument(
        '--frames_gap',
        type=int,
        help='Gaps between frames when reading the video (default: from data)'
    )
    data_group.add_argument(
        '--all_frames',
        action='store_true',
        default=False,
        help='Train and evaluate on all frames in a sequence (default: False)'
    )
    data_group.add_argument(
        '--target_size',
        nargs='+',
        type=int,
        default=[360, 640],
        help='Target size (default: [360, 640])'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='experiment_name',
        help='Name of the current experiment (default: experiment_name)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
        help='Only evaluation (default: False)'
    )
    parser.add_argument(
        '--random',
        nargs='+',
        type=int,
        default=None,
        help='Number of random images to try (default: None)'
    )

    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to the weights of a network'
    )
    architecture_group.add_argument(
        '--frame_based',
        action='store_true',
        default=False,
        help='Make the model frame based (default: False)'
    )

    optimisation_group = parser.add_argument_group('optimisation')
    optimisation_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=0.1,
        help='The learning rate parameter (default: 0.1)'
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
        default=0,
        help='The decay weight parameter (default: 0)'
    )
    optimisation_group.add_argument(
        '-e', '--epochs',
        type=int,
        default=90,
        help='Number of epochs (default: 90)'
    )
    optimisation_group.add_argument(
        '-ts', '--train_samples',
        type=int,
        default=None,
        help='Number of training samples per epoch (default: all)'
    )
    optimisation_group.add_argument(
        '-vs', '--validation_samples',
        type=int,
        default=None,
        help='Number of validation samples per epoch (default: all)'
    )
    optimisation_group.add_argument(
        '--initial_epoch',
        type=int,
        default=0,
        help='The initial epoch number (default: 0)'
    )

    process_group = parser.add_argument_group('process')
    process_group.add_argument(
        '-j', '--workers',
        type=int,
        default=1,
        help='Number of workers for image generator (default: 1)'
    )
    process_group.add_argument(
        '-b', '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: according to dataset)'
    )
    process_group.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[0],
        help='List of GPUs to be used (default: [0])'
    )
    process_group.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='Frequency of reporting (default: 100)'
    )
    return parser
