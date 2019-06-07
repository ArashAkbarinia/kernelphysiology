"""
Options of the GEETUP.
"""

import sys
import argparse


def check_args(parser, argv):
    args = parser.parse_args(argv)
    if args.random is not None:
        args.evaluate = True
    if args.evaluate:
        if args.validation_file is None:
            sys.exit('The validation file should be specified')
        if args.weights is None:
            sys.exit('Models weights most be specified.')
    else:
        if args.train_file is None:
            sys.exit('The training file should be specified')
    return args


def argument_parser():
    parser = argparse.ArgumentParser(description='GEETUP Train/Test')
    parser.add_argument(
        dest='architecture',
        type=str,
        help='Type of architecture to be used')

    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--train_file',
        dest='train_file',
        type=str,
        help='Path to the training file')
    data_group.add_argument(
        '--validation_file',
        dest='validation_file',
        type=str,
        help='Path to the validation file')

    parser.add_argument(
        '--log_dir',
        dest='log_dir',
        type=str,
        default='Ex',
        help='Path to the logging directory (default: Ex)')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
        help='Only evaluation (default: False)')
    parser.add_argument(
        '--random',
        dest='random',
        nargs='+',
        type=int,
        default=None,
        help='Number of random images to try (default: None)')

    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=15,
        help='Number of epochs (default: 15)')

    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument(
        '--weights',
        dest='weights',
        type=str,
        default=None,
        help='Path to the weights of a network')
    architecture_group.add_argument(
        '--frame_based',
        action='store_true',
        default=False,
        help='Make the model frame based (default: False)')

    process_group = parser.add_argument_group('process')
    process_group.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        default=8,
        help='Batch size (default: 8)')
    process_group.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[0],
        help='List of GPUs to be used (default: [0])')
    return parser
