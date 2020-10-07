"""
Supporting arguments for the image decomposition project.
"""

import argparse


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Image Decomposition')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument(
        '--model', type=str, default='single',
        choices=['single', 'multi', 'deeplabv3', 'unet', 'category'],
        help='type of model (default: single)'
    )
    model_parser.add_argument(
        '--target_size', type=int, default=256,
        help='input target size (default: 256)'
    )
    parser.add_argument(
        '--outputs', type=str, nargs='+', default=['input'],
        help='outputs (default: same as input)'
    )
    parser.add_argument(
        '--in_space', type=str, default='rgb',
        help='input colour space (default: rgb)'
    )
    model_parser.add_argument(
        '--hidden', type=int,
        help='number of hidden kernels in decoder/encoder'
    )
    model_parser.add_argument(
        '-k', '--k', type=int,
        help='number of vector in the embedding space'
    )
    model_parser.add_argument(
        '-d', '--d', type=int,
        help='length of vector in the embedding space'
    )
    model_parser.add_argument(
        '--lr', type=float, default=2e-4,
        help='learning rate (default: 2e-4)'
    )
    model_parser.add_argument(
        '--vq_coef', type=float, default=1.0,
        help='vq coefficient in loss (default: 1.0)'
    )
    model_parser.add_argument(
        '--commit_coef', type=float, default=0.5,
        help='commitment coefficient in loss (default 0.5)'
    )

    pipe_parser = parser.add_argument_group('Pipeline Parameters')
    pipe_parser.add_argument(
        '--batch_size', type=int, default=128,
        help='training batch size (default: 128)'
    )
    pipe_parser.add_argument(
        '-j', '--workers', type=int, default=4,
        help='number of workers for image generator (default: 4)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='the path to previous model to be resumed (default: None)'
    )
    parser.add_argument(
        '-ft', '--fine_tune', type=str, default=None,
        help='the path to weights to be fine-tuned (default: None)'
    )
    pipe_parser.add_argument(
        '--dataset', default='imagenet',
        choices=('imagenet', 'celeba', 'touch', 'voc'),
        help='dataset for train/validation'
    )
    pipe_parser.add_argument(
        '--data_dir', type=str, default=None,
        help='the path to dataset (default: None)'
    )
    pipe_parser.add_argument(
        '--epochs', type=int, default=30,
        help='number of training epochs (default: 30)'
    )
    pipe_parser.add_argument(
        '--start_epoch', type=int, default=0,
        help='the initial epoch (default: 0)'
    )
    pipe_parser.add_argument(
        '--train_samples', type=int, default=100000,
        help='max num of training samples per epoch (default: 100000)'
    )
    pipe_parser.add_argument(
        '--test_samples', type=int, default=1000,
        help='max num of testing samples per epoch (default: 1000)'
    )
    pipe_parser.add_argument(
        '--seed', type=int, default=1,
        help='random seed (default: 1)'
    )
    pipe_parser.add_argument(
        '--pred', type=str, default=None,
        help='Only prediction'
    )

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument(
        '--log_interval', type=int, default=10,
        help='how many batches to wait before logging training status'
    )
    logging_parser.add_argument(
        '--results_dir', default='./results',
        help='results dir (default: ./results)'
    )
    logging_parser.add_argument(
        '--experiment_name', default=None,
        help='experiment name (default: current time)'
    )

    return parser.parse_args(args)
