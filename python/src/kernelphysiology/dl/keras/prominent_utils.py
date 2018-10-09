'''
Utility functions for training prominent networks.
'''


import os
import glob
import argparse
import datetime
import time
import numpy as np
from functools import partial

import keras
from keras import backend as K

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train
from kernelphysiology.dl.keras.imagenet import imagenet_train

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet

from kernelphysiology.dl.keras.utils import contrast_generator


def test_prominent_prepares(args):
    args.target_size = (args.target_size, args.target_size)
    # check the input shape
    if K.image_data_format() == 'channels_last':
        args.input_shape = (*args.target_size, 3)
    elif K.image_data_format() == 'channels_first':
        args.input_shape = (3, *args.target_size)

    output_file = None
    if os.path.isdir(args.network):
        dirname = args.network
        output_file = os.path.join(dirname, 'contrast_results')
        networks = sorted(glob.glob(dirname + '*.h5'))
        preprocessings = [args.preprocessing] * len(networks)
    elif os.path.isfile(args.network):
        networks = []
        preprocessings = []
        with open(args.network) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split(',')
                networks.append(tokens[0])
                preprocessings.append(tokens[1])
    else:
        networks = args.network.lower()
        preprocessings = [args.preprocessing]

    if not output_file:
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
        output_file = 'contrast_results_' + current_time

    args.networks = networks
    args.preprocessings = preprocessings
    args.output_file = output_file

    return args


def get_preprocessing_function(preprocessing):
    # switch case of preprocessing functions
    if preprocessing == 'resnet50':
        preprocessing_function = resnet50.preprocess_input
    elif preprocessing == 'inception_v3':
        preprocessing_function = inception_v3.preprocess_input
    elif preprocessing == 'vgg16':
        preprocessing_function = vgg16.preprocess_input
    elif preprocessing == 'vgg19':
        preprocessing_function = vgg19.preprocess_input
    elif preprocessing == 'densenet121' or preprocessing == 'densenet169' or preprocessing == 'densenet201':
        preprocessing_function = densenet.preprocess_input
    return preprocessing_function


def get_top_k_accuracy(k):
    top_k_acc = partial(keras.metrics.top_k_categorical_accuracy, k=k)
    top_k_acc.__name__ = 'top_%d_acc' % k
    return top_k_acc


def train_prominent_prepares(args):
    dataset_name = args.dataset.lower()
    network_name = args.network.lower()

    args.target_size = (args.target_size, args.target_size)
    # check the input shape
    if K.image_data_format() == 'channels_last':
        args.input_shape = (*args.target_size, 3)
    elif K.image_data_format() == 'channels_first':
        args.input_shape = (3, *args.target_size)

    # choosing the preprocessing function
    if not args.preprocessing:
        args.preprocessing = network_name
    args.preprocessing_function = get_preprocessing_function(args.preprocessing)

    # which dataset
    if dataset_name == 'cifar10':
        args = cifar_train.prepare_cifar10_generators(args)
    elif dataset_name == 'cifar100':
        args = cifar_train.prepare_cifar100_generators(args)
    elif dataset_name == 'stl10':
        args = stl_train.prepare_stl10_generators(args)
    elif dataset_name == 'imagenet':
        # TODO: make the path as a parameter
        args.train_dir = '/home/arash/Software/imagenet/raw-data/train/'
        args.validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
        args = imagenet_train.prepare_imagenet(args)

        # FIXME: this is working only for imagenet nwo
        if args.contrast_aug:
            contrast_range = np.array([1, 100]) / 100
            if args.steps is None:
                args.steps = args.train_generator.samples / args.batch_size
            args.train_generator = contrast_generator(args.train_generator, contrast_range)

    # which architecture
    if network_name == 'resnet50':
        args.model = resnet50.ResNet50(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'inception_v3':
        args.model = inception_v3.InceptionV3(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg16':
        args.model = vgg16.VGG16(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg19':
        args.model = vgg19.VGG19(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet121':
        args.model = densenet.DenseNet121(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet169':
        args.model = densenet.DenseNet169(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet201':
        args.model = densenet.DenseNet201(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)

    return args


def common_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network', type=str, help='Which network to be used')

    parser.add_argument('--gpus', dest='gpus', nargs='+', type=int, default=[0], help='List of GPUs to be used (default: [0])')

    parser.add_argument('--batch_size', dest='batch_size', type=int, default=48, help='Batch size (default: 64)')
    parser.add_argument('--target_size', dest='target_size', type=int, default=224, help='Target size (default: 224)')
    parser.add_argument('--preprocessing', dest='preprocessing', type=str, default=None, help='The preprocessing function (default: network preprocessing function)')
    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Accuracy of top K elements (default: 5)')

    return parser


def test_arg_parser(argvs):
    parser = common_arg_parser('Test prominent nets of Keras for different contrasts.')

    parser.add_argument('--contrasts', dest='contrasts', nargs='+', type=int, default=[50, 100], help='List of contrasts to be evaluated (default: [50, 100])')

    return check_args(parser, argvs)


def train_arg_parser(argvs):
    parser = common_arg_parser('Training prominent nets of Keras.')

    # TODO: remove dest with identical names
    parser.add_argument('--area1layers', dest='area1layers', type=int, default=None, help='The number of layers in area 1 (default: 0)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--a1reduction', dest='area1_reduction', action='store_true', default=False, help='Whether to include a reduction layer in area 1 (default: False)')
    parser.add_argument('--a1dilation', dest='area1_dilation', action='store_true', default=False, help='Whether to include dilation in kernels in area 1 (default: False)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--train_contrast', dest='train_contrast', type=int, default=100, help='The level of contrast to be used at training (default: 100)')

    parser.add_argument('--name', dest='experiment_name', type=str, default='Ex', help='The name of the experiment (default: Ex)')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default=None, help='The path to a previous checkpoint to continue (default: None)')

    parser.add_argument('--optimiser', dest='optimiser', type=str, default='sgd', help='The optimiser to be used (default: sgd)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--steps', dest='steps', type=int, default=None, help='Number of steps per epochs (default: number of samples divided by the batch size)')

    parser.add_argument('--horizontal_flip', dest='horizontal_flip', action='store_true', default=False, help='Whether to perform horizontal flip data (default: False)')
    parser.add_argument('--vertical_flip', dest='vertical_flip', action='store_true', default=False, help='Whether to perform vertical flip (default: False)')
    parser.add_argument('--contrast_aug', action='store_true', default=False, help='Whether to perform contrast agumentation (default: False)')

    return check_args(parser, argvs)


def check_args(parser, argvs):
    args = parser.parse_args(argvs)
    # TODO: more checking for GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(e) for e in args.gpus)
    return args