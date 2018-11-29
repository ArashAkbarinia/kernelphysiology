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
import warnings
import socket

import keras
from keras import backend as K
from keras import applications as kmodels

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train
from kernelphysiology.dl.keras.imagenet import imagenet_train

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet

from kernelphysiology.utils.imutils import adjust_contrast, gaussian_blur, adjust_illuminant


# FIXME unifying all augmentations here
def augmentation_preprocessing(img, contrast_range, local_contrast_variation=0, gaussian_sigma=None, preprocessing_function=None):
    img = adjust_contrast(img, np.random.uniform(*contrast_range), local_contrast_variation) * 255
    if gaussian_sigma is not None:
        win_size = (gaussian_sigma, gaussian_sigma)
        img = gaussian_blur(img, win_size) * 255
    if preprocessing_function is not None:
        img = preprocessing_function(img)
    return img


# FIXME: move all preprocessing to one function
def colour_constancy_augmented_preprocessing(img, illuminant_range, contrast_range, preprocessing_function=None):
    # FIXME: make the augmentations smarter: e.g. half normal, half crazy illumiant
    illuminant = np.random.uniform(*illuminant_range, 3)
    img = adjust_illuminant(img, illuminant) * 255
    img = adjust_contrast(img, np.random.uniform(*contrast_range)) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


def contrast_augmented_preprocessing(img, contrast_range, local_contrast_variation=0, preprocessing_function=None):
    img = adjust_contrast(img, np.random.uniform(*contrast_range), local_contrast_variation) * 255
    if preprocessing_function is not None:
        img = preprocessing_function(img)
    return img


def get_input_shape(target_size):
    # check the input shape
    if K.image_data_format() == 'channels_last':
        input_shape = (*target_size, 3)
    elif K.image_data_format() == 'channels_first':
        input_shape = (3, *target_size)
    return input_shape


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def test_prominent_prepares(args):
    output_file = None
    if os.path.isdir(args.network_name):
        dirname = args.network_name
        output_dir = os.path.join(dirname, args.experiment_name)
        create_dir(output_dir)
        output_file = os.path.join(output_dir, 'contrast_results')
        networks = sorted(glob.glob(dirname + '*.h5'))
        preprocessings = [args.preprocessing] * len(networks)
    elif os.path.isfile(args.network_name):
        networks = []
        preprocessings = []
        with open(args.network_name) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split(',')
                networks.append(tokens[0])
                preprocessings.append(tokens[1])
    else:
        networks = [args.network_name.lower()]
        preprocessings = [args.preprocessing]

    if not output_file:
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
        output_dir = args.experiment_name
        create_dir(output_dir)
        output_file = os.path.join(output_dir, 'contrast_results' + current_time)

    args.networks = networks
    args.preprocessings = preprocessings
    args.output_file = output_file

    return args


def get_preprocessing_function(preprocessing):
    # switch case of preprocessing functions
    if preprocessing == 'resnet50':
        preprocessing_function = kmodels.resnet50.preprocess_input
    elif preprocessing == 'inception_v3':
        preprocessing_function = kmodels.inception_v3.preprocess_input
    elif preprocessing == 'inception_resnet_v2':
        preprocessing_function = kmodels.inception_resnet_v2.preprocess_input
    elif preprocessing == 'xception':
        preprocessing_function = kmodels.xception.preprocess_input
    elif preprocessing == 'vgg16':
        preprocessing_function = kmodels.vgg16.preprocess_input
    elif preprocessing == 'vgg19':
        preprocessing_function = kmodels.vgg19.preprocess_input
    elif preprocessing == 'densenet121' or preprocessing == 'densenet169' or preprocessing == 'densenet201':
        preprocessing_function = kmodels.densenet.preprocess_input
    elif preprocessing == 'mobilenet':
        preprocessing_function = kmodels.mobilenet.preprocess_input
    elif preprocessing == 'mobilenet_v2':
        # FIXME: compatibility with version 2.2.0
        preprocessing_function = kmodels.mobilenetv2.preprocess_input
    elif preprocessing == 'nasnetmobile' or preprocessing == 'nasnetlarge':
        preprocessing_function = kmodels.nasnet.preprocess_input
    return preprocessing_function


def get_top_k_accuracy(k):
    top_k_acc = partial(keras.metrics.top_k_categorical_accuracy, k=k)
    top_k_acc.__name__ = 'top_%d_acc' % k
    return top_k_acc


def train_prominent_prepares(args):
    dataset_name = args.dataset.lower()
    network_name = args.network_name.lower()

    # choosing the preprocessing function
    if not args.preprocessing:
        args.preprocessing = network_name

    if args.illuminant_range is not None:
        contrast_range = np.array([args.contrast_range, 100]) / 100
#        local_contrast_variation = args.local_contrast_variation / 100
        illuminant_range = np.array([args.illuminant_range, 1])
        current_augmentation_preprocessing = lambda img: colour_constancy_augmented_preprocessing(img,
                                                                                illuminant_range=illuminant_range, contrast_range=contrast_range,
                                                                                preprocessing_function=get_preprocessing_function(args.preprocessing))
        args.train_preprocessing_function = current_augmentation_preprocessing
    else:
        args.train_preprocessing_function = get_preprocessing_function(args.preprocessing)
    # we don't want contrast augmentation for validation set
    args.validation_preprocessing_function = get_preprocessing_function(args.preprocessing)

    # which dataset
    args = which_dataset(args, dataset_name)

    if args.steps_per_epoch is None:
        args.steps_per_epoch = args.train_samples / args.batch_size
    if args.validation_steps is None:
        args.validation_steps = args.validation_samples / args.batch_size

    if args.load_weights is not None:
        # which network
        args = which_network(args, args.load_weights)
    else:
        # which architecture
        args.model = which_architecture(args)

    return args


def which_dataset(args, dataset_name):
    if dataset_name == 'cifar10':
        args = cifar_train.prepare_cifar10_generators(args)
    elif dataset_name == 'cifar100':
        args = cifar_train.prepare_cifar100_generators(args)
    elif dataset_name == 'stl10':
        args = stl_train.prepare_stl10_generators(args)
    elif dataset_name == 'imagenet':
        # TODO: this is not the nicest way to distinguish between train and validaiton
        if hasattr(args, 'train_preprocessing_function'):
            args = imagenet_train.prepare_imagenet(args)
        else:
            args = imagenet_train.validation_generator(args)
    return args


def which_network(args, network_name):
    # if passed by name we assume the original architectures
    # TODO: make the arguments nicer so in this case no preprocessing can be passed
    # TODO: very ugly work around for target size and input shape
    if network_name == 'resnet50':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.resnet50.ResNet50(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'inception_v3':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.inception_v3.InceptionV3(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'inception_resnet_v2':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.inception_resnet_v2.InceptionResNetV2(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'xception':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.xception.Xception(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'vgg16':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.vgg16.VGG16(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'vgg19':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.vgg19.VGG19(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet121':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet121(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet169':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet169(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet201':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet201(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'mobilenet':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.mobilenet.MobileNet(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'mobilenet_v2':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        # FIXME: compatibility with version 2.2.0
        args.model = kmodels.mobilenetv2.MobileNetV2(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'nasnetmobile':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.nasnet.NASNetMobile(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'nasnetlarge':
        args.target_size = (331, 331)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.nasnet.NASNetLarge(input_shape=args.input_shape, weights='imagenet')
    else:
        args.model = keras.models.load_model(network_name, compile=False)
    return args


def which_architecture(args):
    # TODO: add other architectures of keras
    network_name = args.network_name
    if network_name == 'resnet50':
        model = resnet50.ResNet50(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'inception_v3':
        model = inception_v3.InceptionV3(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg16':
        model = vgg16.VGG16(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg19':
        model = vgg19.VGG19(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet121':
        model = densenet.DenseNet121(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet169':
        model = densenet.DenseNet169(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet201':
        model = densenet.DenseNet201(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    return model


def common_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network_name', type=str, help='Which network to be used')

    parser.add_argument('--name', dest='experiment_name', type=str, default='Ex', help='The name of the experiment (default: Ex)')

    # TODO: this is just now for imagenet
    parser.add_argument('--train_dir', type=str, default=None, help='The path to the train directory (default: None)')
    parser.add_argument('--validation_dir', type=str, default=None, help='The path to the validation directory (default: None)')

    # TODO: make the argument list nicer according to test or train ...
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPUs to be used (default: [0])')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers for image generator (default: 1)')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--target_size', type=int, default=224, help='Target size (default: 224)')
    parser.add_argument('--crop_centre', action='store_true', default=False, help='Crop the image to its centre (default: False)')
    parser.add_argument('--preprocessing', type=str, default=None, help='The preprocessing function (default: network preprocessing function)')
    parser.add_argument('--top_k', type=int, default=5, help='Accuracy of top K elements (default: 5)')

    return parser


def activation_arg_parser(argvs):
    parser = common_arg_parser('Analysing activation of prominent nets of Keras.')

    parser.add_argument('--contrasts', nargs='+', type=float, default=[100], help='List of contrasts to be evaluated (default: [100])')

    return check_args(parser, argvs)


def test_arg_parser(argvs):
    parser = common_arg_parser('Test prominent nets of Keras for different contrasts.')

    image_degradation_group = parser.add_mutually_exclusive_group()
    image_degradation_group.add_argument('--contrasts', nargs='+', type=float, default=None, help='List of contrasts to be evaluated (default: None)')
    image_degradation_group.add_argument('--gaussian_sigma', nargs='+', type=float, default=None, help='List of Gaussian sigmas to be evaluated (default: None)')
    image_degradation_group.add_argument('--s_p_noise', nargs='+', type=float, default=None, help='List of salt and pepper noise to be evaluated (default: None)')
    image_degradation_group.add_argument('--uniform_noise', nargs='+', type=float, default=None, help='List of uniform noise to be evaluated (default: None)')
    image_degradation_group.add_argument('--gammas', nargs='+', type=float, default=None, help='List of gammas to be evaluated (default: None)')
    image_degradation_group.add_argument('--illuminants', nargs='+', type=float, default=None, help='List of illuminations to be evaluated (default: None)')

    return check_args(parser, argvs)


def train_arg_parser(argvs):
    parser = common_arg_parser('Training prominent nets of Keras.')

    # better handling the parameters, e.g. pretrained ones are only for imagenet
    parser.add_argument('--area1layers', type=int, default=None, help='The number of layers in area 1 (default: None)')

    parser.add_argument('--load_weights', type=str, default=None, help='Whether loading weights from a model (default: None)')

    parser.add_argument('--optimiser', type=str, default='adam', help='The optimiser to be used (default: adam)')
    parser.add_argument('--lr', type=float, default=None, help='The learning rate parameter of optimiser (default: None)')
    parser.add_argument('--decay', type=float, default=None, help='The decay weight parameter of optimiser (default: None)')
    parser.add_argument('--exp_decay', type=float, default=None, help='The exponential decay (default: None)')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--initial_epoch', type=int, default=0, help='The initial epoch number (default: 0)')
    parser.add_argument('--log_period', type=int, default=0, help='The period of logging the epochs weights (default: 0)')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='Number of steps per epochs (default: number of samples divided by the batch size)')
    parser.add_argument('--validation_steps', type=int, default=None, help='Number of steps for validations (default: number of samples divided by the batch size)')

    parser.add_argument('--noshuffle', dest='shuffle', action='store_false', default=True, help='Whether to stop shuffling data (default: False)')
    parser.add_argument('--horizontal_flip', action='store_true', default=False, help='Whether to perform horizontal flip data (default: False)')
    parser.add_argument('--vertical_flip', action='store_true', default=False, help='Whether to perform vertical flip (default: False)')
    parser.add_argument('--zoom_range', type=float, default=0, help='Value for zoom agumentation (default: 0)')
    parser.add_argument('--width_shift_range', type=float, default=0, help='Value for width shift agumentation (default: 0)')
    parser.add_argument('--height_shift_range', type=float, default=0, help='Value for height shift agumentation (default: 0)')

    parser.add_argument('--contrast_range', type=float, default=None, help='Value to perform contrast agumentation (default: None)')
    parser.add_argument('--local_contrast_variation', type=float, default=0, help='Value to deviate local contrast augmentation (default: 0)')
    parser.add_argument('--illuminant_range', type=float, default=None, help='Value to perform illumination agumentation (default: None)')
    parser.add_argument('--local_illuminant_variation', type=float, default=0, help='Value to deviate local illumination augmentation (default: 0)')
    parser.add_argument('--gaussian_sigma', type=float, default=None, help='Value to perform Gaussian blurring agumentation (default: None)')

    return check_args(parser, argvs)


def check_args(parser, argvs):
    # NOTE: this is just in order to get rid of EXIF warnigns
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    args = parser.parse_args(argvs)
    # TODO: more checking for GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(e) for e in args.gpus)

    args.target_size = (args.target_size, args.target_size)
    # check the input shape
    args.input_shape = get_input_shape(args.target_size)

    # workers
    if args.workers > 1:
        args.use_multiprocessing = True
    else:
        args.use_multiprocessing = False

    if args.dataset == 'imagenet':
        # TODO: just for the ease of working in my machiens
        if args.train_dir is None:
            args.train_dir = '/home/arash/Software/imagenet/raw-data/train/'
        if args.validation_dir is None:
            if socket.gethostname() == 'awesome':
                args.validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
            else:
                args.validation_dir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/raw-data/validation/'

    return args