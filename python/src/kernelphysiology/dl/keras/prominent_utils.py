'''
Utility functions for training prominent networks.
'''


import os
import sys
import glob
import argparse
import datetime
import time
import numpy as np
import warnings
import math

from kernelphysiology.utils.imutils import adjust_contrast, gaussian_blur, adjust_illuminant, adjust_gamma
from kernelphysiology.utils.imutils import s_p_noise, gaussian_noise, speckle_noise, poisson_noise
from kernelphysiology.utils.path_utils import create_dir

from kernelphysiology.dl.keras.datasets.utils import get_default_dataset_paths, which_dataset

from kernelphysiology.dl.keras.models.utils import which_architecture, which_network
from kernelphysiology.dl.keras.models.utils import get_preprocessing_function

from kernelphysiology.dl.keras.utils import get_input_shape


def convert_to_uni8(img):
    img = img *255
    return img.astype('uint8')


def augmented_preprocessing(img, augmentation_types=None, num_augmentation=0,
                            illuminant_range=None, illuminant_variation=0,
                            contrast_range=None, contrast_variation=0,
                            gaussian_sigma_range=None, salt_pepper_range=None,
                            gaussian_noise_range=None, poisson_range=False,
                            speckle_range=None, gamma_range=None,
                            preprocessing_function=None):
    if num_augmentation is None or num_augmentation == 0:
        order_augmentatoin = ['blur', 'illuminant', 'contrast', 'gamma', 's_p', 'poisson', 'speckle', 'gaussian']
    else:
        rand_inds = np.random.randint(0, augmentation_types.shape[0], size=num_augmentation)
        order_augmentatoin = augmentation_types[rand_inds]

    for aug_type in order_augmentatoin:
        if aug_type == 'blur' and gaussian_sigma_range is not None:
            img = convert_to_uni8(gaussian_blur(img, np.random.uniform(*gaussian_sigma_range)))
        elif aug_type == 'illuminant' and illuminant_range is not None:
            illuminant = np.random.uniform(*illuminant_range, 3)
            img = convert_to_uni8(adjust_illuminant(img, illuminant, illuminant_variation))
        elif aug_type == 'contrast' and contrast_range is not None:
            img = convert_to_uni8(adjust_contrast(img, np.random.uniform(*contrast_range), contrast_variation))
        elif aug_type == 'gamma' and gamma_range is not None:
            img = convert_to_uni8(adjust_gamma(img, np.random.uniform(*gamma_range)))
        elif aug_type == 's_p' and salt_pepper_range is not None:
            img = convert_to_uni8(s_p_noise(img, np.random.uniform(*salt_pepper_range)))
        elif aug_type == 'poisson' and poisson_range:
            img = convert_to_uni8(poisson_noise(img))
        elif aug_type == 'speckle' and speckle_range is not None:
            img = convert_to_uni8(speckle_noise(img, np.random.uniform(*speckle_range)))
        elif aug_type == 'gaussian' and gaussian_noise_range is not None:
            img = convert_to_uni8(gaussian_noise(img, np.random.uniform(*gaussian_noise_range)))
    if preprocessing_function is not None:
        img = preprocessing_function(img)
    return img


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


def prepare_train_augmentation(args):
    if args.num_augmentation is not None:
        augmentation_types = []
        if args.illuminant_range is not None:
            illuminant_range = np.array([args.illuminant_range, 1])
            augmentation_types.append('iluuminant')
        else:
            illuminant_range = None
        if args.contrast_range is not None:
            contrast_range = np.array([args.contrast_range, 1])
            augmentation_types.append('contrast')
        else:
            contrast_range = None
        if args.gaussian_sigma is not None:
            gaussian_sigma_range = np.array([0, args.gaussian_sigma])
            augmentation_types.append('blur')
        else:
            gaussian_sigma_range = None
        if args.s_p_amount is not None:
            salt_pepper_range = np.array([0, args.s_p_amount])
            augmentation_types.append('s_p')
        else:
            salt_pepper_range = None
        if args.gaussian_amount is not None:
            gaussian_noise_range = np.array([0, args.gaussian_amount])
            augmentation_types.append('gaussian')
        else:
            gaussian_noise_range = None
        if args.speckle_amount is not None:
            speckle_range = np.array([0, args.speckle_amount])
            augmentation_types.append('speckle')
        else:
            speckle_range = None
        if args.gamma_range is not None:
            gamma_range = np.array(args.gamma_range[0:2])
            augmentation_types.append('gamma')
        else:
            gamma_range = None
        augmentation_types = np.array(augmentation_types)

        # creating the augmentation lambda
        current_augmentation_preprocessing = lambda img: augmented_preprocessing(img, augmentation_types=augmentation_types, num_augmentation=args.num_augmentation,
                                                                                 illuminant_range=illuminant_range, illuminant_variation=args.local_illuminant_variation,
                                                                                 contrast_range=contrast_range, contrast_variation=args.local_contrast_variation,
                                                                                 gaussian_sigma_range=gaussian_sigma_range,
                                                                                 salt_pepper_range=salt_pepper_range,
                                                                                 gaussian_noise_range=gaussian_noise_range,
                                                                                 poisson_range=args.poisson_noise,
                                                                                 speckle_range=speckle_range,
                                                                                 gamma_range=gamma_range,
                                                                                 preprocessing_function=get_preprocessing_function(args.preprocessing))
    else:
        current_augmentation_preprocessing = get_preprocessing_function(args.preprocessing)

    args.train_preprocessing_function = current_augmentation_preprocessing
    return args


def train_prominent_prepares(args):
    dataset_name = args.dataset.lower()
    network_name = args.network_name.lower()

    # choosing the preprocessing function
    if not args.preprocessing:
        args.preprocessing = network_name

    # which augmentation we're handling
    args = prepare_train_augmentation(args)

    # we don't want augmentation for validation set
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


def common_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network_name', type=str, help='Which network to be used')

    parser.add_argument('--name', dest='experiment_name', type=str, default='Ex', help='The name of the experiment (default: Ex)')

    data_dir_group = parser.add_argument_group('data path')
    data_dir_group.add_argument('--data_dir', type=str, default=None, help='The path to the data directory (default: None)')
    data_dir_group.add_argument('--train_dir', type=str, default=None, help='The path to the train directory (default: None)')
    data_dir_group.add_argument('--validation_dir', type=str, default=None, help='The path to the validation directory (default: None)')

    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPUs to be used (default: [0])')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers for image generator (default: 1)')

    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: according to dataset)')
    parser.add_argument('--target_size', type=int, default=None, help='Target size (default: according to dataset)')
    parser.add_argument('--crop_centre', action='store_true', default=False, help='Crop the image to its centre (default: False)')
    parser.add_argument('--preprocessing', type=str, default=None, help='The preprocessing function (default: network preprocessing function)')
    parser.add_argument('--top_k', type=int, default=5, help='Accuracy of top K elements (default: 5)')

    return parser


def activation_arg_parser(argvs):
    parser = common_arg_parser('Analysing activation of prominent nets of Keras.')

    parser.add_argument('--contrasts', nargs='+', type=float, default=[1], help='List of contrasts to be evaluated (default: [1])')

    return check_args(parser, argvs, 'activation')


def test_arg_parser(argvs):
    parser = common_arg_parser('Test prominent nets of Keras for different contrasts.')

    image_degradation_group = parser.add_mutually_exclusive_group()
    image_degradation_group.add_argument('--contrasts', nargs='+', type=float, default=None, help='List of contrasts to be evaluated (default: None)')
    image_degradation_group.add_argument('--gaussian_sigma', nargs='+', type=float, default=None, help='List of Gaussian sigmas to be evaluated (default: None)')
    image_degradation_group.add_argument('--s_p_noise', nargs='+', type=float, default=None, help='List of salt and pepper noise to be evaluated (default: None)')
    image_degradation_group.add_argument('--speckle_noise', nargs='+', type=float, default=None, help='List of speckle noise to be evaluated (default: None)')
    image_degradation_group.add_argument('--gaussian_noise', nargs='+', type=float, default=None, help='List of Gaussian noise to be evaluated (default: None)')
    image_degradation_group.add_argument('--poisson_noise', action='store_true', default=False, help='Poisson noise to be evaluated (default: False)')
    image_degradation_group.add_argument('--gammas', nargs='+', type=float, default=None, help='List of gammas to be evaluated (default: None)')
    image_degradation_group.add_argument('--illuminants', nargs='+', type=float, default=None, help='List of illuminations to be evaluated (default: None)')

    return check_args(parser, argvs, 'testing')


def train_arg_parser(argvs):
    parser = common_arg_parser('Training prominent nets of Keras.')

    # better handling the parameters, e.g. pretrained ones are only for imagenet
    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument('--area1layers', type=int, default=None, help='The number of layers in area 1 (default: None)')

    initialisation_group = parser.add_argument_group('initialisation')
    weights_group = initialisation_group.add_mutually_exclusive_group()
    weights_group.add_argument('--load_weights', type=str, default=None, help='Whether loading weights from a model (default: None)')
    weights_group.add_argument('--initialise', type=str, default=None, help='Whether using a specific initialisation of weights (default: None)')
    initialisation_group.add_argument('--tog_sigma', type=float, default=1, help='Sigma of ToG (default: 1)')
    initialisation_group.add_argument('--tog_surround', type=float, default=5, help='Surround enlargement in ToG (default: 5)')
    initialisation_group.add_argument('--g_sigmax', type=float, default=1, help='Sigma-x of Gaussian (default: 1)')
    initialisation_group.add_argument('--g_sigmay', type=float, default=None, help='Sigma-y of Gaussian (default: None)')
    initialisation_group.add_argument('--g_meanx', type=float, default=0, help='Mean-x of Gaussian (default: 0)')
    initialisation_group.add_argument('--g_meany', type=float, default=0, help='Mean-y of Gaussian (default: 0)')
    initialisation_group.add_argument('--g_theta', type=float, default=0, help='Theta of Gaussian (default: 0)')
    initialisation_group.add_argument('--gg_sigma', type=float, default=1, help='Sigma of Gaussian gradient (default: 1)')
    initialisation_group.add_argument('--gg_theta', type=float, default=math.pi/2, help='Theta of Gaussian gradient (default: 1)')
    initialisation_group.add_argument('--gg_seta', type=float, default=1, help='Seta of Gaussian gradient (default: 1)')

    optimisation_group = parser.add_argument_group('optimisation')
    optimisation_group.add_argument('--optimiser', type=str, default='adam', help='The optimiser to be used (default: adam)')
    optimisation_group.add_argument('--lr', type=float, default=None, help='The learning rate parameter (default: None)')
    optimisation_group.add_argument('--decay', type=float, default=None, help='The decay weight parameter (default: None)')
    optimisation_group.add_argument('--exp_decay', type=float, default=None, help='The exponential decay (default: None)')
    optimisation_group.add_argument('--lr_schedule', type=str, default=None, help='The custom learning rate scheduler (default: None)')
    optimisation_group.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    optimisation_group.add_argument('--initial_epoch', type=int, default=0, help='The initial epoch number (default: 0)')

    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument('--log_period', type=int, default=0, help='The period of logging the epochs weights (default: 0)')
    logging_group.add_argument('--steps_per_epoch', type=int, default=None, help='Number of steps per epochs (default: number of samples divided by the batch size)')
    logging_group.add_argument('--validation_steps', type=int, default=None, help='Number of steps for validations (default: number of samples divided by the batch size)')

    keras_augmentation_group = parser.add_argument_group('keras augmentation')
    keras_augmentation_group.add_argument('--noshuffle', dest='shuffle', action='store_false', default=True, help='Stop shuffling data (default: False)')
    keras_augmentation_group.add_argument('--horizontal_flip', action='store_true', default=False, help='Perform horizontal flip data (default: False)')
    keras_augmentation_group.add_argument('--vertical_flip', action='store_true', default=False, help='Perform vertical flip (default: False)')
    keras_augmentation_group.add_argument('--zoom_range', type=float, default=0, help='Range of zoom agumentation (default: 0)')
    keras_augmentation_group.add_argument('--width_shift_range', type=float, default=0, help='Range of width shift (default: 0)')
    keras_augmentation_group.add_argument('--height_shift_range', type=float, default=0, help='Range of height shift (default: 0)')

    our_augmentation_group = parser.add_argument_group('our augmentation')
    our_augmentation_group.add_argument('--num_augmentation', type=int, default=None, help='Number of types at each instance (default: None)')
    our_augmentation_group.add_argument('--contrast_range', type=float, default=None, help='Contrast lower limit (default: None)')
    our_augmentation_group.add_argument('--local_contrast_variation', type=float, default=0, help='Contrast local variation (default: 0)')
    our_augmentation_group.add_argument('--illuminant_range', type=float, default=None, help='Lower illuminant limit (default: None)')
    our_augmentation_group.add_argument('--local_illuminant_variation', type=float, default=0, help='Illuminant local variation (default: 0)')
    our_augmentation_group.add_argument('--gaussian_sigma', type=float, default=None, help='Gaussian blurring upper limit (default: None)')
    our_augmentation_group.add_argument('--s_p_amount', type=float, default=None, help='Salt&pepper upper limit (default: None)')
    our_augmentation_group.add_argument('--gaussian_amount', type=float, default=None, help='Gaussian noise upper limit (default: None)')
    our_augmentation_group.add_argument('--speckle_amount', type=float, default=None, help='Speckle noise upper limit (default: None)')
    our_augmentation_group.add_argument('--gamma_range', nargs='+', type=float, default=None, help='Gamma lower and upper limits (default: None)')
    our_augmentation_group.add_argument('--poisson_noise', action='store_true', default=False, help='Poisson noise (default: False)')

    return check_args(parser, argvs, 'training')


def check_args(parser, argvs, script_type):
    # NOTE: this is just in order to get rid of EXIF warnigns
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    args = parser.parse_args(argvs)

    args.script_type = script_type

    # setting the target size
    if args.target_size is None:
        if args.dataset == 'imagenet':
            args.target_size = 224
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'stl10':
            args.target_size = 32
        else:
            sys.exit('target_size is required for dataset %s' % (args.dataset))

    # setting the batch size
    if args.batch_size is None:
        if args.dataset == 'imagenet':
            if args.script_type == 'training':
                args.batch_size = 32
            if args.script_type == 'testing':
                args.batch_size = 64
            if args.script_type == 'activation':
                args.batch_size = 32
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'stl10':
            if args.script_type == 'training':
                args.batch_size = 256
            if args.script_type == 'testing':
                args.batch_size = 512
            if args.script_type == 'activation':
                args.batch_size = 256
        else:
            sys.exit('batch_size is required for dataset %s' % (args.dataset))

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

    # handling the paths
    args = get_default_dataset_paths(args)

    return args
