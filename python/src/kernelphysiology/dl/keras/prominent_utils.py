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
from kernelphysiology.utils.imutils import reduce_chromaticity, reduce_lightness, reduce_yellow_blue, reduce_red_green
from kernelphysiology.utils.path_utils import create_dir

from kernelphysiology.dl.keras.datasets.utils import get_default_target_size, which_dataset
from kernelphysiology.dl.utils import default_configs

from kernelphysiology.dl.keras.models.utils import which_architecture, which_network
from kernelphysiology.dl.keras.models.utils import get_preprocessing_function

from kernelphysiology.dl.keras.utils import get_input_shape


def convert_to_uni8(img):
    img = img * 255
    return img.astype('uint8')


def augmented_preprocessing(img, augmentation_types=None, num_augmentation=0,
                            illuminant_range=None, illuminant_variation=0,
                            contrast_range=None, contrast_variation=0,
                            gaussian_sigma_range=None, salt_pepper_range=None,
                            gaussian_noise_range=None, poisson_range=False,
                            speckle_range=None, gamma_range=None,
                            chromatic_contrast=None, luminance_contrast=None,
                            yellow_blue=None, red_green=None,
                            mask_radius=None, preprocessing_function=None):
    if num_augmentation is None:
        order_augmentatoin = []
    elif num_augmentation == 0:
        colour_augmentatoin = np.array(['blur', 'illuminant', 'contrast', 'gamma'])
        noise_augmentatoin = np.array(['s_p', 'poisson', 'speckle', 'gaussian'])
        order_augmentatoin = [*colour_augmentatoin[np.random.randint(0, colour_augmentatoin.shape[0], size=1)],
                              *noise_augmentatoin[np.random.randint(0, noise_augmentatoin.shape[0], size=1)]]
    else:
        rand_inds = np.random.randint(0, augmentation_types.shape[0], size=num_augmentation)
        order_augmentatoin = augmentation_types[rand_inds]

    transformation_params = {}
    for aug_type in order_augmentatoin:
        if mask_radius is not None:
            mask_radius = np.sign(mask_radius) * np.random.uniform(0, abs(mask_radius))

        if aug_type == 'blur' and gaussian_sigma_range is not None:
            img = convert_to_uni8(gaussian_blur(img, np.random.uniform(*gaussian_sigma_range), mask_radius=mask_radius))
        elif aug_type == 'illuminant' and illuminant_range is not None:
            illuminant = np.random.uniform(*illuminant_range, 3)
            transformation_params['illuminant'] = illuminant
            img = convert_to_uni8(adjust_illuminant(img, illuminant, illuminant_variation, mask_radius=mask_radius))
        elif aug_type == 'contrast' and contrast_range is not None:
            img = convert_to_uni8(adjust_contrast(img, np.random.uniform(*contrast_range), contrast_variation, mask_radius=mask_radius))
        elif aug_type == 'gamma' and gamma_range is not None:
            img = convert_to_uni8(adjust_gamma(img, np.random.uniform(*gamma_range), mask_radius=mask_radius))
        elif aug_type == 's_p' and salt_pepper_range is not None:
            img = convert_to_uni8(s_p_noise(img, np.random.uniform(*salt_pepper_range), mask_radius=mask_radius))
        elif aug_type == 'poisson' and poisson_range:
            img = convert_to_uni8(poisson_noise(img), mask_radius=mask_radius)
        elif aug_type == 'speckle' and speckle_range is not None:
            img = convert_to_uni8(speckle_noise(img, np.random.uniform(*speckle_range), mask_radius=mask_radius))
        elif aug_type == 'gaussian' and gaussian_noise_range is not None:
            img = convert_to_uni8(gaussian_noise(img, np.random.uniform(*gaussian_noise_range), mask_radius=mask_radius))
        elif aug_type == 'chromatic_contrast' and chromatic_contrast is not None:
            img = convert_to_uni8(reduce_chromaticity(img, np.random.uniform(*chromatic_contrast), mask_radius=mask_radius))
        elif aug_type == 'luminance_contrast' and luminance_contrast is not None:
            img = convert_to_uni8(reduce_lightness(img, np.random.uniform(*luminance_contrast), mask_radius=mask_radius))
        elif aug_type == 'yellow_blue' and yellow_blue is not None:
            img = convert_to_uni8(reduce_yellow_blue(img, np.random.uniform(*yellow_blue), mask_radius=mask_radius))
        elif aug_type == 'red_green' and red_green is not None:
            img = convert_to_uni8(reduce_red_green(img, np.random.uniform(*red_green), mask_radius=mask_radius))

    if preprocessing_function is not None:
        img = preprocessing_function(img)
    return (img, transformation_params)


def test_prominent_prepares(args):
    output_file = None
    if os.path.isdir(args.network_name):
        dirname = args.network_name
        output_dir = os.path.join(dirname, args.experiment_name)
        create_dir(output_dir)
        output_file = os.path.join(output_dir, 'results_')
        networks = sorted(glob.glob(dirname + '*.h5'))
        network_names = []
        preprocessings = [args.preprocessing] * len(networks)
    elif os.path.isfile(args.network_name):
        networks = []
        preprocessings = []
        network_names = []
        with open(args.network_name) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split(',')
                networks.append(tokens[0])
                if len(tokens) > 1:
                    preprocessings.append(tokens[1])
                else:
                    preprocessings.append(args.preprocessing)
                # FIXME
                if len(tokens) > 2:
                    network_names.append(tokens[2])
    else:
        networks = [args.network_name.lower()]
        network_names = [args.network_name.lower()]
        # choosing the preprocessing function
        if not args.preprocessing:
            args.preprocessing = args.network_name.lower()
        preprocessings = [args.preprocessing]

    if not output_file:
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
        output_dir = args.experiment_name
        create_dir(output_dir)
        output_file = os.path.join(output_dir, 'results_' + current_time)

    args.networks = networks
    args.network_names = network_names
    args.preprocessings = preprocessings
    args.output_file = output_file

    return args


def prepare_train_augmentation(args):
    if args.num_augmentation is not None:
        # creating the augmentation lambda
        current_augmentation_preprocessing = lambda img: augmented_preprocessing(img, augmentation_types=args.augmentation_types, num_augmentation=args.num_augmentation,
                                                                                 illuminant_range=args.illuminant_range, illuminant_variation=args.local_illuminant_variation,
                                                                                 contrast_range=args.contrast_range, contrast_variation=args.local_contrast_variation,
                                                                                 gaussian_sigma_range=args.gaussian_sigma,
                                                                                 salt_pepper_range=args.s_p_amount,
                                                                                 gaussian_noise_range=args.gaussian_amount,
                                                                                 poisson_range=args.poisson_noise,
                                                                                 speckle_range=args.speckle_amount,
                                                                                 gamma_range=args.gamma_range,
                                                                                 chromatic_contrast=args.chromatic_contrast,
                                                                                 luminance_contrast=args.luminance_contrast,
                                                                                 red_green=args.red_green,
                                                                                 yellow_blue=args.yellow_blue,
                                                                                 mask_radius=args.mask_radius,
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
        args = which_network(args, args.load_weights, args.task_type)
    else:
        # which architecture
        args.model = which_architecture(args)

    return args