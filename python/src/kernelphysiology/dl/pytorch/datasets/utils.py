"""
Loading different datasets for Pytorch.
"""

import numpy as np
import random
import sys

from skimage.util import random_noise

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_validation_dataset(dataset_name, valdir, colour_transformations,
                           other_transformations, chns_transformation,
                           normalize, target_size=224):
    if dataset_name == 'imagenet':
        validation_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(target_size),
                *colour_transformations,
                *other_transformations,
                transforms.ToTensor(),
                *chns_transformation,
                normalize,
            ])
        )
    elif 'wcs_lms' in dataset_name:
        data_loader_validation = lambda x: npy_data_loader(x, False, False)

        validation_dataset = datasets.DatasetFolder(
            valdir,
            data_loader_validation,
            ['.npy'],
            transforms.Compose([
                *chns_transformation,
                normalize,
            ])
        )
    elif 'wcs_jpg' in dataset_name:
        validation_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                *colour_transformations,
                *other_transformations,
                transforms.ToTensor(),
                *chns_transformation,
                normalize,
            ])
        )
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)
    return validation_dataset


def get_train_dataset(dataset_name, traindir, colour_transformations,
                      other_transformations, chns_transformation, normalize):
    if dataset_name == 'imagenet':
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                *colour_transformations,
                *other_transformations,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                *chns_transformation,
                normalize,
            ])
        )
    elif 'wcs_lms' in dataset_name:
        data_loader_train = lambda x: npy_data_loader(x, True, False)

        train_dataset = datasets.DatasetFolder(
            traindir,
            data_loader_train,
            ['.npy'],
            transforms.Compose([
                # TODO: consider other transformation
                *chns_transformation,
                normalize,
            ])
        )
    elif 'wcs_jpg' in dataset_name:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                *colour_transformations,
                *other_transformations,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                *chns_transformation,
                normalize,
            ])
        )
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)
    return train_dataset


def get_default_target_size(dataset_name):
    if dataset_name == 'imagenet':
        target_size = 224
    elif 'wcs_lms' in dataset_name:
        target_size = 128
    elif 'wcs_jpg' in dataset_name:
        target_size = 128
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)
    return target_size


def npy_data_loader(input_path, random_flip=False, gaussian_noise=False):
    lms_image = np.load(input_path).astype(np.float32)
    lms_image = lms_image.transpose([2, 0, 1])
    if random_flip and bool(random.getrandbits(1)):
        lms_image = lms_image[:, ::-1, :].copy()
    if gaussian_noise and bool(random.getrandbits(1)):
        lms_image /= lms_image.max()
        lms_image = random_noise(lms_image, mode='gaussian', var=0.1)
    lms_image = torch.from_numpy(lms_image)
    lms_image = lms_image.type(torch.FloatTensor)
    return lms_image
