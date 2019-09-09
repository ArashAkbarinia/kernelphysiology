"""
Loading different datasets for Pytorch.
"""

import numpy as np
import random
import sys

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset

from kernelphysiology.dl.pytorch.datasets import label_augmentation


def prepare_transformations_train(dataset_name, colour_transformations,
                                  other_transformations, chns_transformation,
                                  normalize, target_size):
    if 'cifar' in dataset_name or dataset_name == 'imagenet':
        if 'cifar' in dataset_name:
            size_transform = transforms.RandomCrop(target_size, padding=4)
        else:
            scale = (0.08, 1.0)
            size_transform = transforms.RandomResizedCrop(
                target_size, scale=scale
            )
        transformations = transforms.Compose([
            size_transform,
            *colour_transformations,
            *other_transformations,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_lms' in dataset_name:
        # FIXME: colour transformation in lms is different from rgb or lab
        transformations = transforms.Compose([
            *other_transformations,
            RandomHorizontalFlip(),
            Numpy2Tensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_jpg' in dataset_name:
        transformations = transforms.Compose([
            *colour_transformations,
            *other_transformations,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    else:
        sys.exit(
            'Transformations for dataset %s is not supported.' % dataset_name
        )
    return transformations


def prepare_transformations_test(dataset_name, colour_transformations,
                                 other_transformations, chns_transformation,
                                 normalize, target_size):
    if 'cifar' in dataset_name or dataset_name == 'imagenet':
        transformations = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            *colour_transformations,
            *other_transformations,
            transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_lms' in dataset_name:
        # FIXME: colour transformation in lms is different from rgb or lab
        transformations = transforms.Compose([
            *other_transformations,
            Numpy2Tensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_jpg' in dataset_name:
        transformations = transforms.Compose([
            *colour_transformations,
            *other_transformations,
            transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    else:
        sys.exit(
            'Transformations for dataset %s is not supported.' % dataset_name
        )
    return transformations


def get_validation_dataset(dataset_name, valdir, colour_transformations,
                           other_transformations, chns_transformation,
                           normalize, target_size):
    transformations = prepare_transformations_test(
        dataset_name, colour_transformations,
        other_transformations, chns_transformation,
        normalize, target_size
    )
    if dataset_name == 'imagenet':
        validation_dataset = datasets.ImageFolder(
            valdir, transformations
        )
    elif dataset_name == 'cifar10':
        validation_dataset = datasets.CIFAR10(
            valdir, train=False, download=False, transform=transformations
        )
    elif dataset_name == 'cifar100':
        validation_dataset = datasets.CIFAR100(
            valdir, train=False, download=False, transform=transformations
        )
    elif 'wcs_lms' in dataset_name:
        # FIXME: colour transformation in lms is different from rgb or lab
        data_loader_validation = lambda x: npy_data_loader(x)

        validation_dataset = datasets.DatasetFolder(
            valdir, data_loader_validation, ['.npy'], transformations
        )
    elif 'wcs_jpg' in dataset_name:
        validation_dataset = datasets.ImageFolder(
            valdir, transformations
        )
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)
    return validation_dataset


# TODO: train and validation merge together
def get_train_dataset(dataset_name, traindir, colour_transformations,
                      other_transformations, chns_transformation,
                      normalize, target_size, augment_labels=False):
    transformations = prepare_transformations_train(
        dataset_name, colour_transformations,
        other_transformations, chns_transformation,
        normalize, target_size
    )
    if dataset_name == 'imagenet':
        train_dataset = datasets.ImageFolder(
            traindir, transformations
        )
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            traindir, train=True, download=False, transform=transformations
        )
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(
            traindir, train=True, download=False, transform=transformations
        )
    elif 'wcs_lms' in dataset_name:
        data_loader_train = lambda x: npy_data_loader(x)

        train_dataset = datasets.DatasetFolder(
            traindir, data_loader_train, ['.npy'], transformations
        )
    elif 'wcs_jpg' in dataset_name:
        train_dataset = datasets.ImageFolder(
            traindir, transformations
        )
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)

    if augment_labels:
        # for label augmentation, we don't want to perform crazy cropping
        augmented_transformations = prepare_transformations_test(
            dataset_name, colour_transformations,
            other_transformations, chns_transformation,
            normalize, target_size
        )
        if dataset_name == 'imagenet':
            augmented_dataset = label_augmentation.AugmentedLabelDataset(
                traindir, augmented_transformations
            )
        elif dataset_name == 'cifar10':
            augmented_dataset = label_augmentation.AugmentedLabelArray(
                train_dataset.data, train_dataset.targets,
                augmented_transformations
            )
        elif dataset_name == 'cifar100':
            augmented_dataset = label_augmentation.AugmentedLabelArray(
                train_dataset.data, train_dataset.targets,
                augmented_transformations
            )
        else:
            sys.exit('Augmented dataset %s is not supported.' % dataset_name)

        train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    return train_dataset


def npy_data_loader(input_path):
    lms_image = np.load(input_path).astype(np.float32)
    return lms_image


def is_dataset_pil_image(dataset_name):
    if 'wcs_lms' in dataset_name:
        return False
    else:
        return True


class Numpy2Tensor(object):
    """Converting a numpy array to a tensor image.
    """

    def __call__(self, img):
        """
        Args:
            img (Numpy matrix): matrix to be converted to tensor.

        Returns:
            Pytorch Tensor: colour channels are stored in axis=0
        """
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        return img

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlip(object):
    """Horizontally flip the given matrix randomly with a given probability.

    Args:
        p (float): probability of the matrix being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Numpy matrix): matrix to be flipped.

        Returns:
            Numpy matrix: Randomly flipped matrix.
        """
        if random.random() < self.p:
            return img[:, ::-1, ].copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given matrix randomly with a given probability.

    Args:
        p (float): probability of the matrix being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Numpy matrix): matrix to be flipped.

        Returns:
            Numpy matrix: Randomly flipped matrix.
        """
        if random.random() < self.p:
            return img[::-1, ].copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)