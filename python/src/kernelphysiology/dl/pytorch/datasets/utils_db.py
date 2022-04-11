"""
Loading different datasets for Pytorch.
"""

import numpy as np
import random
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as torch_transforms

from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils import segmentation_utils
from kernelphysiology.dl.pytorch.datasets import data_loaders as custom_datasets

folder_dbs = ['imagenet', 'fruits', 'leaves', 'land', 'vggface2', 'ecoset']


def prepare_transformations_train(dataset_name, colour_transformations, other_transformations,
                                  chns_transformation, normalize, target_size, random_labels=False):
    if 'cifar' in dataset_name or dataset_name in folder_dbs:
        flip_p = 0.5
        if random_labels:
            size_transform = cv2_transforms.Resize(target_size)
            flip_p = -1
        elif 'cifar' in dataset_name:
            size_transform = cv2_transforms.RandomCrop(target_size, padding=4)
        elif 'imagenet' in dataset_name or 'ecoset' in dataset_name:
            scale = (0.08, 1.0)
            size_transform = cv2_transforms.RandomResizedCrop(target_size, scale=scale)
        else:
            scale = (0.50, 1.0)
            size_transform = cv2_transforms.RandomResizedCrop(target_size, scale=scale)
        transformations = torch_transforms.Compose([
            size_transform,
            *colour_transformations,
            *other_transformations,
            cv2_transforms.RandomHorizontalFlip(p=flip_p),
            cv2_transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_lms' in dataset_name:
        # FIXME: colour transformation in lms is different from rgb or lab
        transformations = torch_transforms.Compose([
            *other_transformations,
            RandomHorizontalFlip(),
            Numpy2Tensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_jpg' in dataset_name:
        transformations = torch_transforms.Compose([
            *colour_transformations,
            *other_transformations,
            cv2_transforms.RandomHorizontalFlip(),
            cv2_transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    else:
        sys.exit('Transformations for dataset %s is not supported.' % dataset_name)
    return transformations


def prepare_transformations_test(dataset_name, colour_transformations, other_transformations,
                                 chns_transformation, normalize, target_size, task=None):
    if 'cifar' in dataset_name or dataset_name in folder_dbs:
        transformations = torch_transforms.Compose([
            cv2_transforms.Resize(target_size),
            cv2_transforms.CenterCrop(target_size),
            *colour_transformations,
            *other_transformations,
            cv2_transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_lms' in dataset_name:
        # FIXME: colour transformation in lms is different from rgb or lab
        transformations = torch_transforms.Compose([
            *other_transformations,
            Numpy2Tensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'wcs_jpg' in dataset_name:
        transformations = torch_transforms.Compose([
            *colour_transformations,
            *other_transformations,
            cv2_transforms.ToTensor(),
            *chns_transformation,
            normalize,
        ])
    elif 'voc' in dataset_name or task == 'segmentation':
        transformations = []
    else:
        sys.exit('Transformations for dataset %s is not supported.' % dataset_name)
    return transformations


def get_validation_dataset(dataset_name, valdir, vision_type, colour_space, other_transformations,
                           normalize, target_size, task=None, target_transform=None):
    colour_transformations = preprocessing.colour_transformation(vision_type, colour_space)
    chns_transformation = preprocessing.channel_transformation(vision_type, colour_space)

    transformations = prepare_transformations_test(
        dataset_name, colour_transformations, other_transformations, chns_transformation,
        normalize, target_size, task=task
    )
    if task == 'segmentation' or 'voc' in dataset_name:
        # TODO: dataset shouldn't return num classes
        data_reading_kwargs = {
            'target_size': target_size,
            'colour_vision': vision_type,
            'colour_space': colour_space,
            'other_tf': other_transformations
        }
        validation_dataset, _ = segmentation_utils.get_dataset(
            dataset_name, valdir, 'val', **data_reading_kwargs
        )
    elif dataset_name in folder_dbs:
        validation_dataset = datasets.ImageFolder(
            valdir, transformations, loader=pil2numpy_loader, target_transform=target_transform
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
            valdir, data_loader_validation, ('.npy',), transformations
        )
    elif 'wcs_jpg' in dataset_name:
        validation_dataset = datasets.ImageFolder(valdir, transformations, loader=pil2numpy_loader)
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)
    return validation_dataset


# TODO: train and validation merge together
def get_train_dataset(dataset_name, traindir, vision_type, colour_space, other_transformations,
                      normalize, target_size, target_transform=None, random_labels=False):
    colour_transformations = preprocessing.colour_transformation(vision_type, colour_space)
    chns_transformation = preprocessing.channel_transformation(vision_type, colour_space)

    transformations = prepare_transformations_train(
        dataset_name, colour_transformations, other_transformations, chns_transformation,
        normalize, target_size, random_labels=random_labels
    )
    if dataset_name in folder_dbs:
        if random_labels:
            train_dataset = custom_datasets.RandomImageNet(
                traindir, transform=transformations, loader=pil2numpy_loader,
                target_transform=target_transform
            )
        else:
            train_dataset = datasets.ImageFolder(
                traindir, transformations, loader=pil2numpy_loader,
                target_transform=target_transform
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
            traindir, data_loader_train, ('.npy',), transformations
        )
    elif 'wcs_jpg' in dataset_name:
        train_dataset = datasets.ImageFolder(traindir, transformations, loader=pil2numpy_loader)
    else:
        sys.exit('Dataset %s is not supported.' % dataset_name)

    return train_dataset


def npy_data_loader(input_path):
    lms_image = np.float32(np.load(input_path))
    lms_image /= lms_image.max()
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


def pil2numpy_loader(path):
    img = datasets.folder.pil_loader(path)
    return np.asarray(img)


class ImagenetCategoryTransform(object):
    def __init__(self, category, cat_dir):
        if category is None:
            self.target_mapper = np.arange(1000)
        elif category == 'natural_manmade':
            self.target_mapper = np.zeros(1000, dtype=int)
            file_path = cat_dir + '/half_nat_idx.txt'
            nat_idx = np.loadtxt(file_path).astype('int')
            self.target_mapper[nat_idx] = 1

    def __call__(self, target):
        return self.target_mapper[target]
