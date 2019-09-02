"""
Loading different datasets for Pytorch.
"""

import numpy as np
import random
import sys

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
        # FIXME: colour transformation in lms is different from rgb or lab
        data_loader_validation = lambda x: npy_data_loader(x)

        validation_dataset = datasets.DatasetFolder(
            valdir,
            data_loader_validation,
            ['.npy'],
            transforms.Compose([
                *other_transformations,
                Numpy2Tensor(),
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


# TODO: train and validation merge together
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
        data_loader_train = lambda x: npy_data_loader(x)

        train_dataset = datasets.DatasetFolder(
            traindir,
            data_loader_train,
            ['.npy'],
            transforms.Compose([
                *other_transformations,
                RandomHorizontalFlip(),
                Numpy2Tensor(),
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


def npy_data_loader(input_path):
    lms_image = np.load(input_path).astype(np.float32)
    return lms_image


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
