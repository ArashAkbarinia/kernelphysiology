import numpy as np
import random

import cv2

import torch
from torch.utils import data as torch_data
from torchvision import datasets as tdatasets
import torchvision.transforms as torch_transforms

from kernelphysiology.utils import imutils
from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.filterfactory import gratings
from kernelphysiology.filterfactory import gaussian
from kernelphysiology.transformations import colour_spaces


def two_pairs_stimuli(img0, img1, contrast0, contrast1, p=0.5):
    imgs_cat = [img0, img1]
    max_contrast = np.argmax([contrast0, contrast1])
    if random.random() < p:
        contrast_target = 0
    else:
        contrast_target = 1
    if max_contrast != contrast_target:
        imgs_cat = imgs_cat[::-1]

    dim = 2
    grey_width = 40
    grey_cols = torch.zeros(
        (img0.shape[0], img0.shape[1], grey_width)
    ).type(img0.type())
    imgs_cat = [grey_cols, imgs_cat[0], grey_cols, imgs_cat[1], grey_cols]
    img_out = torch.cat(imgs_cat, dim)
    dim = 1
    grey_rows = torch.zeros(
        (img0.shape[0], grey_width, img_out.shape[2])
    ).type(img0.type())

    return torch.cat([grey_rows, img_out, grey_rows], dim), contrast_target


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, p=0.5, contrasts=None, same_transforms=False,
                 colour_space='grey', **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        img0 = self.loader(path)
        img0 = np.asarray(img0).copy()
        # TODO: just grey scale images
        img1 = img0.copy()

        if self.contrasts is None:
            contrast0 = random.uniform(0, 1)
            contrast1 = random.uniform(0, 1)
        else:
            contrast0, contrast1 = self.contrasts

        # converting to range 0 to 1
        img0 = img0.astype('float32') / 255
        img1 = img1.astype('float32') / 255

        if 'grey' in self.colour_space:
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            if self.colour_space == 'grey':
                img0 = np.expand_dims(img0, axis=2)
                img1 = np.expand_dims(img1, axis=2)
            elif self.colour_space == 'grey3':
                img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
                img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

        # manipulating the contrast
        img0 = imutils.adjust_contrast(img0, contrast0)
        img1 = imutils.adjust_contrast(img1, contrast1)

        if self.transform is not None:
            if self.same_transforms:
                img0, img1 = self.transform([img0, img1])
            else:
                [img0] = self.transform([img0])
                [img1] = self.transform([img1])

        img_out, contrast_target = two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return img_out, contrast_target, path


class GratingImages(torch_data.Dataset):
    def __init__(self, samples, target_size=(224, 224), p=0.5,
                 transform=None, colour_space='grey', contrast_space=None,
                 gabor_like=False,
                 contrasts=None, theta=None, rho=None, lambda_wave=None):
        super(GratingImages, self).__init__()
        if type(samples) is dict:
            # under this condition one contrast will be zero while the other
            # takes the arguments of samples.
            self.samples, self.settings = self._create_samples(samples)
        else:
            self.samples = samples
            self.settings = None
        if type(target_size) not in [list, tuple]:
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.p = p
        self.transform = transform
        self.colour_space = colour_space
        self.contrast_space = contrast_space
        self.contrasts = contrasts
        self.theta = theta
        self.rho = rho
        self.lambda_wave = lambda_wave
        self.gabor_like = gabor_like
        if self.gabor_like:
            self.gauss_img = gaussian.gaussian_kernel2(
                120 / (256 / target_size[0]), max_width=target_size[0]
            )
            self.gauss_img /= self.gauss_img.max()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        if self.settings is None:
            if self.contrasts is None:
                contrast0 = random.uniform(0, 1)
                contrast1 = random.uniform(0, 1)
            else:
                contrast0, contrast1 = self.contrasts

            # randomising the parameters
            if self.theta is None:
                theta = random.uniform(0, np.pi)
            else:
                theta = self.theta
            if self.rho is None:
                rho = random.uniform(0, np.pi)
            else:
                rho = self.rho
            if self.lambda_wave is None:
                lambda_wave = random.uniform(np.pi / 2, np.pi * 10)
            else:
                lambda_wave = self.lambda_wave
        else:
            inds = np.unravel_index(index, self.settings['lenghts'])
            contrast0 = self.settings['amp'][inds[0]]
            lambda_wave = self.settings['lambda_wave'][inds[1]]
            theta = self.settings['theta'][inds[2]]
            rho = self.settings['rho'][inds[3]]
            self.p = self.settings['side'][inds[4]]
            contrast1 = 0
        omega = [np.cos(theta), np.sin(theta)]

        # generating the gratings
        sinusoid_param = {
            'amp': contrast0, 'omega': omega, 'rho': rho,
            'img_size': self.target_size, 'lambda_wave': lambda_wave
        }
        img0 = (gratings.sinusoid(**sinusoid_param) + 1) / 2
        sinusoid_param['amp'] = contrast1
        img1 = (gratings.sinusoid(**sinusoid_param) + 1) / 2

        # multiply it by gaussian
        if self.gabor_like:
            img0 *= self.gauss_img
            img1 *= self.gauss_img

        # if target size is even, the generated stimuli is 1 pixel larger.
        if np.mod(self.target_size[0], 2) == 0:
            img0 = img0[:-1]
            img1 = img1[:-1]
        if np.mod(self.target_size[1], 2) == 0:
            img0 = img0[:, :-1]
            img1 = img1[:, :-1]

        if self.colour_space != 'grey':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
            if self.contrast_space == 'red':
                img0[:, :, [1, 2]] = 0.5
                img1[:, :, [1, 2]] = 0.5
            elif self.contrast_space == 'green':
                img0[:, :, [0, 2]] = 0.5
                img1[:, :, [0, 2]] = 0.5
            elif self.contrast_space == 'blue':
                img0[:, :, [0, 1]] = 0.5
                img1[:, :, [0, 1]] = 0.5
            elif self.contrast_space == 'rg':
                img0[:, :, [0, 1]] = 0.5
                img0 = colour_spaces.yog012rgb01(img0)
                img1[:, :, [0, 1]] = 0.5
                img1 = colour_spaces.yog012rgb01(img1)
            elif self.contrast_space == 'yb':
                img0[:, :, [0, 2]] = 0.5
                img0 = colour_spaces.yog012rgb01(img0)
                img1[:, :, [0, 2]] = 0.5
                img1 = colour_spaces.yog012rgb01(img1)

        if self.transform is not None:
            img0, img1 = self.transform([img0, img1])

        img_out, contrast_target = two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )

        item_settings = np.array([contrast0, lambda_wave, theta, rho, self.p])
        return img_out, contrast_target, item_settings

    def __len__(self):
        return self.samples

    def _create_samples(self, samples):
        settings = samples
        settings['lenghts'] = (
            len(settings['amp']), len(settings['lambda_wave']),
            len(settings['theta']), len(settings['rho']), len(settings['side'])
        )
        num_samples = np.prod(np.array(settings['lenghts']))
        return num_samples, settings


def train_set(db, target_size, mean, std, extra_transformation=None,
              natural_kwargs=None, gratings_kwargs=None):
    if extra_transformation is None:
        extra_transformation = []
    all_dbs = []
    shared_transforms = [
        *extra_transformation,
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ]
    if db is None or db == 'both':
        scale = (0.08, 1.0)
        size_transform = cv2_transforms.RandomResizedCrop(
            target_size, scale=scale
        )
        transforms = torch_transforms.Compose([
            size_transform, *shared_transforms
        ])
        all_dbs.append(
            ImageFolder(
                transform=transforms, **natural_kwargs
            ))
    if db in ['both', 'gratings']:
        transforms = torch_transforms.Compose(shared_transforms)
        all_dbs.append(
            GratingImages(
                transform=transforms, target_size=target_size, **gratings_kwargs
            )
        )
    return torch_data.ConcatDataset(all_dbs)


def validation_set(db, target_size, mean, std, extra_transformation=None,
                   natural_kwargs=None, gratings_kwargs=None):
    if extra_transformation is None:
        extra_transformation = []
    all_dbs = []
    shared_transforms = [
        *extra_transformation,
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ]
    if db is None or db == 'both':
        transforms = torch_transforms.Compose([
            cv2_transforms.Resize(target_size),
            cv2_transforms.CenterCrop(target_size),
            *shared_transforms
        ])
        all_dbs.append(
            ImageFolder(
                transform=transforms, **natural_kwargs
            ))
    if db in ['both', 'gratings']:
        transforms = torch_transforms.Compose(shared_transforms)
        all_dbs.append(
            GratingImages(
                transform=transforms, target_size=target_size, **gratings_kwargs
            )
        )
    return torch_data.ConcatDataset(all_dbs)
