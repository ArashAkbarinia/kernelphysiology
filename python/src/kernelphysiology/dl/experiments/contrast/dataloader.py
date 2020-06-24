import numpy as np
import random

import torch
from torch.utils import data as torch_data
from torchvision import datasets as tdatasets
import torchvision.transforms as torch_transforms

from kernelphysiology.utils import imutils
from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.filterfactory import gratings


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
    grey_cols = torch.zeros((3, img0.shape[1], 40)).type(img0.type())
    imgs_cat = [grey_cols, imgs_cat[0], grey_cols, imgs_cat[1], grey_cols]
    img_out = torch.cat(imgs_cat, dim)
    dim = 1
    grey_rows = torch.zeros((3, 40, img_out.shape[2])).type(img0.type())
    return torch.cat([grey_rows, img_out, grey_rows], dim), contrast_target


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, p=0.5, contrasts=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.p = p
        self.contrasts = contrasts

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

        img0 = imutils.adjust_contrast(img0, contrast0)
        img1 = imutils.adjust_contrast(img1, contrast1)

        if self.transform is not None:
            img0, img1 = self.transform([img0, img1])

        img_out, contrast_target = two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )
        # print([contrast0, contrast1], max_contrast, contrast_target)

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return img_out, contrast_target, path


class GratingImages(torch_data.Dataset):
    def __init__(self, samples, p=0.5, contrasts=None, transform=None):
        super(GratingImages, self).__init__()
        self.samples = samples
        self.p = p
        self.contrasts = contrasts
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        if self.contrasts is None:
            contrast0 = random.uniform(0, 1)
            contrast1 = random.uniform(0, 1)
        else:
            contrast0, contrast1 = self.contrasts

        # randomising the parameters
        theta = random.uniform(0, np.pi)
        omega = [np.cos(theta), np.sin(theta)]
        rho = random.uniform(0, np.pi)
        lambda_wave = random.uniform(np.pi / 8, np.pi * 8)

        sinusoid_param = {
            'amp': contrast0, 'omega': omega, 'rho': rho,
            'img_size': (512, 512), 'lambda_wave': lambda_wave
        }
        img0 = (gratings.sinusoid(**sinusoid_param) + 1) / 2
        img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
        sinusoid_param['amp'] = contrast1
        img1 = (gratings.sinusoid(**sinusoid_param) + 1) / 2
        img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

        if self.transform is not None:
            img0, img1 = self.transform([img0, img1])

        img_out, contrast_target = two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )

        return img_out, contrast_target, "path"

    def __len__(self):
        return self.samples


def train_set(db, train_dir, target_size, mean, std):
    all_dbs = []
    scale = (0.08, 1.0)
    size_transform = cv2_transforms.RandomResizedCrop(
        target_size, scale=scale
    )
    transforms = torch_transforms.Compose([
        size_transform,
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])
    if db is None or db == 'both':
        all_dbs.append(ImageFolder(root=train_dir, transform=transforms))
    if db in ['both', 'gratings']:
        all_dbs.append(GratingImages(samples=10000, transform=transforms))
    return torch_data.ConcatDataset(all_dbs)


def validation_set(db, validation_dir, target_size, mean, std):
    all_dbs = []
    transforms = torch_transforms.Compose([
        cv2_transforms.Resize(target_size),
        cv2_transforms.CenterCrop(target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])
    if db is None or db == 'both':
        all_dbs.append(ImageFolder(root=validation_dir, transform=transforms))
    if db in ['both', 'gratings']:
        all_dbs.append(GratingImages(samples=1000, transform=transforms))
    return torch_data.ConcatDataset(all_dbs)
