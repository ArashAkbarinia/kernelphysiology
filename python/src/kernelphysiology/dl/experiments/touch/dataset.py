import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.datasets.folder import pil_loader


class TouchDataset(Dataset):
    def __init__(self, img_dir, gt_dir, all_txt, test_inds, image_set,
                 transforms=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.all_imgs = np.loadtxt(all_txt, delimiter=',', dtype=int)
        self.inputs = []
        self.targets = []
        for img_info in self.all_imgs:
            img_name = str(img_info[0]) + '.png'
            if img_info[1] in test_inds and image_set == 'test':
                self.inputs.append(img_name)
                self.targets.append(img_name)
            elif img_info[1] not in test_inds and image_set == 'train':
                self.inputs.append(img_name)
                self.targets.append(img_name)
        print('set %s has %d images' % (image_set, len(self.inputs)))
        self.transforms = transforms
        self.data_loader = pil_loader

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.inputs[index])
        gt_path = os.path.join(self.gt_dir, self.targets[index])

        img = self.data_loader(img_path)
        gt = self.data_loader(gt_path)

        if self.transforms is not None:
            img, gt = self.transforms(img, gt)
        img = img[0].unsqueeze(0)
        gt = gt[0].unsqueeze(0)
        # if gt.max() > 0:
        #     gt /= gt.max()

        return img, gt

    def __len__(self):
        return len(self.inputs)


def get_train_dataset(img_dir, gt_dir, all_txt, test_inds,
                      trans_funcs, mean, std, target_size):
    train_transforms = Compose([
        *trans_funcs,
        RandomHorizontalFlip(),
        RandomCrop(target_size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    train_dataset = TouchDataset(
        img_dir, gt_dir, all_txt, test_inds, 'train', train_transforms
    )

    return train_dataset


def get_val_dataset(img_dir, gt_dir, all_txt, test_inds, trans_funcs, mean, std,
                    target_size):
    train_transforms = Compose([
        *trans_funcs,
        CenterCrop(target_size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    train_dataset = TouchDataset(
        img_dir, gt_dir, all_txt, test_inds, 'test', train_transforms
    )

    return train_dataset


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img
