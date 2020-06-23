import numpy as np
import random

import torch
from torchvision import datasets as tdatasets
import torchvision.transforms as torch_transforms

from kernelphysiology.utils import imutils
from kernelphysiology.dl.pytorch.utils import cv2_transforms


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        img_l = self.loader(path)
        img_l = np.asarray(img_l).copy()
        img_r = img_l.copy()

        contrast_l = random.uniform(0, 1)
        contrast_r = random.uniform(0, 1)

        img_l = imutils.adjust_contrast(img_l, contrast_l)
        img_r = imutils.adjust_contrast(img_r, contrast_r)

        if self.transform is not None:
            img_l, img_r = self.transform([img_l, img_r])

        dim = 2
        if random.randint(0, 1):
            img_out = torch.cat([img_l, img_r], dim)
            contrast_target = np.argmax([contrast_l, contrast_r])
            # print('LR', contrast_l, contrast_r)
        else:
            img_out = torch.cat([img_r, img_l], dim)
            contrast_target = np.argmax([contrast_r, contrast_l])
            # print('RL', contrast_r, contrast_l)

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return img_out, contrast_target, path


def train_set(train_dir, target_size, mean, std):
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
    return ImageFolder(root=train_dir, transform=transforms)


def validation_set(validation_dir, target_size, mean, std):
    transforms = torch_transforms.Compose([
        cv2_transforms.Resize(target_size),
        cv2_transforms.CenterCrop(target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])
    return ImageFolder(root=validation_dir, transform=transforms)
