from torchvision import datasets as tdatasets
import torchvision.transforms as torch_transforms

import numpy as np
import cv2

from kernelphysiology.dl.pytorch.utils import cv2_transforms


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float') / 255
    return img


def rgb2xopic(rgb, org_img):
    if org_img:
        light_con = -1
    else:
        light_con = np.random.randint(10)
    gray = np.mean(rgb, axis=2)
    xopic = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    if light_con == -1:
        # simulation of photopic
        xopic[:, :, :3] = rgb
    elif light_con < 3:
        # simulation of mesopic
        div_fac = np.random.uniform(1e2, 1e3)
        xopic[:, :, :3] = rgb / div_fac
        xopic[:, :, 3] = gray / div_fac
    else:
        # simulation of scotopic
        div_fac = np.random.uniform(1e3, 1e6)
        xopic[:, :, 3] = gray / div_fac
    return xopic


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, folder_kwargs):
        tdatasets.ImageFolder.__init__(self, **folder_kwargs)
        self.loader = cv2_loader

        self.real_num_imgs = len(self.samples)
        # duplicating the dataset for mesopic and scotopic
        self.samples = [*self.samples, *self.samples]
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # the mesopic ot scotopic simulation
        sample = rgb2xopic(sample, index < self.real_num_imgs)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_train_dataset(train_dir, target_size, preprocess):
    mean, std = preprocess
    normalise = cv2_transforms.Normalize(mean=mean, std=std)
    scale = (0.08, 1.0)
    size_transform = cv2_transforms.RandomResizedCrop(
        target_size, scale=scale
    )
    transform = torch_transforms.Compose([
        size_transform,
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        normalise,
    ])
    train_dataset = ImageFolder({'root': train_dir, 'transform': transform})
    return train_dataset


def get_val_dataset(val_dir, target_size, preprocess):
    mean, std = preprocess
    normalise = cv2_transforms.Normalize(mean=mean, std=std)
    transform = torch_transforms.Compose([
        cv2_transforms.Resize(target_size),
        cv2_transforms.CenterCrop(target_size),
        cv2_transforms.ToTensor(),
        normalise,
    ])
    train_dataset = ImageFolder({'root': val_dir, 'transform': transform})
    return train_dataset
