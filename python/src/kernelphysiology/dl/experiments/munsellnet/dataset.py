"""

"""

import numpy as np
import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from kernelphysiology.dl.pytorch.datasets import utils_db


class MunsellNetDataset(Dataset):
    def __init__(self, data_dir, sub_type, transforms=None):
        self.data_dir = '%s/%s/' % (data_dir, sub_type)
        self.inputs = glob.glob('%s/*.npy' % self.data_dir)
        self.targets = []
        for img in self.inputs:
            img_parsed = img.split('/')[-1].split('.')
            gt = [int(img_parsed[1]), int(img_parsed[2]), int(img_parsed[3])]
            self.targets.append(gt)
        self.targets = torch.tensor(self.targets)
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.inputs[index]
        targets = self.targets[index]

        img = utils_db.npy_data_loader(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

    def __len__(self):
        return len(self.inputs)


def get_train_val_dataset(data_dir, other_transformations, normalize):
    train_transforms = transforms.Compose([
        *other_transformations,
        utils_db.RandomHorizontalFlip(),
        utils_db.Numpy2Tensor(),
        normalize,
    ])
    train_dataset = MunsellNetDataset(data_dir, 'train', train_transforms)
    val_transforms = transforms.Compose([
        *other_transformations,
        utils_db.Numpy2Tensor(),
        normalize,
    ])
    val_dataset = MunsellNetDataset(data_dir, 'val', val_transforms)

    # db_data = np.loadtxt(data_dir + '/ds.csv', delimiter=',', dtype='str')

    return train_dataset, val_dataset
