"""

"""

import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from kernelphysiology.dl.pytorch.datasets import utils_db


class MunsellNetDataset(Dataset):
    def __init__(self, data_dir, sub_type, transforms=None):
        self.is_pill_img = 'wcs_xyz_png_1600' in data_dir

        self.data_dir = '%s/%s/' % (data_dir, sub_type)
        if self.is_pill_img:
            self.inputs = glob.glob('%s/*.png' % self.data_dir)
            from torchvision.datasets.folder import pil_loader
            self.data_loader = pil_loader
        else:
            self.inputs = glob.glob('%s/*.npy' % self.data_dir)
            self.data_loader = utils_db.npy_data_loader
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

        img = self.data_loader(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

    def __len__(self):
        return len(self.inputs)


def get_train_val_dataset(data_dir, train_transformations, val_transformations,
                          normalize):
    is_pill_img = 'wcs_xyz_png_1600' in data_dir
    if is_pill_img:
        train_transforms = transforms.Compose([
            *train_transformations,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            *val_transformations,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = transforms.Compose([
            *train_transformations,
            utils_db.RandomHorizontalFlip(),
            utils_db.Numpy2Tensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            *val_transformations,
            utils_db.Numpy2Tensor(),
            normalize,
        ])

    train_dataset = MunsellNetDataset(data_dir, 'train', train_transforms)
    val_dataset = MunsellNetDataset(data_dir, 'val', val_transforms)

    # db_data = np.loadtxt(data_dir + '/ds.csv', delimiter=',', dtype='str')

    return train_dataset, val_dataset
