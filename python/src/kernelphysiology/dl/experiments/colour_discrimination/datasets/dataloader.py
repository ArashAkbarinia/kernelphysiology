"""

"""

import numpy as np
import glob
import random
import ntpath

from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from skimage import io
import scipy.stats as ss

from . import cv2_transforms


def _normal_dist_munsell_int(max_diff):
    diffs = np.arange(-max_diff + 1, max_diff)
    x_u, x_l = diffs + 0.5, diffs - 0.5
    probs = ss.norm.cdf(x_u, scale=3) - ss.norm.cdf(x_l, scale=3)

    ind0 = np.where(diffs == 0)
    diffs = np.delete(diffs, ind0)
    probs = np.delete(probs, ind0)
    # normalise the probabilities so their sum is 1
    probs = probs / probs.sum()
    return diffs, probs


class OddOneOutTrain(torch_data.Dataset):

    def __init__(self, root, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.num_stimuli = 4
        img_info = np.loadtxt(self.root + 'img_info.txt', delimiter=',', dtype=str)
        self.munsells = (int(img_info[0]), int(img_info[1]) + 1)
        self.rotations = (int(img_info[2]), int(img_info[3]) + 1)
        self.objects = img_info[4:]
        self.imgdir = '%s/img/' % self.root
        self.img_paths = sorted(glob.glob(self.imgdir + '*.png'))
        self.muns_diffs, self.muns_probs = _normal_dist_munsell_int(self.munsells[1] - 1)

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        target_parts = ntpath.basename(target_path).split('_')

        # munsell pool
        munsell_ind = int(target_parts[0].replace('MunsellNo', ''))
        # muns_pool = np.arange(*self.munsells).tolist()
        # muns_pool.remove(munsell_ind)
        # random.shuffle(muns_pool)
        # same_munsells_ind = muns_pool[0]

        # selecting a munsell close to current munsell chip with a gaussian dist
        munsell_diff = np.random.choice(self.muns_diffs, size=1, p=self.muns_probs)[0]
        munsell_pool = munsell_ind + munsell_diff
        min_mun = self.munsells[0]
        max_mun = self.munsells[1] - 1
        if munsell_pool > max_mun:
            munsell_pool = munsell_pool - max_mun
        elif munsell_pool < min_mun:
            munsell_pool = max_mun + munsell_pool

        # rotation pool
        rots_pool = np.arange(*self.rotations).tolist()
        random.shuffle(rots_pool)

        identical_munsell_paths = [
            '%s/MunsellNo%d_rot%d_%s' % (self.imgdir, munsell_pool, rots_pool[0], target_parts[-1]),
            '%s/MunsellNo%d_rot%d_%s' % (self.imgdir, munsell_pool, rots_pool[1], target_parts[-1]),
            '%s/MunsellNo%d_rot%d_%s' % (self.imgdir, munsell_pool, rots_pool[2], target_parts[-1]),
        ]

        imgs = [
            io.imread(target_path),
            io.imread(identical_munsell_paths[0]),
            io.imread(identical_munsell_paths[1]),
            io.imread(identical_munsell_paths[2])
        ]

        if self.transform is not None:
            imgs = self.transform(imgs)

        inds = np.arange(0, self.num_stimuli).tolist()
        random.shuffle(inds)
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target

    def __len__(self):
        return len(self.img_paths)


class OddOneOutVal(torch_data.Dataset):

    def __init__(self, root, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.num_stimuli = 4
        data_file_path = '%s/simulationOrder_validation.csv' % self.root
        data_file = np.loadtxt(data_file_path, delimiter=',', dtype=str)
        # the first row are comments
        self.data_file = data_file[1:]
        self.imgdir = '%s/img' % (self.root)

    def __getitem__(self, item):
        current_test = self.data_file[item]
        imgs = [
            io.imread('%s/%s' % (self.imgdir, current_test[0])),
            io.imread('%s/%s' % (self.imgdir, current_test[1])),
            io.imread('%s/%s' % (self.imgdir, current_test[2])),
            io.imread('%s/%s' % (self.imgdir, current_test[3])),
        ]

        if self.transform is not None:
            imgs = self.transform(imgs)

        inds = np.roll(np.arange(4), np.mod(item, 4)).tolist()
        target = inds.index(0)
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target

    def __len__(self):
        return len(self.data_file)


def train_set(root, target_size, preprocess, **kwargs):
    mean, std = preprocess

    scale = (0.8, 1.0)
    transform = torch_transforms.Compose([
        cv2_transforms.RandomResizedCrop(target_size, scale=scale),
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    return OddOneOutTrain(root, transform, **kwargs)


def val_set(root, target_size, preprocess, **kwargs):
    mean, std = preprocess

    transform = torch_transforms.Compose([
        cv2_transforms.CenterCrop(target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    return OddOneOutVal(root, transform, **kwargs)
