"""

"""

import os
import numpy as np
import glob
import random
import ntpath

from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from skimage import io
import cv2
import scipy.stats as ss

from . import cv2_transforms


def _normal_dist_ints(max_diff, scale=3):
    diffs = np.arange(-max_diff + 1, max_diff)
    x_u, x_l = diffs + 0.5, diffs - 0.5
    probs = ss.norm.cdf(x_u, scale=scale) - ss.norm.cdf(x_l, scale=scale)

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
        self.muns_diffs, self.muns_probs = _normal_dist_ints(self.munsells[1] - 1)

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


class ShapeDataset(torch_data.Dataset):
    def __init__(self, root, transform=None, background=None, same_rotation=None, **kwargs):
        self.root = root
        self.transform = transform
        self.target_size = (224, 224)
        self.mask_size = (128, 128)
        self.imgdir = '%s/shape2D/' % self.root
        self.bg = background
        self.same_rotation = same_rotation

    def _prepare_out_imgs(self, masks, others_colour, target_colour, place_fun):
        imgs = []
        for mask_ind, mask in enumerate(masks):
            mask = cv2.resize(mask, self.mask_size, interpolation=cv2.INTER_NEAREST)
            current_colour = target_colour if mask_ind == 0 else others_colour
            if os.path.exists(self.bg):
                bg_img = io.imread(self.bg)
                mask_img = cv2.resize(bg_img, self.mask_size, interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(bg_img, self.target_size, interpolation=cv2.INTER_NEAREST)
            elif self.bg == 'rnd':
                mask_img = np.random.randint(0, 256, (*self.mask_size, 3), dtype='uint8')
                img = np.random.randint(0, 256, (*self.target_size, 3), dtype='uint8')
            else:
                mask_img = np.zeros((*self.mask_size, 3), dtype='uint8') + int(self.bg)
                img = np.zeros((*self.target_size, 3), dtype='uint8') + int(self.bg)
            # converting images to float
            mask_img = mask_img.astype('float32') / 255
            img = img.astype('float32') / 255

            for chn_ind in range(3):
                current_chn = mask_img[:, :, chn_ind]
                current_chn[mask == 255] = current_colour[chn_ind]

            srow, scol = place_fun(self.mask_size, self.target_size)
            erow = srow + self.mask_size[0]
            ecol = scol + self.mask_size[1]
            img[srow:erow, scol:ecol] = mask_img
            imgs.append(img)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs


class ShapeTrain(ShapeDataset):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeDataset.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 'rnd'
        if self.same_rotation is None:
            self.same_rotation = False
        self.angles = (1, 11)
        self.rgb_diffs, self.rgb_probs = _normal_dist_ints(25, scale=5)
        self.img_paths = sorted(glob.glob(self.imgdir + '*.png'))
        self.colour_dist = colour_dist
        if self.colour_dist is not None:
            self.colour_dist = np.loadtxt(self.colour_dist, delimiter=',', dtype=int)

    def _prepare_train_imgs(self, masks, others_colour, target_colour):
        others_colour = np.array(others_colour).astype('float32') / 255
        target_colour = np.array(target_colour).astype('float32') / 255
        imgs = self._prepare_out_imgs(masks, others_colour, target_colour, _random_place)
        return imgs

    def _get_target_colour(self):
        if self.colour_dist is not None:
            rand_row = random.randint(0, len(self.colour_dist) - 1)
            target_colour = self.colour_dist[rand_row]
        else:
            target_colour = [random.randint(0, 255) for _ in range(3)]
        return target_colour

    def _get_others_colour(self, target_colour):
        others_colour = []
        # others_diff = np.random.choice(self.rgb_diffs, size=3, p=self.rgb_probs)
        others_diff = [random.choice([1, -1]) * random.randint(1, 128) for _ in range(3)]
        for chn_ind in range(3):
            chn_colour = target_colour[chn_ind] + others_diff[chn_ind]
            if chn_colour < 0 or chn_colour > 255:
                chn_colour = target_colour[chn_ind] - others_diff[chn_ind]
            others_colour.append(chn_colour)
        return others_colour

    def _prepare_angle_paths(self, path, samples):
        angle = int(ntpath.basename(path[:-4]).split('_')[-1].replace('angle', ''))
        ang_pool = np.arange(*self.angles).tolist()
        ang_pool.remove(angle)
        random.shuffle(ang_pool)
        org_angle = 'angle%d.png' % angle
        angle_paths = [path.replace(org_angle, 'angle%d.png' % ang_pool[i]) for i in range(samples)]
        return angle_paths

    def __len__(self):
        return len(self.img_paths)


class ShapeVal(ShapeDataset):

    def __init__(self, root, transform=None, target_colour=None, others_colour=None, **kwargs):
        ShapeDataset.__init__(self, root, transform=transform, **kwargs)
        if self.bg is None:
            self.bg = 128
        if self.same_rotation is None:
            self.same_rotation = True
        stimuli_path = '%s/validation.cvs' % self.root
        self.stimuli = np.loadtxt(stimuli_path, delimiter=',', dtype=int)
        self.target_colour = target_colour
        self.others_colour = others_colour

    def _prepare_test_imgs(self, masks):
        others_colour = self.others_colour.squeeze()
        target_colour = self.target_colour.squeeze()
        imgs = self._prepare_out_imgs(masks, others_colour, target_colour, _centre_place)
        return imgs

    def __len__(self):
        return len(self.stimuli)


class ShapeOddOneOutTrain(ShapeTrain):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, colour_dist=colour_dist, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        if self.same_rotation:
            other_paths = [target_path, target_path, target_path]
        else:
            other_paths = self._prepare_angle_paths(target_path, 3)

        masks = [io.imread(target_path), *[io.imread(opath) for opath in other_paths]]

        # set the colours
        target_colour = self._get_target_colour()
        others_colour = self._get_others_colour(target_colour)

        imgs = self._prepare_train_imgs(masks, others_colour, target_colour)

        inds = np.arange(0, self.num_stimuli).tolist()
        random.shuffle(inds)
        # the target is always added the first element in the imgs list
        target = inds.index(0)
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target


class ShapeOddOneOutVal(ShapeVal):

    def __init__(self, root, transform=None, **kwargs):
        ShapeVal.__init__(self, root, transform=transform, **kwargs)
        self.num_stimuli = 4

    def __getitem__(self, item):
        # image names start from 1
        imgi = item + 1
        base_path = '%s/img_shape%d_angle' % (self.imgdir, imgi)
        target_path = '%s%d.png' % (base_path, self.stimuli[item, 0])
        if self.same_rotation:
            other_paths = [target_path, target_path, target_path]
        else:
            other_paths = ['%s%d.png' % (base_path, self.stimuli[item, i]) for i in range(3)]
        masks = [io.imread(target_path), *[io.imread(opath) for opath in other_paths]]

        imgs = self._prepare_test_imgs(masks)

        # the target is always added the first element in the imgs list
        target = self.stimuli[item, -1]
        inds = np.arange(0, self.num_stimuli).tolist()
        tmp_img = imgs[target]
        imgs[target] = imgs[0]
        imgs[0] = tmp_img
        return imgs[inds[0]], imgs[inds[1]], imgs[inds[2]], imgs[inds[3]], target


class Shape2AFCTrain(ShapeTrain):

    def __init__(self, root, transform=None, colour_dist=None, **kwargs):
        ShapeTrain.__init__(self, root, transform=transform, colour_dist=colour_dist, **kwargs)

    def __getitem__(self, item):
        target_path = self.img_paths[item]
        if self.same_rotation:
            other_paths = target_path
        else:
            other_paths = self._prepare_angle_paths(target_path, 1)

        masks = [io.imread(target_path), io.imread(other_paths)]

        # set the colours
        target_colour = self._get_target_colour()
        if random.random() < 0.5:
            target = 1
            others_colour = target_colour
        else:
            target = 0
            others_colour = self._get_others_colour(target_colour)

        imgs = self._prepare_train_imgs(masks, others_colour, target_colour)

        return imgs[0], imgs[1], target


class Shape2AFCVal(ShapeVal):

    def __init__(self, root, transform=None, **kwargs):
        ShapeVal.__init__(self, root, transform=transform, **kwargs)

    def __getitem__(self, item):
        # image names start from 1
        imgi = item + 1
        target_path = '%s/img_shape%d_angle%d.png' % (self.imgdir, imgi, self.stimuli[item, 0])
        if self.same_rotation:
            other_paths = target_path
        else:
            other_paths = '%s/img_shape%d_angle%d.png' % (self.imgdir, imgi, self.stimuli[item, 1])
        masks = [io.imread(target_path), io.imread(other_paths)]

        imgs = self._prepare_test_imgs(masks)

        # target doesn't have a meaning in this test, it's always False
        target = 0
        return imgs[0], imgs[1], target


def _centre_place(mask_size, target_size):
    srow = int((target_size[0] - mask_size[0]) / 2)
    scol = int((target_size[1] - mask_size[1]) / 2)

    return srow, scol


def _random_place(mask_size, target_size):
    srow = random.randint(0, target_size[0] - mask_size[0])
    scol = random.randint(0, target_size[1] - mask_size[1])

    return srow, scol


def train_set(root, target_size, preprocess, task, **kwargs):
    mean, std = preprocess

    scale = (0.8, 1.0)
    transform = torch_transforms.Compose([
        cv2_transforms.RandomResizedCrop(target_size, scale=scale),
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    # return OddOneOutTrain(root, transform, **kwargs)
    if task == 'odd4':
        return ShapeOddOneOutTrain(root, transform, **kwargs)
    else:
        return Shape2AFCTrain(root, transform, **kwargs)


def val_set(root, target_size, preprocess, task, **kwargs):
    mean, std = preprocess

    transform = torch_transforms.Compose([
        cv2_transforms.CenterCrop(target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    # return OddOneOutVal(root, transform, **kwargs)
    if task == 'odd4':
        return ShapeOddOneOutVal(root, transform, **kwargs)
    else:
        return Shape2AFCVal(root, transform, **kwargs)
