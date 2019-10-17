"""
Transformations on tensors without going to CPU.
"""

import numpy as np
import warnings
import random
import math
from scipy import linalg

from PIL import Image

import torch
import torchvision
from torchvision.transforms.transforms import _pil_interpolation_to_str
from torchvision.transforms import functional as F

np_xyz_from_rgb = [[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]]
xyz_from_rgb = torch.tensor(np_xyz_from_rgb)

np_rgb_from_xyz = linalg.inv(np_xyz_from_rgb)
rgb_from_xyz = torch.tensor(np_rgb_from_xyz)

xyz_ref_white = (0.95047, 1., 1.08883)


def rgb2xyz(img_rgb):
    arr = img_rgb.clone()
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055).pow(2.4)
    arr[~mask] /= 12.92

    img_xyz = torch.empty(img_rgb.shape, dtype=torch.float)
    img_xyz = img_xyz.cuda(img_rgb.get_device(), non_blocking=False)
    for i in range(3):
        x_r = arr[:, 0:1, ] * xyz_from_rgb[i, 0]
        y_g = arr[:, 1:2, ] * xyz_from_rgb[i, 1]
        z_b = arr[:, 2:3, ] * xyz_from_rgb[i, 2]
        img_xyz[:, i:i + 1, ] = x_r + y_g + z_b

    return img_xyz


def xyz2rgb(img_xyz):
    arr = img_xyz.clone()
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    img_rgb = torch.empty(img_xyz.shape, dtype=torch.float)
    img_rgb = img_rgb.cuda(img_xyz.get_device(), non_blocking=False)
    for i in range(3):
        x_r = arr[:, 0:1, ] * rgb_from_xyz[i, 0]
        y_g = arr[:, 1:2, ] * rgb_from_xyz[i, 1]
        z_b = arr[:, 2:3, ] * rgb_from_xyz[i, 2]
        img_rgb[:, i:i + 1, ] = x_r + y_g + z_b

    mask = img_rgb > 0.0031308
    img_rgb[mask] = 1.055 * img_rgb[mask].pow(1 / 2.4) - 0.055
    img_rgb[~mask] *= 12.92
    img_rgb = img_rgb.clamp(0, 1)
    return img_rgb


def xyz2lab(img_xyz):
    arr = img_xyz.clone()

    # scale by CIE XYZ tristimulus values of the reference white point
    for i in range(3):
        arr[:, i:i + 1, ] /= xyz_ref_white[i]

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = arr[mask].pow(1 / 3)
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x = arr[:, 0:1, ]
    y = arr[:, 1:2, ]
    z = arr[:, 2:3, ]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    img_lab = torch.empty(img_xyz.shape, dtype=torch.float)
    img_lab = img_lab.cuda(img_xyz.get_device(), non_blocking=False)
    img_lab[:, 0:1, ] = L
    img_lab[:, 1:2, ] = a
    img_lab[:, 2:3, ] = b
    return img_lab


def lab2xyz(img_lab):
    arr = img_lab.clone()

    L = arr[:, 0:1, ]
    a = arr[:, 1:2, ]
    b = arr[:, 2:3, ]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    invalid = torch.nonzero(z < 0)
    if invalid.shape[0] > 0:
        warnings.warn(
            'Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
        z[invalid] = 0

    img_xyz = torch.empty(img_lab.shape, dtype=torch.float)
    img_xyz = img_xyz.cuda(img_lab.get_device(), non_blocking=False)
    img_xyz[:, 0:1, ] = x
    img_xyz[:, 1:2, ] = y
    img_xyz[:, 2:3, ] = z

    mask = img_xyz > 0.2068966
    img_xyz[mask] = img_xyz[mask].pow(3.)
    img_xyz[~mask] = (img_xyz[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    for i in range(3):
        img_xyz[:, i:i + 1, ] *= xyz_ref_white[i]
    return img_xyz


def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))


def lab2rgb(lab):
    return xyz2rgb(lab2xyz(lab))


def inverse_mean_std(mean, std):
    mean = np.array(mean)
    std = np.array(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    return mean_inv, std_inv


def normalize_inverse(tensor, mean, std, inplace=False):
    mean_inv, std_inv = inverse_mean_std(mean, std)
    return torchvision.transforms.functional.normalize(
        tensor, mean_inv, std_inv, inplace=inplace
    )


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input
    domain.
    """

    def __init__(self, mean, std):
        mean_inv, std_inv = inverse_mean_std(mean, std)
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is
    made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio
                           cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized images.
        """
        i, j, h, w = self.get_params(imgs[0][0], self.scale, self.ratio)
        out_imgs = []
        for img_list in imgs:
            inner_list = []
            for img in img_list:
                inner_list.append(
                    F.resized_crop(
                        img, i, j, h, w, self.size, self.interpolation
                    )
                )
            out_imgs.append(inner_list)
        return out_imgs

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be flipped.

        Returns:
            PIL Image: Randomly flipped images.
        """
        if random.random() < self.p:
            out_imgs = []
            for img_list in imgs:
                inner_list = []
                for img in img_list:
                    inner_list.append(F.hflip(img))
                out_imgs.append(inner_list)
            return out_imgs
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
