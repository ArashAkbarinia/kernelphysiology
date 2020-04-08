"""
Implementation of PyTorch transforms functions in a recursive fashion to support
more flexible type of inputs.
"""

import math
import random
import warnings

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import _pil_interpolation_to_str

from kernelphysiology.dl.pytorch.utils.transformations import Iterable


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

    def _call_recursive(self, imgs, i, j, h, w):
        if type(imgs) is list:
            inner_list = []
            for img in imgs:
                inner_list.append(self._call_recursive(img, i, j, h, w))
            return inner_list
        else:
            return F.resized_crop(
                imgs, i, j, h, w, self.size, self.interpolation
            )

    def _find_first_image_recursive(self, imgs):
        if type(imgs) is list:
            return self._find_first_image_recursive(imgs[0])
        else:
            return imgs

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized images.
        """
        i, j, h, w = self.get_params(
            self._find_first_image_recursive(imgs), self.scale, self.ratio
        )
        out_imgs = self._call_recursive(imgs, i, j, h, w)
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

    def _call_recursive(self, imgs):
        if type(imgs) is list:
            inner_list = []
            for img in imgs:
                inner_list.append(self._call_recursive(img))
            return inner_list
        else:
            return F.hflip(imgs)

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): List of images to be flipped.

        Returns:
            PIL Image: Randomly flipped images.
        """
        if random.random() < self.p:
            out_imgs = self._call_recursive(imgs)
            return out_imgs
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert (isinstance(size, int) or
                (isinstance(size, Iterable) and len(size) == 2))
        self.size = size
        self.interpolation = interpolation

    def _call_recursive(self, imgs):
        if type(imgs) is list:
            inner_list = []
            for img in imgs:
                inner_list.append(self._call_recursive(img))
            return inner_list
        else:
            return F.resize(imgs, self.size, self.interpolation)

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        out_imgs = self._call_recursive(imgs)
        return out_imgs

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str
        )
