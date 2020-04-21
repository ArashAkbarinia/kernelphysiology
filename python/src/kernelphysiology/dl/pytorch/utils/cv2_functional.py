from __future__ import division
import torch
import cv2
import numpy as np
import numbers
import collections

INTER_MODE = {
    'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR,
    'BICUBIC': cv2.INTER_CUBIC
}
PAD_MOD = {
    'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_DEFAULT, 'symmetric': cv2.BORDER_REFLECT
}


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor,
         (H x W x C[RGB]).

    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic):
        if len(pic.shape) == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 1:
            return img.float().div(255)
        else:
            return img
    elif _is_tensor_image(pic):
        return pic

    else:
        try:
            return to_tensor(np.array(pic))
        except Exception:
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if _is_tensor_image(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean)
                ) / np.array(std)
    else:
        raise RuntimeError('Undefined type')


def resize(img, size, interpolation='BILINEAR'):
    """Resize the input CV Image to the given size.

    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
         (h, w), the output size will be matched to this. If size is an int,
         the smaller edge of the image will be matched to this number  maintaing
         the aspect ratio. i.e, if height > width, then image will be rescaled
         to (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is
         ``BILINEAR``

    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (
            isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh),
                              interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh),
                              interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(
            img, dsize=(int(ow), int(oh)),
            interpolation=INTER_MODE[interpolation]
        )


def crop(img, x, y, h, w):
    """Crop the given CV Image.

    Args:
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(
        type(img))
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x + h), round(y + w)

    try:
        check_point1 = img[x1, y1, ...]
        check_point2 = img[x2 - 1, y2 - 1, ...]
    except IndexError:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0),
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) * 0.5))
    j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


def resized_crop(img, top, left, height, width, size, interpolation='BILINEAR'):
    """Crop the given CV2 Image and resize it to desired size. Notably used in
    RandomResizedCrop.

    Args:
        img (np.ndarray): Image to be cropped.
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (str, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        np.ndarray: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given CV2 Image.

    Args:
        img (np.ndarray): Image to be flipped.

    Returns:
        np.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV2 Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given CV2 Image.

    Args:
        img (CV2 Image): Image to be flipped.

    Returns:
        CV2 Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV2 Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)
