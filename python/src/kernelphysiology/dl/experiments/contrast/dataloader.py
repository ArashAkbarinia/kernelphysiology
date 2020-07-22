import numpy as np
import random
import os

import cv2

import torch
from torch.utils import data as torch_data
from torchvision import datasets as tdatasets
import torchvision.transforms as torch_transforms

from kernelphysiology.utils import imutils
from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.filterfactory import gratings
from kernelphysiology.transformations import colour_spaces


def _two_pairs_stimuli(img0, img1, contrast0, contrast1, p=0.5):
    imgs_cat = [img0, img1]
    max_contrast = np.argmax([contrast0, contrast1])
    if random.random() < p:
        contrast_target = 0
    else:
        contrast_target = 1
    if max_contrast != contrast_target:
        imgs_cat = imgs_cat[::-1]

    dim = 2
    grey_width = 40
    grey_cols = torch.zeros(
        (img0.shape[0], img0.shape[1], grey_width)
    ).type(img0.type())
    imgs_cat = [grey_cols, imgs_cat[0], grey_cols, imgs_cat[1], grey_cols]
    img_out = torch.cat(imgs_cat, dim)
    dim = 1
    grey_rows = torch.zeros(
        (img0.shape[0], grey_width, img_out.shape[2])
    ).type(img0.type())

    return torch.cat([grey_rows, img_out, grey_rows], dim), contrast_target


def _random_mask_params():
    mask_params = dict()
    mask_params['mask_type'] = random.choice(['circle', 'square'])
    mask_params['mask_length'] = random.random()
    return mask_params


def _prepare_stimuli(img0, colour_space, vision_type, contrasts, mask_image,
                     transform, same_transforms, p):
    if 'grey' not in colour_space and vision_type != 'trichromat':
        dkl0 = colour_spaces.rgb2dkl(img0)
        if vision_type == 'dichromat_rg':
            dkl0[:, :, 1] = 0
        elif vision_type == 'dichromat_yb':
            dkl0[:, :, 2] = 0
        img0 = colour_spaces.dkl2rgb(dkl0)

    img1 = img0.copy()

    if contrasts is None:
        contrast0 = random.uniform(0, 1)
        contrast1 = random.uniform(0, 1)
    else:
        contrast0, contrast1 = contrasts

    # converting to range 0 to 1
    img0 = img0.astype('float32') / 255
    img1 = img1.astype('float32') / 255

    if 'grey' in colour_space:
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if colour_space == 'grey':
            img0 = np.expand_dims(img0, axis=2)
            img1 = np.expand_dims(img1, axis=2)
        elif colour_space == 'grey3':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)

    # manipulating the contrast
    img0 = imutils.adjust_contrast(img0, contrast0)
    img1 = imutils.adjust_contrast(img1, contrast1)

    if mask_image == 'gaussian':
        img0 -= 0.5
        img0 = img0 * _get_gauss(img0.shape)
        img0 += 0.5
        img1 -= 0.5
        img1 = img1 * _get_gauss(img1.shape)
        img1 += 0.5
    elif mask_image == 'shapes':
        img0 = imutils.mask_image(img0, 0.5, **_random_mask_params())
        img1 = imutils.mask_image(img1, 0.5, **_random_mask_params())

    if transform is not None:
        if same_transforms:
            img0, img1 = transform([img0, img1])
        else:
            [img0] = transform([img0])
            [img1] = transform([img1])

    img_out, contrast_target = _two_pairs_stimuli(
        img0, img1, contrast0, contrast1, p
    )
    return img_out, contrast_target


def _get_gauss(img_size):
    midx = np.floor(img_size[1] / 2) + 1
    midy = np.floor(img_size[0] / 2) + 1
    y = np.linspace(img_size[0], 0, img_size[0]) - midy
    x = np.linspace(0, img_size[1], img_size[1]) - midx
    [x, y] = np.meshgrid(x, y)
    sigma = min(img_size[0], img_size[1]) / 4
    gauss_img = np.exp(
        -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))
    )

    gauss_img = gauss_img / np.max(gauss_img)
    if len(img_size) > 2:
        gauss_img = np.repeat(gauss_img[:, :, np.newaxis], img_size[2], axis=2)
    return gauss_img


class CelebA(tdatasets.CelebA):
    def __init__(self, p=0.5, contrasts=None, same_transforms=False,
                 colour_space='grey', vision_type='trichromat',
                 mask_image=None, **kwargs):
        super(CelebA, self).__init__(**kwargs)
        self.loader = tdatasets.folder.pil_loader
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space
        self.vision_type = vision_type
        self.mask_image = mask_image

    def __getitem__(self, index):
        path = os.path.join(
            self.root, self.base_folder, "img_align_celeba",
            self.filename[index]
        )
        img0 = self.loader(path)
        img0 = np.asarray(img0).copy()

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t)
                )
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts,
            self.mask_image, self.transform, self.same_transforms, self.p
        )

        return img_out, contrast_target, path


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, p=0.5, contrasts=None, same_transforms=False,
                 colour_space='grey', vision_type='trichromat',
                 mask_image=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.p = p
        self.contrasts = contrasts
        self.same_transforms = same_transforms
        self.colour_space = colour_space
        self.vision_type = vision_type
        self.mask_image = mask_image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        img0 = self.loader(path)
        img0 = np.asarray(img0).copy()
        img_out, contrast_target = _prepare_stimuli(
            img0, self.colour_space, self.vision_type, self.contrasts,
            self.mask_image, self.transform, self.same_transforms, self.p
        )

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return img_out, contrast_target, path


def _model_fest_stimuli(target_size, contrast, theta, rho, lambda_wave):
    midn = np.floor(target_size[0] / 2) + 1
    y = np.linspace(target_size[0], 0, target_size[0]) - midn
    x = np.linspace(0, target_size[0], target_size[0]) - midn
    [x, y] = np.meshgrid(x, y)
    grating = contrast * np.sin(
        2 * np.pi * (
                (x * np.cos(theta) + y * np.sin(theta)) / lambda_wave
        ) - rho
    )

    sigma = 60.1264858771449
    gauss_img = np.exp(
        -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))
    )

    gauss_img = gauss_img / np.max(gauss_img)
    img = gauss_img * grating
    return img


class GratingImages(torch_data.Dataset):
    def __init__(self, samples, target_size=(224, 224), p=0.5,
                 transform=None, colour_space='grey', contrast_space=None,
                 vision_type='trichromat', gabor_like='fixed_size',
                 contrasts=None, theta=None, rho=None, lambda_wave=None):
        super(GratingImages, self).__init__()
        if type(samples) is dict:
            # under this condition one contrast will be zero while the other
            # takes the arguments of samples.
            self.samples, self.settings = self._create_samples(samples)
        else:
            self.samples = samples
            self.settings = None
        if type(target_size) not in [list, tuple]:
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.p = p
        self.transform = transform
        self.colour_space = colour_space
        self.contrast_space = contrast_space
        self.vision_type = vision_type
        self.contrasts = contrasts
        self.theta = theta
        self.rho = rho
        self.lambda_wave = lambda_wave
        self.gabor_like = gabor_like

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_l, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        if self.settings is None:
            if self.contrasts is None:
                contrast0 = random.uniform(0, 1)
                contrast1 = random.uniform(0, 1)
            else:
                contrast0, contrast1 = self.contrasts

            # randomising the parameters
            if self.theta is None:
                theta = random.uniform(0, np.pi)
            else:
                theta = self.theta
            if self.rho is None:
                rho = random.uniform(0, np.pi)
            else:
                rho = self.rho
            if self.lambda_wave is None:
                lambda_wave = random.uniform(np.pi / 2, np.pi * 10)
            else:
                lambda_wave = self.lambda_wave
        else:
            inds = np.unravel_index(index, self.settings['lenghts'])
            contrast0 = self.settings['amp'][inds[0]]
            lambda_wave = self.settings['lambda_wave'][inds[1]]
            theta = self.settings['theta'][inds[2]]
            rho = self.settings['rho'][inds[3]]
            self.p = self.settings['side'][inds[4]]
            contrast1 = 0
        omega = [np.cos(theta), np.sin(theta)]

        if self.gabor_like == 'model_fest':
            img0 = _model_fest_stimuli(
                self.target_size, contrast0, theta, rho, lambda_wave
            )
            img1 = _model_fest_stimuli(
                self.target_size, contrast1, theta, rho, lambda_wave
            )
        else:
            # generating the gratings
            sinusoid_param = {
                'amp': contrast0, 'omega': omega, 'rho': rho,
                'img_size': self.target_size, 'lambda_wave': lambda_wave
            }
            img0 = gratings.sinusoid(**sinusoid_param)
            sinusoid_param['amp'] = contrast1
            img1 = gratings.sinusoid(**sinusoid_param)

            # multiply it by gaussian
            if self.gabor_like == 'fixed_size':
                radius = (
                    int(self.target_size[0] / 2.0),
                    int(self.target_size[1] / 2.0)
                )
                [x, y] = np.meshgrid(
                    range(-radius[0], radius[0] + 1),
                    range(-radius[1], radius[1] + 1)
                )
                x1 = +x * np.cos(theta) + y * np.sin(theta)
                y1 = -x * np.sin(theta) + y * np.cos(theta)

                k = 2
                o1 = 8
                o2 = o1 / 2
                omg = (1 / 8) * (np.pi ** 2 / lambda_wave)
                gauss_img = omg ** 2 / (o2 * np.pi * k ** 2) * np.exp(
                    -omg ** 2 / (o1 * k ** 2) * (1 * x1 ** 2 + y1 ** 2))

                gauss_img = gauss_img / np.max(gauss_img)
                img0 *= gauss_img
                img1 *= gauss_img
            elif self.gabor_like == 'fixed_cycle':
                radius = (
                    int(self.target_size[0] / 2.0),
                    int(self.target_size[1] / 2.0)
                )
                [x, y] = np.meshgrid(
                    range(-radius[0], radius[0] + 1),
                    range(-radius[1], radius[1] + 1)
                )

                sigma = self.target_size[0] / 4.25
                gauss_img = np.exp(
                    -(np.power(x, 2) + np.power(y, 2)) / (
                                2 * np.power(sigma, 2))
                )

                gauss_img = gauss_img / np.max(gauss_img)
                img0 *= gauss_img
                img1 *= gauss_img

        img0 = (img0 + 1) / 2
        img1 = (img1 + 1) / 2

        # if target size is even, the generated stimuli is 1 pixel larger.
        if np.mod(self.target_size[0], 2) == 0:
            img0 = img0[:-1]
            img1 = img1[:-1]
        if np.mod(self.target_size[1], 2) == 0:
            img0 = img0[:, :-1]
            img1 = img1[:, :-1]

        if self.colour_space != 'grey':
            img0 = np.repeat(img0[:, :, np.newaxis], 3, axis=2)
            img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
            if self.contrast_space == 'red':
                img0[:, :, [1, 2]] = 0.5
                img1[:, :, [1, 2]] = 0.5
            elif self.contrast_space == 'green':
                img0[:, :, [0, 2]] = 0.5
                img1[:, :, [0, 2]] = 0.5
            elif self.contrast_space == 'blue':
                img0[:, :, [0, 1]] = 0.5
                img1[:, :, [0, 1]] = 0.5
            elif self.contrast_space == 'yb':
                img0[:, :, [0, 1]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [0, 1]] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)
            elif self.contrast_space == 'rg':
                img0[:, :, [0, 2]] = 0.5
                img0 = colour_spaces.dkl012rgb01(img0)
                img1[:, :, [0, 2]] = 0.5
                img1 = colour_spaces.dkl012rgb01(img1)

        if 'grey' not in self.colour_space and self.vision_type != 'trichromat':
            dkl0 = colour_spaces.rgb2dkl(img0)
            dkl1 = colour_spaces.rgb2dkl(img1)
            if self.vision_type == 'dichromat_rg':
                dkl0[:, :, 1] = 0
                dkl1[:, :, 1] = 0
            elif self.vision_type == 'dichromat_yb':
                dkl0[:, :, 2] = 0
                dkl1[:, :, 2] = 0
            img0 = colour_spaces.dkl2rgb01(dkl0)
            img1 = colour_spaces.dkl2rgb01(dkl1)

        if self.transform is not None:
            img0, img1 = self.transform([img0, img1])

        img_out, contrast_target = _two_pairs_stimuli(
            img0, img1, contrast0, contrast1, self.p
        )

        item_settings = np.array([contrast0, lambda_wave, theta, rho, self.p])
        return img_out, contrast_target, item_settings

    def __len__(self):
        return self.samples

    def _create_samples(self, samples):
        settings = samples
        settings['lenghts'] = (
            len(settings['amp']), len(settings['lambda_wave']),
            len(settings['theta']), len(settings['rho']), len(settings['side'])
        )
        num_samples = np.prod(np.array(settings['lenghts']))
        return num_samples, settings


def train_set(db, target_size, mean, std, extra_transformation=None,
              natural_kwargs=None, gratings_kwargs=None):
    if extra_transformation is None:
        extra_transformation = []
    all_dbs = []
    shared_transforms = [
        *extra_transformation,
        cv2_transforms.RandomHorizontalFlip(),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ]
    if db in ['imagenet', 'celeba']:
        scale = (0.08, 1.0)
        size_transform = cv2_transforms.RandomResizedCrop(
            target_size, scale=scale
        )
        transforms = torch_transforms.Compose([
            size_transform, *shared_transforms
        ])
        if db == 'imagenet':
            natural_kwargs['root'] = os.path.join(
                natural_kwargs['root'], 'train'
            )
            current_db = ImageFolder(
                transform=transforms, **natural_kwargs
            )
        elif db == 'celeba':
            current_db = CelebA(
                transform=transforms, split='train', **natural_kwargs
            )
        all_dbs.append(current_db)
    if db in ['gratings']:
        transforms = torch_transforms.Compose(shared_transforms)
        all_dbs.append(
            GratingImages(
                transform=transforms, target_size=target_size, **gratings_kwargs
            )
        )
    return torch_data.ConcatDataset(all_dbs)


def validation_set(db, target_size, mean, std, extra_transformation=None,
                   natural_kwargs=None, gratings_kwargs=None):
    if extra_transformation is None:
        extra_transformation = []
    all_dbs = []
    shared_transforms = [
        *extra_transformation,
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ]
    if db in ['imagenet', 'celeba']:
        transforms = torch_transforms.Compose([
            cv2_transforms.Resize(target_size),
            cv2_transforms.CenterCrop(target_size),
            *shared_transforms
        ])
        if db == 'imagenet':
            natural_kwargs['root'] = os.path.join(
                natural_kwargs['root'], 'validation'
            )
            current_db = ImageFolder(
                transform=transforms, **natural_kwargs
            )
        elif db == 'celeba':
            current_db = CelebA(
                transform=transforms, split='test', **natural_kwargs
            )
        all_dbs.append(current_db)
    if db in ['gratings']:
        transforms = torch_transforms.Compose(shared_transforms)
        all_dbs.append(
            GratingImages(
                transform=transforms, target_size=target_size, **gratings_kwargs
            )
        )
    return torch_data.ConcatDataset(all_dbs)
