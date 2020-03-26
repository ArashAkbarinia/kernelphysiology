"""
Label augmentation.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

import numpy as np
import glob
import random

from PIL import Image


def _initialize_neg_labels_correct(targets, v_total_images, neg_labels,
                                   num_images_label_org):
    num_images_label_c1 = num_images_label_org.copy()

    correct = 1
    new_labels = []
    for i in v_total_images:
        new_label, num_images_label_c1 = _select_new_label(
            targets[i], neg_labels, num_images_label_c1, num_images_label_org
        )

        if new_label == -1:
            print('No luck!!!!')
            return 0, num_images_label_org

        # Add it to target list
        new_labels.append(new_label)

    return correct, new_labels


def _select_new_label(current_label, neg_labels, num_images_label,
                      num_images_label_orig):
    aux = neg_labels.copy()
    aux[current_label] = -1
    aux = aux[aux != -1]

    aux_im = num_images_label.copy()
    aux_im[current_label] = -1
    aux_im = aux_im[aux_im != -1]

    data = random.sample(range(0, len(aux)), 1)

    # Select random label
    while num_images_label[aux[data] - neg_labels[0]] <= 0:
        data = random.sample(range(0, len(aux)), 1)
        if aux_im.sum() == 0:
            return -1, num_images_label_orig

    num_images_label[aux[data] - neg_labels[0]] -= 1

    return aux[data], num_images_label


def _get_negative_sample_array(targets):
    total_images = np.arange(0, len(targets))
    neg_labels = np.arange(0, max(targets) + 1)
    return total_images, neg_labels


def _get_new_labels(num_samples_label, targets, v_total_samples, neg_labels):
    # In case of emergency (not lucky) use one copy
    num_images_label_c1 = num_samples_label.copy()

    correct = 0
    while correct == 0:
        correct, new_labels = _initialize_neg_labels_correct(
            targets, v_total_samples, neg_labels, num_images_label_c1
        )

    sort_i = sorted(range(len(new_labels)), key=lambda k: new_labels[k])
    return sort_i, new_labels


class ExplicitNegativeLabelArray(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets_pos = targets
        self.transform = transform
        self.target_transform = target_transform
        # max is correct, given targets start from 0
        self.num_pos_classes = max(targets)

    def __getitem__(self, index):
        img = self.data[index]
        target_pos = self.targets_pos[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_pos = self.target_transform(target_pos)

        target_neg = np.ones(self.num_pos_classes)
        # we consider the last label as negative for all
        if target_pos < self.num_pos_classes:
            target_neg[target_pos] = 0

        target_neg = torch.tensor(target_neg, dtype=torch.float32)

        return img, target_pos, target_neg

    def __len__(self):
        return len(self.targets_pos)


class RandomNegativeLabelArray(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        self.num_samples_label = self.compute_images_per_label()
        self.shuffle_augmented_labels()

    def compute_images_per_label(self):
        unique_labels = np.unique(np.array(self.targets))
        num_samples_label = []
        for label in unique_labels:
            num_samples_label.append(self.targets.count(label))
        return np.array(num_samples_label)

    def shuffle_augmented_labels(self):
        # Indirect labels (implicit labels)
        ind_map_to_org, targets_neg = self.initialize_neg_labels()

        self.ind_map_to_org = ind_map_to_org.copy()
        self.targets_neg = targets_neg.copy()

    def initialize_neg_labels(self):
        v_total_samples, neg_labels = _get_negative_sample_array(self.targets)

        ind_map_to_org, new_labels = _get_new_labels(
            self.num_samples_label, self.targets, v_total_samples, neg_labels
        )

        targets_neg = []
        for i in v_total_samples:
            targets_neg.append(new_labels[ind_map_to_org[i]].item())
        return ind_map_to_org, targets_neg

    def __getitem__(self, index):
        img = self.data[self.ind_map_to_org[index]]
        target_neg = self.targets_neg[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_neg = self.target_transform(target_neg)

        return img, target_neg

    def __len__(self):
        return len(self.targets_neg)


IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp'
]


class ExplicitNegativeLabelFolder(Dataset):
    def __init__(self, data_root, negative_root, transform=None,
                 target_transform=None, loader=pil_loader, extensions=None):
        if extensions is None:
            extensions = IMG_EXTENSIONS
        self.data_root = data_root
        self.negative_root = negative_root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        (self.image_paths,
         self.targets,
         self.num_samples_label) = self.read_real_labels()
        self.num_pos_classes = len(self.num_samples_label) - 1

    def read_real_labels(self):
        all_subfolders = sorted(glob.glob(self.data_root + '/*/'))
        # adding the nagative folder
        all_subfolders.append(self.negative_root)
        image_paths = []
        targets = []
        num_images_label = []
        for target_id, sub_folder in enumerate(all_subfolders):
            sub_folder_imgs = []
            # reading all extensions
            for extension in self.extensions:
                sub_folder_imgs.extend(
                    sorted(glob.glob(sub_folder + '*' + extension))
                )
                # with upper case
                sub_folder_imgs.extend(
                    sorted(glob.glob(sub_folder + '*' + extension.upper()))
                )
            for image_path in sub_folder_imgs:
                image_paths.append(image_path)
                targets.append(target_id)
            num_images_label.append(len(sub_folder_imgs))
        return image_paths, targets, np.array(num_images_label)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img, target_pos = self.loader(path), self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_pos = self.target_transform(target_pos)

        target_neg = np.ones(self.num_pos_classes)
        if target_pos < self.num_pos_classes:
            target_neg[target_pos] = 0

        target_neg = torch.tensor(target_neg, dtype=torch.float32)

        return img, target_pos, target_neg

    def __len__(self):
        return len(self.targets)


class RandomNegativeLabelFolder(Dataset):

    def __init__(self, data_root, transform=None, target_transform=None,
                 loader=pil_loader, extensions=None):
        if extensions is None:
            extensions = IMG_EXTENSIONS
        self.data_root = data_root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform

        (self.image_paths,
         self.targets,
         self.num_samples_label) = self.read_real_labels()
        self.shuffle_augmented_labels()

    def read_real_labels(self):
        all_subfolders = sorted(glob.glob(self.data_root + '/*/'))
        image_paths = []
        targets = []
        num_images_label = []
        for target_id, sub_folder in enumerate(all_subfolders):
            sub_folder_imgs = []
            # reading all extensions
            for extension in self.extensions:
                sub_folder_imgs.extend(
                    sorted(glob.glob(sub_folder + '*' + extension))
                )
                # with upper case
                sub_folder_imgs.extend(
                    sorted(glob.glob(sub_folder + '*' + extension.upper()))
                )
            for image_path in sub_folder_imgs:
                image_paths.append(image_path)
                targets.append(target_id)
            num_images_label.append(len(sub_folder_imgs))
        return image_paths, targets, np.array(num_images_label)

    def shuffle_augmented_labels(self):
        # Indirect labels (implicit labels)
        ind_map_to_org, targets_neg = self.initialize_neg_labels()

        self.ind_map_to_org = ind_map_to_org.copy()
        self.targets_neg = targets_neg.copy()

    def initialize_neg_labels(self):
        v_total_samples, neg_labels = _get_negative_sample_array(self.targets)

        ind_map_to_org, new_labels = _get_new_labels(
            self.num_samples_label, self.targets, v_total_samples, neg_labels
        )

        targets_neg = []
        for i in v_total_samples:
            targets_neg.append(new_labels[ind_map_to_org[i]].item())
        return ind_map_to_org, targets_neg

    def __getitem__(self, index):
        path = self.image_paths[self.ind_map_to_org[index]]
        img, target_neg = self.loader(path), self.targets_neg[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_neg = self.target_transform(target_neg)
        return img, target_neg

    def __len__(self):
        return len(self.image_paths)
