"""
Label augmentation.
"""

from torch.utils.data import Dataset

import numpy as np
import glob
import random

from PIL import Image


def pil_loader(path):
    # copied from torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def initialize_neglabels_correct(targets, v_total_images, neg_labels,
                                 num_images_label, num_images_label_orig):
    correct = 1
    newlabels = []
    for i in v_total_images:
        newlabel, num_images_label = select_newlabel(
            targets[i], neg_labels, num_images_label, num_images_label_orig
        )

        if newlabel == -1:
            print('No luck!!!!')
            correct = 0
            return correct, num_images_label_orig

        # Add it to target list
        newlabels.append(newlabel)

    return correct, newlabels


def select_newlabel(current_label, neg_labels, num_images_label,
                    num_images_label_orig):
    aux = neg_labels.copy()
    aux[current_label] = False
    aux = aux[aux != False]

    aux_im = num_images_label.copy()
    aux_im[current_label] = False
    aux_im = aux_im[aux_im != False]

    data = random.sample(range(0, len(aux)), 1)

    # Select random label
    while num_images_label[aux[data] - neg_labels[0]] <= 0:
        data = random.sample(range(0, len(aux)), 1)
        if aux_im.sum() == 0:
            return -1, num_images_label_orig

    num_images_label[aux[data] - neg_labels[0]] -= 1

    return aux[data], num_images_label


class AugmentedLabelDataset(Dataset):

    def __init__(self, data_root, transform=None, loader=pil_loader):
        self.data_root = data_root
        self.loader = loader
        self.transform = transform

        # Direct labels (explicit labels)
        self.image_paths = []
        self.targets = []

        self.num_images_label = []

        self.read_real_labels()
        self.shuffle_augmented_labels()

    def read_real_labels(self):
        all_subfolders = sorted(glob.glob(self.data_root + '/*/'))
        for target_id, sub_folder in enumerate(all_subfolders):
            sub_folder_imgs = sorted(glob.glob(sub_folder + '*.JPEG'))
            for image_path in sub_folder_imgs:
                self.image_paths.append(image_path)
                self.targets.append(target_id)
            self.num_images_label.append(len(sub_folder_imgs))

    def shuffle_augmented_labels(self):
        # Indirect labels (implicit labels)
        image_paths_neg, targets_neg = self.initialize_neglabels()

        self.all_image_paths = image_paths_neg.copy()
        self.all_targets = targets_neg.copy()

    def initialize_neglabels(self):
        v_total_images = np.arange(0, len(self.targets))
        neg_labels = np.arange(
            max(self.targets) + 1, (max(self.targets) + 1) * 2
        )

        # Go through all labels and define the range for selecting random images
        num_images_label = np.array(self.num_images_label)

        # In case of emergency (not lucky) use one copy
        num_images_label_orig = num_images_label.copy()

        correct, newlabels = initialize_neglabels_correct(
            self.targets, v_total_images, neg_labels,
            num_images_label, num_images_label_orig
        )

        while correct == 0:
            correct, newlabels = initialize_neglabels_correct(
                self.targets, v_total_images, neg_labels,
                num_images_label_orig.copy(), num_images_label_orig
            )

        sort_i = sorted(range(len(newlabels)), key=lambda k: newlabels[k])

        image_paths_neg = []
        targets_neg = []
        for i in v_total_images:
            image_paths_neg.append(self.image_paths[sort_i[i]])
            targets_neg.append(newlabels[sort_i[i]].item())
        return image_paths_neg, targets_neg

    def __getitem__(self, index):
        path = self.all_image_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        target = self.all_targets[index]
        return img, target

    def __len__(self):
        return len(self.all_image_paths)
