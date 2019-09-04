import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import glob
import random
import time

from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def initialize_neglabels_correct(targets, v_total_images, neg_labels, num_images_label, num_images_label_orig):
    correct = 1
    newlabels = []
    for i in v_total_images:

        newlabel, num_images_label = select_newlabel(targets[i], neg_labels, num_images_label,
                                                     num_images_label_orig)

        if newlabel == -1:
            print('No luck!!!!')
            correct = 0
            return correct, num_images_label_orig

        # Add it to target list
        newlabels.append(newlabel)
        # print( newlabel )

    return correct, newlabels


def select_newlabel(current_label, neg_labels, num_images_label, num_images_label_orig):
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


class AugmentedLabelDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, transform=None, loader=pil_loader):
        self.data_root = data_root
        self.loader = loader
        self.transform = transform

        all_subfolders = sorted(glob.glob(self.data_root + '/*/'))

        # Direct labels (explicit labels)
        self.image_paths = []
        self.targets = []

        self.image_paths_neg = []
        self.targets_neg = []

        num_images_label = []

        for target_id, sub_folder in enumerate(all_subfolders):
            count_images_in = 0
            for image_path in glob.glob(sub_folder + '*.JPEG'):
                self.image_paths.append(image_path)
                self.targets.append(target_id)
                count_images_in += 1

            num_images_label.append(count_images_in)

        # Indirect labels (implicit labels)
        # Go through all labels and define the range for selecting random images
        v_total_images = np.arange(0, len(self.targets))
        neg_labels = np.arange(max(self.targets) + 1, (max(self.targets) + 1) * 2)
        num_images_label = np.array(num_images_label)

        # In case of emergency (not lucky) use one copy
        num_images_label_orig = num_images_label.copy()

        initialize_neglabels(self, v_total_images, neg_labels, num_images_label, num_images_label_orig)

        for i in v_total_images:
            self.image_paths.append(self.image_paths_neg[i])
            self.targets.append(np.asscalar(self.targets_neg[i]))

    def initialize_neglabels(self, v_total_images, neg_labels, num_images_label, num_images_label_orig):

        correct, newlabels = initialize_neglabels_correct(self.targets, v_total_images, neg_labels, num_images_label,
                                                          num_images_label_orig)

        while correct == 0:
            correct, newlabels = initialize_neglabels_correct(self.targets, v_total_images, neg_labels,
                                                              num_images_label_orig.copy(), num_images_label_orig)

        sort_i = sorted(range(len(newlabels)), key=lambda k: newlabels[k])

        for i in v_total_images:
            self.image_paths_neg.append(self.image_paths[sort_i[i]])
            self.targets_neg.append(newlabels[sort_i[i]])

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        target = self.targets[index]
        return img, target

    def __len__(self):
        n = len(self.image_paths)
        return n
