"""
Creating dataset objects for GEETUP in PyTorch environment.
"""

import numpy as np
import pickle
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from torchvision.transforms.functional import hflip

from kernelphysiology.dl.geetup.geetup_utils import map_point_to_image_size
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.utils.controls import isint
from kernelphysiology.utils.imutils import heat_map_from_point
from kernelphysiology.filterfactory.gaussian import gaussian_kernel2


class HeatMapFixationPoint(object):
    def __init__(self, target_size, org_size, gaussian_sigma=25):
        self.target_size = target_size
        self.org_size = org_size
        self.gaussian_kernel = gaussian_kernel2(gaussian_sigma)

    def __call__(self, point):
        point = map_point_to_image_size(point, self.target_size, self.org_size)
        img = heat_map_from_point(
            point, target_size=self.target_size, g_kernel=self.gaussian_kernel
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(target_size={})'.format(
            self.target_size
        )


def get_train_dataset(pickle_file, target_size):
    mean, std = get_preprocessing_function('rgb', 'trichromat')
    normalise = transforms.Normalize(mean=mean, std=std)
    img_transform = transforms.Compose([
        transforms.Resize(target_size), transforms.ToTensor(), normalise,
    ])
    target_transform = transforms.Compose([
        HeatMapFixationPoint(target_size, (360, 640)), transforms.ToTensor()
    ])
    common_transforms = []
    train_dataset = GeetupDataset(
        pickle_file, img_transform, target_transform, common_transforms
    )
    return train_dataset


def get_validation_dataset(pickle_file, target_size):
    mean, std = get_preprocessing_function('rgb', 'trichromat')
    normalise = transforms.Normalize(mean=mean, std=std)
    img_transform = transforms.Compose([
        transforms.Resize(target_size), transforms.ToTensor(), normalise,
    ])
    target_transform = transforms.Compose([
        HeatMapFixationPoint(target_size, (360, 640)), transforms.ToTensor()
    ])
    common_transforms = None
    train_dataset = GeetupDataset(
        pickle_file, img_transform, target_transform, common_transforms
    )
    return train_dataset


def _init_videos(video_list):
    video_paths = []
    all_videos = []
    num_sequences = 0
    for f, video_info in enumerate(video_list):
        video_paths.append(video_info[0])
        num_sequences += video_info[2]
        for j in range(video_info[2]):
            all_videos.append([f, video_info[1], j])
    return all_videos, num_sequences, video_paths


class GeetupDataset(Dataset):
    def __init__(self, pickle_file, transform=None, target_transform=None,
                 common_transforms=None, all_gts=False, frames_gap=None,
                 sequence_length=None):
        super(GeetupDataset, self).__init__()

        self.pickle_file = pickle_file
        self.transform = transform
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.frames_gap = frames_gap
        self.sequence_length = sequence_length
        self.all_gts = all_gts

        (self.all_videos,
         self.num_sequences,
         self.video_paths) = self.__read_pickle()

    def __read_pickle(self):
        f = open(self.pickle_file, 'rb')
        f_data = pickle.load(f)
        f.close()

        if self.frames_gap is None:
            self.frames_gap = f_data['frames_gap']
        if self.sequence_length is None:
            self.sequence_length = f_data['sequence_length']
        video_list = f_data['video_list']
        all_videos, num_sequences, video_paths = _init_videos(video_list)
        print('Read %d sequences' % num_sequences)
        return all_videos, num_sequences, video_paths

    def __getitem__(self, idx):
        vid_info = self.all_videos[idx]

        # read data
        segment_dir = self.video_paths[vid_info[0]]
        video_num = vid_info[1]
        frame_0 = vid_info[2]
        frame_n = frame_0 + self.sequence_length * self.frames_gap
        all_frames = [i for i in range(frame_0, frame_n, self.frames_gap)]

        video_path = '%s/CutVid_%s/' % (segment_dir, video_num)
        f_selected = '%s/SELECTED_IMGS_%s.txt' % (segment_dir, video_num)
        selected_imgs = np.loadtxt(f_selected, dtype=str, delimiter=',')

        x_item = []
        y_item = []

        # TODO: for now, just horizontal flipping
        do_for_entire_sequence = False
        if self.common_transforms is not None:
            if random.random() < 0.5:
                do_for_entire_sequence = True

        for j, frame_num in enumerate(all_frames):
            image_path = os.path.join(video_path, selected_imgs[frame_num][0])
            img = pil_loader(image_path)

            if do_for_entire_sequence:
                img = hflip(img)

            if self.transform is not None:
                img = self.transform(img)
            x_item.append(img)

            if self.all_gts or j == len(all_frames) - 1:
                gt = selected_imgs[frame_num][1]
                gt = gt.replace('[', '').replace(']', '').split(' ')
                gt = [int(i) for i in gt if isint(i)]
                # in the file it's stored as x and y, rather than rows and cols
                gt = gt[::-1]

                if do_for_entire_sequence:
                    gt[1] = img.shape[2] - gt[1]

                if self.target_transform is not None:
                    gt = self.target_transform(gt)
                y_item.append(gt)
        x_item = torch.stack(x_item, dim=0)
        y_item = torch.stack(y_item, dim=0)
        y_item = torch.squeeze(y_item)

        return x_item, y_item

    def __len__(self):
        return len(self.all_videos)
