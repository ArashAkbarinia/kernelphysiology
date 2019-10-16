"""
Creating dataset objects for GEETUP in PyTorch environment.
"""

import numpy as np
import pickle
import os
import random

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from torchvision.transforms.functional import hflip

from kernelphysiology.dl.geetup.geetup_utils import map_point_to_image_size
from kernelphysiology.dl.geetup.geetup_utils import parse_gt_line
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.utils.imutils import heat_map_from_point
from kernelphysiology.filterfactory.gaussian import gaussian_kernel2


class HeatMapFixationPoint(object):
    def __init__(self, target_size, org_size, gaussian_sigma=None):
        self.target_size = target_size
        self.org_size = org_size
        max_width = min(target_size)
        if gaussian_sigma is None:
            gaussian_sigma = max_width * 0.1
        self.gaussian_kernel = gaussian_kernel2(
            gaussian_sigma, max_width=max_width
        )

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


def get_train_dataset(pickle_file, target_size, mean_std=None):
    if mean_std is None:
        mean, std = get_preprocessing_function('rgb', 'trichromat')
    else:
        mean, std = mean_std
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


def get_validation_dataset(pickle_file, target_size, mean_std=None):
    if mean_std is None:
        mean, std = get_preprocessing_function('rgb', 'trichromat')
    else:
        mean, std = mean_std
    normalise = transforms.Normalize(mean=mean, std=std)
    img_transform = transforms.Compose([
        transforms.Resize(target_size), transforms.ToTensor(), normalise,
    ])
    target_transform = transforms.Compose([
        HeatMapFixationPoint(target_size, (360, 640)), transforms.ToTensor()
    ])
    common_transforms = None
    validation_dataset = GeetupDataset(
        pickle_file, img_transform, target_transform, common_transforms
    )
    return validation_dataset


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


def _npy_loader(input_path):
    img = np.load(input_path).astype(np.float32)
    img = Image.fromarray(img, mode='F')
    # FIXME: this is not a proper solution
    img = img.resize((360, 640)[::-1])
    return img


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
        self.extension = None
        self.data_loader = pil_loader

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
        self.base_path_img = f_data['base_path_img']
        self.base_path_txt = f_data['base_path_txt']
        self.prefix = f_data['prefix']
        if 'extension' in f_data:
            self.extension = f_data['extension']
            if self.extension == '.npy':
                self.data_loader = _npy_loader
        all_videos, num_sequences, video_paths = _init_videos(video_list)
        print('Read %d sequences' % num_sequences)
        return all_videos, num_sequences, video_paths

    def _prepare_item(self, idx):
        vid_info = self.all_videos[idx]

        # read data
        segment_dir = self.video_paths[vid_info[0]]
        video_num = vid_info[1]
        frame_0 = vid_info[2]
        frame_n = frame_0 + self.sequence_length * self.frames_gap
        all_frames = [i for i in range(frame_0, frame_n, self.frames_gap)]

        video_path = '%s/%s/CutVid_%s/%s' % (
            self.base_path_img, segment_dir, video_num, self.prefix
        )
        f_selected = '%s/%s/SELECTED_IMGS_%s.txt' % (
            self.base_path_txt, segment_dir, video_num
        )
        selected_imgs = np.loadtxt(f_selected, dtype=str, delimiter=',')
        return all_frames, video_path, selected_imgs

    def __getitem__(self, idx):
        all_frames, video_path, selected_imgs = self._prepare_item(idx)

        x_item = []
        y_item = []

        # TODO: for now, just horizontal flipping
        do_for_entire_sequence = False
        if self.common_transforms is not None:
            if random.random() < 0.5:
                do_for_entire_sequence = True

        for j, frame_num in enumerate(all_frames):
            file_name = selected_imgs[frame_num][0]
            # this is a hack to change the extension easily to .png or .npy
            if self.extension is not None:
                file_name = file_name[:-4] + self.extension
            image_path = os.path.join(video_path, file_name)
            img = self.data_loader(image_path)

            if do_for_entire_sequence:
                img = hflip(img)

            if self.transform is not None:
                img = self.transform(img)
            x_item.append(img)

            if self.all_gts or j == len(all_frames) - 1:
                gt = parse_gt_line(selected_imgs[frame_num][1])

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


class GeetupDatasetInformative(GeetupDataset):
    def __getitem__(self, idx):
        all_frames, video_path, selected_imgs = self._prepare_item(idx)

        x_item = []
        y_item = []

        for j, frame_num in enumerate(all_frames):
            image_path = os.path.join(video_path, selected_imgs[frame_num][0])
            x_item.append(image_path)

            if self.all_gts or j == len(all_frames) - 1:
                gt = parse_gt_line(selected_imgs[frame_num][1])
                y_item.append(gt)

        return x_item, y_item
