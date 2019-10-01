"""
Creating dataset objects for GEETUP in PyTorch environment.
"""

import numpy as np
import pickle
import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

from kernelphysiology.utils.controls import isint


def _init_videos(video_list):
    all_videos = []
    num_sequences = 0
    for f, video_info in enumerate(video_list):
        num_sequences += video_info[2]
        for j in range(video_info[2]):
            all_videos.append([f, video_info[1], j])
    return all_videos, num_sequences


class GeetupDataset(Dataset):
    def __init__(self, pickle_file, transform=None, target_transform=None,
                 all_gts=True, frames_gap=None, sequence_length=None):
        super(GeetupDataset, self).__init__()

        self.pickle_file = pickle_file
        self.transform = transform
        self.target_transform = target_transform
        self.frames_gap = frames_gap
        self.sequence_length = sequence_length
        self.all_gts = all_gts

        self.all_videos, self.num_sequences = self.__read_pickle()

    def __read_pickle(self):
        f = open(self.pickle_file, 'rb')
        f_data = pickle.load(f)
        f.close()

        if self.frames_gap is None:
            self.frames_gap = f_data['frames_gap']
        if self.sequence_length is None:
            self.sequence_length = f_data['sequence_length']
        video_list = f_data['video_list']
        all_videos, num_sequences = _init_videos(video_list)
        return all_videos, num_sequences

    def __getitem__(self, idx):
        vid_info = self.all_videos[idx]

        # read data
        segment_dir = vid_info[0]
        video_num = vid_info[1]
        frame_0 = vid_info[2]
        frame_n = frame_0 + self.sequence_length * self.frames_gap
        all_frames = [i for i in range(frame_0, frame_n, self.frames_gap)]

        video_path = '%s/CutVid_%s/' % (segment_dir, video_num)
        f_selected = '%s/SELECTED_IMGS_%s.txt/' % (segment_dir, video_num)
        selected_imgs = np.loadtxt(f_selected, dtype=str, delimiter=',')

        x_item = []
        y_item = []
        for j, frame_num in enumerate(all_frames):
            image_path = os.path.join(video_path, selected_imgs[frame_num][0])
            img = pil_loader(image_path)
            if self.transform is not None:
                img = self.transform(img)
            x_item.append(img)

            if self.all_gts or j == len(all_frames) - 1:
                gt = selected_imgs[frame_num][1]
                gt = gt.replace('[', '').replace(']', '').split(' ')
                gt = [int(i) for i in gt if isint(i)]
                if self.target_transform is not None:
                    gt = self.target_transform(gt)
                y_item.append(gt)
        x_item = np.array(x_item)
        y_item = np.array(y_item)

        return x_item, y_item

    def __len__(self):
        return len(self.all_videos)
