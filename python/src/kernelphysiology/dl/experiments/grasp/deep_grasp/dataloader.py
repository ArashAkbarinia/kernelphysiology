"""
The dataloader routines for the grasp project.
"""
import cv2
import numpy as np
import os
import sys
import glob

from scipy.io import loadmat

import torch
from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

from kernelphysiology.transformations.normalisations import min_max_normalise


def _read_trial(file_path):
    # Columns 1-2-3 are x-y-z for thumb
    # Columns 4-5-6 are x-y-z for finger
    # Columns 7-8-9 are x-y-z for marker 1 on the object
    # Columns 10-11-12 are x-y-z for marker 2 on the object
    # Columns 13-14-15 are x-y-z for marker 3 on the object
    trial_data = np.loadtxt(file_path)
    return trial_data


def nan2neighbour(data):
    mask = np.isnan(data)
    data[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask]
    )
    return data


def trial2img(trial, which_xyz):
    chns = len(which_xyz)
    n_times = trial.shape[0]
    img = np.zeros((n_times, 3, chns))

    for i, xyz in enumerate(which_xyz):
        if xyz == 'thumb':
            xyz_data = trial[:, 0:3]
        elif xyz == 'index':
            xyz_data = trial[:, 3:6]
        elif xyz == 'marker1':
            xyz_data = trial[:, 6:9]
        elif xyz == 'marker2':
            xyz_data = trial[:, 9:12]
        elif xyz == 'marker3':
            xyz_data = trial[:, 12:15]
        else:
            sys.exit('Unsupported which_xyz %s' % which_xyz)
        # setting the NaNs to the closest value
        for c in range(3):
            xyz_data[:, c] = nan2neighbour(xyz_data[:, c])
        img[:, :, i] = xyz_data

    return img


def trial2voxel(trial, which_xyz, img_size=128):
    n_times = trial.shape[0]
    voxel = np.zeros((n_times, img_size, img_size, img_size))

    min_vals = trial.min(axis=0)
    max_vals = trial.max(axis=0)
    trial_normalised = min_max_normalise(trial, 0, 1, min_vals, max_vals)

    trial_int = np.round(trial_normalised * (img_size - 1)).astype('int')

    for i, t_ind in enumerate(trial_int):
        if 'thumb' in which_xyz:
            voxel[i, t_ind[0], t_ind[1], t_ind[2]] = 1
        if 'index' in which_xyz:
            voxel[i, t_ind[3], t_ind[4], t_ind[5]] = 1
        if 'marker1' in which_xyz:
            voxel[i, t_ind[6], t_ind[7], t_ind[8]] = 1
        if 'marker2' in which_xyz:
            voxel[i, t_ind[9], t_ind[10], t_ind[11]] = 1
        if 'marker3' in which_xyz:
            voxel[i, t_ind[12], t_ind[13], t_ind[14]] = 1
        else:
            sys.exit('Unsupported which_xyz %s' % which_xyz)

    return voxel


class SingleParticipant(torch_data.Dataset):

    def __init__(self, root, participant, condition, which_xyz=None,
                 time_interval=None, transform=None):
        self.root = root
        self.participant = participant
        self.condition = condition
        self.which_xyz = which_xyz
        if self.which_xyz is None:
            self.which_xyz = ['thumb', 'index']

        self.time_interval = time_interval
        self.transform = transform

        # creating the root paths
        self.kinematic_root = os.path.join(self.root, 'kinematic', participant, condition)
        self.startend_root = os.path.join(self.root, 'start_end', participant, condition)

        # reading the mat info
        mat_path = '%s/stimlist_%s.mat' % (self.kinematic_root, condition)
        matdata = loadmat(mat_path)
        # trial target_position mass_distribution intensity time response
        self.stimuli_data = matdata['stimlist']
        self.valid_trials = glob.glob(self.startend_root + '/trial_*')
        self.num_trials = len(self.valid_trials)

    def __getitem__(self, item):
        se_file_path = self.valid_trials[item]
        tnum = int(se_file_path.split('_')[-1][:-4])

        start_end = np.loadtxt(se_file_path).astype(int)

        file_path = '%s/trial_%d' % (self.kinematic_root, tnum)
        trial_data = np.loadtxt(file_path)

        trial_img = trial2img(trial_data, self.which_xyz)

        # only considering the interval from start_end file
        sind, eind = start_end
        if (eind - sind) < 10:
            print(start_end, se_file_path)
        trial_img = trial_img[sind:eind]

        if self.time_interval is not None:
            sind, eind = self.time_interval
            trial_img = trial_img[sind:eind]
        if self.transform is not None:
            trial_img = self.transform(trial_img)

        # -1 because in Matlab they're stored from 1
        mass_dist = self.stimuli_data[tnum][2] - 1
        intensity = self.stimuli_data[tnum][-3]
        # -1 because in Matlab they're stored as 1 and 2
        response = self.stimuli_data[tnum][-1] - 1

        # converting to tensor
        trial_img = torch.tensor(trial_img.transpose((2, 0, 1))).type(torch.FloatTensor)
        intensity = torch.tensor([intensity]).type(torch.FloatTensor)

        # FIXME
        # response -1 means the participant hasn't responded
        # in this case we'll assume that zhe hasn't felt it.
        if response == -1:
            # print('Shouldnt happen', se_file_path)
            response = 1
        return trial_img, intensity, mass_dist, response, tnum

    def __len__(self):
        return self.num_trials


def _random_train_val_sets(train_percent):
    all_sets = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23, 24]
    num_sets = len(all_sets)
    np.random.shuffle(all_sets)
    break_ind = round(train_percent * num_sets)
    train_group = all_sets[:break_ind]
    val_group = all_sets[break_ind:]
    return train_group, val_group


def get_val_set(root, condition, target_size, val_group, **kwargs):
    v_transform = torch_transforms.Compose([
        TimeResize(target_size)
    ])
    val_set = []
    for vg in val_group:
        val_set.append(
            SingleParticipant(
                root, str(vg), condition, transform=v_transform, **kwargs
            )
        )
    val_db = torch_data.dataset.ConcatDataset(val_set)
    return val_db


def train_val_sets(root, condition, target_size, train_group=None, val_group=None, **kwargs):
    if train_group is None:
        train_group = 0.8
        val_group = 0.2
    # computing the portion of train and validation
    if not type(train_group) is list:
        train_group, val_group = _random_train_val_sets(train_group)

    t_transform = torch_transforms.Compose([
        #        ClipTime(target_size, place=None),
        TimeResize(target_size)
    ])
    train_set = []
    for tg in train_group:
        train_set.append(
            SingleParticipant(
                root, str(tg), condition, transform=t_transform, **kwargs
            )
        )
    train_db = torch_data.dataset.ConcatDataset(train_set)

    val_db = get_val_set(root, condition, target_size, val_group, **kwargs)

    return train_db, val_db


class TimeResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, trial_img):
        trial_img = cv2.resize(trial_img, (trial_img.shape[1], self.size))
        return trial_img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ClipTime(object):
    def __init__(self, size, place=None):
        self.size = size
        self.place = place

    def __call__(self, trial_img):
        if self.place is None:
            max_time = abs(len(trial_img) - self.size)
            if max_time == 0:
                sind = 0
            else:
                sind = np.random.randint(0, max_time)
        else:
            sind = self.place
        eind = sind + self.size
        trial_img = trial_img[sind:eind]
        return trial_img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
