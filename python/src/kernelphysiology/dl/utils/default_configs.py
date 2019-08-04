"""
Default configurations of this project.
"""

import os
import sys
import socket

from kernelphysiology import commons


def get_default_dataset_paths(
        dataset_name,
        train_dir=None,
        validation_dir=None,
        data_dir=None):
    pre_path = '/home/arash/Software/'
    if dataset_name == 'imagenet':
        # NOTE: just for the ease of working in my machiens
        if train_dir is None:
            train_dir = '%simagenet/raw-data/train/' % pre_path
        if validation_dir is None:
            if (socket.gethostname() == 'awesome' or
                    socket.gethostname() == 'nickel' or
                    socket.gethostname() == 'nyanza'):
                validation_dir = '%simagenet/raw-data/validation/' % pre_path
            else:
                validation_dir = '%srepositories/kernelphysiology/data/' \
                                 'computervision/ilsvrc/ilsvrc2012/' \
                                 'raw-data/validation/' % pre_path
    elif dataset_name == 'cifar10':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root,
                'data/datasets/cifar/cifar10/')
    elif dataset_name == 'cifar100':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root,
                'data/datasets/cifar/cifar100/')
    elif dataset_name == 'stl10':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root,
                'data/datasets/stl/stl10/')
    elif dataset_name == 'leaf' or dataset_name == 'fruits':
        if train_dir is None:
            train_dir = '%sdatasets/misc/%s/train' % (pre_path,
                                                      dataset_name)
        if validation_dir is None:
            validation_dir = '%sdatasets/misc/%s/validation/' % (pre_path,
                                                                 dataset_name)
    elif dataset_name == 'coco':
        # NOTE: just for the ease of working in my machiens
        if data_dir is None:
            if (socket.gethostname() == 'awesome' or
                    socket.gethostname() == 'nickel' or
                    socket.gethostname() == 'nyanza'):
                data_dir = '/home/arash/Software/coco/'
            else:
                validation_dir = '%srepositories/kernelphysiology/' \
                                 'data/computervision/coco/' % pre_path
    elif 'wcs' in dataset_name:
        # NOTE: just for the ease of working in my machiens
        if train_dir is None:
            train_dir = '%sdatasets/wcs/%s/train/' % (pre_path, dataset_name)
        if validation_dir is None:
            if (socket.gethostname() == 'awesome' or
                    socket.gethostname() == 'nickel' or
                    socket.gethostname() == 'nyanza'):
                validation_dir = '%sdatasets/wcs/%s/validation/' % \
                                 (pre_path, dataset_name)
    else:
        sys.exit('Unsupported dataset %s' % dataset_name)
    return train_dir, validation_dir, data_dir
