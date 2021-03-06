"""
Default configurations of this project.
"""

import os
import sys
import socket

from kernelphysiology import commons


def get_num_classes(dataset_name, num_classes=None):
    if num_classes is not None:
        return num_classes

    if 'voc' in dataset_name:
        num_classes = 21
    elif dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'imagenet':
        num_classes = 1000
    elif '1600' in dataset_name:
        num_classes = 1600
    elif '330' in dataset_name:
        num_classes = 330
    else:
        sys.exit(
            'Dataset %s not recognised. num_classes must be provided' %
            dataset_name
        )
    return num_classes


def get_default_target_size(dataset_name, target_size=None):
    if target_size is not None:
        return target_size

    # default target size for a set of commonly used datasets
    if 'voc' in dataset_name:
        target_size = 480
    elif dataset_name in ['imagenet', 'leaves', 'fruits', 'land', 'vggface2']:
        target_size = 224
    elif 'wcs_lms' in dataset_name:
        target_size = 128
    elif 'wcs_jpg' in dataset_name:
        target_size = 128
    elif 'cifar' in dataset_name or 'stl' in dataset_name:
        target_size = 32
    else:
        sys.exit(
            'Default target_size is not defined for dataset %s' % dataset_name
        )

    return target_size


def get_default_dataset_paths(dataset_name, train_dir=None, validation_dir=None,
                              data_dir=None, script_type=None):
    pre_path = '/home/arash/Software/'
    if script_type == 'testing' and validation_dir is not None:
        return train_dir, validation_dir, data_dir
    elif (script_type == 'training' and train_dir is not None and
          validation_dir is not None):
        return train_dir, validation_dir, data_dir
    if dataset_name == 'imagenet':
        # NOTE: just for the ease of working in my machines
        if train_dir is None:
            train_dir = '%simagenet/raw-data/train/' % pre_path
        if validation_dir is None:
            if _is_server_known():
                validation_dir = '%simagenet/raw-data/validation/' % pre_path
            else:
                validation_dir = '%srepositories/kernelphysiology/data/' \
                                 'computervision/ilsvrc/ilsvrc2012/' \
                                 'raw-data/validation/' % pre_path
    elif dataset_name == 'cifar10':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root, 'data/datasets/cifar/cifar10/'
            )
    elif dataset_name == 'cifar100':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root, 'data/datasets/cifar/cifar100/'
            )
    elif dataset_name == 'stl10':
        if data_dir is None:
            data_dir = os.path.join(
                commons.python_root, 'data/datasets/stl/stl10/'
            )
    elif dataset_name in ['fruits', 'leaves']:
        db_path = '%sdatasets/misc/%s' % (pre_path, dataset_name)
        if train_dir is None:
            train_dir = '%s/train' % db_path
        if validation_dir is None:
            validation_dir = '%s/validation' % db_path
    elif dataset_name == 'coco':
        # NOTE: just for the ease of working in my machiens
        if data_dir is None:
            if _is_server_known():
                data_dir = '/home/arash/Software/coco/'
            else:
                validation_dir = '%srepositories/kernelphysiology/' \
                                 'data/computervision/coco/' % pre_path
    elif 'wcs' in dataset_name:
        db_path = '%sdatasets/wcs/%s' % (pre_path, dataset_name)
        if train_dir is None:
            train_dir = '%s/train/' % db_path
        if validation_dir is None:
            validation_dir = '%s/validation/' % db_path
    elif dataset_name == 'voc_org':
        if data_dir is None:
            if _is_server_known():
                data_dir = '%s/datasets/' % pre_path
    elif dataset_name == 'voc_coco':
        if data_dir is None:
            if _is_server_known():
                data_dir = '%s/datasets/coco/' % pre_path
    else:
        sys.exit('Unsupported dataset %s' % dataset_name)
    return train_dir, validation_dir, data_dir


def optimisation_params(task_type, args=None):
    if args is None:
        args = {}
    # TODO: other tasks
    if task_type == 'classification':
        if 'lr' not in args or args.lr is None:
            lr = 0.1
        else:
            lr = args.lr
        if 'weight_decay' not in args or args.weight_decay is None:
            weight_decay = 1e-4
        else:
            weight_decay = args.weight_decay
        return lr, weight_decay


def _is_server_known():
    hostname = socket.gethostname()
    if hostname in ['awesome', 'nickel', 'nyanza']:
        return True
    return False
