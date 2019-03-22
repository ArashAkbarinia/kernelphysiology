'''
Default configurations of this project.
'''


import os
import sys
import socket

from kernelphysiology import commons


def get_default_dataset_paths(dataset_name, train_dir=None, validation_dir=None, data_dir=None):
    if dataset_name == 'imagenet':
        # NOTE: just for the ease of working in my machiens
        if train_dir is None:
            train_dir = '/home/arash/Software/imagenet/raw-data/train/'
        if validation_dir is None:
            if socket.gethostname() == 'awesome' or socket.gethostname() == 'nickel' or socket.gethostname() == 'nyanza':
                validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
            else:
                validation_dir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/raw-data/validation/'
    elif dataset_name == 'cifar10':
        if data_dir is None:
            data_dir = os.path.join(commons.python_root, 'data/datasets/cifar/cifar10/')
    elif dataset_name == 'cifar100':
        if data_dir is None:
            data_dir = os.path.join(commons.python_root, 'data/datasets/cifar/cifar100/')
    elif dataset_name == 'stl10':
        if data_dir is None:
            data_dir = os.path.join(commons.python_root, 'data/datasets/stl/stl10/')
    elif dataset_name == 'coco':
        # NOTE: just for the ease of working in my machiens
        if data_dir is None:
            if socket.gethostname() == 'awesome' or socket.gethostname() == 'nickel' or socket.gethostname() == 'nyanza':
                data_dir = '/home/arash/Software/coco/'
            else:
                validation_dir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/coco/'
    else:
        sys.exit('Unsupported dataset %s' % (dataset_name))
    return (train_dir, validation_dir, data_dir)