'''
The utility functoins for datasets.
'''


import commons

import socket
import sys
import os


from kernelphysiology.dl.keras.datasets.cifar import cifar_train
from kernelphysiology.dl.keras.datasets.stl import stl_train
from kernelphysiology.dl.keras.datasets.imagenet import imagenet_train


def which_dataset(args, dataset_name):
    if dataset_name == 'cifar10':
        if args.script_type == 'training':
            args = cifar_train.prepare_cifar10_generators(args)
        else:
            args = cifar_train.cifar10_validatoin_generator(args)
    elif dataset_name == 'cifar100':
        if args.script_type == 'training':
            args = cifar_train.prepare_cifar100_generators(args)
        else:
            args = cifar_train.cifar100_validatoin_generator(args)
    elif dataset_name == 'stl10':
        if args.script_type == 'training':
            args = stl_train.prepare_stl10_generators(args)
        else:
            args = stl_train.stl10_validation_generator(args)
    elif dataset_name == 'imagenet':
        if args.script_type == 'training':
            args = imagenet_train.prepare_imagenet(args)
        else:
            args = imagenet_train.validation_generator(args)
    return args


def get_default_dataset_paths(args):
    if args.dataset == 'imagenet':
        # NOTE: just for the ease of working in my machiens
        if args.train_dir is None:
            args.train_dir = '/home/arash/Software/imagenet/raw-data/train/'
        if args.validation_dir is None:
            if socket.gethostname() == 'awesome':
                args.validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
            else:
                args.validation_dir = '/home/arash/Software/repositories/kernelphysiology/data/computervision/ilsvrc/ilsvrc2012/raw-data/validation/'
    elif args.dataset == 'cifar10':
        if args.data_dir is None:
            args.data_dir = os.path.join(commons.python_root, 'data/datasets/cifar/cifar10/')
    elif args.dataset == 'cifar100':
        if args.data_dir is None:
            args.data_dir = os.path.join(commons.python_root, 'data/datasets/cifar/cifar100/')
    elif args.dataset == 'stl10':
        if args.data_dir is None:
            args.data_dir = os.path.join(commons.python_root, 'data/datasets/stl/stl10/')
    else:
        sys.exit('Unsupported dataset %s' % (args.dataset))
    return args