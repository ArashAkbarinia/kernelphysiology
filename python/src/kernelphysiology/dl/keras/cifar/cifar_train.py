'''
Train a simple DNN on CIFAR 10 or 100.
'''


import os

from kernelphysiology.dl.keras.cifar import cifar_utils
from kernelphysiology.dl.keras.cifar import cifar10
from kernelphysiology.dl.keras.cifar import cifar100


def train_cifar10(args):
    args.num_classes = 10
    confs = cifar_utils.CifarConfs(args=args)

    # The data, split between train and test sets:
    (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar10.load_data(os.path.join(confs.python_root, 'data/datasets/cifar/cifar10/'))

    cifar_utils.start_training(confs)


def train_cifar100(args):
    args.num_classes = 100
    confs = cifar_utils.CifarConfs(args=args)

    # The data, split between train and test sets:
    (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar100.load_data('fine', os.path.join(confs.python_root, 'data/datasets/cifar/cifar100/'))

    cifar_utils.start_training(confs)
