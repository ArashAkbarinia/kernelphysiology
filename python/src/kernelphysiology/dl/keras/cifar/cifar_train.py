'''
Train a simple DNN on CIFAR 10 or 100.
'''


import os

from kernelphysiology.dl.keras.cifar import cifar_utils
from kernelphysiology.dl.keras.cifar import cifar10
from kernelphysiology.dl.keras.cifar import cifar100


def train_cifar10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar10.load_data()

    cifar_utils.start_training(args)


def train_cifar100(args):
    args.num_classes = 100

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar100.load_data()

    cifar_utils.start_training(args)
