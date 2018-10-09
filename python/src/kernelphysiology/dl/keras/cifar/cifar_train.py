'''
Train a simple DNN on CIFAR 10 or 100.
'''


from kernelphysiology.dl.keras.cifar import cifar10
from kernelphysiology.dl.keras.cifar import cifar100

from kernelphysiology.dl.keras.utils import get_generators


def prepare_cifar10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar10.load_data()

    return args


def prepare_cifar10_generators(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    args = get_generators(args, x_train, y_train, x_test, y_test)
    return args


def prepare_cifar100(args):
    args.num_classes = 100

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar100.load_data()

    return args


def prepare_cifar100_generators(args):
    args.num_classes = 100

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    args = get_generators(args, x_train, y_train, x_test, y_test)
    return args