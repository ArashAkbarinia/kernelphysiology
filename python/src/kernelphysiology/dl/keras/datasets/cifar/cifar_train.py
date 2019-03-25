'''
Train a simple DNN on CIFAR 10 or 100.
'''


from kernelphysiology.dl.keras.datasets.cifar import cifar10
from kernelphysiology.dl.keras.datasets.cifar import cifar100

from kernelphysiology.dl.keras.utils import get_generators, get_validatoin_generator


def prepare_cifar10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar10.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    return args


def prepare_cifar10_generators(args, train_preprocessing_function, validation_preprocessing_function):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    args = get_generators(args, x_train, y_train, x_test, y_test, train_preprocessing_function, validation_preprocessing_function)
    return args


def cifar10_validatoin_generator(args, validation_preprocessing_function):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    args = get_validatoin_generator(args, x_test, y_test, validation_preprocessing_function)
    return args


def prepare_cifar100(args):
    args.num_classes = 100

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = cifar100.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    return args


def prepare_cifar100_generators(args, train_preprocessing_function, validation_preprocessing_function):
    args.num_classes = 100

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    args = get_generators(args, x_train, y_train, x_test, y_test, train_preprocessing_function, validation_preprocessing_function)
    return args


def cifar100_validatoin_generator(args, validation_preprocessing_function):
    args.num_classes = 100

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(dirname=args.data_dir, which_outputs=args.output_types)

    args = get_validatoin_generator(args, x_test, y_test, validation_preprocessing_function)
    return args