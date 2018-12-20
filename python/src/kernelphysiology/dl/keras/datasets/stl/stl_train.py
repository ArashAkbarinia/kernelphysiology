'''
Train a simple DNN on STL 10.
'''


from kernelphysiology.dl.keras.datasets.stl import stl10

from kernelphysiology.dl.keras.utils import get_generators, get_validatoin_generator


def prepare_stl10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = stl10.load_data(dirname=args.data_dir)

    return args


def prepare_stl10_generators(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = stl10.load_data(dirname=args.data_dir)

    args = get_generators(args, x_train, y_train, x_test, y_test)
    return args


def stl10_validation_generator(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = stl10.load_data(dirname=args.data_dir)

    args = get_validatoin_generator(args, x_test, y_test)
    return args