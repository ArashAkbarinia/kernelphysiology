'''
Train a simple DNN on STL 10.
'''


from kernelphysiology.dl.keras.stl import stl10

from kernelphysiology.dl.keras.utils import resize_generator


def prepare_stl10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = stl10.load_data()

    return args


def prepare_stl10_generators(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = stl10.load_data()

    args = get_generators(args, x_train, y_train, x_test, y_test)
    return args


def get_generators(args, x_train, y_train, x_test, y_test):
    (args.train_generator, args.train_samples) = resize_generator(x_train, y_train, batch_size=args.batch_size,
                                            target_size=args.target_size, preprocessing_function=args.preprocessing_function,
                                            horizontal_flip=args.horizontal_flip, vertical_flip=args.vertical_flip)


    (args.validation_generator, args.validation_samples) = resize_generator(x_test, y_test, batch_size=args.batch_size,
                                            target_size=args.target_size, preprocessing_function=args.preprocessing_function)

    return args