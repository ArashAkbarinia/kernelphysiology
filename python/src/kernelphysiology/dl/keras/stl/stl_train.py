'''
Train a simple DNN on STL 10.
'''


from kernelphysiology.dl.keras.stl import stl10


def prepare_stl10(args):
    args.num_classes = 10

    # The data, split between train and test sets:
    (args.x_train, args.y_train), (args.x_test, args.y_test) = stl10.load_data()

    return args