'''
Testing a Keras model of CIFAR or STL against different levels of contrast.
'''


import os
import sys
import numpy as np
import glob

import commons
from kernelphysiology.dl.keras.cifar import cifar10
from kernelphysiology.dl.keras.cifar import cifar100
from kernelphysiology.dl.keras.stl import stl10
from kernelphysiology.dl.keras import contrast_net

import keras

from kernelphysiology.utils.imutils import adjust_contrast


if __name__ == "__main__":
    args = sys.argv[2:]

    if os.path.isdir(args[0]):
        dirname = args[0]
        args = sorted(glob.glob(dirname + '*.h5'))

    dataset = sys.argv[1]
    if dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data(os.path.join(commons.python_root, 'data/datasets/cifar/cifar10/'))
    elif dataset.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine', os.path.join(commons.python_root, 'data/datasets/cifar/cifar100/'))
    elif dataset.lower() == 'stl10':
        (x_train, y_train), (x_test, y_test) = stl10.load_data(os.path.join(commons.python_root, 'data/datasets/stl/stl10/'))

    contrasts = np.array([1, 3, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]) / 100

    results = np.zeros((np.size(args, 0), np.size(contrasts, 0)))
    i = 0
    for model_path in args:
        print(model_path)
        # loading the model
        model = keras.models.load_model(model_path)
        j = 0
        for contrast in contrasts:
            # reduce the contrast of image
            x_test_contrast = adjust_contrast(x_test, contrast)
            # score trained model
            x_test_contrast = contrast_net.preprocess_input(x_test_contrast)
            scores = model.evaluate(x_test_contrast, y_test, verbose=0)
            results[i, j] = scores[1]
            j += 1
        i += 1
    print(*results)
