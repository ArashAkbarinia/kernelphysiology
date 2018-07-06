'''
Testing a Keras model of IMAGENET against different levels of contrast.
'''


import os
import sys
import numpy as np
import glob

import commons
import imagenet

import keras
from keras.applications.vgg16 import VGG16, preprocess_input

from utils.imutils import adjust_contrast


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if os.path.isdir(args[0]):
        dirname = args[0]
        args = sorted(glob.glob(dirname + '*.h5'))
    
    num_classes = 10

    # The data, split between train and test sets:
    (x_test, y_test) = imagenet.load_test_data('/home/ImageNet/Val_Images_RGB/')
    
    x_test = preprocess_input(x_test)
#    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    contrasts = np.array([80, 100]) / 100 # np.array([1, 3, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]) / 100
    
    results = np.zeros((np.size(args, 0), np.size(contrasts, 0)))
    i = 0
    for model_path in args:
        print(model_path)
        # loading the model
#        model = keras.models.load_model(model_path)
        model = VGG16()
        j = 0
        for contrast in contrasts:
            # Score trained model.
            predicts = model.predict(y_test)
#            scores = model.evaluate(adjust_contrast(x_test, contrast), y_test, verbose=1)
#            results[i, j] = scores[1]
            j += 1
        i += 1
    print(results)
