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
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess

from utils.imutils import adjust_contrast


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if os.path.isdir(args[0]):
        dirname = args[0]
        args = sorted(glob.glob(dirname + '*.h5'))
    
    num_classes = 1000
    
    contrasts = np.array([80, 100]) / 100 # np.array([1, 3, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]) / 100

    # TODO put other models as well
    model = VGG16()
    results = np.zeros((np.size(args, 0), np.size(contrasts, 0)))
    i = 0
    chunk_size = 1000
    ntests = 50000
    for start in range(0, ntests, chunk_size):
        # TODO: read the data by chunks
        end = min(ntests, start + chunk_size)
        which_chunk = (start, end)
        print('Reading chunk: ' + str(start) + '-' + str(end))
        
        (x_test, y_test) = imagenet.load_test_data('/home/ImageNet/Val_Images_RGB/', which_chunk=which_chunk)
        x_test = vgg_preprocess(x_test)
        #    y_test = keras.utils.to_categorical(y_test, num_classes)
        for model_path in args:
            print(model_path)
            # loading the model
    #        model = keras.models.load_model(model_path)
            j = 0
            for contrast in contrasts:
                # Score trained model.
                predicts = model.predict(adjust_contrast(x_test, contrast))
    #            scores = model.evaluate(adjust_contrast(x_test, contrast), y_test, verbose=1)
    #            results[i, j] = scores[1]
                j += 1
            i += 1
    print(results)
