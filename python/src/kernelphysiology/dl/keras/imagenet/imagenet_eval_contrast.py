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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3

from kernelphysiology.utils.imutils import adjust_contrast
from kernelphysiology.dl.keras.models import ResNet50


if __name__ == "__main__":
    args = sys.argv[1:]

    if os.path.isdir(args[0]):
        dirname = args[0]
        args = sorted(glob.glob(dirname + '*.h5'))

    num_classes = 1000

    # TODO put correct contrasts
    # TODO image size for inception and others might be different
    contrasts = np.array([5, 15, 50, 100]) / 100 # [1, 3, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]

    chunk_size = 5000
    ntests = 50000

    results_top1 = np.zeros((np.size(args, 0), np.size(contrasts, 0), ntests))
    results_top5 = np.zeros((np.size(args, 0), np.size(contrasts, 0), ntests))

    # iterating through the models
    i = 0
    for model_path in args:
        print('Processing model %s ' % (model_path))
        if model_path.lower() == 'vgg16':
            model_name = 'vgg16'
            model = VGG16()
            decode_predictions = keras.applications.vgg16.decode_predictions
            preprocess_input = keras.applications.vgg16.preprocess_input
        elif model_path.lower() == 'vgg19':
            model_name = 'vgg19'
            model = VGG19()
            decode_predictions = keras.applications.vgg19.decode_predictions
            preprocess_input = keras.applications.vgg19.preprocess_input
        elif model_path.lower() == 'resnet50':
            model_name = 'resnet50'
            model = keras.applications.resnet50.ResNet50()
            decode_predictions = keras.applications.resnet50.decode_predictions
            preprocess_input = keras.applications.resnet50.preprocess_input
        elif model_path.lower() == 'inception':
            model_name = 'inception'
            model = InceptionV3()
            decode_predictions = keras.applications.inception_v3.decode_predictions
            preprocess_input = keras.applications.inception_v3.preprocess_input
        else:
            model_name = os.path.basename(model_path)[0:-3]
            if model_name == 'org':
                area1layers = 1
            else:
                area1layers = 2
            model = ResNet50.ResNet50(area1layers=area1layers)
            model.load_weights(model_path)
            # TODO: fix me with correct preprocessings
            decode_predictions = ResNet50.decode_predictions
            preprocess_input = ResNet50.preprocess_input

        for start in range(0, ntests, chunk_size):
            # read the data by chunks
            end = min(ntests, start + chunk_size)
            which_chunk = (start, end)
            print('Reading chunk: ' + str(start) + '-' + str(end))
    
            (x_test, y_test) = imagenet.load_test_data('/home/ImageNet/Val_Images_RGB/', which_chunk=which_chunk)
            x_test = x_test.astype('float32')
            x_test /= 255

            # iterating through the contrasts
            j = 0
            for contrast in contrasts:
                # evaluate the model
                x_test_contrast = adjust_contrast(x_test, contrast)
                x_test_contrast *= 255
                x_test_contrast = preprocess_input(x_test_contrast)
                predicts = model.predict(x_test_contrast)
                model_outs = decode_predictions(predicts, top=5)
                model_outs = np.array(model_outs)
                for c in range(0, chunk_size):
                    if y_test[c] == model_outs[c, 0, 0]:
                        results_top1[i, j, start + c] = 1
                        results_top5[i, j, start + c] = 1
                    elif any(y_test[c] in s for s in model_outs[c, :, 0]):
                        results_top5[i, j, start + c] = 1
                j += 1
        if end == ntests:
            np.savetxt('top1%s.txt' % (model_name), np.transpose(results_top1[i, :, :], (1, 0)), fmt='%d')
            np.savetxt('top5%s.txt' % (model_name), np.transpose(results_top5[i, :, :], (1, 0)), fmt='%d')
            print('%s: Top1' % (model_name), *np.mean(results_top1[i, :, :], axis=1))
            print('%s: Top5' % (model_name), *np.mean(results_top5[i, :, :], axis=1))
        i += 1
