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

from utils.imutils import adjust_contrast


if __name__ == "__main__":
    args = sys.argv[1:]

    if os.path.isdir(args[0]):
        dirname = args[0]
        args = sorted(glob.glob(dirname + '*.h5'))

    num_classes = 1000

    contrasts = np.array([1, 3, 5, 10, 15, 20, 30, 40, 50, 65, 80, 100]) / 100

    chunk_size = 1000
    ntests = 50000

    results_top1 = np.zeros((np.size(args, 0), np.size(contrasts, 0), ntests))
    results_top5 = np.zeros((np.size(args, 0), np.size(contrasts, 0), ntests))
    for start in range(0, ntests, chunk_size):
        # read the data by chunks
        end = min(ntests, start + chunk_size)
        which_chunk = (start, end)
        print('Reading chunk: ' + str(start) + '-' + str(end))

        (x_test, y_test) = imagenet.load_test_data('/home/ImageNet/Val_Images_RGB/', which_chunk=which_chunk)
        # iterating through the models
        i = 0
        for model_path in args:
            print(model_path)
            if model_path.lower() == 'vgg16':
                model_name = 'vgg16'
                from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess, decode_predictions
                model = VGG16()
                x_test = vgg_preprocess(x_test)
            else:
                model_name = os.path.basename(model_path)[0:-3]
                model = keras.models.load_model(model_path)
            # iterating through the contrasts
            j = 0
            for contrast in contrasts:
                # evaluate the model
                predicts = model.predict(adjust_contrast(x_test, contrast))
                model_outs = decode_predictions(predicts, top=5)
                model_outs = np.array(model_outs)
                for c in range(0, chunk_size):
                    if y_test[c] == model_outs[c, 0, 0]:
                        results_top1[i, j, start + c] = 1
                        results_top5[i, j, start + c] = 1
                    elif any(y_test[c] in s for s in model_outs[c,:,0]):
                        results_top5[i, j, start + c] = 1
                j += 1
            if end == ntests:
                print(results_top1[i, :, :], file=open('top1%s.txt' % model_name, 'w'))
                print(results_top5[i, :, :], file=open('top5%s.txt' % model_name, 'w'))
                print('%s: Top1' % (model_name), np.mean(results_top1[i, :, :], axis=0), 'Top5', np.mean(results_top5[i, :, :], axis=0))
            i += 1
