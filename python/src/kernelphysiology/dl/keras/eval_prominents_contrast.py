'''
Testing a Keras model of CIFAR or STL against different levels of contrast.
'''


import commons

import sys
import numpy as np
import time
import datetime

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train
from kernelphysiology.dl.keras.imagenet import imagenet_train

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet

from kernelphysiology.dl.keras.prominent_utils import test_prominent_prepares, test_arg_parser
from kernelphysiology.dl.keras.prominent_utils import get_preprocessing_function, get_top_k_accuracy
from kernelphysiology.utils.imutils import adjust_contrast


def contrast_preprocessing(img, contrast, preprocessing_function=None):
    img = adjust_contrast(img, contrast) * 255
    if preprocessing_function:
        img = preprocessing_function(img)
    return img


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = test_arg_parser(sys.argv[1:])
    args = test_prominent_prepares(args)

    dataset_name = args.dataset.lower()

    contrasts = np.array(args.contrasts) / 100
    results_top1 = np.zeros((contrasts.shape[0], len(args.networks)))
    results_topk = np.zeros((contrasts.shape[0], len(args.networks)))
    for i, contrast in enumerate(contrasts):
        # maybe if only one preprocessing is used, the generators can be called only once
        for j, network_name in enumerate(args.networks):
            preprocessing = args.preprocessings[j]
            current_contrast_preprocessing = lambda img : contrast_preprocessing(img, contrast=contrast, preprocessing_function=get_preprocessing_function(preprocessing))
            args.validation_preprocessing_function = current_contrast_preprocessing
            # which dataset
            if dataset_name == 'cifar10':
                args = cifar_train.prepare_cifar10_generators(args)
            elif dataset_name == 'cifar100':
                args = cifar_train.prepare_cifar100_generators(args)
            elif dataset_name == 'stl10':
                args = stl_train.prepare_stl10_generators(args)
            elif dataset_name == 'imagenet':
                args.train_dir = '/home/arash/Software/imagenet/raw-data/train/'
                args.validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
                args = imagenet_train.validation_generator(args)
            print('Processing network %s and contrast %f' % (network_name, contrast))

            # which architecture
            if network_name == 'resnet50':
                args.model = resnet50.ResNet50(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'inception_v3':
                args.model = inception_v3.InceptionV3(classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'vgg16':
                args.model = vgg16.VGG16(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'vgg19':
                args.model = vgg19.VGG19(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'densenet121':
                args.model = densenet.DenseNet121(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'densenet169':
                args.model = densenet.DenseNet169(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            elif network_name == 'densenet201':
                args.model = densenet.DenseNet201(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
            else:
                args.model = keras.models.load_model(network_name, compile=False)

            top_k_acc = get_top_k_accuracy(args.top_k)
            metrics = ['accuracy', top_k_acc]
            opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)
            if len(args.gpus) == 1:
                # the compilation being necessary is a bug of keras
                args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                current_results = args.model.evaluate_generator(generator=args.validation_generator, verbose=1)
            else:
                with tf.device('/cpu:0'):
                    args.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                parallel_model = multi_gpu_model(args.model, gpus=args.gpus)
                parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
                current_results = parallel_model.evaluate_generator(generator=args.validation_generator, verbose=1)
            results_top1[i, j] = current_results[1]
            results_topk[i, j] = current_results[2]

    # saving the results in a CSV format
    np.savetxt(args.output_file + '_top1.csv', results_top1, delimiter=',')
    np.savetxt(args.output_file + '_top%d.csv' % args.top_k, results_topk, delimiter=',')

    finish_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
    print('Finishing at: ' + finish_time)
