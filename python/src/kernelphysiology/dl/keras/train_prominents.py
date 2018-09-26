'''
Train prominent DNN architectures on various different datasets.
'''


import os
import commons
import time
import datetime
import sys

import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train
from kernelphysiology.dl.keras.imagenet import imagenet_train

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet

from kernelphysiology.dl.keras.utils import common_arg_parser


def start_training_generator(args):

    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    checkpoint_logger = ModelCheckpoint(os.path.join(args.log_dir, 'model_weights.h5'), monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'), append=False, separator=';')
    args.callbacks = [csv_logger, checkpoint_logger]

    # TODO: put a switch case according to each network
#    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
    opt = keras.optimizers.SGD(lr=1e-1, momentum=0.9, decay=1e-4)

    model = args.model
    if args.multi_gpus == None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        parallel_model = None
    else:
        with tf.device('/cpu:0'):
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        parallel_model = multi_gpu_model(model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)
    else:
        model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)

    # save model and weights
    model_name = args.model_name + '.h5'
    model_path = os.path.join(args.save_dir, model_name)
    model.save_weights(model_path)


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    args = common_arg_parser(sys.argv[1:])
    dataset_name = args.dataset.lower()
    network_name = args.network.lower()

    # preparing arguments
    network_dir = os.path.join(commons.python_root, 'data/nets/%s/%s/%s/' % (''.join([i for i in dataset_name if not i.isdigit()]), dataset_name, network_name))
    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)
    args.save_dir = os.path.join(network_dir, args.experiment_name)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # preparing the name of the model
    args.model_name = 'keras_%s_%s_area_%d_contrast_%d' % (dataset_name, network_name, args.area1layers, args.train_contrast)
    if args.area1_batchnormalise:
        args.model_name += '_bnr'
    if args.area1_activation:
        args.model_name += '_act'
    if args.area1_reduction:
        args.model_name += '_red'
    if args.area1_dilation:
        args.model_name += '_dil'
    if args.add_dog:
        args.model_name += '_dog'
        args.dog_path = os.path.join(args.save_dir, 'dog.h5')

    args.target_size = (args.target_size, args.target_size)
    # check the input shape
    if K.image_data_format() == 'channels_last':
        args.input_shape = (*args.target_size, 3)
    elif K.image_data_format() == 'channels_first':
        args.input_shape = (3, *args.target_size)

    # choosing the preprocessing function
    preprocessing = args.preprocessing
    if not preprocessing:
        preprocessing = network_name
    # switch case of preprocessing functions
    if preprocessing == 'resnet50':
        args.preprocessing_function = resnet50.preprocess_input
    elif preprocessing == 'inception_v3':
        args.preprocessing_function = inception_v3.preprocess_input
    elif preprocessing == 'vgg16':
        args.preprocessing_function = vgg16.preprocess_input
    elif preprocessing == 'vgg19':
        args.preprocessing_function = vgg19.preprocess_input
    elif preprocessing == 'densenet121' or network_name == 'densenet169' or network_name == 'densenet201':
        args.preprocessing_function = densenet.preprocess_input

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
        args = imagenet_train.prepare_imagenet(args)

    # which architecture
    if network_name == 'resnet50':
        args.model = resnet50.ResNet50(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'inception_v3':
        args.model = inception_v3.InceptionV3(classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'vgg16':
        args.model = vgg16.VGG16(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'vgg19':
        args.model = vgg19.VGG19(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'densenet121':
        args.model = densenet.DenseNet121(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'densenet169':
        args.model = densenet.DenseNet169(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))
    elif network_name == 'densenet201':
        args.model = densenet.DenseNet201(input_shape=args.input_shape, classes=args.num_classes, area1layers=int(args.area1layers))

    start_training_generator(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))