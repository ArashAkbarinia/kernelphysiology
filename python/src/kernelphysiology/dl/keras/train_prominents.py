'''
Train prominent DNN architectures on various different datasets.
'''


import os
import commons
import time
import datetime
import argparse

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train
from kernelphysiology.dl.keras.imagenet import imagenet_train

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3


# TODO: for CIFAR and STL
def start_training(args):
    return


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
            args.model = model
        parallel_model = multi_gpu_model(args.model, gpus=args.multi_gpus)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if not parallel_model == None:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)
    else:
        parallel_model.fit_generator(generator=args.train_generator, steps_per_epoch=args.steps, epochs=args.epochs, verbose=1, validation_data=args.validation_generator,
                                     callbacks=args.callbacks)

    # save model and weights
    model_name = args.model_name + '.h5'
    model_path = os.path.join(args.save_dir, model_name)
    model.save_weights(model_path)


if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    parser = argparse.ArgumentParser(description='Training prominent nets of Keras.')
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network', type=str, help='Which network to be used')
    parser.add_argument('--area1layers', dest='area1layers', type=int, default=0, help='The number of layers in area 1 (default: 0)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--a1reduction', dest='area1_reduction', action='store_true', default=False, help='Whether to include a reduction layer in area 1 (default: False)')
    parser.add_argument('--a1dilation', dest='area1_dilation', action='store_true', default=False, help='Whether to include dilation in kernels in area 1 (default: False)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')
    parser.add_argument('--name', dest='experiment_name', type=str, default='Ex', help='The name of the experiment (default: Ex)')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, default=None, help='The path to a previous checkpoint to continue (default: None)')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--steps', dest='steps', type=int, default=None, help='Number of steps per epochs (default: number of samples divided by the batch size)')
    parser.add_argument('--train_contrast', dest='train_contrast', type=int, default=100, help='The level of contrast to be used at training (default: 100)')
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', default=False, help='Whether to augment data (default: False)')

    args = parser.parse_args()
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

    # TODO: maybe make it dynamic
    args.target_size = (224, 224)

    # TODO: don't add model to args and pass it to imagenet
    if network_name == 'resnet50':
        args.preprocessing_function = resnet50.preprocess_input
        args.model = resnet50.ResNet50(area1layers=int(args.area1layers))
    elif network_name == 'inception_v3':
        args.preprocessing_function = inception_v3.preprocess_input
        args.model = inception_v3.InceptionV3(area1layers=int(args.area1layers))

    # which model to run
    if dataset_name == 'cifar10':
        args = cifar_train.prepare_cifar10(args)
    elif dataset_name == 'cifar100':
        args = cifar_train.prepare_cifar100(args)
    elif dataset_name == 'stl10':
        args = stl_train.prepare_stl10(args)
    elif dataset_name == 'imagenet':
        args.train_dir = '/home/arash/Software/imagenet/raw-data/train/'
        args.validation_dir = '/home/arash/Software/imagenet/raw-data/validation/'
        args = imagenet_train.prepare_imagenet(args)

    if dataset_name == 'imagenet':
        start_training_generator(args)
    else:
        start_training(args)

    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))