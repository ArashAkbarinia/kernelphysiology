'''
Train a simple DNN on CIFAR 10 or 100.
'''


import os
import argparse

import cifar
import cifar10
import cifar100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training CIFAR.')
    parser.add_argument('--n', dest='num_classes', type=int, help='Number of classes either 10 or 100')
    parser.add_argument('--a', dest='area1_nlayers', type=int, default=1, help='The number of layers in area 1 (default: 1)')
    parser.add_argument('--dog', dest='add_dog', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')
    parser.add_argument('--b', dest='add_batch_elu', default=False, help='Whether to include batch normalisation between layers of area 1 (default: False)')

    args = parser.parse_args()

    num_classes = args.num_classes
    if num_classes == 10:
        confs = cifar.CifarConfs(args=args)
    
        # The data, split between train and test sets:
        (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar10.load_data(os.path.join(confs.project_root, 'data/datasets/cifar/cifar10/'))
    elif num_classes == 100:
        confs = cifar.CifarConfs(args=args)
    
        # The data, split between train and test sets:
        (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar100.load_data('fine', os.path.join(confs.project_root, 'data/datasets/cifar/cifar100/'))
    
    cifar.start_training(confs)
