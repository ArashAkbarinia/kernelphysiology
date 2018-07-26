'''
Train a simple DNN on CIFAR 10 or 100.
'''


import commons

import argparse

from kernelphysiology.dl.keras.cifar import cifar_train
from kernelphysiology.dl.keras.stl import stl_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training CNET.')
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument('--a1', dest='area1_nlayers', type=int, default=1, help='The number of layers in area 1 (default: 1)')
    parser.add_argument('--a1nb', dest='area1_batchnormalise', action='store_false', default=True, help='Whether to include batch normalisation between layers of area 1 (default: True)')
    parser.add_argument('--a1na', dest='area1_activation', action='store_false', default=True, help='Whether to include activation between layers of area 1 (default: True)')
    parser.add_argument('--a1reduction', dest='area1_reduction', action='store_true', default=False, help='Whether to include a reduction layer in area 1 (default: False)')
    parser.add_argument('--dog', dest='add_dog', action='store_true', default=False, help='Whether to add a DoG layer (default: False)')
    parser.add_argument('--mg', dest='multi_gpus', type=int, default=None, help='The number of GPUs to be used (default: None)')
    parser.add_argument('--name', dest='experiment_name', type=str, default='', help='The name of the experiment (default: None)')

    args = parser.parse_args()

    if args.dataset.lower() == 'cifar10':
        cifar_train.train_cifar10(args)
    elif args.dataset.lower() == 'cifar100':
        cifar_train.train_cifar100(args)
    elif args.dataset.lower() == 'stl10':
        stl_train.train_stl10(args)