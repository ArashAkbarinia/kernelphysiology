'''
Train a simple DNN on CIFAR 10 or 100.
'''


import os
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
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size (default: 64)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='NUmber of epochs (default: 50)')
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true', default=False, help='Whether to augment data (default: False)')

    args = parser.parse_args()
    which_model = args.dataset.lower()
    # preparing arguments
    args.save_dir = os.path.join(commons.python_root, 'data/nets/%s/%s/' % (''.join([i for i in which_model if not i.isdigit()]), which_model))
    args.dog_path = os.path.join(args.save_dir, 'dog.h5')

    # preparing the name of the model
    args.model_name = 'keras_%s_area_%s_' % (which_model, args.experiment_name)
    if args.area1_batchnormalise:
        args.model_name += 'bnr_'
    if args.area1_activation:
        args.model_name += 'act_'
    if args.area1_activation:
        args.model_name += 'red_'
    if args.add_dog:
        args.model_name += 'dog_'
    # other parameters
    args.log_period = round(args.epochs / 4)

    # which model to run
    if which_model == 'cifar10':
        cifar_train.train_cifar10(args)
    elif which_model == 'cifar100':
        cifar_train.train_cifar100(args)
    elif which_model == 'stl10':
        stl_train.train_stl10(args)
