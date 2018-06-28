'''
Train a simple DNN on CIFAR 10 or 100.
'''


import os
import sys
import cifar
import cifar10
import cifar100


if __name__ == "__main__":
    args = sys.argv[2:]
    num_classes = int(sys.argv[1])
    if num_classes == 10:
        confs = cifar.CifarConfs(num_classes=num_classes, args=args)
    
        # The data, split between train and test sets:
        (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar10.load_data(os.path.join(confs.project_root, 'data/datasets/cifar/cifar10/'))
    elif num_classes == 100:
        confs = cifar.CifarConfs(num_classes=num_classes, args=args)
    
        # The data, split between train and test sets:
        (confs.x_train, confs.y_train), (confs.x_test, confs.y_test) = cifar100.load_data('fine', os.path.join(confs.project_root, 'data/datasets/cifar/cifar100/'))
    
    cifar.start_training(confs)
