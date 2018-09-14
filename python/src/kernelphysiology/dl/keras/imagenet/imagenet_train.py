'''
Training the IMAGENET dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from kernelphysiology.dl.keras.imagenet import imagenet


def prepare_imagenet(args):
    args.num_classes = 1000

    args.train_generator = imagenet.train_generator(args.train_dir, batch_size=32, target_size=(224, 224), preprocessing_function=args.preprocessing_function)

    args.validation_generator = imagenet.validation_generator(args.validation_dir, batch_size=32, target_size=(224, 224), preprocessing_function=args.preprocessing_function)

    return args