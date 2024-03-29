'''
Training the IMAGENET dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from kernelphysiology.dl.keras.datasets.imagenet import imagenet


def prepare_imagenet(args, train_preprocessing_function, validation_preprocessing_function):
    args.num_classes = 1000

    args.train_generator = imagenet.train_generator(args.train_dir, batch_size=args.batch_size,
                                                    target_size=args.target_size,
                                                    preprocessing_function=train_preprocessing_function,
                                                    crop_type=args.crop_type, shuffle=args.shuffle,
                                                    horizontal_flip=args.horizontal_flip,
                                                    vertical_flip=args.vertical_flip,
                                                    zoom_range=args.zoom_range,
                                                    width_shift_range=args.width_shift_range, height_shift_range=args.height_shift_range)
    args.train_samples = args.train_generator.samples

    args.validation_generator = imagenet.validation_generator(args.validation_dir, batch_size=args.batch_size,
                                                              target_size=args.target_size,
                                                              preprocessing_function=validation_preprocessing_function,
                                                              crop_type=args.validation_crop_type)
    args.validation_samples = args.validation_generator.samples

    return args


def validation_generator(args, validation_preprocessing_function):
    args.num_classes = 1000

    args.validation_generator = imagenet.validation_generator(args.validation_dir, batch_size=args.batch_size,
                                                              target_size=args.target_size,
                                                              preprocessing_function=validation_preprocessing_function,
                                                              crop_type=args.validation_crop_type)
    args.validation_samples = args.validation_generator.samples

    return args