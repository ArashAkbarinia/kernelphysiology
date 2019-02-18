'''
Reading the COCO dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons

from kernelphysiology.dl.keras.datasets.coco.utils import CocoDataset, CocoConfig


def train_config(args):
    config = CocoConfig()
    # FIXME move it
    # FIXME apecify which model is for which dataset
    from kernelphysiology.dl.keras.models import mrcnn as modellib
    import os
    args.save_dir = '/home/arash/Software/repositories/'
    args.model_name = 'mrcnn'
    args.log_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.log_dir)
    
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = CocoDataset()
    dataset_train.load_coco(args.data_dir, 'train', year=2017, auto_download=False)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    val_type = 'val'
    dataset_val.load_coco(args.data_dir, val_type, year=2017, auto_download=False)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    import imgaug
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=40, layers='heads', augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=120, layers='4+', augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=160, layers='all', augmentation=augmentation)
    return args


def validation_config(args):
    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
    config = InferenceConfig()

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco(args.data_dir, 'val', year=2017, return_coco=True, auto_download=False)
    dataset_val.prepare()
    args.validation_set = dataset_val
    args.config = config
    args.coco = coco

    return args
