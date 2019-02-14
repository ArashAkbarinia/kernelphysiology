'''
Reading the COCO dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons

from kernelphysiology.dl.keras.datasets.coco.utils import CocoDataset, CocoConfig
from kernelphysiology.dl.keras.datasets.coco.evaluation import evaluate_coco


def train_config(args):
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
    val_type= 'val2017'
    coco = dataset_val.load_coco(args.validation_dir, val_type, year=2017, return_coco=True, auto_download=False)
    dataset_val.prepare()
    args.validation_set = dataset_val
    args.config = config
    args.coco = coco

    # FIXME move it
    # FIXME apecify which model is for which dataset
    from kernelphysiology.dl.keras.models import mrcnn as modellib
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)
    model_path = '/home/arash/Software/repositories/Mask_RCNN/mask_rcnn_coco.h5'
    model.load_weights(model_path, by_name=True)
    evaluate_coco(model, dataset_val, coco, 'bbox', limit=10)

    return args