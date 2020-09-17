import sys

rep_path = '/home/arash/Software/repositories/others/Segment_Transparent_Objects/'
sys.path.append(rep_path)

from segmentron.config import cfg
from segmentron.models.model_zoo import get_segmentation_model


def get_transparency_model():
    config_file = rep_path + '/configs/trans10K/translab.yaml'

    model_path = rep_path + '/weights/16.pth'

    cfg.update_from_file(config_file)
    cfg.PHASE = 'test'
    cfg.DATASET.NAME = 'trans10k_extra'
    cfg.TEST.TEST_MODEL_PATH = model_path
    # cfg.check_and_freeze()

    return get_segmentation_model()
