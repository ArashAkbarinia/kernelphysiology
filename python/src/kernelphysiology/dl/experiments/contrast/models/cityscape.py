import sys

rep_path = '/home/arash/Software/repositories/others/semantic-segmentation/'
sys.path.append(rep_path)

from network import deepv3
from optimizer import restore_snapshot


def citymode():
    model = deepv3.DeepWV3Plus(21)

    model_path = rep_path + '/pretrained_models/cityscapes_best.pth'
    model, _ = restore_snapshot(model, optimizer=None, snapshot=model_path,
                                restore_optimizer_bool=False)
    model.cpu()

    return model
