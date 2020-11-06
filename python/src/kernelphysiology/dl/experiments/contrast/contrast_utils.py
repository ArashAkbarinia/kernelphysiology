import numpy as np
from scipy import stats

import torch
import torch.nn as nn

from torchvision.models import resnet as presnet
from kernelphysiology.dl.pytorch.models import resnet as cresnet
from kernelphysiology.dl.pytorch import models as custom_models
from kernelphysiology.dl.pytorch.models import pretrained_features


class AFCModel(nn.Module):
    def __init__(self, arch_name, layer_name, num_classes=2, weights_path=None):
        super(AFCModel, self).__init__()

        self.features = pretrained_features.ResNetIntermediate(
            arch_name=arch_name, layer_name=layer_name,
            weights_path=weights_path
        )
        # making the features unlearnable
        # for p in self.features.parameters():
        #     p.requires_grad = False

        in_planes = self.features.get_num_kernels()
        nkernels = 256
        self.cvon1x1 = nn.Conv2d(in_planes, nkernels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nkernels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cvon1x1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_afc_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    arch_name = checkpoint['arch']
    customs = checkpoint['customs']
    model = AFCModel(
        arch_name, customs['layer_name'], num_classes=customs['num_classes']
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model


class NewClassificationModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(NewClassificationModel, self).__init__()

        last_layer = list(original_model.children())[-1][0]
        if isinstance(last_layer, (cresnet.Bottleneck, presnet.Bottleneck)):
            org_classes = last_layer.conv3.out_channels
        else:
            org_classes = last_layer.conv2.out_channels
        self.features = nn.Sequential(*list(original_model.children()))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(org_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FaceModel(nn.Module):
    def __init__(self, network_name, transfer_weights):
        super(FaceModel, self).__init__()

        if 'resnet18' in network_name:
            backbone = 'resnet18'
        elif 'resnet50' in network_name:
            backbone = 'resnet50'
        model = custom_models.__dict__['deeplabv3_resnet'](
            backbone, num_classes=21, aux_loss=False
        )
        model = model.backbone

        model = NewClassificationModel(model, 500)
        face_net = torch.load(transfer_weights[0], map_location='cpu')
        model.load_state_dict(face_net['state_dict'])
        self.backbone = model.features


def _voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _normalise_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def _spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def report_jnd(all_gts, all_diffs):
    gt_array = np.array(all_gts)
    result_array = np.array(all_diffs)

    sorted_inds = np.argsort(result_array)
    sames_sorted = gt_array[sorted_inds]

    tps = np.cumsum(sames_sorted)
    fps = np.cumsum(1 - sames_sorted)
    fns = np.sum(sames_sorted) - tps

    precs = tps / (tps + fps)
    recs = tps / (tps + fns)
    mAP = _voc_ap(recs, precs)

    p_corr, p_pval = stats.pearsonr(gt_array, result_array)
    s_corr, s_pval = stats.spearmanr(gt_array, result_array)
    report = {
        'map': mAP, 'pearsonr': p_corr, 'spearmanr': s_corr
    }
    return report


def compute_2afc_score(d0s, d1s, gt):
    d0_smaller = (d0s < d1s) * (1.0 - gt)
    d1_smaller = (d1s < d0s) * gt
    scores = d0_smaller + d1_smaller + (d1s == d0s) * 0.5
    return scores
