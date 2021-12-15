"""

"""

import sys
import argparse

import numpy as np

import torch
from torchvision import models as pmodels
import torchvision.transforms as torch_transforms

from skimage import io

from kernelphysiology.dl.pytorch.utils import cv2_transforms


def _get_activation(name, acts_dict):
    def hook(model, input_x, output_y):
        acts_dict[name] = output_y.detach()

    return hook


def _create_resnet_hooks(model):
    act_dict = dict()
    rfhs = dict()
    for attr_name in ['maxpool', 'contrast_pool']:
        if hasattr(model, attr_name):
            area0 = getattr(model, attr_name)
            rfhs['area0'] = area0.register_forward_hook(_get_activation('area0', act_dict))
    for i in range(1, 5):
        attr_name = 'layer%d' % i
        act_name = 'area%d' % i
        area_i = getattr(model, attr_name)
        rfhs[act_name] = area_i.register_forward_hook(_get_activation(act_name, act_dict))
        for j in range(len(area_i)):
            for k in range(1, 4):
                attr_name = 'bn%d' % k
                if hasattr(area_i[j], attr_name):
                    act_name = 'area%d.%d_%d' % (i, j, k)
                    area_ijk = getattr(area_i[j], attr_name)
                    rfhs[act_name] = area_ijk.register_forward_hook(
                        _get_activation(act_name, act_dict)
                    )
    return act_dict, rfhs


def main(argv):
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        '-aname', '--architecture',
        required=True,
        type=str,
        help='Name of the architecture or network'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='The path to the data directory (default: None)'
    )
    args = parser.parse_args(argv)

    # creating the model
    model = pmodels.__dict__[args.architecture](pretrained=True)
    # creating the hooks
    act_dict, rfhs = _create_resnet_hooks(model)

    # transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = torch_transforms.Compose([
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    # preparing the data
    data_file_path = '%s/simulationOrder_preliminary.csv' % args.data_dir
    data_file = np.loadtxt(data_file_path, delimiter=',', dtype=str)
    data_file = data_file[1:]
    test_colour = data_file[:, 4].astype('int')
    shape = data_file[:, 5].astype('int')

    n_comps = 4
    img_dir = '%s/img' % (args.data_dir)
    all_preds = []
    # looping over all the data
    for test_ind, current_test in enumerate(data_file):
        print('Doing [%d/%d]' % (test_ind, len(data_file)))
        img_target = io.imread('%s/%s' % (img_dir, current_test[0]))
        img_ref0 = io.imread('%s/%s' % (img_dir, current_test[1]))
        img_ref1 = io.imread('%s/%s' % (img_dir, current_test[2]))
        img_ref2 = io.imread('%s/%s' % (img_dir, current_test[3]))

        # making it pytorch friendly
        img_batch = torch.stack(
            [transform(img_target), transform(img_ref0), transform(img_ref1), transform(img_ref2)]
        )

        # computing the activations
        _ = model(img_batch)

        all_distances = []
        for key, val in act_dict.items():
            current_acts = val.clone().cpu().numpy().squeeze()

            key_distance_mat = np.zeros((n_comps, n_comps))
            for i in range(0, n_comps - 1):
                for j in range(i, n_comps):
                    if i == j:
                        continue
                    diff = current_acts[i] - current_acts[j]
                    diff = (diff ** 2).sum() ** 0.5
                    key_distance_mat[i, j] = diff
                    key_distance_mat[j, i] = diff
            all_distances.append(key_distance_mat)
        all_distances = np.array(all_distances)
        distance_mat = np.mean(all_distances, axis=0)
        pred = distance_mat.mean(axis=0).argmax()
        all_preds.append(pred)

    # saving the results
    out_file = '%s.csv' % args.architecture
    np.savetxt(out_file, np.array(all_preds), delimiter=',')


if __name__ == '__main__':
    main(sys.argv[1:])
