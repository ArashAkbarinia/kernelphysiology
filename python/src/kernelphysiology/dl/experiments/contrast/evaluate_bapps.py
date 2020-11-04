import argparse
import sys
import numpy as np

import torch
import torchvision.transforms as torch_transforms

from kernelphysiology.dl.pytorch.models import model_utils

from kernelphysiology.dl.experiments.contrast import pretrained_models
from kernelphysiology.dl.experiments.contrast import contrast_utils
from kernelphysiology.utils import path_utils
from kernelphysiology.dl.pytorch.datasets import image_quality
from kernelphysiology.dl.pytorch.utils import cv2_transforms

DISTORTIONS_2AFC = [
    'cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional'
]
DISTORTIONS_JND = [
    'cnn', 'traditional'
]


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model_name', type=str)
    model_parser.add_argument('--model_path', type=str, default=None)
    model_parser.add_argument('--activation_layer', type=str)
    model_parser.add_argument('--db_dir', type=str)
    model_parser.add_argument('--split', type=str, default='val')
    model_parser.add_argument('--task', type=str, choices=['2afc', 'jnd'])
    model_parser.add_argument(
        '--distortion', type=str, default=None, choices=DISTORTIONS_2AFC
    )
    model_parser.add_argument('--out_file', type=str)
    model_parser.add_argument('--target_size', type=int)
    model_parser.add_argument('--colour_space', type=str, default='rgb')
    model_parser.add_argument('--batch_size', type=int, default=16)
    model_parser.add_argument('-j', '--workers', type=int, default=4)
    model_parser.add_argument('--print', action='store_true', default=False)
    model_parser.add_argument('--vision_type', type=str, default='trichromat')
    return parser.parse_args(args)


def run_jnd(db_loader, model, print_val):
    with torch.no_grad():
        all_diffs = []
        all_gts = []
        num_batches = db_loader.__len__()
        for i, (img0, img1, gt) in enumerate(db_loader):
            img0 = img0.cuda()
            img1 = img1.cuda()
            gt = gt.squeeze()

            out0 = model(img0)
            out1 = model(img1)

            # normalise the activations
            out0 = contrast_utils._normalise_tensor(out0)
            out1 = contrast_utils._normalise_tensor(out1)

            # compute the difference
            diffs = (out0 - out1) ** 2

            # collapse the differences
            diffs = contrast_utils._spatial_average(
                diffs.sum(dim=1, keepdim=True), keepdim=True
            ).squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)

            all_diffs.extend(diffs.detach().cpu().numpy())
            all_gts.extend(gt.detach().numpy())

            num_tests = num_batches * img0.shape[0]
            test_num = i * img0.shape[0]
            percent = float(test_num) / float(num_tests)
            if print_val is not None:
                print(
                    '%s %.2f [%d/%d]' % (
                        print_val, percent, test_num, num_tests
                    )
                )
    all_scores = contrast_utils.report_jnd(all_gts, all_diffs)
    return {'diff': all_diffs, 'score': all_scores, 'gt': all_gts}


def run_2afc(db_loader, model, print_val):
    with torch.no_grad():
        all_results = []
        num_batches = db_loader.__len__()
        for i, (img_ref, img_p0, img_p1, gt) in enumerate(db_loader):
            img_ref = img_ref.cuda()
            img_p0 = img_p0.cuda()
            img_p1 = img_p1.cuda()
            gt = gt.squeeze().cuda()

            out_ref = model(img_ref)
            out_p0 = model(img_p0)
            out_p1 = model(img_p1)

            # normalise the activations
            out_ref = contrast_utils._normalise_tensor(out_ref)
            out_p0 = contrast_utils._normalise_tensor(out_p0)
            out_p1 = contrast_utils._normalise_tensor(out_p1)

            # compute the difference
            d0s = (out_ref - out_p0) ** 2
            d1s = (out_ref - out_p1) ** 2

            # collapse the differences
            d0s = contrast_utils._spatial_average(
                d0s.sum(dim=1, keepdim=True), keepdim=True
            ).squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)
            d1s = contrast_utils._spatial_average(
                d1s.sum(dim=1, keepdim=True), keepdim=True
            ).squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)

            scores = contrast_utils.compute_2afc_score(d0s, d1s, gt)
            all_results.extend(scores.detach().cpu().numpy())

            num_tests = num_batches * img_ref.shape[0]
            test_num = i * img_ref.shape[0]
            percent = float(test_num) / float(num_tests)
            if print_val is not None:
                print(
                    '%s %.2f [%d/%d]' % (
                        print_val, percent, test_num, num_tests
                    )
                )
    return all_results


def save_results(eval_results, out_file):
    save_path = out_file + '.pickle'
    path_utils.write_pickle(save_path, eval_results)
    return


def main(args):
    args = parse_arguments(args)
    if args.model_path is None:
        args.model_path = args.model_name
    colour_space = args.colour_space
    target_size = args.target_size

    # loading the model
    is_segmentation = False
    if 'deeplabv3_resnet' in args.model_name or 'fcn_resnet' in args.model_name:
        is_segmentation = True
    transfer_weights = [args.model_path, None, is_segmentation]
    model = pretrained_models.get_pretrained_model(
        args.model_name, transfer_weights
    )

    # selecting the layer
    model = pretrained_models.LayerActivation(
        pretrained_models.get_backbones(args.model_name, model),
        args.activation_layer
    )
    model = model.eval()
    model.cuda()

    mean, std = model_utils.get_preprocessing_function(
        colour_space, 'trichromat'
    )
    transform = torch_transforms.Compose([
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std),
    ])

    if args.task == '2afc':
        default_dist = DISTORTIONS_2AFC
        db_class = image_quality.BAPPS2afc
        run_fun = run_2afc
    else:
        default_dist = DISTORTIONS_JND
        db_class = image_quality.BAPPSjnd
        run_fun = run_jnd
    distortions = default_dist if args.distortion is None else [args.distortion]

    eval_results = dict()
    for dist in distortions:
        print('Starting with %s' % dist)
        db = db_class(
            root=args.db_dir, split=args.split, distortion=dist,
            transform=transform
        )
        db_loader = torch.utils.data.DataLoader(
            db, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print_val = dist if args.print else None
        eval_results[dist] = run_fun(db_loader, model, print_val)
    save_results(eval_results, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
