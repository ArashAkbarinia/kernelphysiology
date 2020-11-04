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

PARTS = ['blurjpeg', 'blurnoise']


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model_name', type=str)
    model_parser.add_argument('--model_path', type=str, default=None)
    model_parser.add_argument('--activation_layer', type=str)
    model_parser.add_argument('--db_dir', type=str)
    model_parser.add_argument(
        '--part', type=str, choices=PARTS, default=None
    )
    model_parser.add_argument('--out_file', type=str)
    model_parser.add_argument('--target_size', type=int)
    model_parser.add_argument('--colour_space', type=str, default='rgb')
    model_parser.add_argument('--batch_size', type=int, default=16)
    model_parser.add_argument('-j', '--workers', type=int, default=4)
    model_parser.add_argument('--print', action='store_true', default=False)
    model_parser.add_argument('--vision_type', type=str, default='trichromat')
    return parser.parse_args(args)


def run_live(db_loader, model, print_val):
    with torch.no_grad():
        all_diffs = []
        all_moses = []
        all_zscores = []
        num_batches = db_loader.__len__()
        for i, (img0, img1, mos, zscore) in enumerate(db_loader):
            img0 = img0.cuda()
            img1 = img1.cuda()

            slice_diffs = 0
            for srow in range(0, img0.shape[2], 256):
                erow = srow + 256
                if erow > img0.shape[2]:
                    erow = img0.shape[2]
                    srow = erow - 256
                for scol in range(0, img0.shape[3], 512):
                    ecol = scol + 512
                    if ecol > img0.shape[3]:
                        ecol = img0.shape[3]
                        scol = ecol - 512
                    out0 = model(img0[:, :, srow:erow, scol:ecol])
                    out1 = model(img1[:, :, srow:erow, scol:ecol])

                    # normalise the activations
                    out0 = contrast_utils._normalise_tensor(out0)
                    out1 = contrast_utils._normalise_tensor(out1)

                    # compute the difference
                    diffs = (out0 - out1) ** 2

                    # collapse the differences
                    diffs = contrast_utils._spatial_average(
                        diffs.sum(dim=1, keepdim=True), keepdim=True
                    ).squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)
                    slice_diffs += diffs
            diffs = slice_diffs

            all_diffs.extend(diffs.detach().cpu().numpy())
            all_moses.extend(mos.detach().numpy())
            all_zscores.extend(zscore.detach().numpy())

            num_tests = num_batches * img0.shape[0]
            test_num = i * img0.shape[0]
            percent = float(test_num) / float(num_tests)
            if print_val is not None:
                print(
                    '%s %.2f [%d/%d]' % (
                        print_val, percent, test_num, num_tests
                    )
                )

    all_scores = {
        'mos': contrast_utils.report_jnd(all_moses, all_diffs),
        'zscore': contrast_utils.report_jnd(all_zscores, all_diffs),
    }
    all_gts = {'mos': all_moses, 'zscore': all_zscores}
    return {'diff': all_diffs, 'score': all_scores, 'gt': all_gts}


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

    parts = PARTS if args.part is None else [args.part]

    eval_results = dict()
    for part in parts:
        print('Starting with %s' % part)
        db = image_quality.LIVE(
            root=args.db_dir, part=part, transform=transform
        )
        db_loader = torch.utils.data.DataLoader(
            db, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print_val = part if args.print else None
        eval_results[part] = run_live(db_loader, model, print_val)
    save_results(eval_results, args.out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
