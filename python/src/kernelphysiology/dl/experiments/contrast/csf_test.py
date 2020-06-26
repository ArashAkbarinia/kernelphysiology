import numpy as np
import argparse
import sys

import torch

from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.experiments.contrast import dataloader
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.utils import imutils


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model_path', type=str)
    model_parser.add_argument('--db', type=str)
    model_parser.add_argument('--out_file', type=str)
    model_parser.add_argument('--imagenet_dir', type=str, default=None)
    model_parser.add_argument('--batch_size', type=int, default=1)
    model_parser.add_argument('--noise', nargs='+', type=str, default=None)
    model_parser.add_argument('--contrasts', nargs='+', type=str, default=None)
    return parser.parse_args(args)


def run_gratings(db, model, out_file, contrasts):
    grating_db = 0
    grating_ind = 1

    test_sfs = np.linspace(np.pi / 4, np.pi * 16, 32)
    if contrasts is None:
        test_contrasts = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.50, 1.00]
    else:
        test_contrasts = contrasts
    test_thetas = np.linspace(0, np.pi, 7)
    test_rhos = np.linspace(0, np.pi, 3)
    test_ps = [0.0, 1.0]

    num_tests = len(test_sfs) * len(test_contrasts) * len(test_thetas) * len(
        test_rhos) * len(test_ps)
    test_num = 0
    all_results = []
    header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
    for tcon in test_contrasts:
        db.datasets[grating_db].contrasts = [tcon, 0.00]
        for tsf in test_sfs:
            db.datasets[grating_db].lambda_wave = tsf
            for ttheta in test_thetas:
                db.datasets[grating_db].theta = ttheta
                for trho in test_rhos:
                    db.datasets[grating_db].rho = trho
                    for tp in test_ps:
                        db.datasets[grating_db].p = tp
                        test_img, target, _ = db.__getitem__(grating_ind)
                        test_img = test_img.cuda()
                        with torch.no_grad():
                            out = model(test_img.unsqueeze(0))
                            pred = out.cpu().argmax().numpy() == target

                        params = [tcon, tsf, ttheta, trho, tp, pred]
                        all_results.append(params)
                        percent = float(test_num) / float(num_tests)
                        print('%.2f [%d/%d]' % (percent, test_num, num_tests))
                        test_num += 1

    save_path = out_file + '.csv'
    np.savetxt(save_path, np.array(all_results), delimiter=',', header=header)


def main(args):
    args = parse_arguments(args)
    if args.imagenet_dir is None:
        args.imagenet_dir = '/home/arash/Software/imagenet/raw-data/validation/'
    vision_type = 'trichromat'
    colour_space = 'rgb'
    target_size = 224

    mean, std = model_utils.get_preprocessing_function(
        colour_space, vision_type
    )
    gratings_args = {'samples': 1000}
    noise_transformation = []
    if args.noise is not None:
        noise_kwargs = {'amount': float(args.noise[1])}
        noise_transformation.append(
            cv2_preprocessing.UniqueTransformation(
                imutils.gaussian_noise, **noise_kwargs
            )
        )
    db = dataloader.validation_set(
        args.db, args.imagenet_dir, target_size, mean, std,
        noise_transformation, **gratings_args
    )

    model, _ = model_utils.which_network_classification(args.model_path, 2)
    model.eval()
    model.cuda()

    if args.db == 'gratings':
        run_gratings(db, model, args.out_file, args.contrasts)


if __name__ == "__main__":
    main(sys.argv[1:])
