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
    model_parser.add_argument('--target_size', type=int)
    model_parser.add_argument('--imagenet_dir', type=str, default=None)
    model_parser.add_argument('--colour_space', type=str, default='grey')
    model_parser.add_argument('--contrast_space', type=str, default=None)
    model_parser.add_argument('--batch_size', type=int, default=16)
    model_parser.add_argument('-j', '--workers', type=int, default=4)
    model_parser.add_argument('--noise', nargs='+', type=str, default=None)
    model_parser.add_argument('--contrasts', nargs='+', type=float,
                              default=None)
    model_parser.add_argument('--freqs', nargs='+', type=float,
                              default=None)
    model_parser.add_argument('--print', action='store_true', default=False)
    model_parser.add_argument('--gabor', action='store_true', default=False)
    model_parser.add_argument('--mosaic_pattern', type=str, default=None)
    model_parser.add_argument('--vision_type', type=str, default='trichromat')
    return parser.parse_args(args)


def run_gratings(db_loader, model, out_file, update=False):
    with torch.no_grad():
        header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
        all_results = []
        num_batches = db_loader.__len__()
        for i, (test_img, targets, item_settings) in enumerate(db_loader):
            test_img = test_img.cuda()

            out = model(test_img)
            preds = out.cpu().numpy().argmax(axis=1)
            targets = targets.numpy()
            item_settings = item_settings.numpy()

            for j in range(len(preds)):
                current_settings = item_settings[j]
                params = [*current_settings, preds[j] == targets[j]]
                all_results.append(params)
            num_tests = num_batches * test_img.shape[0]
            test_num = i * test_img.shape[0]
            percent = float(test_num) / float(num_tests)
            if update:
                print('%.2f [%d/%d]' % (percent, test_num, num_tests))

    save_path = out_file + '.csv'
    np.savetxt(save_path, np.array(all_results), delimiter=',', header=header)


def main(args):
    args = parse_arguments(args)
    if args.imagenet_dir is None:
        args.imagenet_dir = '/home/arash/Software/imagenet/raw-data/validation/'
    vision_type = 'trichromat'
    colour_space = args.colour_space
    target_size = args.target_size

    mean, std = model_utils.get_preprocessing_function(
        colour_space, vision_type
    )
    extra_transformations = []
    if args.noise is not None:
        noise_kwargs = {'amount': float(args.noise[1])}
        extra_transformations.append(
            cv2_preprocessing.UniqueTransformation(
                imutils.gaussian_noise, **noise_kwargs
            )
        )
    if args.mosaic_pattern is not None:
        mosaic_trans = cv2_preprocessing.MosaicTransformation(
            args.mosaic_pattern
        )
        extra_transformations.append(mosaic_trans)

    # testing setting
    freqs = args.freqs
    if freqs is None:
        t4 = target_size / 4
        t2 = target_size / 2
        sf_base = ((target_size / 2) / np.pi)
        test_sfs = [
            sf_base / e for e in
            [*np.arange(1, 21), *np.arange(21, 61, 5),
             *np.arange(61, t4, 25), t4, t2]
        ]
    else:
        if len(freqs) == 3:
            test_sfs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
        else:
            test_sfs = freqs
    contrasts = args.contrasts
    if contrasts is None:
        test_contrasts = [
            0.001, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.023, 0.026, 0.029,
            0.032, 0.036, 0.040, 0.045, 0.050, 0.100, 0.200
        ]
    else:
        test_contrasts = contrasts
    test_thetas = np.linspace(0, np.pi, 7)
    test_rhos = np.linspace(0, np.pi, 3)
    test_ps = [0.0, 1.0]
    test_samples = {
        'amp': test_contrasts, 'lambda_wave': test_sfs,
        'theta': test_thetas, 'rho': test_rhos, 'side': test_ps
    }
    gratings_args = {
        'samples': test_samples, 'colour_space': colour_space,
        'contrast_space': args.contrast_space, 'vision_type': args.vision_type,
        'gabor_like': args.gabor
    }

    db = dataloader.validation_set(
        args.db, target_size, mean, std, extra_transformations,
        gratings_kwargs=gratings_args
    )
    db_loader = torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model, _ = model_utils.which_network_classification(args.model_path, 2)
    model.eval()
    model.cuda()

    if args.db == 'gratings':
        run_gratings(
            db_loader, model, args.out_file, args.print
        )


if __name__ == "__main__":
    main(sys.argv[1:])
