import numpy as np
import argparse
import sys

import torch

from skimage import io

from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.experiments.contrast import dataloader
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.utils import imutils
from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor

from kernelphysiology.dl.experiments.contrast import pretrained_models
from kernelphysiology.dl.experiments.contrast import models_csf


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model_path', type=str)
    model_parser.add_argument('--db', type=str)
    model_parser.add_argument('--out_file', type=str)
    model_parser.add_argument('--target_size', type=int)
    model_parser.add_argument('--imagenet_dir', type=str, default=None)
    model_parser.add_argument('--colour_space', type=str, default='rgb')
    model_parser.add_argument('--contrast_space', type=str, default=None)
    model_parser.add_argument('--batch_size', type=int, default=1)
    model_parser.add_argument('-j', '--workers', type=int, default=4)
    model_parser.add_argument('--noise', nargs='+', type=str, default=None)
    model_parser.add_argument('--contrasts', nargs='+', type=float,
                              default=None)
    model_parser.add_argument('--freqs', nargs='+', type=float,
                              default=None)
    model_parser.add_argument('--print', action='store_true', default=False)
    model_parser.add_argument('--gabor', type=str, default=None)
    model_parser.add_argument('--visualise', action='store_true', default=False)
    model_parser.add_argument('--model_fest', action='store_true',
                              default=False)
    model_parser.add_argument('--mosaic_pattern', type=str, default=None)
    model_parser.add_argument('--vision_type', type=str, default='trichromat')
    model_parser.add_argument('--pretrained', action='store_true',
                              default=False)
    model_parser.add_argument('--repeat', action='store_true', default=False)
    model_parser.add_argument('--grey_width', default=0, choices=[0, 40],
                              type=int)
    model_parser.add_argument('--avg_illuminant', default=0, type=float)
    model_parser.add_argument('--side_by_side', action='store_true',
                              default=False)
    model_parser.add_argument('--scale_factor', default=None, type=float)
    return parser.parse_args(args)


def run_gratings(db_loader, model, out_file, update=False, mean_std=None,
                 old_results=None):
    with torch.no_grad():
        header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
        new_results = []
        num_batches = db_loader.__len__()
        for i, (test_img, targets, item_settings) in enumerate(db_loader):
            test_img = test_img.cuda()

            out = model(test_img)
            preds = out.cpu().numpy().argmax(axis=1)
            targets = targets.numpy()
            item_settings = item_settings.numpy()

            if mean_std is not None:
                img_inv = inv_normalise_tensor(test_img, mean_std[0],
                                               mean_std[1])
                img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                img_inv = np.concatenate(img_inv, axis=1)
                save_path = '%s%.5d.png' % (out_file, i)
                img_inv = np.uint8((img_inv.squeeze() * 255))
                io.imsave(save_path, img_inv)

            for j in range(len(preds)):
                current_settings = item_settings[j]
                params = [*current_settings, preds[j] == targets[j]]
                new_results.append(params)
            num_tests = num_batches * test_img.shape[0]
            test_num = i * test_img.shape[0]
            percent = float(test_num) / float(num_tests)
            if update:
                print('%.2f [%d/%d]' % (percent, test_num, num_tests))

    save_path = out_file + '.csv'
    if old_results is not None:
        all_results = [*old_results, *new_results]
    else:
        all_results = new_results
    all_results = np.array(all_results)
    np.savetxt(save_path, all_results, delimiter=',', header=header)
    return np.array(new_results), all_results


def run_gratings_separate(db_loader, model, out_file, update=False,
                          mean_std=None, old_results=None):
    with torch.no_grad():
        header = 'Contrast,SpatialFrequency,Theta,Rho,Side,Prediction'
        new_results = []
        num_batches = db_loader.__len__()
        for i, (timg0, timg1, targets, item_settings) in enumerate(db_loader):
            timg0 = timg0.cuda()
            timg1 = timg1.cuda()

            out = model(timg0, timg1)
            preds = out.cpu().numpy().argmax(axis=1)
            targets = targets.numpy()
            item_settings = item_settings.numpy()

            if mean_std is not None:
                timgs = torch.cat([timg0, timg1], dim=2)
                img_inv = inv_normalise_tensor(timgs, mean_std[0], mean_std[1])
                img_inv = img_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
                img_inv = np.concatenate(img_inv, axis=1)
                save_path = '%s%.5d.png' % (out_file, i)
                img_inv = np.uint8((img_inv.squeeze() * 255))
                io.imsave(save_path, img_inv)

            for j in range(len(preds)):
                current_settings = item_settings[j]
                params = [*current_settings, preds[j] == targets[j]]
                new_results.append(params)
            num_tests = num_batches * timg0.shape[0]
            test_num = i * timg0.shape[0]
            percent = float(test_num) / float(num_tests)
            if update:
                print('%.2f [%d/%d]' % (percent, test_num, num_tests))

    save_path = out_file + '.csv'
    if old_results is not None:
        all_results = [*old_results, *new_results]
    else:
        all_results = new_results
    all_results = np.array(all_results)
    np.savetxt(save_path, all_results, delimiter=',', header=header)
    return np.array(new_results), all_results


def sensitivity_sf(result_mat, sf, varname='all', th=0.75, low=0, high=1):
    result_mat = result_mat[result_mat[:, 1] == sf, :]
    unique_contrast = np.unique(result_mat[:, 0])
    accs = []
    for contrast in unique_contrast:
        accs.append(result_mat[result_mat[:, 0] == contrast, -1].mean())

    max_ind = 0
    diff_acc = accs[max_ind] - th
    contrast_i = unique_contrast[max_ind]
    if abs(diff_acc) < 0.005:
        return None, 0, 1
    elif diff_acc > 0:
        return (low + contrast_i) / 2, low, contrast_i
    else:
        return (high + contrast_i) / 2, contrast_i, high


def main(args):
    args = parse_arguments(args)
    if args.imagenet_dir is None:
        args.imagenet_dir = '/home/arash/Software/imagenet/raw-data/validation/'
    colour_space = args.colour_space
    target_size = args.target_size
    if args.repeat:
        target_size = int(args.target_size / 2)

    mean, std = model_utils.get_preprocessing_function(
        colour_space, 'trichromat'
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
        if args.model_fest:
            test_sfs = [
                107.558006181068, 60.2277887428434, 42.5666842683747,
                30.1176548488805, 21.2841900445655, 15.0589946730047,
                10.6611094334144, 7.5293662203794, 5.33055586478039,
                4.01568430264343
            ]
            args.gabor = 'model_fest'
        else:
            if target_size == 256:
                t4s = [
                    target_size / 2, target_size / 2.5, target_size / 3,
                    target_size / 3.5, target_size / 3.75, target_size / 4,
                ]
            else:
                # assuming 128
                t4s = [64]

            sf_base = ((target_size / 2) / np.pi)
            test_sfs = [
                sf_base / e for e in
                [*np.arange(1, 21), *np.arange(21, 61, 5),
                 *np.arange(61, t4s[-1], 25), *t4s]
            ]
    else:
        if len(freqs) == 3:
            test_sfs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
        else:
            test_sfs = freqs
    # so the sfs gets sorted
    test_sfs = np.unique(test_sfs)
    contrasts = args.contrasts
    if contrasts is None:
        test_contrasts = [0.5]
    else:
        test_contrasts = contrasts
    test_thetas = np.linspace(0, np.pi, 7)
    test_rhos = np.linspace(0, np.pi, 4)
    test_ps = [0.0, 1.0]

    if args.pretrained:
        if args.side_by_side:
            if args.scale_factor is None:
                scale_factor = (args.target_size / 256) ** 2
            else:
                scale_factor = args.scale_factor
            model = pretrained_models.NewClassificationModel(
                args.model_path, grey_width=args.grey_width == 40,
                scale_factor=scale_factor
            )
        else:
            if args.scale_factor is None:
                scale_factor = (args.target_size / 128) ** 2
            else:
                scale_factor = args.scale_factor
            model = models_csf.ContrastDiscrimination(
                args.model_path, grey_width=args.grey_width == 40,
                scale_factor=scale_factor
            )
    else:
        model, _ = model_utils.which_network_classification(args.model_path, 2)
    model.eval()
    model.cuda()

    mean_std = None
    if args.visualise:
        mean_std = (mean, std)

    if args.avg_illuminant < 0:
        max_high = 1 + 2 * args.avg_illuminant
    elif args.avg_illuminant > 0:
        max_high = 1 + -2 * args.avg_illuminant
    else:
        max_high = 1.0
    mid_contrast = (0 + max_high) / 2

    all_results = None
    csf_flags = [mid_contrast for _ in test_sfs]

    if args.db == 'gratings':
        for i in range(len(csf_flags)):
            low = 0
            high = max_high
            j = 0
            while csf_flags[i] is not None:
                print(
                    '%.2d %.3d Doing %f - %f %f %f' % (
                        i, j, test_sfs[i], csf_flags[i], low, high
                    )
                )

                test_samples = {
                    'amp': [csf_flags[i]], 'lambda_wave': [test_sfs[i]],
                    'theta': test_thetas, 'rho': test_rhos, 'side': test_ps,
                    'avg_illuminant': args.avg_illuminant
                }
                db_params = {
                    'colour_space': colour_space,
                    'vision_type': args.vision_type, 'repeat': args.repeat,
                    'mask_image': args.gabor, 'grey_width': args.grey_width,
                    'side_by_side': args.side_by_side
                }

                db = dataloader.validation_set(
                    args.db, target_size, mean, std, extra_transformations,
                    data_dir=test_samples, **db_params
                )
                db.contrast_space = args.contrast_space

                db_loader = torch.utils.data.DataLoader(
                    db, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                )

                if args.side_by_side:
                    new_results, all_results = run_gratings(
                        db_loader, model, args.out_file,
                        args.print, mean_std=mean_std, old_results=all_results
                    )
                else:
                    new_results, all_results = run_gratings_separate(
                        db_loader, model, args.out_file,
                        args.print, mean_std=mean_std, old_results=all_results
                    )
                new_contrast, low, high = sensitivity_sf(
                    new_results, test_sfs[i], varname='all', th=0.75,
                    low=low, high=high
                )
                if (
                        abs(csf_flags[i] - max_high) < 1e-3
                        or new_contrast == csf_flags[i]
                        or j == 20
                ):
                    print('had to skip', csf_flags[i])
                    csf_flags[i] = None
                else:
                    csf_flags[i] = new_contrast
                j += 1


if __name__ == "__main__":
    main(sys.argv[1:])
