import numpy as np
import sys
import cv2
from skimage import io
import glob

from kernelphysiology.utils import path_utils
from kernelphysiology.transformations import normalisations


def angular_diff(alpha, beta):
    ang_diff = alpha - beta
    ang_diff[ang_diff > 180] = 360 - ang_diff[ang_diff > 180]
    ang_diff[ang_diff < -180] = -360 - ang_diff[ang_diff < -180]
    return ang_diff


def hist(a, bins, data_range):
    hist_array, _ = np.histogram(a, bins, data_range)
    return hist_array


def report_vector_influence(img_org, img_full, img_lesion):
    img_org = normalisations.rgb2double(img_org)
    img_org_op = cv2.cvtColor(img_org, cv2.COLOR_RGB2LAB)
    img_org_hsv = cv2.cvtColor(img_org, cv2.COLOR_RGB2HSV)
    img_full = normalisations.rgb2double(img_full)
    img_full_op = cv2.cvtColor(img_full, cv2.COLOR_RGB2LAB)
    img_full_hsv = cv2.cvtColor(img_full, cv2.COLOR_RGB2HSV)
    img_lesion = normalisations.rgb2double(img_lesion)
    img_lesion_op = cv2.cvtColor(img_lesion, cv2.COLOR_RGB2LAB)
    img_lesion_hsv = cv2.cvtColor(img_lesion, cv2.COLOR_RGB2HSV)

    img_diff = img_full - img_lesion
    img_diff_op = img_full_op - img_lesion_op
    img_diff_hsv = img_full_hsv - img_lesion_hsv
    img_diff_hsv[:, :, 0] = angular_diff(img_full_hsv[:, :, 0],
                                         img_lesion_hsv[:, :, 0])

    white_pixs = img_org_op[:, :, 0] > 75
    black_pixs = img_org_op[:, :, 0] < 25
    red_pixs = img_org_op[:, :, 1] > 0
    green_pixs = img_org_op[:, :, 1] < 0
    yellow_pixs = img_org_op[:, :, 2] > 0
    blue_pixs = img_org_op[:, :, 2] < 0
    satuated_pixs = img_org_hsv[:, :, 1] > 0.75
    achromatic_pixs = img_org_hsv[:, :, 1] < 0.25

    report = dict()
    report['num_pixels'] = {
        'white_pixs': white_pixs.sum(),
        'black_pixs': black_pixs.sum(),
        'red_pixs': red_pixs.sum(),
        'green_pixs': green_pixs.sum(),
        'yellow_pixs': yellow_pixs.sum(),
        'blue_pixs': blue_pixs.sum(),
        'satuated_pixs': satuated_pixs.sum(),
        'achromatic_pixs': achromatic_pixs.sum()
    }
    report['diff_rgb'] = img_diff.mean(axis=(0, 1))
    report['hist_rgb_full'] = [
        hist(img_full[:, :, 0], 10, [0, 1]),
        hist(img_full[:, :, 1], 10, [0, 1]),
        hist(img_full[:, :, 2], 10, [0, 1])
    ]
    report['hist_rgb_lesion'] = [
        hist(img_lesion[:, :, 0], 10, [0, 1]),
        hist(img_lesion[:, :, 1], 10, [0, 1]),
        hist(img_lesion[:, :, 2], 10, [0, 1])
    ]

    ab_rng = [
        -127, -40, -30, -20, -10, 0, 10, 20, 30, 40, 128
    ]
    report['diff_op'] = img_diff_op.mean(axis=(0, 1))
    report['hist_op_full'] = [
        hist(img_full_op[:, :, 0], 10, [0, 100]),
        hist(img_full_op[:, :, 1], ab_rng, [-127, 128]),
        hist(img_full_op[:, :, 2], ab_rng, [-127, 128])
    ]
    report['hist_op_lesion'] = [
        hist(img_lesion_op[:, :, 0], 10, [0, 100]),
        hist(img_lesion_op[:, :, 1], ab_rng, [-127, 128]),
        hist(img_lesion_op[:, :, 2], ab_rng, [-127, 128])
    ]
    report['white_pixs_op'] = img_diff_op[white_pixs].mean(axis=(0))
    report['black_pixs_op'] = img_diff_op[black_pixs].mean(axis=(0))
    report['red_pixs_op'] = img_diff_op[red_pixs].mean(axis=(0))
    report['green_pixs_op'] = img_diff_op[green_pixs].mean(axis=(0))
    report['yellow_pixs_op'] = img_diff_op[yellow_pixs].mean(axis=(0))
    report['blue_pixs_op'] = img_diff_op[blue_pixs].mean(axis=(0))
    report['saturated_pixs_op'] = img_diff_op[satuated_pixs].mean(axis=(0))
    report['achromatic_pixs_op'] = img_diff_op[achromatic_pixs].mean(axis=(0))

    report['diff_hsv'] = img_diff_hsv.mean(axis=(0, 1))
    report['hist_hsv_full'] = [
        hist(img_full_hsv[:, :, 0], 12, [0, 360]),
        hist(img_full_hsv[:, :, 1], 10, [0, 1]),
        hist(img_full_hsv[:, :, 2], 10, [0, 1])
    ]
    report['hist_hsv_lesion'] = [
        hist(img_lesion_hsv[:, :, 0], 12, [0, 360]),
        hist(img_lesion_hsv[:, :, 1], 10, [0, 1]),
        hist(img_lesion_hsv[:, :, 2], 10, [0, 1])
    ]
    report['white_pixs_hsv'] = img_diff_hsv[white_pixs].mean(axis=(0))
    report['black_pixs_hsv'] = img_diff_hsv[black_pixs].mean(axis=(0))
    report['red_pixs_hsv'] = img_diff_hsv[red_pixs].mean(axis=(0))
    report['green_pixs_hsv'] = img_diff_hsv[green_pixs].mean(axis=(0))
    report['yellow_pixs_hsv'] = img_diff_hsv[yellow_pixs].mean(axis=(0))
    report['blue_pixs_hsv'] = img_diff_hsv[blue_pixs].mean(axis=(0))
    report['saturated_pixs_hsv'] = img_diff_hsv[satuated_pixs].mean(axis=(0))
    report['achromatic_pixs_hsv'] = img_diff_hsv[achromatic_pixs].mean(axis=(0))
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--org_dir', type=str)
    parser.add_argument('--colour_space', type=str)
    args = parser.parse_args(sys.argv[1:])

    img_org_paths = sorted(glob.glob(args.org_dir + '/*.*'))
    img_full_paths = sorted(glob.glob(
        '%s/%s/val2017/%s/*.*' % (
            args.in_dir, 'ful_11111111', args.colour_space)
    ))

    for i in range(8):
        lesion_dir_name = [1] * 8
        lesion_dir_name[i] = 0
        lesion_dir_name = 'zro_' + ''.join(str(e) for e in lesion_dir_name)
        img_lesion_paths = sorted(glob.glob(
            '%s/%s/val2017/%s/*.*' % (
                args.in_dir, lesion_dir_name, args.colour_space)
        ))
        if len(img_lesion_paths) == 0:
            print('UPS, no images were found', lesion_dir_name)
            continue
        all_reports = []
        for img_ind in range(len(img_org_paths)):
            print(img_lesion_paths[img_ind])
            img_org = io.imread(img_org_paths[img_ind])
            img_full = io.imread(img_full_paths[img_ind])
            img_lesion = io.imread(img_lesion_paths[img_ind])
            current_report = report_vector_influence(
                img_org, img_full, img_lesion
            )
            all_reports.append(current_report)

        path_utils.write_pickle(
            '%s/%s_%s.pickle' % (
                args.in_dir, lesion_dir_name, args.colour_space
            ),
            all_reports
        )
