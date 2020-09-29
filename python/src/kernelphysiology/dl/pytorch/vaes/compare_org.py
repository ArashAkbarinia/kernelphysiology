import sys
import argparse
import glob
import numpy as np
from skimage import io
from skimage import color
from skimage import metrics
import cv2


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    parser.add_argument(
        '--org_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    parser.add_argument(
        '--res_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    parser.add_argument(
        '--colour_space',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    parser.add_argument(
        '--de',
        action='store_true',
        default=False,
        help='Compute DeltaE (default: False)'
    )

    return parser.parse_args(args)


def main(args):
    args = parse_arguments(args)
    org_img_paths = sorted(glob.glob(args.org_dir + '/*.jpg'))
    res_img_paths = sorted(glob.glob(args.res_dir + '/*.jpg'))

    all_des = []
    all_ssim = []
    all_psnr = []
    for i in range(len(org_img_paths)):
        print(i, org_img_paths[i])
        img_org = io.imread(org_img_paths[i])
        if len(img_org.shape) == 2:
            img_org = np.repeat(img_org[:, :, np.newaxis], 3, axis=2)
        img_res = io.imread(res_img_paths[i])

        if img_org.shape != img_res.shape:
            img_org = cv2.resize(img_org, (img_res.shape[1], img_res.shape[0]))

        ssim = metrics.structural_similarity(img_org, img_res,
                                             multichannel=True)
        all_ssim.append(ssim)
        psnr = metrics.peak_signal_noise_ratio(img_org, img_res)
        all_psnr.append(psnr)
        if args.de:
            img_org = color.rgb2lab(img_org)
            img_res = color.rgb2lab(img_res)
            de = color.deltaE_ciede2000(img_org, img_res)
            all_des.append([np.mean(de), np.median(de), np.max(de)])

    np.savetxt(args.out_dir + '/ssim_' + args.colour_space + '.txt',
               np.array(all_ssim))
    np.savetxt(args.out_dir + '/psnr_' + args.colour_space + '.txt',
               np.array(all_psnr))
    if args.de:
        np.savetxt(args.out_dir + '/' + args.colour_space + '.txt',
                   np.array(all_des))


if __name__ == "__main__":
    main(sys.argv[1:])
