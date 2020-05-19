import sys
import argparse
import glob
import numpy as np
from skimage import io
from skimage import color
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

    return parser.parse_args(args)


def main(args):
    args = parse_arguments(args)
    org_img_paths = sorted(glob.glob(args.org_dir + '/*.jpg'))
    res_img_paths = sorted(glob.glob(args.res_dir + '/*.jpg'))

    all_des = []
    for i in range(len(org_img_paths)):
        print(i, org_img_paths[i])
        img_org = io.imread(org_img_paths[i])
        if len(img_org.shape) == 2:
            img_org = np.repeat(img_org[:, :, np.newaxis], 3, axis=2)
        img_res = io.imread(res_img_paths[i])

        if img_org.shape != img_res.shape:
            img_org = cv2.resize(img_org, (img_res.shape[1], img_res.shape[0]))

        img_org = color.rgb2lab(img_org)
        img_res = color.rgb2lab(img_res)
        de = color.deltaE_ciede2000(img_org, img_res)
        all_des.append([np.mean(de), np.median(de), np.max(de)])

    np.savetxt(args.out_dir + '/' + args.colour_space + '.txt',
               np.array(all_des))


if __name__ == "__main__":
    main(sys.argv[1:])
