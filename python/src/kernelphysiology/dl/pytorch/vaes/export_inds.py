import numpy as np
import os
import sys
import argparse

import torch
from torchvision import transforms

from kernelphysiology.dl.pytorch.vaes import data_loaders

from kernelphysiology.dl.pytorch.vaes import model as vqmodel

from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.utils import imutils


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument(
        '--model_path',
        help='autoencoder variant to use: vae | vqvae'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)'
    )
    parser.add_argument('--k', type=int, dest='k', metavar='K',
                        help='number of atoms in dictionary')
    parser.add_argument('--kl', type=int, dest='kl',
                        help='number of atoms in dictionary')
    parser.add_argument('--cos_dis', action='store_true',
                        default=False, help='cosine distance')
    parser.add_argument('--exclude', type=int, default=0, metavar='K',
                        help='number of atoms in dictionary')
    parser.add_argument('--colour_space', type=str, default=None,
                        help='The type of output colour space.')
    parser.add_argument('--manipulation', type=str, nargs='+', default=None,
                        help='The type of output colour space.')

    parser.add_argument(
        '--dataset', default=None,
        help='dataset to use: mnist | cifar10 | imagenet | coco | custom'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='The specific category (default: None)'
    )
    parser.add_argument(
        '--validation_dir',
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
    parser.add_argument('--colour_space', type=str, default=None,
                        help='The type of output colour space.')

    return parser.parse_args(args)


def main(args):
    args = parse_arguments(args)
    weights_rgb = torch.load(args.model_path, map_location='cpu')
    network = vqmodel.VQ_CVAE(128, k=args.k, kl=args.kl, in_chns=3,
                              cos_distance=args.cos_dis)
    network.load_state_dict(weights_rgb)
    if args.exclude > 0:
        which_vec = [args.exclude - 1]
        print(which_vec)
        network.state_dict()['emb.weight'][:, which_vec] = 0
    elif args.exclude < 0:
        which_vec = [*range(8)]
        which_vec.remove(abs(args.exclude) - 1)
        print(which_vec)
        network.state_dict()['emb.weight'][:, which_vec] = 0
    network.cuda()
    network.eval()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    manipulation_func = []
    args.suffix = ''
    if args.manipulation is not None:
        man_func = None
        parameters = None
        if args.manipulation[0] == 'contrast':
            man_func = imutils.adjust_contrast
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'gamma':
            man_func = imutils.adjust_gamma
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'luminance':
            man_func = imutils.reduce_lightness
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'chromaticity':
            man_func = imutils.reduce_chromaticity
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'red_green':
            man_func = imutils.reduce_red_green
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'yellow_blue':
            man_func = imutils.reduce_yellow_blue
            parameters = {'amount': float(args.manipulation[1])}
        elif args.manipulation[0] == 'illuminant':
            man_func = imutils.adjust_illuminant
            parameters = {'illuminant': [
                float(args.manipulation[1]),
                float(args.manipulation[2]),
                float(args.manipulation[3])
            ]}
        if man_func is None:
            sys.exit('Unsupported function %s' % args.manipulation[0])
        args.suffix = '_' + args.manipulation[0] + '_' + ''.join(
            e for e in args.manipulation[1:]
        )
        manipulation_func.append(cv2_preprocessing.UniqueTransformation(
            man_func, **parameters
        ))

    args.in_colour_space = args.colour_space[:3]
    args.out_colour_space = args.colour_space[4:]

    intransform_funs = []
    intransform_funs.append(*manipulation_func)
    if args.in_colour_space != ' rgb':
        intransform_funs.append(
            cv2_preprocessing.ColourTransformation(None, args.in_colour_space)
        )
    intransform = transforms.Compose(intransform_funs)
    transform_funcs = transforms.Compose([
        # cv2_transforms.Resize(256), cv2_transforms.CenterCrop(224),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ])

    if args.dataset == 'imagenet':
        test_loader = torch.utils.data.DataLoader(
            data_loaders.ImageFolder(
                root=args.validation_dir,
                intransform=intransform,
                outtransform=None,
                transform=transform_funcs
            ),
            batch_size=args.batch_size, shuffle=False
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            data_loaders.CategoryImages(
                root=args.validation_dir,
                # FIXME
                category=args.category,
                intransform=intransform,
                outtransform=None,
                transform=transform_funcs
            ),
            batch_size=args.batch_size, shuffle=False
        )
    export(test_loader, network, mean, std, args)


def export(data_loader, model, mean, std, args):
    hists = []
    bins = [*range(model.state_dict()['emb.weight'].shape[0] + 1)]
    hist_rng = [0, model.state_dict()['emb.weight'].shape[0] - 1]
    with torch.no_grad():
        for i, (img_readies, img_target, img_paths) in enumerate(data_loader):
            img_readies = img_readies.cuda()
            out_rgb = model(img_readies)
            out_rgb = out_rgb[3].detach().cpu().numpy()

            for img_ind in range(out_rgb.shape[0]):
                current_hist, edges = np.histogram(
                    out_rgb[img_ind], bins, hist_rng, density=True
                )
                hists.append(current_hist)

            np.savetxt(
                args.out_dir + '/' + args.colour_space + args.suffix + '.txt',
                np.array(hists)
            )


if __name__ == "__main__":
    main(sys.argv[1:])
