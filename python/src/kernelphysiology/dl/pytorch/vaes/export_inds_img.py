import os
import sys

import torch
from torchvision import transforms

from skimage import io
from kernelphysiology.dl.pytorch.datasets import data_loaders

from kernelphysiology.dl.pytorch.vaes import model as vqmodel

from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
import argparse


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

    args.in_colour_space = args.colour_space[:3]
    args.out_colour_space = args.colour_space[4:]

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform_funcs = transforms.Compose([
        # cv2_transforms.Resize(256), cv2_transforms.CenterCrop(224),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ])

    intransform_funs = []
    if args.in_colour_space != ' rgb':
        intransform_funs.append(
            cv2_preprocessing.ColourSpaceTransformation(args.in_colour_space)
        )
    intransform = transforms.Compose(intransform_funs)

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
    export(test_loader, network, args)


def export(data_loader, model, args):
    with torch.no_grad():
        for i, (img_readies, img_target, img_paths) in enumerate(data_loader):
            img_readies = img_readies.cuda()
            out_rgb = model(img_readies)
            out_rgb = out_rgb[3].detach().cpu()

            for img_ind in range(out_rgb.shape[0]):
                img_path = img_paths[img_ind]

                cat_in_dir = os.path.dirname(img_path) + '/'

                cat_out_dir = cat_in_dir.replace(args.validation_dir,
                                                 args.out_dir)
                if not os.path.exists(cat_out_dir):
                    os.mkdir(cat_out_dir)

                rec_dir = cat_out_dir + '/' + args.colour_space + '/'
                if not os.path.exists(rec_dir):
                    os.mkdir(rec_dir)

                rec_img_tmp = out_rgb[img_ind].numpy().squeeze()

                out_file = img_path.replace(cat_in_dir, rec_dir)[:-4]
                out_file = out_file + '.png'
                print(out_file)
                io.imsave(out_file, rec_img_tmp)


if __name__ == "__main__":
    main(sys.argv[1:])
