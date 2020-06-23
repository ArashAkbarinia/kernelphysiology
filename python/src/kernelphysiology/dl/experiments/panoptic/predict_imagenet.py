import numpy as np
import os
import sys

import torch
from torchvision import transforms

import cv2

from kernelphysiology.dl.pytorch.vaes import model as vqmodel
from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor
from kernelphysiology.transformations import colour_spaces

from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.transformations import normalisations
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy_preds
import argparse

from torchvision import datasets as tdatasets


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgin, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()
        imgout = imgin.copy()
        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return imgin, imgout, path, class_target


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument(
        '--model_path',
        help='autoencoder variant to use: vae | vqvae'
    )
    parser.add_argument('--imagenet_model', help='path to the imagenet model')
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
    args.out_colour_space = args.colour_space[4:7]

    (imagenet_model, target_size) = model_utils.which_network(
        args.imagenet_model, 'classification', num_classes=1000,
    )
    imagenet_model.cuda()
    imagenet_model.eval()

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform_funcs = transforms.Compose([
        # cv2_transforms.Resize(256), cv2_transforms.CenterCrop(224),
        cv2_transforms.Resize(512),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ])

    imagenet_transformations = transforms.Compose([
        cv2_transforms.Resize(256), cv2_transforms.CenterCrop(224),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    intransform_funs = []
    if args.in_colour_space != ' rgb':
        intransform_funs.append(
            cv2_preprocessing.VisionTypeTransformation(
                None, args.in_colour_space
            )
        )
    intransform = transforms.Compose(intransform_funs)

    test_loader = torch.utils.data.DataLoader(
        ImageFolder(
            root=args.validation_dir,
            intransform=intransform,
            outtransform=None,
            transform=transform_funcs
        ),
        batch_size=args.batch_size, shuffle=False
    )
    top1, top5, prediction_output = export(test_loader, network, mean, std,
                                           imagenet_model,
                                           imagenet_transformations, args)
    output_file = '%s/%s.csv' % (args.out_dir, args.colour_space)
    np.savetxt(output_file, prediction_output, delimiter=',', fmt='%i')


def export(data_loader, model, mean, std, imagenet_model,
           imagenet_transformations, args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        all_predictions = []
        for i, (img_readies, img_target, img_paths, targets) in enumerate(
                data_loader):
            img_readies = img_readies.cuda()
            out_rgb = model(img_readies)
            out_rgb = out_rgb[0].detach().cpu()
            img_readies = img_readies.detach().cpu()
            targets = targets.cuda()

            for img_ind in range(out_rgb.shape[0]):
                img_ready = img_readies[img_ind].unsqueeze(0)

                org_img_tmp = inv_normalise_tensor(img_ready, mean, std)
                org_img_tmp = org_img_tmp.numpy().squeeze().transpose(1, 2, 0)
                org_img_tmp = org_img_tmp * 255
                org_img_tmp = org_img_tmp.astype('uint8')
                # org_img.append(org_img_tmp)

                rec_img_tmp = inv_normalise_tensor(
                    out_rgb[img_ind].unsqueeze(0), mean, std)
                rec_img_tmp = rec_img_tmp.numpy().squeeze().transpose(1, 2, 0)
                rec_img_tmp = cv2.resize(
                    rec_img_tmp, (org_img_tmp.shape[1], org_img_tmp.shape[0])
                )
                if args.out_colour_space == 'lab':
                    rec_img_tmp = rec_img_tmp * 255
                    rec_img_tmp = rec_img_tmp.astype('uint8')
                    rec_img_tmp = cv2.cvtColor(rec_img_tmp, cv2.COLOR_LAB2RGB)
                elif args.out_colour_space == 'hsv':
                    rec_img_tmp = colour_spaces.hsv012rgb(rec_img_tmp)
                elif args.out_colour_space == 'lms':
                    rec_img_tmp = colour_spaces.lms012rgb(rec_img_tmp)
                elif args.out_colour_space == 'yog':
                    rec_img_tmp = colour_spaces.yog012rgb(rec_img_tmp)
                elif args.out_colour_space == 'dkl':
                    rec_img_tmp = colour_spaces.dkl012rgb(rec_img_tmp)
                else:
                    rec_img_tmp = normalisations.uint8im(rec_img_tmp)

                # imagenet stuff
                img_imagenet = imagenet_transformations(rec_img_tmp).unsqueeze(
                    0)
                output = imagenet_model(img_imagenet.cuda())

                # measure accuracy and record loss
                ((acc1, acc5), (corrects1, corrects5)) = accuracy_preds(
                    output, targets[img_ind:img_ind + 1], topk=(1, 5)
                )
                corrects1 = corrects1.cpu().numpy()
                corrects5 = corrects5.cpu().numpy().sum(axis=0)

                pred_outs = np.zeros((corrects1.shape[1], 3))
                pred_outs[:, 0] = corrects1
                pred_outs[:, 1] = corrects5
                pred_outs[:, 2] = output.cpu().numpy().argmax(axis=1)

                # I'm not sure if this is all necessary, copied from keras
                if not isinstance(pred_outs, list):
                    pred_outs = [pred_outs]

                if not all_predictions:
                    for _ in pred_outs:
                        all_predictions.append([])

                for j, out in enumerate(pred_outs):
                    all_predictions[j].append(out)

                top1.update(acc1[0], img_imagenet.size(0))
                top5.update(acc5[0], img_imagenet.size(0))
            print(i, top1.avg, top5.avg)

            if np.mod(i, 100) == 0:
                if len(all_predictions) == 1:
                    prediction_output = np.concatenate(all_predictions[0])
                else:
                    prediction_output = [np.concatenate(out) for out in
                                         all_predictions]
                output_file = '%s/%s.csv' % (args.out_dir, args.colour_space)
                np.savetxt(output_file, prediction_output, delimiter=',',
                           fmt='%i')
        if len(all_predictions) == 1:
            prediction_output = np.concatenate(all_predictions[0])
        else:
            prediction_output = [np.concatenate(out) for out in all_predictions]
    return top1.avg, top5.avg, prediction_output


if __name__ == "__main__":
    main(sys.argv[1:])
