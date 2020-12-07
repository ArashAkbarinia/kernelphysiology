"""
The main script to decompose an image into several meaningful entities.
"""

import numpy as np
import os
import sys
import time
import logging
from functools import partial

from torch import optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from kernelphysiology.dl.experiments.decomposition import util as vae_util
from kernelphysiology.dl.experiments.decomposition import model_vqvae
from kernelphysiology.dl.experiments.decomposition import model_multi
from kernelphysiology.dl.experiments.decomposition import model_segmentation
from kernelphysiology.dl.experiments.decomposition import arguments
from kernelphysiology.dl.experiments.decomposition import data_loaders
from kernelphysiology.dl.experiments.decomposition import ColourTransformer
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.dl.pytorch.utils import cv2_transforms

from kernelphysiology.utils import random_imutils
from kernelphysiology.transformations.colour_spaces import all2rgb

datasets_classes = {
    'imagenet': data_loaders.ImageFolder,
    'ccvr': data_loaders.ColourConstancyVR,
    'celeba': data_loaders.CelebA,
    'touch': data_loaders.TouchRelief,
    'voc': data_loaders.VOCSegmentation,
    'coco': data_loaders.COCOPanoptic,
}
dataset_target_size = {
    'imagenet': 256,
    'ccvr': 256,
    'celeba': 64,
    'touch': 256,
    'voc': 256,
    'coco': 256,
}


def main(args):
    args = arguments.parse_arguments(args)

    # determining the number of input channels
    args.in_chns = 3

    out_chns = 3
    args.out_chns = out_chns

    args.mean = 0.5
    args.std = 0.5
    target_size = args.target_size or dataset_target_size[args.dataset]

    if args.dataset == 'ccvr':
        pre_shared_transforms = [
            cv2_transforms.Resize(target_size + 32),
            cv2_transforms.RandomCrop(target_size),
        ]
    else:
        pre_shared_transforms = [
            cv2_transforms.Resize(target_size + 32),
            cv2_transforms.CenterCrop(target_size),
        ]
    post_shared_transforms = [
        cv2_transforms.ToTensor(),
        # cv2_transforms.Normalize(args.mean, args.std)
    ]

    pre_dataset_transforms = dict()
    post_dataset_transforms = dict()
    for key in datasets_classes.keys():
        pre_dataset_transforms[key] = transforms.Compose(
            pre_shared_transforms
        )
        post_dataset_transforms[key] = transforms.Compose(
            post_shared_transforms
        )

    save_path = vae_util.setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.pred is not None:
        checkpoint = torch.load(args.pred, map_location='cpu')
        model_vae = model_vqvae.DecomposeNet(**checkpoint['arch_params'])
        model_vae.load_state_dict(checkpoint['state_dict'])
    else:
        # FIXME: archs_param should be added to resume and fine_tune
        arch_params = {'k': args.k, 'd': args.d, 'hidden': args.hidden}
        model_vae = model_vqvae.DecomposeNet(
            hidden=args.hidden, k=args.k, d=args.d, in_chns=args.in_chns,
            out_chns=args.out_chns
        )
    model_vae = model_vae.cuda()
    model_vae.tanh = False

    # FIXME make it only for one single output
    if args.lab_init:
        distortion = [
            116.0 / 500, 16.0 / 500, 500.0 / 500, 200.0 / 500,
            0.2068966
        ]
        trans_mat = [[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]]

        ref_white = (0.95047, 1., 1.08883)
    else:
        trans_mat = None
        ref_white = None
        distortion = None
    model_cst = ColourTransformer.LabTransformer(
        trans_mat=trans_mat, ref_white=ref_white,
        distortion=distortion, linear=args.linear
    )
    model_cst = model_cst.cuda()

    vae_params = [
        {'params': [p for p in model_vae.parameters() if p.requires_grad]},
    ]
    cst_params = [
        {'params': [p for p in model_cst.parameters() if p.requires_grad]},
    ]
    optimizer_vae = optim.Adam(vae_params, lr=args.lr)
    optimizer_cst = optim.Adam(cst_params, lr=args.lr)
    scheduler_vae = optim.lr_scheduler.StepLR(
        optimizer_vae, int(args.epochs / 3), 0.5
    )
    scheduler_cst = optim.lr_scheduler.StepLR(
        optimizer_cst, int(args.epochs / 3), 0.5
    )

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_vae.load_state_dict(checkpoint['state_dict'])
        model_vae = model_vae.cuda()
        args.start_epoch = checkpoint['epoch'] + 1
        scheduler_vae.load_state_dict(checkpoint['scheduler_vae'])
        optimizer_vae.load_state_dict(checkpoint['optimizer_vae'])
        scheduler_cst.load_state_dict(checkpoint['scheduler_cst'])
        optimizer_cst.load_state_dict(checkpoint['optimizer_cst'])
    elif args.fine_tune is not None:
        weights = torch.load(args.fine_tune, map_location='cpu')
        model_vae.load_state_dict(weights, strict=False)
        model_vae = model_vae.cuda()

    intransform_funs = []
    if args.in_space.lower() == 'cgi':
        augmentation_settings = [
            {
                'function': random_imutils.adjust_contrast,
                'kwargs': {'amount': np.array([0.2, 1.0]), 'channel_wise': True}
            },
            {
                'function': random_imutils.adjust_gamma,
                'kwargs': {'amount': np.array([0.2, 5.0]), 'channel_wise': True}
            },
            {
                'function': random_imutils.adjust_illuminant,
                'kwargs': {'illuminant': np.array([0.0, 1.0])}
            }
        ]
        intransform_funs.append(
            cv2_preprocessing.RandomAugmentationTransformation(
                augmentation_settings, num_augmentations=1
            )
        )
    elif args.in_space.lower() != 'rgb':
        intransform_funs.append(
            cv2_preprocessing.DecompositionTransformation(args.in_space.lower())
        )
    intransform = transforms.Compose(intransform_funs)

    outtransform_funs = []
    outtransform = transforms.Compose(outtransform_funs)

    args.outs_dict = {'rgb': {'vis_fun': None}}

    # preparing the dataset
    transforms_kwargs = {
        'intransform': intransform,
        'outtransform': outtransform,
        'pre_transform': pre_dataset_transforms[args.dataset],
        'post_transform': post_dataset_transforms[args.dataset]
    }
    if args.dataset in ['celeba', 'touch', 'ccvr']:
        train_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='train', **transforms_kwargs
        )
        test_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='test', **transforms_kwargs
        )
    elif args.dataset in ['coco']:
        train_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='train', **transforms_kwargs
        )
        test_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='val', **transforms_kwargs
        )
    elif args.dataset in ['voc']:
        train_dataset = datasets_classes[args.dataset](
            root=args.data_dir, image_set='train', **transforms_kwargs
        )
        test_dataset = datasets_classes[args.dataset](
            root=args.data_dir, image_set='val', **transforms_kwargs
        )
    else:
        train_dataset = datasets_classes[args.dataset](
            root=os.path.join(args.data_dir, 'train'), **transforms_kwargs
        )
        test_dataset = datasets_classes[args.dataset](
            root=os.path.join(args.data_dir, 'validation'), **transforms_kwargs
        )

    loader_kwargs = {
        'batch_size': args.batch_size, 'num_workers': args.workers,
        'pin_memory': True
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **loader_kwargs
    )

    if args.pred is not None:
        predict(model_vae, test_loader, save_path, args)
        return

    # starting to train
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(args.start_epoch, args.epochs):
        train_losses = train(
            epoch, model_vae, model_cst, train_loader,
            (optimizer_vae, optimizer_cst), save_path, args
        )
        test_losses = test_net(
            epoch, model_vae, model_cst, test_loader, save_path, args
        )
        for k in train_losses.keys():
            name = k.replace('_trn', '')
            train_name = k
            test_name = k.replace('_trn', '_val')
            writer.add_scalars(
                name, {
                    'train': train_losses[train_name],
                    'test': test_losses[test_name]
                }, epoch
            )
        scheduler_vae.step()
        scheduler_cst.step()
        vae_util.save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model_vae.state_dict(),
                'colour_transformer': model_cst.state_dict(),
                'optimizer_vae': optimizer_vae.state_dict(),
                'scheduler_vae': scheduler_vae.state_dict(),
                'optimizer_cst': optimizer_cst.state_dict(),
                'scheduler_cst': scheduler_cst.state_dict(),
                'arch': args.model,
                'arch_params': {
                    **arch_params,
                    'in_chns': args.in_chns, 'out_chns': args.out_chns,
                },
                'transformer_params': {'linear': args.linear}
            },
            save_path
        )


def train(epoch, model_vae, model_cst, train_loader, optimizers, save_path,
          args):
    (optimizer_vae, optimizer_cst) = optimizers
    model_vae.train()
    model_cst.train()
    vae_loss_dict = model_vae.latest_losses()
    batch_losses = {k + '_trn_vae': 0 for k, v in vae_loss_dict.items()}
    epoch_losses = {k + '_trn_vae': 0 for k, v in vae_loss_dict.items()}
    cst_loss_dict = model_cst.latest_losses()
    for k, v in cst_loss_dict.items():
        batch_losses[k + '_trn_cst'] = 0
        epoch_losses[k + '_trn_cst'] = 0

    num_batches = len(train_loader)
    start_time = time.time()
    for bidx, loader_data in enumerate(train_loader):
        data = loader_data[0]
        data = data.cuda()

        # optimise the VAE
        model_cst.zero_grad()
        new_target_space = model_cst(data)
        outputs = model_vae(data)

        loss_vae = model_vae.loss_function(new_target_space, *outputs)
        loss_vae.backward()
        optimizer_vae.step()

        # optimise the colour space transformer network
        model_vae.zero_grad()
        new_target_space = model_cst(data)
        outputs = model_vae(data)

        loss_cst = model_cst.loss_function(new_target_space, data, outputs[0])
        loss_cst.backward()
        optimizer_cst.step()

        latest_losses = model_vae.latest_losses()
        for key in latest_losses:
            batch_losses[key + '_trn_vae'] += float(latest_losses[key])
            epoch_losses[key + '_trn_vae'] += float(latest_losses[key])
        ct_latest_losses = model_cst.latest_losses()
        for key in ct_latest_losses:
            batch_losses[key + '_trn_cst'] += float(ct_latest_losses[key])
            epoch_losses[key + '_trn_cst'] += float(ct_latest_losses[key])

        if bidx % args.log_interval == 0:
            for key in batch_losses.keys():
                batch_losses[key] /= args.log_interval
            loss_string = ' '.join(
                ['{}: {:.6f}'.format(k, v) for k, v in batch_losses.items()]
            )
            logging.info(
                'Train Epoch: {epoch} [{batch:5d}/{total_batch} '
                '({percent:2d}%)]   time: {time:3.2f}   {loss}'.format(
                    epoch=epoch, batch=bidx * len(data),
                    total_batch=num_batches * len(data),
                    percent=int(100. * bidx / num_batches),
                    time=time.time() - start_time, loss=loss_string
                )
            )
            start_time = time.time()
            for key in batch_losses.keys():
                batch_losses[key] = 0
        if bidx in list(np.linspace(0, num_batches - 1, 4).astype('int')):
            out_rgb = {'rgb': model_cst.rnd2rgb(outputs[0].clone(), clip=True)}
            target_rgb = {'rgb': model_cst.rnd2rgb(new_target_space, clip=True)}
            vae_util.grid_save_reconstructions_noinv(
                args.outs_dict, target_rgb, out_rgb, args.mean, args.std, epoch,
                save_path, 'reconstruction_train%.5d' % bidx, inputs=data
            )

        if bidx * len(data) > args.train_samples:
            break

    print(model_cst.ref_white)
    print(model_cst.trans_mat)
    for key in epoch_losses:
        epoch_losses[key] /= (
                len(train_loader.dataset) / train_loader.batch_size
        )
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()]
    )
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    return epoch_losses


def test_net(epoch, model_vae, model_cst, test_loader, save_path, args):
    model_vae.eval()
    model_cst.eval()
    loss_dict = model_vae.latest_losses()
    losses = {k + '_val_vae': 0 for k, v in loss_dict.items()}
    ct_loss_dict = model_cst.latest_losses()
    for k, v in ct_loss_dict.items():
        losses[k + '_val_cst'] = 0

    num_batches = len(test_loader)
    with torch.no_grad():
        for bidx, loader_data in enumerate(test_loader):
            data = loader_data[0]
            data = data.cuda()

            new_target_space = model_cst(data)
            outputs = model_vae(data)
            model_vae.loss_function(new_target_space, *outputs)
            model_cst.loss_function(new_target_space, data, outputs[0])
            latest_losses = model_vae.latest_losses()
            for key in latest_losses:
                losses[key + '_val_vae'] += float(latest_losses[key])
            ct_latest_losses = model_cst.latest_losses()
            for key in ct_latest_losses:
                losses[key + '_val_cst'] += float(ct_latest_losses[key])
            if bidx in list(np.linspace(0, num_batches - 1, 4).astype('int')):
                out_rgb = {
                    'rgb': model_cst.rnd2rgb(outputs[0].clone(), clip=True)}
                target_rgb = {
                    'rgb': model_cst.rnd2rgb(new_target_space, clip=True)}
                vae_util.grid_save_reconstructions_noinv(
                    args.outs_dict, target_rgb, out_rgb, args.mean, args.std,
                    epoch, save_path, 'reconstruction_test%.5d' % bidx,
                    inputs=data
                )
            if bidx * len(data) > args.test_samples:
                break

    for key in losses:
        losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
    loss_string = ' '.join(
        ['{}: {:.6f}'.format(k, v) for k, v in losses.items()]
    )
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def predict(model, test_loader, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    with torch.no_grad():
        img_ind = 0
        for bidx, loader_data in enumerate(test_loader):
            data = loader_data[0]
            data = data.cuda()
            target = loader_data[1]
            for key in target.keys():
                target[key] = target[key].cuda()

            outputs = model(data)
            model.loss_function(target, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])

            vae_util.individual_save_reconstructions(
                args.outs_dict, target, outputs[0], args.mean, args.std,
                img_ind, save_path, 'reconstruction_test%.5d' % bidx
            )
            img_ind += len(data)

    for key in losses:
        losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
    loss_string = ' '.join(
        ['{}: {:.6f}'.format(k, v) for k, v in losses.items()]
    )
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


if __name__ == "__main__":
    main(sys.argv[1:])
