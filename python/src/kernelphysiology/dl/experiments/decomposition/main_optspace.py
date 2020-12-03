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
from kernelphysiology.dl.experiments.decomposition import model_single
from kernelphysiology.dl.experiments.decomposition import model_category
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
    in_size = 1
    if args.in_space == 'gry':
        args.in_chns = 1
    elif args.in_space == 'db1':
        args.in_chns = 4
        in_size = 0.5
    else:
        args.in_chns = 3

    # FIXME
    # preparing the output dictionary
    args.outs_dict = dict()
    for out_type in args.outputs:
        if out_type == 'input':
            out_shape = [1 / in_size, 1 / in_size, args.in_chns]
        elif out_type == 'gry':
            out_shape = [1 / in_size, 1 / in_size, 1]
        elif out_type == 'db1':
            # TODO: just assuming numbers of square 2
            out_shape = [0.5 / in_size, 0.5 / in_size, 4]
        else:
            out_shape = [1 / in_size, 1 / in_size, 3]
        args.outs_dict[out_type] = {'shape': out_shape}

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
        vae_model = (
            model_single if checkpoint['model'] == 'single' else model_multi
        )
        model = vae_model.DecomposeNet(**checkpoint['arch_params'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # FIXME: add to prediction, right now it's hard-coded for resnet18
        if args.model == 'deeplabv3':
            backbone = {
                'arch': 'resnet_bottleneck_custom',
                'customs': {
                    'pooling_type': 'max',
                    'in_chns': args.in_chns,
                    'blocks': [2, 2, 2, 2], 'num_kernels': 64,
                    'num_classes': 1000
                }
            }
            # FIXME out_shape is defined far above, this is just a hack
            arch_params = {'backbone': backbone, 'num_classes': out_shape[-1]}
            model = model_segmentation.deeplabv3_resnet(
                backbone, num_classes=out_shape[-1], outs_dict=args.outs_dict
            )
        elif 'unet' in args.model:
            from segmentation_models import unet
            encoder_name = args.model.split('_')[-1]
            model = unet.model.Unet(
                in_channels=args.in_chns, encoder_name=encoder_name,
                encoder_weights=None,
                outs_dict=args.outs_dict, classes=out_shape[-1]
            )
            arch_params = {'encoder_name': encoder_name}
        elif args.model == 'category':
            # FIXME: archs_param should be added to resume and fine_tune
            arch_params = {'k': args.k, 'd': args.d, 'hidden': args.hidden}
            vae_model = model_category
            model = vae_model.DecomposeNet(
                hidden=args.hidden, k=args.k, d=args.d, in_chns=args.in_chns,
                outs_dict=args.outs_dict
            )
        else:
            # FIXME: archs_param should be added to resume and fine_tune
            arch_params = {'k': args.k, 'd': args.d, 'hidden': args.hidden}
            vae_model = model_single if args.model == 'single' else model_multi
            model = vae_model.DecomposeNet(
                hidden=args.hidden, k=args.k, d=args.d, in_chns=args.in_chns,
                outs_dict=args.outs_dict
            )
    model = model.cuda()
    model.tanh = False

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
    colour_transformer = ColourTransformer.LabTransformer(
        trans_mat=trans_mat, ref_white=ref_white,
        distortion=distortion, linear=args.linear
    )
    colour_transformer = colour_transformer.cuda()

    params_to_optimize = [
        {'params': [p for p in model.parameters() if p.requires_grad]},
        {'params': [p for p in colour_transformer.parameters() if
                    p.requires_grad]},
    ]
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(args.epochs / 3), 0.5)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        args.start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.fine_tune is not None:
        weights = torch.load(args.fine_tune, map_location='cpu')
        model.load_state_dict(weights, strict=False)
        model = model.cuda()

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

    outtransform_funs = [
        cv2_preprocessing.MultipleOutputTransformation(args.outputs)
    ]
    outtransform = transforms.Compose(outtransform_funs)

    # FIXME
    for out_type in args.outputs:
        if out_type == 'input':
            vis_fun = None
        elif out_type == 'gry':
            vis_fun = None
        elif out_type == 'db1':
            vis_fun = vae_util.wavelet_visualise
        elif args.vis_rgb:
            vis_fun = partial(all2rgb, src_space=out_type)
        else:
            vis_fun = None
        args.outs_dict[out_type]['vis_fun'] = vis_fun

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
        predict(model, test_loader, save_path, args)
        return

    # starting to train
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(args.start_epoch, args.epochs):
        train_losses = train(
            epoch, model, colour_transformer, train_loader,
            optimizer, save_path, args
        )
        test_losses = test_net(
            epoch, model, colour_transformer, test_loader, save_path, args
        )
        for k in train_losses.keys():
            name = k.replace('_t', '')
            train_name = k
            test_name = k.replace('_t', '_v')
            writer.add_scalars(
                name, {
                    'train': train_losses[train_name],
                    'test': test_losses[test_name]
                }, epoch
            )
        scheduler.step()
        vae_util.save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'colour_transformer': colour_transformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'arch': args.model,
                'arch_params': {
                    **arch_params,
                    'in_chns': args.in_chns, 'outs_dict': args.outs_dict,
                },
                'transformer_params': {'linear': args.linear}
            },
            save_path
        )


def train(epoch, model, colour_transformer, train_loader, optimizer, save_path,
          args):
    model.train()
    colour_transformer.train()
    loss_dict = model.latest_losses()
    losses = {k + '_t_m': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_t_m': 0 for k, v in loss_dict.items()}
    ct_loss_dict = colour_transformer.latest_losses()
    for k, v in ct_loss_dict.items():
        losses[k + '_t_ct'] = 0
        epoch_losses[k + '_t_ct'] = 0

    num_batches = len(train_loader)
    start_time = time.time()
    for bidx, loader_data in enumerate(train_loader):
        data = loader_data[0]
        data = data.cuda()
        target = loader_data[1]
        for key in target.keys():
            target[key] = target[key].cuda()

        optimizer.zero_grad()
        new_target_space = colour_transformer(data)
        # FIXME: clean it up no multiple outputs
        target['rgb'] = new_target_space
        outputs = model(data)

        loss = (
                model.loss_function(target, *outputs) +
                colour_transformer.loss_function(target, data, outputs[0])
        )
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_t_m'] += float(latest_losses[key])
            epoch_losses[key + '_t_m'] += float(latest_losses[key])
        ct_latest_losses = colour_transformer.latest_losses()
        for key in ct_latest_losses:
            losses[key + '_t_ct'] += float(ct_latest_losses[key])
            epoch_losses[key + '_t_ct'] += float(ct_latest_losses[key])

        if bidx % args.log_interval == 0:
            for key in losses.keys():
                losses[key] /= args.log_interval
            loss_string = ' '.join(
                ['{}: {:.6f}'.format(k, v) for k, v in losses.items()]
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
            for key in losses.keys():
                losses[key] = 0
        if bidx in list(np.linspace(0, num_batches - 1, 4).astype('int')):
            out_rgb = outputs[0].copy()
            out_rgb['rgb'] = colour_transformer.rnd2rgb(
                out_rgb['rgb'], clip=True)
            target_rgb = target.copy()
            target_rgb['rgb'] = colour_transformer.rnd2rgb(
                target_rgb['rgb'], clip=True)
            vae_util.grid_save_reconstructions_noinv(
                args.outs_dict, target_rgb, out_rgb, args.mean, args.std, epoch,
                save_path, 'reconstruction_train%.5d' % bidx, inputs=data
            )

        if bidx * len(data) > args.train_samples:
            break

    print(colour_transformer.ref_white)
    print(colour_transformer.trans_mat)
    for key in epoch_losses:
        epoch_losses[key] /= (
                len(train_loader.dataset) / train_loader.batch_size
        )
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()]
    )
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    return epoch_losses


def test_net(epoch, model, colour_transformer, test_loader, save_path, args):
    model.eval()
    colour_transformer.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_v_m': 0 for k, v in loss_dict.items()}
    ct_loss_dict = colour_transformer.latest_losses()
    for k, v in ct_loss_dict.items():
        losses[k + '_v_ct'] = 0

    num_batches = len(test_loader)
    with torch.no_grad():
        for bidx, loader_data in enumerate(test_loader):
            data = loader_data[0]
            data = data.cuda()
            target = loader_data[1]
            for key in target.keys():
                target[key] = target[key].cuda()

            new_target_space = colour_transformer(data)
            # FIXME: clean it up no multiple outputs
            target['rgb'] = new_target_space
            outputs = model(data)
            model.loss_function(target, *outputs)
            colour_transformer.loss_function(
                target, data, outputs[0]
            )
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_v_m'] += float(latest_losses[key])
            ct_latest_losses = colour_transformer.latest_losses()
            for key in ct_latest_losses:
                losses[key + '_v_ct'] += float(ct_latest_losses[key])
            if bidx in list(np.linspace(0, num_batches - 1, 4).astype('int')):
                out_rgb = outputs[0].copy()
                out_rgb['rgb'] = colour_transformer.rnd2rgb(
                    out_rgb['rgb'], clip=True)
                target_rgb = target.copy()
                target_rgb['rgb'] = colour_transformer.rnd2rgb(
                    target_rgb['rgb'], clip=True)
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
