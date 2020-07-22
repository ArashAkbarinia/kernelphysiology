"""
The main script to decompose an image into several meaningful entities.
"""

import numpy as np
import os
import sys
import time
import logging

from torch import optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from kernelphysiology.dl.experiments.decomposition import util as vae_util
from kernelphysiology.dl.experiments.decomposition import model_single
from kernelphysiology.dl.experiments.decomposition import model_multi
from kernelphysiology.dl.experiments.decomposition import arguments
from kernelphysiology.dl.experiments.decomposition import data_loaders
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.dl.pytorch.utils import cv2_transforms

datasets_classes = {
    'imagenet': data_loaders.ImageFolder,
    'celeba': data_loaders.CelebA,
}
dataset_target_size = {
    'imagenet': 256,
    'celeba': 64,
}


def main(args):
    args = arguments.parse_arguments(args)

    # determining the number of input channels
    if args.in_space == 'grey':
        args.in_chns = 1
    else:
        args.in_chns = 3

    # FIXME
    # preparing the output dictionary
    args.outs_dict = dict()
    for out_type in args.outputs:
        if out_type == 'input':
            out_shape = [1, 1, args.in_chns]
        elif out_type == 'gry':
            out_shape = [1, 1, 1]
        elif out_type == 'db1':
            # TODO: just assuming numbers of square 2
            out_shape = [0.5, 0.5, 4]
        else:
            out_shape = [1, 1, args.in_chns]
        args.outs_dict[out_type] = {'shape': out_shape}

    args.mean = 0.5
    args.std = 0.5
    target_size = args.target_size or dataset_target_size[args.dataset]

    pre_shared_transforms = [
        cv2_transforms.Resize(target_size + 32),
        cv2_transforms.CenterCrop(target_size),
    ]
    post_shared_transforms = [
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(args.mean, args.std)
    ]

    pre_dataset_transforms = {
        'imagenet': transforms.Compose(pre_shared_transforms),
        'celeba': transforms.Compose(pre_shared_transforms),
    }
    post_dataset_transforms = {
        'imagenet': transforms.Compose(post_shared_transforms),
        'celeba': transforms.Compose(post_shared_transforms),
    }

    save_path = vae_util.setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vae_model = model_single if args.model == 'single' else model_multi
    model = vae_model.DecomposeNet(
        hidden=args.hidden, k=args.k, d=args.d, in_chns=args.in_chns,
        outs_dict=args.outs_dict
    )
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    if args.in_space.lower() != 'rgb':
        intransform_funs.append(
            cv2_preprocessing.ColourSpaceTransformation(args.in_space.lower())
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
    if args.dataset == 'celeba':
        train_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='train', **transforms_kwargs
        )
        test_dataset = datasets_classes[args.dataset](
            root=args.data_dir, split='test', **transforms_kwargs
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

    # starting to train
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(args.start_epoch, args.epochs):
        train_losses = train(
            epoch, model, train_loader, optimizer, save_path, args
        )
        test_losses = test_net(
            epoch, model, test_loader, save_path, args
        )
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
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
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'arch': {
                    'k': args.k, 'd': args.d, 'hidden': args.hidden,
                    'in_chns': args.in_chns, 'outs_dict': args.outs_dict
                }
            },
            save_path
        )


def train(epoch, model, train_loader, optimizer, save_path, args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    num_batches = len(train_loader)
    start_time = time.time()
    for bidx, loader_data in enumerate(train_loader):
        data = loader_data[0]
        data = data.cuda()
        target = loader_data[1]
        for key in target.keys():
            target[key] = target[key].cuda()

        optimizer.zero_grad()
        outputs = model(data)

        loss = model.loss_function(target, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if bidx % args.log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= args.log_interval
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
            for key in latest_losses:
                losses[key + '_train'] = 0
        if bidx in list(np.linspace(0, num_batches - 1, 4).astype(int)):
            vae_util.grid_save_reconstructions(
                args.outs_dict, target, outputs[0], args.mean, args.std, epoch,
                save_path, 'reconstruction_train%.5d' % bidx
            )

        if bidx * len(data) > args.train_samples:
            break

    for key in epoch_losses:
        epoch_losses[key] /= (
                len(train_loader.dataset) / train_loader.batch_size
        )
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()]
    )
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    return epoch_losses


def test_net(epoch, model, test_loader, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    num_batches = len(test_loader)
    with torch.no_grad():
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
            if bidx in list(np.linspace(0, num_batches - 1, 4).astype(int)):
                vae_util.grid_save_reconstructions(
                    args.outs_dict, target, outputs[0], args.mean, args.std,
                    epoch, save_path, 'reconstruction_test%.5d' % bidx
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


if __name__ == "__main__":
    main(sys.argv[1:])
