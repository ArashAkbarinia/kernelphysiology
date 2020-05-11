import os
import sys
import time
import logging

from torch import optim
from torch import nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional
from torchvision import datasets, transforms

from kernelphysiology.dl.pytorch.vaes import util as vae_util
from kernelphysiology.dl.pytorch.vaes import model as vae_model
from kernelphysiology.dl.pytorch.vaes import wavenet_vae
from kernelphysiology.dl.pytorch.vaes import data_loaders
from kernelphysiology.dl.pytorch.vaes.arguments import parse_arguments
from kernelphysiology.dl.pytorch.utils import cv2_preprocessing
from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.transformations import colour_spaces
from kernelphysiology.transformations import normalisations

import cv2

models = {
    'custom': {'vqvae': vae_model.VQ_CVAE},
    'imagenet': {'vqvae': vae_model.VQ_CVAE},
    'bsds': {'vqvae': vae_model.VQ_CVAE},
    'voc': {'vqvae': vae_model.VQ_CVAE},
    'coco': {'vqvae': vae_model.VQ_CVAE},
    'cifar10': {'vae': vae_model.CVAE, 'vqvae': vae_model.VQ_CVAE},
    'mnist': {'vae': vae_model.VAE, 'vqvae': vae_model.VQ_CVAE},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': data_loaders.ImageFolder,
    'bsds': data_loaders.BSDSEdges,
    'coco': torch.utils.data.DataLoader,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'bsds': {},
    'voc': {},
    'coco': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'bsds': {},
    'voc': {},
    'coco': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'bsds': 3,
    'voc': 3,
    'coco': 3,
    'cifar10': 3,
    'mnist': 1,
}
dataset_target_size = {
    'imagenet': 224,
}
default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'bsds': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'voc': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'coco': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64}
}


def generic_inv_fun(x, colour_space):
    if colour_space == 'hsv':
        x = colour_spaces.hsv012rgb(x)
    elif colour_space == 'dkl':
        x = colour_spaces.dkl012rgb(x)
    elif colour_space == 'lab':
        x *= 255
        x = x.astype('uint8')
        x = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
    return x


def removekey(d, r_key):
    r = dict(d)
    for key, val in d.items():
        if r_key in key:
            del r[key]
    return r


def main(args):
    args = parse_arguments(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.mean = (0.5, 0.5, 0.5)
    args.std = (0.5, 0.5, 0.5)
    normalise = transforms.Normalize(args.mean, args.std)

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = args.num_channels or dataset_n_channels[args.dataset]
    target_size = args.target_size or dataset_target_size[args.dataset]

    dataset_transforms = {
        'custom': transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224),
             transforms.ToTensor(), normalise]),
        'coco': transforms.Compose([normalise]),
        'voc': transforms.Compose(
            [cv2_transforms.RandomResizedCropSegmentation(target_size,
                                                          scale=(0.50, 1.0)),
             cv2_transforms.ToTensorSegmentation(),
             cv2_transforms.NormalizeSegmentation(args.mean, args.std)]),
        'bsds': transforms.Compose(
            [cv2_transforms.RandomResizedCropSegmentation(target_size,
                                                          scale=(0.50, 1.0)),
             cv2_transforms.ToTensorSegmentation(),
             cv2_transforms.NormalizeSegmentation(args.mean, args.std)]),
        'imagenet': transforms.Compose(
            [cv2_transforms.Resize(target_size + 32),
             cv2_transforms.CenterCrop(target_size),
             cv2_transforms.ToTensor(),
             cv2_transforms.Normalize(args.mean, args.std)]),
        'cifar10': transforms.Compose(
            [transforms.ToTensor(), normalise]),
        'mnist': transforms.ToTensor()
    }

    save_path = vae_util.setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    if args.model == 'wavenet':
        # model = wavenet_vae.wavenet_bottleneck(
        #     latent_dim=k, in_channels=num_channels
        # )
        task = None
        out_chns = 3
        if 'voc' in args.dataset:
            task = 'segmentation'
            out_chns = 21
        from kernelphysiology.dl.pytorch.models import \
            model_utils as model_utils
        from torchvision.models import resnet
        backbone = resnet.__dict__['resnet50'](
            pretrained=True,
            replace_stride_with_dilation=[False, True, True]
        )
        from torchvision.models._utils import IntermediateLayerGetter
        return_layers = {'layer4': 'out'}
        resnet = IntermediateLayerGetter(
            backbone, return_layers=return_layers
        )
        model = vae_model.ResNet_VQ_CVAE(
            hidden, k=k, resnet=resnet, num_channels=num_channels,
            colour_space=args.colour_space, task=task,
            out_chns=out_chns
        )
    else:
        task = None
        out_chns = 3
        if 'voc' in args.dataset:
            task = 'segmentation'
            out_chns = 21
        elif 'bsds' in args.dataset:
            task = 'segmentation'
            out_chns = 1
        model = models[args.dataset][args.model](
            hidden, k=k, kl=args.kl, num_channels=num_channels,
            colour_space=args.colour_space, task=task,
            out_chns=out_chns, cos_distance=args.cos_dis
        )
    if args.cuda:
        model.cuda()

    if args.load_encoder is not None:
        params_to_optimize = [
            {'params': [p for p in model.decoder.parameters() if
                        p.requires_grad]},
            {'params': [p for p in model.fc.parameters() if
                        p.requires_grad]},
        ]
        optimizer = optim.Adam(params_to_optimize, lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, int(args.epochs / 3), 0.5
    )

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        args.start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.fine_tune is not None:
        weights = torch.load(args.fine_tune, map_location='cpu')
        model.load_state_dict(weights, strict=False)
        model.cuda()
    elif args.load_encoder is not None:
        weights = torch.load(args.load_encoder, map_location='cpu')
        weights = removekey(weights, 'decode')
        model.load_state_dict(weights, strict=False)
        model.cuda()

    intransform_funs = []
    if args.mosaic_pattern is not None:
        intransform_funs.append(
            cv2_preprocessing.MosaicTransformation(args.mosaic_pattern)
        )
    intransform = transforms.Compose(intransform_funs)
    outtransform_funs = []
    args.inv_func = None
    if args.colour_space is not None:
        outtransform_funs.append(
            cv2_preprocessing.ColourTransformation(None, args.colour_space)
        )
        if args.vis_rgb:
            args.inv_func = lambda x: generic_inv_fun(x, args.colour_space)
    outtransform = transforms.Compose(outtransform_funs)

    if args.data_dir is not None:
        args.train_dir = os.path.join(args.data_dir, 'train')
        args.validation_dir = os.path.join(args.data_dir, 'validation')
    else:
        args.train_dir = args.train_dir
        args.validation_dir = args.validation_dir
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    args.vis_func = vae_util.grid_save_reconstructed_images
    if args.dataset == 'coco':
        train_loader = panoptic_utils.get_coco_train(
            args.batch_size, args.opts, args.cfg_file
        )
        test_loader = panoptic_utils.get_coco_test(
            args.batch_size, args.opts, args.cfg_file
        )
    elif 'voc' in args.dataset:
        train_loader = torch.utils.data.DataLoader(
            data_loaders.VOCSegmentation(
                root=args.data_dir,
                image_set='train',
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_train_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            data_loaders.VOCSegmentation(
                root=args.data_dir,
                image_set='val',
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_test_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    elif args.category is not None:
        train_loader = torch.utils.data.DataLoader(
            data_loaders.CategoryImages(
                root=args.train_dir,
                category=args.category,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_train_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            data_loaders.CategoryImages(
                root=args.validation_dir,
                category=args.category,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_test_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    elif args.dataset == 'bsds':
        args.vis_func = vae_util.grid_save_reconstructed_bsds
        train_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](
                root=args.data_dir,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_train_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](
                root=args.data_dir,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_test_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](
                root=args.train_dir,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_train_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](
                root=args.validation_dir,
                intransform=intransform,
                outtransform=outtransform,
                transform=dataset_transforms[args.dataset],
                **dataset_test_args[args.dataset]
            ),
            batch_size=args.batch_size, shuffle=False, **kwargs
        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(args.start_epoch, args.epochs):
        train_losses = train(
            epoch, model, train_loader, optimizer, args.cuda, args.log_interval,
            save_path, args
        )
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path,
                               args)
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(
                name, {'train': train_losses[train_name],
                       'test': test_losses[test_name]}, epoch
            )
        scheduler.step()
        vae_util.save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'arch': {'k': args.k, 'hidden': args.hidden}
            },
            save_path
        )


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path,
          args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, loader_data in enumerate(train_loader):
        if args.dataset == 'coco':
            data = []
            for batch_data in loader_data:
                current_image = batch_data['image'][[2, 1, 0], :, :].clone()
                current_image = current_image.unsqueeze(0)
                current_image = current_image.type('torch.FloatTensor')
                current_image = nn.functional.interpolate(
                    current_image, (224, 224)
                )
                current_image /= 255
                current_image[0] = functional.normalize(
                    current_image[0], args.mean, args.std, False
                )
                data.append(current_image)
            data = torch.cat(data, dim=0)
            max_len = len(train_loader.dataset)
        else:
            data = loader_data[0]
            target = loader_data[1]
            max_len = len(train_loader)
            target = target.cuda()

        data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)

        loss = model.loss_function(target, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(
                ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info(
                'Train Epoch: {epoch} [{batch:5d}/{total_batch} '
                '({percent:2d}%)]   time: {time:3.2f}   {loss}'
                    .format(epoch=epoch, batch=batch_idx * len(data),
                            total_batch=max_len * len(data),
                            percent=int(100. * batch_idx / max_len),
                            time=time.time() - start_time, loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx in [18, 180, 1650, max_len - 1]:
            args.vis_func(
                target, outputs, args.mean, args.std, epoch, save_path,
                'reconstruction_train%.5d' % batch_idx, args.inv_func
            )

        if args.dataset in ['imagenet', 'coco', 'custom'] and batch_idx * len(
                data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (max_len / data.shape[0])
        else:
            epoch_losses[key] /= (
                    len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    # writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
    # model.print_atom_hist(outputs[3])
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, loader_data in enumerate(test_loader):
            if args.dataset == 'coco':
                data = []
                for batch_data in loader_data:
                    current_image = batch_data['image'][[2, 1, 0], :, :].clone()
                    current_image = current_image.unsqueeze(0)
                    current_image = current_image.type('torch.FloatTensor')
                    current_image = nn.functional.interpolate(
                        current_image, (224, 224)
                    )
                    current_image /= 255
                    current_image[0] = functional.normalize(
                        current_image[0], args.mean, args.std, False
                    )
                    data.append(current_image)
                data = torch.cat(data, dim=0)
            else:
                data = loader_data[0]
                target = loader_data[1]
                target = target.cuda()
            data = data.cuda()
            outputs = model(data)
            model.loss_function(target, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i in [0, 100, 200, 300, 400]:
                args.vis_func(
                    target, outputs, args.mean, args.std, epoch, save_path,
                    'reconstruction_test%.5d' % i, args.inv_func
                )
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'coco', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(
        ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


if __name__ == "__main__":
    main(sys.argv[1:])
