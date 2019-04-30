import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from kernelphysiology.utils.imutils import get_colour_inds
from kernelphysiology.dl.pytorch.utils import transformations
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy
from kernelphysiology.dl.pytorch.utils.misc import adjust_learning_rate
from kernelphysiology.dl.pytorch.utils.misc import save_checkpoint
from kernelphysiology.dl.pytorch.utils.preprocessing import normalise_tensor
from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor
from kernelphysiology.dl.pytorch.models.utils import which_network

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--experiment_name', type=str, default='Ex',
                    help='The name of the experiment (default: Ex)')
parser.add_argument('--pos_net_path', type=str, required=True,
                    help='The path to network to be maximised.')
parser.add_argument('--neg_net_path', type=str, required=True,
                    help='The path to network to be minimised.')
parser.add_argument('--pos_colour', type=str, default='trichromat',
                    help='Preprocessing colour transformation to pos net.')
parser.add_argument('--neg_colour', type=str, default='trichromat',
                    help='Preprocessing colour transformation to neg net.')

best_acc1 = 0

warnings.filterwarnings(
    'ignore',
    '.*is a low contrast image.*',
    UserWarning)


def main():
    args = parser.parse_args()
    # args.out_dir = prepare_training.prepare_output_directories(
    #     dataset_name='imagenet',
    #     network_name=args.arch,
    #     optimiser='sgd',
    #     load_weights=False,
    #     experiment_name=args.experiment_name,
    #     framework='pytorch')
    args.out_dir = args.experiment_name
    from kernelphysiology.utils.path_utils import create_dir
    create_dir(args.out_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


class SimpleModel(nn.Module):
    def __init__(self, pos_net, neg_net, pos_tra, neg_tra, mean, std):
        super(SimpleModel, self).__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 1),
            nn.Tanh()
        )
        self.pos_net = pos_net
        self.neg_net = neg_net
        self.pos_tra = pos_tra
        self.neg_tra = neg_tra
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = self.generator(x)
        x_pos = x.detach()
        if self.pos_tra is not None:
            x_pos = prepare_dichromat(
                    inv_normalise_tensor(x_pos, self.mean, self.std), self.pos_tra)
            x_pos = normalise_tensor(x_pos, self.mean, self.std)
        pos_out = self.pos_net(x_pos)
        x_neg = x.detach()
        if self.neg_tra is not None:
            x_neg = prepare_dichromat(
                    inv_normalise_tensor(x_neg, self.mean, self.std), self.neg_tra)
            x_neg = normalise_tensor(x_neg, self.mean, self.std)
        neg_out = self.neg_net(x_neg)
        return x, pos_out, neg_out, x_pos, x_neg


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # HINT TWO networks introduced
    args.n2_transform = 1
    (negative_network, _) = which_network(args.pos_net_path,
                                          'classification', '')
    (positive_network, _) = which_network(args.neg_net_path,
                                          'classification', '')
    if args.gpu is not None:
        negative_network = negative_network.cuda(args.gpu)
        positive_network = positive_network.cuda(args.gpu)

    for param in positive_network.parameters():
        param.requires_grad = False
    for param in negative_network.parameters():
        param.requires_grad = False

    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        # TODO make the model more smart
        pos_tra = get_colour_inds(args.pos_colour)
        neg_tra = get_colour_inds(args.neg_colour)
        model = SimpleModel(pos_net=positive_network, neg_net=negative_network,
                            pos_tra=pos_tra, neg_tra=neg_tra,
                            mean=args.mean, std=args.std)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda(args.gpu) # MSELoss

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'validation')  # FIXME change it to train
    valdir = os.path.join(args.data, 'validation')
    normalize = transforms.Normalize(mean=args.mean, std=args.std)

    target_size = 224
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_progress = []
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    file_path = os.path.join(args.out_dir, 'model_progress.csv')

    args.criterion_pos = nn.CrossEntropyLoss().cuda(args.gpu)
    args.criterion_neg = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
#            transforms.RandomResizedCrop(target_size),
            transforms.Resize(256),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer, epoch,
                          args)

        # evaluate on validation set
#        validation_log = validate(val_loader, model, criterion, args)
        validation_log = train_log[1:]
        model_progress.append([*train_log, *validation_log])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[3] - validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'target_size': target_size,
            }, is_best, out_folder=args.out_dir)
        np.savetxt(file_path, np.array(model_progress), delimiter=',')


def prepare_dichromat(imgs_rgb, which_inds):
    imgs_lab = transformations.rgb2lab(imgs_rgb)
    imgs_lab[:, which_inds, ] = 0
    output = transformations.lab2rgb(imgs_lab)
    return output


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_overall = AverageMeter()
    losses_gen = AverageMeter()
    losses_neg = AverageMeter()
    losses_pos = AverageMeter()
    top1_neg = AverageMeter()
    top1_pos = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_imgs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input_imgs = input_imgs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        model.zero_grad()
        # compute output
        output_imgs, pos_out, neg_out, x_pos, x_neg = model(input_imgs)
        loss_gen = criterion(output_imgs, input_imgs)
        losses_gen.update(loss_gen.item(), input_imgs.size(0))

        if i % 200 == 0:
            save_sample_imgs(args.out_dir, input_imgs, output_imgs,
                             x_neg, args.mean, args.std)

        output = neg_out
        loss_neg = args.criterion_neg(output, target)
        acc1_neg, acc5 = accuracy(output, target, topk=(1, 5))
        losses_neg.update(loss_neg.item(), input_imgs.size(0))
        top1_neg.update(acc1_neg[0], input_imgs.size(0))

        output = pos_out
        loss_pos = args.criterion_pos(output, target)
        acc1_pos, acc5 = accuracy(output, target, topk=(1, 5))
        losses_pos.update(loss_pos.item(), input_imgs.size(0))
        top1_pos.update(acc1_pos[0], input_imgs.size(0))

#        loss = 0.2 * loss_gen + 0.3 * (loss_pos - loss_neg) + 0.5 * (loss_pos - 1.1)
#        loss = 0.5 * loss_gen + 0.5 * ((loss_pos - loss_neg) / (loss_pos - loss_neg))
        loss = 0.2 * loss_gen + 0.3 * (loss_pos / loss_neg) + 0.5 * (loss_pos - 1.1)
        loss.backward()
        losses_overall.update(loss.item(), input_imgs.size(0))
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'L {losses_overall.val:.3f} ({losses_overall.avg:.3f})\t'
                  'L@I {losses_gen.val:.3f} ({losses_gen.avg:.3f})\t'
                  'L@N {losses_neg.val:.3f} ({losses_neg.avg:.3f})\t'
                  'L@P {losses_pos.val:.3f} ({losses_pos.avg:.3f})\t'
                  'A@N {top1_neg.val:.3f} ({top1_neg.avg:.3f})\t'
                  'A@P {top1_pos.val:.3f} ({top1_pos.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                losses_overall=losses_overall,
                losses_gen=losses_gen,
                losses_neg=losses_neg,
                losses_pos=losses_pos,
                top1_neg=top1_neg, top1_pos=top1_pos))
    return [epoch, batch_time.avg, losses_overall.avg,
            top1_neg.avg, top1_pos.avg]


from skimage import io


def save_sample_imgs(out_dir, input_imgs, output_imgs, imgs_dichromat,
                     mean, std):
    input_imgs = inv_normalise_tensor(input_imgs.data, mean, std).cpu()
    output_imgs = inv_normalise_tensor(output_imgs.data, mean, std).cpu()
    imgs_dichromat = inv_normalise_tensor(imgs_dichromat.data, mean, std).cpu()
    for i in range(min(input_imgs.shape[0], 10)):
        tmp_img = input_imgs.numpy()[i,].squeeze()
        tmp_img = np.transpose(tmp_img, (1, 2, 0))
        io.imsave(out_dir + '/img%04d_org.jpg' % i, tmp_img)
        tmp_img = output_imgs.numpy()[i,].squeeze()
        tmp_img = np.transpose(tmp_img, (1, 2, 0))
        io.imsave(out_dir + '/img%04d_out.jpg' % i, tmp_img)
        tmp_img = imgs_dichromat.numpy()[i,].squeeze()
        tmp_img = np.transpose(tmp_img, (1, 2, 0))
        io.imsave(out_dir + '/img%04d_out_di.jpg' % i, tmp_img)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses_overall = AverageMeter()
    losses_gen = AverageMeter()
    losses_neg = AverageMeter()
    losses_pos = AverageMeter()
    top1_neg = AverageMeter()
    top1_pos = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            if args.gpu is not None:
                input_imgs = input_imgs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output_imgs, pos_out, neg_out, _, _ = model(input_imgs)
            loss_gen = criterion(output_imgs, input_imgs)
            losses_gen.update(loss_gen.item(), input_imgs.size(0))

            output = neg_out
            loss_neg = args.criterion_neg(output, target)
            acc1_neg, acc5 = accuracy(output, target, topk=(1, 5))
            losses_neg.update(loss_neg.item(), input_imgs.size(0))
            top1_neg.update(acc1_neg[0], input_imgs.size(0))

            output = pos_out
            loss_pos = args.criterion_pos(output, target)
            acc1_pos, acc5 = accuracy(output, target, topk=(1, 5))
            losses_pos.update(loss_pos.item(), input_imgs.size(0))
            top1_pos.update(acc1_pos[0], input_imgs.size(0))

#            loss = 0.2 * loss_gen + 0.3 * (loss_pos - loss_neg) + 0.5 * (loss_pos - 1.1)
            loss = 0.5 * loss_gen + 0.5 * ((loss_pos - loss_neg) / (loss_pos - loss_neg))
            losses_overall.update(loss.item(), input_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'L {losses_overall.val:.3f} ({losses_overall.avg:.3f})\t'
                      'L@I {losses_gen.val:.3f} ({losses_gen.avg:.3f})\t'
                      'L@N {losses_neg.val:.3f} ({losses_neg.avg:.3f})\t'
                      'L@P {losses_pos.val:.3f} ({losses_pos.avg:.3f})\t'
                      'A@N {top1_neg.val:.3f} ({top1_neg.avg:.3f})\t'
                      'A@P {top1_pos.val:.3f} ({top1_pos.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    losses_overall=losses_overall,
                    losses_gen=losses_gen,
                    losses_neg=losses_neg,
                    losses_pos=losses_pos,
                    top1_neg=top1_neg,
                    top1_pos=top1_pos))
        print(' * A@N {top1_neg.avg:.3f} A@P {top1_pos.avg:.3f}'
              .format(top1_neg=top1_neg,
                      top1_pos=top1_pos))

    return [batch_time.avg, losses_overall.avg, top1_neg.avg,
            top1_pos.avg]


if __name__ == '__main__':
    main()
