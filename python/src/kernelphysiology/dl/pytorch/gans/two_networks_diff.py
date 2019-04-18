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

from kernelphysiology.dl.pytorch.utils import transformations
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
parser.add_argument('--net_folder', type=str, required=True,
                    help='The folder containing two networks')

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
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        return x


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
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        # TODO make the model more smart
        model = SimpleModel()

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
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  args.gpu])
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
    criterion = nn.MSELoss().cuda(args.gpu)

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
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

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

    # HINT TWO networks introduced
    args.n2_transform = 1
    (trichromat_network, _) = which_network(
        args.net_folder + '/nets/pytorch/imagenet'
                          '/imagenet/resnet50/sgd/scratch/original_b64/model_best.pth.tar',
        'classification', '')
    (dichromat_network, _) = which_network(
        args.net_folder + '/nets/pytorch/imagenet'
                          '/imagenet/resnet50/sgd/scratch/deficiency_red_green/model_best.pth.tar',
        'classification', '')
    if args.gpu is not None:
        trichromat_network = trichromat_network.cuda(args.gpu)
        dichromat_network = dichromat_network.cuda(args.gpu)
    args.dichromat_network = dichromat_network
    args.trichromat_network = trichromat_network

    args.criterion_dichromat = nn.CrossEntropyLoss().cuda(args.gpu)
    args.criterion_trichromat = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(target_size),
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
        validation_log = validate(val_loader, model, criterion, args)
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


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def normalise_tensor(tensor, mean, std):
    # normalising the channels
    for i in range(tensor.shape[1]):
        tensor[:, i, ] = (tensor[:, i, ] - mean[i]) / std[i]
    return tensor


def prepare_dichromat(imgs_rgb, which_inds):
    imgs_lab = transformations.rgb2lab(imgs_rgb)
    imgs_lab[which_inds,] = 0
    output = transformations.lab2rgb(imgs_lab)
    return output


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_overall = AverageMeter()
    losses_gen = AverageMeter()
    losses_dichromat = AverageMeter()
    losses_trichromat = AverageMeter()
    top1_dichromat = AverageMeter()
    top1_trichromat = AverageMeter()

    dichromat_network = args.dichromat_network
    trichromat_network = args.trichromat_network

    # switch to train mode
    model.train()
    dichromat_network.eval()
    trichromat_network.eval()

    end = time.time()
    for i, (input_imgs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input_imgs = input_imgs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output_imgs = model(input_imgs)
        loss = criterion(output_imgs, input_imgs)
        losses_gen.update(loss.item(), input_imgs.size(0))

        # predicting by other networks
        output_imgs = to_img(output_imgs.data)
        imgs_dichromat = prepare_dichromat(output_imgs, args.n2_transform)
        if i % 200 == 0:
            save_sample_imgs(args.out_dir, input_imgs, output_imgs,
                             imgs_dichromat)
        imgs_dichromat = normalise_tensor(imgs_dichromat, args.mean, args.std)
        imgs_trichromat = normalise_tensor(output_imgs, args.mean, args.std)
        with torch.no_grad():
            output = dichromat_network(imgs_dichromat)
            loss_dichromat = args.criterion_dichromat(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_dichromat.update(loss_dichromat.item(), input_imgs.size(0))
            top1_dichromat.update(acc1[0], input_imgs.size(0))

            output = trichromat_network(imgs_trichromat)
            loss_trichromat = args.criterion_trichromat(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_trichromat.update(loss_trichromat.item(), input_imgs.size(0))
            top1_trichromat.update(acc1[0], input_imgs.size(0))

        loss = loss + (loss_trichromat / loss_dichromat)
        losses_overall.update(loss.item(), input_imgs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'L {losses_overall.val:.3f} ({losses_overall.avg:.3f})\t'
                  'L {losses_gen.val:.3f} ({losses_gen.avg:.3f})\t'
                  'L@D {losses_dichromat.val:.3f} ({losses_dichromat.avg:.3f})\t'
                  'L@T {losses_trichromat.val:.3f} ({losses_trichromat.avg:.3f})\t'
                  'A@D {top1_dichromat.val:.3f} ({top1_dichromat.avg:.3f})\t'
                  'A@T {top1_trichromat.val:.3f} ({top1_trichromat.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                losses_overall=losses_overall,
                losses_gen=losses_gen,
                losses_dichromat=losses_dichromat,
                losses_trichromat=losses_trichromat,
                top1_dichromat=top1_dichromat, top1_trichromat=top1_trichromat))
    return [epoch, batch_time.avg, losses_overall.avg,
            top1_dichromat.avg, top1_trichromat.avg]


from skimage import io


def save_sample_imgs(out_dir, input_imgs, output_imgs, imgs_dichromat):
    input_imgs = to_img(input_imgs.data).cpu()
    output_imgs = output_imgs.cpu()
    imgs_dichromat = imgs_dichromat.cpu()
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
    losses_dichromat = AverageMeter()
    losses_trichromat = AverageMeter()
    top1_dichromat = AverageMeter()
    top1_trichromat = AverageMeter()

    dichromat_network = args.dichromat_network
    trichromat_network = args.trichromat_network

    # switch to evaluate mode
    model.eval()
    dichromat_network.eval()
    trichromat_network.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            if args.gpu is not None:
                input_imgs = input_imgs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output_imgs = model(input_imgs)
            loss = criterion(output_imgs, input_imgs)
            losses_gen.update(loss.item(), input_imgs.size(0))

            # predicting by other networks
            output_imgs = to_img(output_imgs.data)
            imgs_dichromat = prepare_dichromat(output_imgs, args.n2_transform)
            imgs_dichromat = normalise_tensor(imgs_dichromat, args.mean,
                                              args.std)
            imgs_trichromat = normalise_tensor(output_imgs, args.mean, args.std)

            output = dichromat_network(imgs_dichromat)
            loss_dichromat = args.criterion_dichromat(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_dichromat.update(loss_dichromat.item(),
                                    input_imgs.size(0))
            top1_dichromat.update(acc1[0], input_imgs.size(0))

            output = trichromat_network(imgs_trichromat)
            loss_trichromat = args.criterion_trichromat(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_trichromat.update(loss_trichromat.item(),
                                     input_imgs.size(0))
            top1_trichromat.update(acc1[0], input_imgs.size(0))

            loss = loss + (loss_trichromat / loss_dichromat)
            losses_overall.update(loss.item(), input_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'L {losses_overall.val:.3f} ({losses_overall.avg:.3f})\t'
                      'L {losses_gen.val:.3f} ({losses_gen.avg:.3f})\t'
                      'L@D {losses_dichromat.val:.3f} ({losses_dichromat.avg:.3f})\t'
                      'L@T {losses_trichromat.val:.3f} ({losses_trichromat.avg:.3f})\t'
                      'A@D {top1_dichromat.val:.3f} ({top1_dichromat.avg:.3f})\t'
                      'A@T {top1_trichromat.val:.3f} ({top1_trichromat.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    losses_overall=losses_overall,
                    losses_gen=losses_gen,
                    losses_dichromat=losses_dichromat,
                    losses_trichromat=losses_trichromat,
                    top1_dichromat=top1_dichromat,
                    top1_trichromat=top1_trichromat))
        print(' * A@D {top1_dichromat.avg:.3f} A@T {top1_trichromat.avg:.3f}'
              .format(top1_dichromat=top1_dichromat,
                      top1_trichromat=top1_trichromat))

    return [batch_time.avg, losses_overall.avg, top1_dichromat.avg,
            top1_trichromat.avg]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
                    out_folder=''):
    filename = os.path.join(out_folder, filename)
    torch.save(state, filename)
    if is_best:
        model_best_path = os.path.join(out_folder, 'model_best.pth.tar')
        shutil.copyfile(filename, model_best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
