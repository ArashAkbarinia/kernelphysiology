import argparse
import os
import random
import time
import warnings
import numpy as np

from skimage import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn import functional as F

from kernelphysiology.utils.imutils import get_colour_inds
from kernelphysiology.dl.pytorch.utils import transformations
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy
from kernelphysiology.dl.pytorch.utils.misc import adjust_learning_rate
from kernelphysiology.dl.pytorch.utils.misc import save_checkpoint
from kernelphysiology.dl.pytorch.utils.preprocessing import normalise_tensor
from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor
from kernelphysiology.dl.pytorch.models.model_utils import \
    which_network_classification
from kernelphysiology.utils.path_utils import create_dir

from model import SimpleVAE

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument(
    '-a', '--arch', metavar='ARCH', default='resnet18',
    choices=model_names,
    help='model architecture: | '.join(model_names) + ' (default: resnet18)'
)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b', '--batch-size', default=256, type=int, metavar='N',
    help='mini-batch size (default: 256)'
)
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--target_size', default=224, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
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
    args.out_dir = args.experiment_name
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

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, img_shape):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_shape),
                                 reduction='sum')
    # print('ARASH', recon_x.min(), recon_x.max())
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # HINT TWO networks introduced
    args.n2_transform = 1
    (negative_network, _) = which_network_classification(
        args.pos_net_path, num_classes=args.num_classes
    )
    (positive_network, _) = which_network_classification(
        args.neg_net_path, num_classes=args.num_classes
    )
    if args.gpu is not None:
        negative_network = negative_network.cuda(args.gpu)
        positive_network = positive_network.cuda(args.gpu)

    for param in positive_network.parameters():
        param.requires_grad = False
    for param in negative_network.parameters():
        param.requires_grad = False

    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.img_shape = args.target_size * args.target_size * 3

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        # TODO make the model more smart
        pos_tra = get_colour_inds(args.pos_colour)
        neg_tra = get_colour_inds(args.neg_colour)
        model = SimpleVAE(img_shape=args.img_shape)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # TODO: should it go to GPU?
    criterion = loss_function

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

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

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.target_size),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_progress = []
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
            transforms.CenterCrop(args.target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

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

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'target_size': args.target_size,
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
        output_imgs, mu, logvar = model(input_imgs)
        loss_gen = criterion(
            output_imgs, input_imgs, mu, logvar, args.img_shape
        )
        losses_gen.update(loss_gen.item(), input_imgs.size(0))

        if i % 200 == 0:
            save_sample_imgs(
                args.out_dir, input_imgs, output_imgs, args
            )

        # output = neg_out
        # loss_neg = args.criterion_neg(output, target)
        # acc1_neg, acc5 = accuracy(output, target, topk=(1, 5))
        # losses_neg.update(loss_neg.item(), input_imgs.size(0))
        # top1_neg.update(acc1_neg[0], input_imgs.size(0))
        #
        # output = pos_out
        # loss_pos = args.criterion_pos(output, target)
        # acc1_pos, acc5 = accuracy(output, target, topk=(1, 5))
        # losses_pos.update(loss_pos.item(), input_imgs.size(0))
        # top1_pos.update(acc1_pos[0], input_imgs.size(0))

        #        loss = 0.2 * loss_gen + 0.3 * (loss_pos - loss_neg) + 0.5 * (loss_pos - 1.1)
        #        loss = 0.5 * loss_gen + 0.5 * ((loss_pos - loss_neg) / (loss_pos - loss_neg))
        # loss = 0.2 * loss_gen + 0.3 * (loss_pos / loss_neg) + 0.5 * ( loss_pos - 1.1)
        loss = loss_gen
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


def save_sample_imgs(out_dir, input_imgs, output_imgs, args):
    mean = args.mean
    std = args.std
    output_imgs = output_imgs.view(
        output_imgs.shape[0], 3, args.target_size, args.target_size
    )
    input_imgs = input_imgs.detach().cpu()
    output_imgs = output_imgs.detach().cpu()
    # input_imgs = inv_normalise_tensor(input_imgs.data, mean, std).cpu()
    # output_imgs = inv_normalise_tensor(output_imgs.data, mean, std).cpu()
    for i in range(min(input_imgs.shape[0], 10)):
        tmp_img = input_imgs.numpy()[i,].squeeze()
        tmp_img = np.transpose(tmp_img, (1, 2, 0))
        tmp_img = (tmp_img * 255).astype('uint8')
        io.imsave(out_dir + '/img%04d_org.jpg' % i, tmp_img)
        tmp_img = output_imgs.numpy()[i,].squeeze()
        tmp_img = np.transpose(tmp_img, (1, 2, 0))
        tmp_img = (tmp_img * 255).astype('uint8')
        io.imsave(out_dir + '/img%04d_out.jpg' % i, tmp_img)


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
            loss = 0.5 * loss_gen + 0.5 * (
                    (loss_pos - loss_neg) / (loss_pos - loss_neg))
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
