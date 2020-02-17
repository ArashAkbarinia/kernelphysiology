"""
PyTorch training script for various datasets and image manipulations.
"""

import os
import sys
import random
import warnings
import numpy as np

import time

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
import torchvision.models as models

from kernelphysiology.dl.experiments.munsellnet import resnet
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter, accuracy
from kernelphysiology.dl.pytorch.utils.misc import adjust_learning_rate
from kernelphysiology.dl.pytorch.utils.misc import save_checkpoint
from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.pytorch.datasets import utils_db
from kernelphysiology.dl.utils.default_configs import get_default_target_size
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.utils.path_utils import create_dir

from kernelphysiology.dl.experiments.munsellnet.dataset import \
    get_train_val_dataset

best_acc1 = 0


def validate_on_data(val_loader, model, criterion, args):
    losses = AverageMeter()
    batch_time = AverageMeter()

    losses_obj = AverageMeter()
    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    losses_mun = AverageMeter()
    top1_mun = AverageMeter()
    top5_mun = AverageMeter()
    losses_ill = AverageMeter()
    top1_ill = AverageMeter()
    top5_ill = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_image, targets) in enumerate(val_loader):
            if args.gpus is not None:
                input_image = input_image.cuda(args.gpus, non_blocking=True)
            targets = targets.cuda(args.gpus, non_blocking=True)

            # compute output
            out_obj, out_mun, out_ill = model(input_image)
            loss_obj = criterion(out_obj, targets[:, 0])
            loss_mun = criterion(out_mun, targets[:, 1])
            loss_ill = criterion(out_ill, targets[:, 2])

            loss = loss_obj + loss_mun + loss_ill
            losses.update(loss.item(), input_image.size(0))

            # measure accuracy and record loss
            acc1_obj, acc5_obj = accuracy(out_obj, targets[:, 0], topk=topks)
            losses_obj.update(loss_obj.item(), input_image.size(0))
            top1_obj.update(acc1_obj[0], input_image.size(0))
            top5_obj.update(acc5_obj[0], input_image.size(0))

            acc1_mun, acc5_mun = accuracy(out_mun, targets[:, 1], topk=topks)
            losses_mun.update(loss_mun.item(), input_image.size(0))
            top1_mun.update(acc1_mun[0], input_image.size(0))
            top5_mun.update(acc5_mun[0], input_image.size(0))

            acc1_ill, acc5_ill = accuracy(out_ill, targets[:, 2], topk=topks)
            losses_ill.update(loss_ill.item(), input_image.size(0))
            top1_ill.update(acc1_ill[0], input_image.size(0))
            top5_ill.update(acc5_ill[0], input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                    'LO {obj_loss.val:.2f} ({obj_loss.avg:.2f})\t'
                    'LM {mun_loss.val:.2f} ({mun_loss.avg:.2f})\t'
                    'LI {ill_loss.val:.2f} ({ill_loss.avg:.2f})\t'
                    'Ao {obj_acc.val:.2f} ({obj_acc.avg:.2f})\t'
                    'AM {mun_acc.val:.2f} ({mun_acc.avg:.2f})\t'
                    'AI {ill_acc.val:.2f} ({ill_acc.avg:.2f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, obj_loss=losses_obj,
                        mun_loss=losses_mun, ill_loss=losses_ill,
                        obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
                    )
                )
        # printing the accuracy of the epoch
        print(
            ' * AccObj {obj_acc.avg:.2f} AccMun {mun_acc.avg:.2f}'
            ' AccIll {ill_acc.avg:.2f}'.format(
                obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
            )
        )

    return [batch_time.avg, losses.avg, losses_obj.avg, losses_mun.avg,
            losses_ill.avg, top1_obj.avg, top1_mun.avg, top1_ill.avg]


def train_on_data(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_obj = AverageMeter()
    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    losses_mun = AverageMeter()
    top1_mun = AverageMeter()
    top5_mun = AverageMeter()
    losses_ill = AverageMeter()
    top1_ill = AverageMeter()
    top5_ill = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_image, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            input_image = input_image.cuda(args.gpus, non_blocking=True)
        targets = targets.cuda(args.gpus, non_blocking=True)

        # compute output
        out_obj, out_mun, out_ill = model(input_image)
        import pdb
        # pdb.set_trace()
        loss_obj = criterion(out_obj, targets[:, 0])
        loss_mun = criterion(out_mun, targets[:, 1])
        loss_ill = criterion(out_ill, targets[:, 2])

        loss = loss_obj + loss_mun + loss_ill
        losses.update(loss.item(), input_image.size(0))

        # measure accuracy and record loss
        acc1_obj, acc5_obj = accuracy(out_obj, targets[:, 0], topk=topks)
        losses_obj.update(loss_obj.item(), input_image.size(0))
        top1_obj.update(acc1_obj[0], input_image.size(0))
        top5_obj.update(acc5_obj[0], input_image.size(0))

        acc1_mun, acc5_mun = accuracy(out_mun, targets[:, 1], topk=topks)
        losses_mun.update(loss_mun.item(), input_image.size(0))
        top1_mun.update(acc1_mun[0], input_image.size(0))
        top5_mun.update(acc5_mun[0], input_image.size(0))

        acc1_ill, acc5_ill = accuracy(out_ill, targets[:, 2], topk=topks)
        losses_ill.update(loss_ill.item(), input_image.size(0))
        top1_ill.update(acc1_ill[0], input_image.size(0))
        top5_ill.update(acc5_ill[0], input_image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # printing the accuracy at certain intervals
        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                'LO {obj_loss.val:.2f} ({obj_loss.avg:.2f})\t'
                'LM {mun_loss.val:.2f} ({mun_loss.avg:.2f})\t'
                'LI {ill_loss.val:.2f} ({ill_loss.avg:.2f})\t'
                'Ao {obj_acc.val:.2f} ({obj_acc.avg:.2f})\t'
                'AM {mun_acc.val:.2f} ({mun_acc.avg:.2f})\t'
                'AI {ill_acc.val:.2f} ({ill_acc.avg:.2f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, obj_loss=losses_obj,
                    mun_loss=losses_mun, ill_loss=losses_ill,
                    obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
                )
            )
    return [epoch, batch_time.avg, losses.avg, losses_obj.avg, losses_mun.avg,
            losses_ill.avg, top1_obj.avg, top1_mun.avg, top1_ill.avg]


def main(argv):
    args = argument_handler.train_arg_parser(argv)
    if args.lr is None:
        args.lr = 0.1
    if args.weight_decay is None:
        args.weight_decay = 1e-4
    # FIXME: cant take more than one GPU
    args.gpus = args.gpus[0]

    # TODO: why load weights is False?
    args.out_dir = prepare_training.prepare_output_directories(
        dataset_name=args.dataset, network_name=args.network_name,
        optimiser='sgd', load_weights=False,
        experiment_name=args.experiment_name, framework='pytorch'
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting from checkpoints.'
        )

    if args.gpus is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely '
            'disable data parallelism.'
        )

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
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
        )
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1

    mean, std = model_utils.get_preprocessing_function(
        args.colour_space, args.colour_transformation
    )

    # preparing the output folder
    create_dir(args.out_dir)

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpus
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    # create model
    if args.transfer_weights is not None:
        print('Transferred model!')
        (model, _) = model_utils.which_network(
            args.transfer_weights, args.task_type, num_classes=args.old_classes
        )
        model = model_utils.NewClassificationModel(model, args.num_classes)
    elif args.custom_arch:
        print('Custom model!')
        if (args.network_name == 'resnet_basic_custom' or
                args.network_name == 'resnet_bottleneck_custom'):
            outputs = {
                'objects': {'num_classes': 2100, 'area': 'b4'},
                'munsells': {'num_classes': 1600, 'area': 'b3'},
                'illuminants': {'num_classes': 280, 'area': 'b2'}
            }
            model = resnet.__dict__[args.network_name](
                args.blocks, pooling_type=args.pooling_type,
                in_chns=len(mean), inplanes=args.num_kernels,
                outputs=outputs
            )
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.network_name))
        model = models.__dict__[args.network_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.network_name))
        model = models.__dict__[args.network_name]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpus is not None:
            torch.cuda.set_device(args.gpus)
            model.cuda(args.gpus)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpus]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpus is not None:
        torch.cuda.set_device(args.gpus)
        model = model.cuda(args.gpus)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if (args.network_name.startswith('alexnet') or
                args.network_name.startswith('vgg')):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpus)

    # optimiser
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    model_progress = []
    model_progress_path = os.path.join(args.out_dir, 'model_progress.csv')
    # optionally resume from a checkpoint
    # TODO: it would be best if resume load the architecture from this file
    # TODO: merge with which_architecture
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.initial_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpus)
                model = model.cuda(args.gpus)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']
                )
            )
            if os.path.exists(model_progress_path):
                model_progress = np.loadtxt(model_progress_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=mean, std=std)

    other_transformations = []
    if args.num_augmentations != 0:
        augmentations = preprocessing.RandomAugmentationTransformation(
            args.augmentation_settings, args.num_augmentations,
            utils_db.is_dataset_pil_image(args.dataset)
        )
        other_transformations.append(augmentations)

    target_size = get_default_target_size(args.dataset, args.target_size)

    train_dataset, validation_dataset = get_train_val_dataset(
        args.data_dir, other_transformations, normalize
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = train_on_data(
            train_loader, model, criterion, optimizer, epoch, args
        )

        # evaluate on validation set
        validation_log = validate_on_data(
            val_loader, model, criterion, args
        )

        model_progress.append([*train_log, *validation_log])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.network_name,
                    'customs': {
                        'pooling_type': args.pooling_type,
                        'in_chns': len(mean),
                        'num_classes': args.num_classes,
                        'blocks': args.blocks,
                        'num_kernels': args.num_kernels,
                        'outputs': outputs
                    },
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'target_size': target_size,
                },
                is_best, out_folder=args.out_dir
            )
        # TODO: get this header directly as a dictionary keys
        header = 'epoch,t_time,t_loss,t_lo,t_lm,t_li,t_ao,t_am,t_ai,' \
                 'v_time,v_loss,v_lo,v_lm,v_li,v_ao,v_am,v_ai'
        np.savetxt(
            model_progress_path, np.array(model_progress),
            delimiter=',', header=header
        )


if __name__ == '__main__':
    main(sys.argv[1:])
