"""
PyTorch classification training script for various datasets.
"""

import os
import sys
import random
import warnings
import numpy as np
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from kernelphysiology.dl.pytorch import models as custom_models
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.pytorch.utils import misc as misc_utils
from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.utils import default_configs
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.utils.path_utils import create_dir

from kernelphysiology.dl.experiments.contrast import dataloader
from kernelphysiology.dl.experiments.contrast import pretrained_models


def main(argv):
    args = argument_handler.train_arg_parser(argv, extra_args_fun)
    args.lr, args.weight_decay = default_configs.optimisation_params(
        'classification', args
    )
    args.num_classes = 2
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

    json_file_name = os.path.join(args.out_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)

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
    mean, std = model_utils.get_preprocessing_function(
        args.colour_space, args.vision_type
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
        model = pretrained_models.NewClassificationModel(
            args.network_name, args.transfer_weights, args.grey_width == 40
        )
    elif args.custom_arch:
        print('Custom model!')
        supported_customs = ['resnet_basic_custom', 'resnet_bottleneck_custom']
        if args.network_name in supported_customs:
            model = custom_models.__dict__[args.network_name](
                args.blocks, pooling_type=args.pooling_type,
                in_chns=len(mean), num_classes=args.num_classes,
                inplanes=args.num_kernels, kernel_size=args.kernel_size
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
    if args.transfer_weights is None:
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        # for p in model.features.parameters():
        #     p.requires_grad = False
        params_to_optimize = [
            # {'params': [p for p in model.features.parameters()], 'lr': 1e-6},
            {'params': [p for p in model.fc.parameters()]},
        ]
        optimizer = torch.optim.SGD(
            params_to_optimize, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )

    model_progress = []
    model_progress_path = os.path.join(args.out_dir, 'model_progress.csv')
    # optionally resume from a checkpoint
    # TODO: it would be best if resume load the architecture from this file
    # TODO: merge with which_architecture
    best_acc1 = 0
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

    train_trans = []
    valid_trans = []
    both_trans = []
    if args.mosaic_pattern is not None:
        mosaic_trans = preprocessing.mosaic_transformation(args.mosaic_pattern)
        both_trans.append(mosaic_trans)

    if args.num_augmentations != 0:
        augmentations = preprocessing.random_augmentation(
            args.augmentation_settings, args.num_augmentations
        )
        train_trans.append(augmentations)

    target_size = default_configs.get_default_target_size(
        args.dataset, args.target_size
    )

    # loading the training set
    train_trans = [*both_trans, *train_trans]
    db_params = {
        'colour_space': args.colour_space,
        'vision_type': args.vision_type,
        'mask_image': args.mask_image,
        'grey_width': args.grey_width
    }
    if args.dataset in dataloader.NATURAL_DATASETS:
        path_or_sample = args.data_dir
    else:
        path_or_sample = args.train_samples
    train_dataset = dataloader.train_set(
        args.dataset, target_size, mean, std, extra_transformation=train_trans,
        data_dir=path_or_sample, **db_params
    )
    if args.dataset == 'natural':
        train_dataset.num_crops = args.batch_size
        args.batch_size = 1

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

    # loading validation set
    valid_trans = [*both_trans, *valid_trans]
    validation_dataset = dataloader.validation_set(
        args.dataset, target_size, mean, std, extra_transformation=valid_trans,
        data_dir=path_or_sample, **db_params
    )
    if args.dataset == 'natural':
        validation_dataset.num_crops = train_dataset.num_crops
        args.batch_size = 1

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        misc_utils.adjust_learning_rate(optimizer, epoch, args)

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

        if misc_utils.is_saving_node(
                args.multiprocessing_distributed, args.rank, ngpus_per_node
        ):
            misc_utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.network_name,
                    'customs': {
                        'pooling_type': args.pooling_type,
                        'in_chns': len(mean),
                        'num_classes': args.num_classes,
                        'blocks': args.blocks,
                        'num_kernels': args.num_kernels,
                        'kernel_size': args.kernel_size
                    },
                    'transfer_weights': args.transfer_weights,
                    'preprocessing': {'mean': mean, 'std': std},
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'target_size': target_size,
                },
                is_best, out_folder=args.out_dir
            )
            # TODO: get this header directly as a dictionary keys
            header = 'epoch,t_time,t_loss,t_top1,t_top5,v_time,v_loss,v_top1,v_top5'
            np.savetxt(
                model_progress_path, np.array(model_progress),
                delimiter=',', header=header
            )


def train_on_data(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = misc_utils.AverageMeter()
    data_time = misc_utils.AverageMeter()
    losses = misc_utils.AverageMeter()
    top1 = misc_utils.AverageMeter()
    top5 = misc_utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_image, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.dataset == 'natural':
            input_image = input_image[0]
            target = target[0]

        if args.gpus is not None:
            input_image = input_image.cuda(args.gpus, non_blocking=True)
        target = target.cuda(args.gpus, non_blocking=True)

        # compute output
        output = model(input_image)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = misc_utils.accuracy(output, target)
        acc5 = acc1.copy()
        losses.update(loss.item(), input_image.size(0))
        top1.update(acc1[0].cpu().numpy()[0], input_image.size(0))
        top5.update(acc5[0].cpu().numpy()[0], input_image.size(0))

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
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5
                )
            )
        if i * len(input_image) > args.train_samples:
            break
    return [epoch, batch_time.avg, losses.avg, top1.avg, top5.avg]


def validate_on_data(val_loader, model, criterion, args):
    batch_time = misc_utils.AverageMeter()
    losses = misc_utils.AverageMeter()
    top1 = misc_utils.AverageMeter()
    top5 = misc_utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_image, target, _) in enumerate(val_loader):
            if args.dataset == 'natural':
                input_image = input_image[0]
                target = target[0]
            if args.gpus is not None:
                input_image = input_image.cuda(args.gpus, non_blocking=True)
            target = target.cuda(args.gpus, non_blocking=True)

            # compute output
            output = model(input_image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = misc_utils.accuracy(output, target)
            acc5 = acc1.copy()
            losses.update(loss.item(), input_image.size(0))
            top1.update(acc1[0].cpu().numpy()[0], input_image.size(0))
            top5.update(acc5[0].cpu().numpy()[0], input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5
                    )
                )
            if i * len(input_image) > args.val_samples:
                break
        # printing the accuracy of the epoch
        print(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5
            )
        )

    return [batch_time.avg, losses.avg, top1.avg, top5.avg]


def extra_args_fun(parser):
    specific_group = parser.add_argument_group('Contrast specific')

    # specific_group.add_argument('-db', '--db', default=None, type=str)
    specific_group.add_argument('--train_samples', default=10000, type=int)
    specific_group.add_argument('--val_samples', default=1000, type=int)
    specific_group.add_argument('--grey_width', default=40, choices=[0, 40],
                                type=int)
    specific_group.add_argument('--mask_image', default=None, type=str)


if __name__ == '__main__':
    main(sys.argv[1:])
