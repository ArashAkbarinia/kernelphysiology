"""
The train and test script.
"""

import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from . import dataloader, system_utils, model as grasp_model, grasp_utils

from kernelphysiology.dl.pytorch.utils import misc as report_utils


def main(argv):
    args = grasp_utils.main_arg_parser(argv)
    system_utils.set_random_environment(args.random_seed)

    # it's a binary classification
    args.num_classes = 2

    # preparing the output folder
    args.output_dir = '%s/networks/%s/%s/' % (
        args.output_dir, args.architecture, args.experiment_name
    )
    system_utils.create_dir(args.output_dir)

    if args.train_group is None:
        args.train_group, args.val_group = dataloader._random_train_val_sets(0.8)

    # dumping all passed arguments to a json file
    system_utils.save_arguments(args)

    _main_worker(args)


def _main_worker(args):
    mean, std = (0.5, 0.5)

    # create model
    kwargs = {'in_chns': len(args.which_xyz), 'inplanes': args.num_kernels}
    model = grasp_model.__dict__[args.architecture](args.blocks, **kwargs)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    params_to_optimize = [{'params': [p for p in model.parameters()]}]
    # optimiser
    optimizer = torch.optim.SGD(
        params_to_optimize, lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    model_progress = []
    model_progress_path = os.path.join(args.output_dir, 'model_progress.csv')

    # optionally resume from a checkpoint
    best_acc1 = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']
                )
            )

            args.initial_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            best_acc1 = best_acc1.to(args.gpu)
            model = model.cuda(args.gpu)

            optimizer.load_state_dict(checkpoint['optimizer'])

            if os.path.exists(model_progress_path):
                model_progress = np.loadtxt(model_progress_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    shuffle = True

    # loading the training set
    db_kwargs = {
        'time_interval': args.time_interval,
        'which_xyz': args.which_xyz
    }

    train_db, val_db = dataloader.train_val_sets(
        args.data_dir, args.condition, args.target_size,
        args.train_group, args.val_group, **db_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_db, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    # loading validation set
    val_loader = torch.utils.data.DataLoader(
        val_db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        _adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = _train_val(
            train_loader, model, criterion, optimizer, epoch, args
        )

        # evaluate on validation set
        validation_log = _train_val(
            val_loader, model, criterion, None, epoch, args
        )

        model_progress.append([*train_log, *validation_log[1:]])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save the checkpoints
        system_utils.save_checkpoint(
            {
                'epoch': epoch,
                'arch': args.architecture,
                'transfer_weights': args.transfer_weights,
                'preprocessing': {'mean': mean, 'std': std},
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'target_size': args.target_size,
            },
            is_best, args
        )
        header = 'epoch,t_time,t_loss,t_top1,v_time,v_loss,v_top1'
        np.savetxt(
            model_progress_path, np.array(model_progress),
            delimiter=',', header=header
        )


def _adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // (args.epochs / 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _train_val(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = report_utils.AverageMeter()
    data_time = report_utils.AverageMeter()
    losses = report_utils.AverageMeter()
    top1 = report_utils.AverageMeter()

    is_train = optimizer is not None

    if is_train:
        model.train()
        num_samples = args.train_samples
    else:
        model.eval()
        num_samples = args.val_samples

    end = time.time()
    with torch.set_grad_enabled(is_train):
        for i, (kinematic, intensity, mass_dist, response) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            kinematic = kinematic.cuda(args.gpu, non_blocking=True)
            intensity = intensity.cuda(args.gpu, non_blocking=True)
            response = response.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(kinematic, intensity)
            loss = criterion(output, response)

            # measure accuracy and record loss
            acc1 = report_utils.accuracy(output, response)
            losses.update(loss.item(), kinematic.size(0))
            top1.update(acc1[0].cpu().numpy()[0], kinematic.size(0))

            if is_train:
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
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1
                    )
                )
            if num_samples is not None and i * len(kinematic) > num_samples:
                break
        if not is_train:
            # printing the accuracy of the epoch
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return [epoch, batch_time.avg, losses.avg, top1.avg]