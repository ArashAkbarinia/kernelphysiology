"""
The train and test script.
"""

import os
import sys

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
    if args.out_type == 'intensity':
        args.num_classes = 2
    elif args.out_type == 'mass':
        args.num_classes = 3
    else:
        sys.exit('Unsupported out_type %s' % args.out_type)

    if args.only_test:
        if args.val_group is None:
            sys.exit('With testing the val_group must be passed.')

        # preparing the output folder
        args.output_dir = '%s/tests/%s/' % (args.output_dir, args.condition)
        system_utils.create_dir(args.output_dir)
        _test_network(args)
        return
    elif args.train_group is None:
        args.train_group, args.val_group = dataloader._random_train_val_sets(0.8)

    # preparing the output folder
    args.output_dir = '%s/networks/%s/%s/' % (
        args.output_dir, args.architecture, args.experiment_name
    )
    system_utils.create_dir(args.output_dir)

    # dumping all passed arguments to a json file
    system_utils.save_arguments(args)

    _main_worker(args)


def _test_network(args):
    torch.cuda.set_device(args.gpu)

    model = grasp_model.load_pretrained(args.architecture)
    model = model.cuda(args.gpu)

    # loading the training set
    db_kwargs = {
        'time_interval': args.time_interval,
        'which_xyz': args.which_xyz
    }

    val_db = dataloader.get_val_set(
        args.data_dir, args.condition, args.target_size,
        args.val_group, **db_kwargs
    )

    # loading validation set
    val_loader = torch.utils.data.DataLoader(
        val_db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    with torch.set_grad_enabled(False):
        outputs = []
        for i, (kinematic, inten_torch, _, _, trial_num) in enumerate(val_loader):
            kinematic = kinematic.cuda(args.gpu, non_blocking=True)
            trial_num = trial_num.numpy()

            for intensity in args.test_intensity:
                inten_torch = inten_torch.fill_(intensity)
                inten_torch = inten_torch.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(kinematic, inten_torch)
                output = output.detach().cpu().numpy()
                output = np.argmax(output, axis=1)
                for j in range(output.shape[0]):
                    outputs.append([trial_num[j], intensity, output[j]])
    out_path = os.path.join(args.output_dir, '%s.csv' % args.experiment_name)
    header = 'trial,intensity,output'
    np.savetxt(out_path, np.array(outputs), delimiter=',', header=header)


def _main_worker(args):
    mean, std = (0.5, 0.5)

    # create model
    net_kwargs = {
        'num_classes': args.num_classes,
        'in_chns': len(args.which_xyz),
        'inplanes': args.num_kernels,
        'intensity_length': 1 if args.out_type == 'intensity' else 0
    }
    model = grasp_model.__dict__[args.architecture](args.blocks, **net_kwargs)

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
        num_workers=args.workers, pin_memory=False, sampler=None
    )

    # loading validation set
    val_loader = torch.utils.data.DataLoader(
        val_db, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
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
                'net_info': {
                    'blocks': args.blocks,
                    'kwargs': net_kwargs
                },
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
    lr_epoch = (epoch // (args.epochs / args.lr_step_size))
    lr = args.learning_rate * (0.1 ** lr_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _train_val(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = report_utils.AverageMeter()
    data_time = report_utils.AverageMeter()
    losses = report_utils.AverageMeter()
    top1 = report_utils.AverageMeter()

    is_train = optimizer is not None

    if is_train:
        train_test_str = 'Train'
        model.train()
        num_samples = args.train_samples
    else:
        train_test_str = 'Test'
        model.eval()
        num_samples = args.val_samples

    end = time.time()
    with torch.set_grad_enabled(is_train):
        for i, (
                kinematic, intensity, mass_dist, response, trial_name
        ) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            kinematic = kinematic.cuda(args.gpu, non_blocking=True)
            if args.out_type == 'intensity':
                intensity = intensity.cuda(args.gpu, non_blocking=True)
                response = response.cuda(args.gpu, non_blocking=True)
            else:
                intensity = None
                response = mass_dist.cuda(args.gpu, non_blocking=True)

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
                    '%s: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1
                    ) % train_test_str
                )
            if num_samples is not None and i * len(kinematic) > num_samples:
                break
        if not is_train:
            # printing the accuracy of the epoch
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return [epoch, batch_time.avg, losses.avg, top1.avg]
