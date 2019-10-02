"""
The main script for GEETUP.
"""

import time
import sys
import os
import logging
import numpy as np

import torch
import torch.nn as nn

from kernelphysiology.dl.pytorch.geetup import geetup_net, geetup_db
from kernelphysiology.dl.geetup import geetup_opts
from kernelphysiology.dl.geetup import geetup_visualise
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.transformations import NormalizeInverse
from kernelphysiology.utils.path_utils import create_dir


def euclidean_error(x, y):
    cumulative_error = 0
    for i in range(x.shape[0]):
        max_x = torch.argmax(x[i].squeeze())
        max_y = torch.argmax(y[i].squeeze())
        max_x = [max_x / x.shape[2], max_x % x.shape[2]]
        max_y = [max_y / x.shape[2], max_y % x.shape[2]]
        sum_error = (max_x[0] - max_y[0]) ** 2 + (max_x[1] - max_y[1]) ** 2
        cumulative_error += torch.sqrt(sum_error.float())
    return cumulative_error / x.shape[0]


def epochs(model, train_loader, validation_loader, optimizer, args):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.33), int(args.epochs * 0.66)],
        last_epoch=args.initial_epoch - 1
    )

    criterion = args.criterion
    for epoch in range(args.initial_epoch, args.epochs):
        scheduler.step(epoch=epoch)
        train(model, train_loader, optimizer, criterion, epoch, args)


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    eucs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # TODO: it's too much to do for all
    for step, (x_input, y_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_input = x_input.to(args.gpus)
        y_target = y_target.to(args.gpus)

        output = model(x_input)
        loss = criterion(output, y_target)

        # measure accuracy and record loss
        losses.update(loss.item(), x_input.size(0))
        eucs.update(euclidean_error(y_target, output), x_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # printing the accuracy at certain intervals
        if step % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Euc {euc.val:.4f} ({euc.avg:.4f})'.format(
                    epoch, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, euc=eucs
                )
            )


def process_random_image(model, validation_loader, normalize_inverse, args):
    for step, (x_input, y_target) in enumerate(validation_loader):
        x_input = x_input.to(args.gpus)

        # inversing the normalisation done before calling the network
        x_input = x_input.clone().detach().cpu()

        y_target = y_target.numpy()

        for b in range(y_target.shape[0]):
            file_name = '%s/image_%d_%d.jpg' % (args.log_dir, step, b)
            # PyTorch has this order: batch, frame, channel, width, height
            current_image = normalize_inverse(x_input[b, -1].squeeze()).numpy()
            current_image = np.transpose(current_image, (1, 2, 0))
            current_image = (current_image * 255).astype('uint8')
            gt = y_target[b].squeeze()
            pred = gt
            _ = geetup_visualise.draw_circle_results(
                current_image, gt, pred, file_name
            )

        # TODO: make it nicer
        if step == 0:
            break


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(e) for e in args.gpus)
    gpus = [*range(len(args.gpus))]
    # FIXME: cant take more than one GPU
    args.gpus = gpus[0]

    create_dir(args.log_dir)
    # for training organise the output file
    if args.evaluate is False:
        # add architecture to directory
        args.log_dir = os.path.join(args.log_dir, args.architecture)
        create_dir(args.log_dir)
        # add frame based or time integration to directory
        if args.frame_based:
            time_or_frame = 'frame_based'
        else:
            time_or_frame = 'time_integration'
        args.log_dir = os.path.join(args.log_dir, time_or_frame)
        create_dir(args.log_dir)
        # add scratch or fine tune to directory
        if args.weights is None:
            new_or_tune = 'scratch'
        else:
            new_or_tune = 'fine_tune'
        args.log_dir = os.path.join(args.log_dir, new_or_tune)
        create_dir(args.log_dir)
    # add experiment name to directory
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    create_dir(args.log_dir)

    logging.basicConfig(
        filename=args.log_dir + '/experiment_info.log', filemode='w',
        format='%(levelname)s: %(message)s', level=logging.INFO
    )

    # creating the model
    model = geetup_net.which_architecture()
    torch.cuda.set_device(args.gpus)
    model = model.cuda(args.gpus)

    args.target_size = [360, 640]

    validation_pickle = os.path.join(args.data_dir, args.validation_file)
    validation_dataset = geetup_db.get_validation_dataset(
        validation_pickle, target_size=args.target_size
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.random is not None:
        mean, std = get_preprocessing_function('rgb', 'trichromat')
        normalize_inverse = NormalizeInverse(mean, std)
        process_random_image(model, validation_loader, normalize_inverse, args)
        return

    # FIXME: the evaluation
    if args.evaluate:
        return

    training_pickle = os.path.join(args.data_dir, args.train_file)
    train_dataset = geetup_db.get_train_dataset(
        training_pickle, target_size=args.target_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    # optimiser
    args.lr = 0.1
    args.momentum = 0.9
    args.weight_decay = 0
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    args.initial_epoch = 0
    args.print_freq = 10
    args.criterion = nn.KLDivLoss().cuda(args.gpus)
    epochs(model, train_loader, validation_loader, optimizer, args)


if __name__ == "__main__":
    parser = geetup_opts.argument_parser()
    args = geetup_opts.check_args(parser, sys.argv[1:])
    main(args)
