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

from kernelphysiology.dl.geetup import geetup_opts
from kernelphysiology.dl.geetup import geetup_visualise
from kernelphysiology.dl.pytorch.geetup import geetup_net, geetup_db
from kernelphysiology.dl.pytorch.models.model_utils import get_preprocessing_function
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import save_checkpoint
from kernelphysiology.dl.pytorch.utils.misc import prepare_device
from kernelphysiology.dl.pytorch.utils.cv2_transforms import NormalizeInverse
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.utils.system_utils import set_visible_gpus
from kernelphysiology.utils.path_utils import write_pickle


def euclidean_error_with_point(x, y):
    max_x = torch.argmax(x)
    max_y = torch.argmax(y)
    max_x = [max_x / x.shape[1], max_x % x.shape[1]]
    max_y = [max_y / x.shape[1], max_y % x.shape[1]]
    sum_error = (max_x[0] - max_y[0]) ** 2 + (max_x[1] - max_y[1]) ** 2
    return torch.sqrt(sum_error.float()), max_x, max_y


def euclidean_error(x, y):
    euc_distance, _, _ = euclidean_error_with_point(x, y)
    return euc_distance


def euclidean_error_batch(x, y):
    cumulative_error = 0
    for i in range(x.shape[0]):
        cumulative_error += euclidean_error(x[i].squeeze(), y[i].squeeze())
    return cumulative_error / x.shape[0]


def epochs(model, train_loader, validation_loader, optimizer, args):
    model_progress = []
    best_euc = np.inf
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.initial_epoch = checkpoint['epoch']
            best_euc = checkpoint['best_euc']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(args.device)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']
                )
            )
            model_progress_path = args.resume.replace(
                'checkpoint.pth.tar', 'model_progress.csv'
            )
            if os.path.exists(model_progress_path):
                model_progress = np.loadtxt(model_progress_path, delimiter=',')
                model_progress = model_progress.tolist()

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.33), int(args.epochs * 0.66)],
        last_epoch=args.initial_epoch - 1
    )

    file_path = os.path.join(args.out_dir, 'model_progress.csv')
    criterion = args.criterion
    for epoch in range(args.initial_epoch, args.epochs):
        scheduler.step(epoch=epoch)
        train_log = train(
            train_loader, model, optimizer, criterion, epoch, args
        )

        validation_log = validate(
            validation_loader, model, criterion, args
        )

        model_progress.append([*train_log, *validation_log])

        # remember best euclidean distance and save checkpoint
        euc_distance = validation_log[2]
        is_best = euc_distance < best_euc
        best_euc = max(euc_distance, best_euc)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.architecture,
                'kwargs': args.architecture_kwargs,
                'state_dict': model.state_dict(),
                'best_euc': best_euc,
                'optimizer': optimizer.state_dict(),
                'target_size': args.target_size,
            },
            is_best, out_folder=args.out_dir
        )
        # TODO: get this header directly as a dictionary keys
        header = 'epoch,t_time,t_loss,t_euc,v_time,v_loss,v_euc'
        np.savetxt(
            file_path, np.array(model_progress), delimiter=',', header=header
        )


def validate(validation_loader, model, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    eucs = AverageMeter()

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (x_input, y_target) in enumerate(validation_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            x_input = x_input.to(args.device)
            y_target = y_target.to(args.device)

            output = model(x_input)
            loss = criterion(output, y_target)

            # measure accuracy and record loss
            losses.update(loss.item(), x_input.size(0))
            eucs.update(euclidean_error_batch(y_target, output),
                        x_input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if step % args.print_freq == 0:
                print(
                    'Step: {0}/{1}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Euc {euc.val:.4f} ({euc.avg:.4f})'.format(
                        step, len(validation_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, euc=eucs
                    )
                )
            if (args.validation_samples is not None and
                    (step * args.batch_size) >= args.validation_samples):
                break
        # printing the accuracy of the epoch
        print(
            ' * Loss {loss.avg:.3f} Euc {euc.avg:.3f}'.format(
                loss=losses, euc=eucs
            )
        )
    return [batch_time.avg, losses.avg, eucs.avg]


def predict(validation_loader, model, criterion, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    eucs = AverageMeter()

    # switch to evaluation mode
    model.eval()

    all_eucs = np.zeros(validation_loader.dataset.num_sequences)
    all_preds = np.zeros((validation_loader.dataset.num_sequences, 2))

    with torch.no_grad():
        end = time.time()
        out_ind = 0
        for step, (x_input, y_target) in enumerate(validation_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            x_input = x_input.to(args.device)
            y_target = y_target.to(args.device)

            output = model(x_input)
            loss = criterion(output, y_target)

            # measure accuracy and record loss
            losses.update(loss.item(), x_input.size(0))
            eucs.update(euclidean_error_batch(y_target, output),
                        x_input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            for b in range(y_target.shape[0]):
                gt = y_target[b].squeeze()
                pred = output[b].squeeze()
                euc_dis, _, pred_p = euclidean_error_with_point(gt, pred)
                all_eucs[out_ind] = euc_dis
                all_preds[out_ind] = np.array(pred_p)
                out_ind += 1

            # printing the accuracy at certain intervals
            if step % args.print_freq == 0:
                print(
                    'Step: {0}/{1}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Euc {euc.val:.4f} ({euc.avg:.4f})'.format(
                        step, len(validation_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, euc=eucs
                    )
                )
            if (args.validation_samples is not None and
                    (step * args.batch_size) >= args.validation_samples):
                break
        # printing the accuracy of the epoch
        print(
            ' * Loss {loss.avg:.3f} Euc {euc.avg:.3f}'.format(
                loss=losses, euc=eucs
            )
        )
    preds_out = {'eucs': all_eucs, 'preds': all_preds}
    return preds_out


def train(train_loader, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    eucs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for step, (x_input, y_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_input = x_input.to(args.device)
        y_target = y_target.to(args.device)

        output = model(x_input)
        loss = criterion(output, y_target)

        # measure accuracy and record loss
        losses.update(loss.item(), x_input.size(0))
        eucs.update(euclidean_error_batch(y_target, output), x_input.size(0))

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
        if (args.train_samples is not None and
                (step * args.batch_size) >= args.train_samples):
            break
    return [epoch, batch_time.avg, losses.avg, eucs.avg]


def process_random_image(model, validation_loader, normalize_inverse, args):
    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for step, (x_input, y_target) in enumerate(validation_loader):
            x_input = x_input.to(args.device)
            output = model(x_input)
            output = output.clone().detach().cpu().numpy()

            # inversing the normalisation done before calling the network
            x_input = x_input.clone().detach().cpu()

            y_target = y_target.numpy()

            for b in range(y_target.shape[0]):
                gt = y_target[b].squeeze()
                pred = output[b].squeeze()
                if args.draw_results:
                    # PyTorch has this order: batch, frame, channel, width, height
                    current_image = normalize_inverse(
                        x_input[b, -1].squeeze()
                    ).numpy()
                    current_image = np.transpose(current_image, (1, 2, 0))
                    current_image = np.uint8(current_image * 255)
                    file_name = '%s/image_%d_%d.jpg' % (args.out_dir, step, b)
                    _ = geetup_visualise.draw_circle_results(
                        current_image, gt, pred, file_name
                    )
                if args.save_gt:
                    file_name = '%s/image_%d_%d_g.jpg' % (args.out_dir, step, b)
                    geetup_visualise.save_heatmap(gt, file_name)
                if args.save_pred:
                    file_name = '%s/image_%d_%d_p.jpg' % (args.out_dir, step, b)
                    geetup_visualise.save_heatmap(pred, file_name)

            # TODO: make it nicer
            if step == 0:
                break


def _get_multi_sensory_paths(data_dirs, data_file):
    # TODO: think of a more robust solution
    all_paths = []
    for current_dir in data_dirs:
        all_paths.append(os.path.join(current_dir, data_file))
    return all_paths


def main(args):
    args.gpus = set_visible_gpus(args.gpus)
    args.device = prepare_device(args.gpus)

    args.architecture_kwargs = {
        'in_chns': args.in_chns
    }
    # TODO: this is not a nice solution here
    if 'wavenet' in args.architecture:
        args.architecture_kwargs['inplanes'] = args.num_kernels
        args.architecture_kwargs['planes'] = args.blocks

    # creating the model
    model, architecture, mean_std = geetup_net.which_network(
        args.architecture, **args.architecture_kwargs
    )
    model = model.to(args.device)

    args.out_dir = prepare_training.prepare_output_directories(
        dataset_name='geetup_' + args.dataset, network_name=architecture,
        optimiser='sgd', load_weights=False,
        experiment_name=args.experiment_name, framework='pytorch'
    )

    logging.basicConfig(
        filename=args.out_dir + '/experiment_info.log', filemode='w',
        format='%(levelname)s: %(message)s', level=logging.INFO
    )

    validation_pickle = _get_multi_sensory_paths(
        args.data_dir, args.validation_file
    )
    validation_dataset = geetup_db.get_validation_dataset(
        validation_pickle, args.target_size, mean_std
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

    args.criterion = nn.BCELoss().to(args.device)
    if args.evaluate:
        predict_outs = predict(
            validation_loader, model, args.criterion, args
        )
        for key, item in predict_outs.items():
            file_name = '_'.join(e for e in [key, *args.out_prefix])
            result_file = '%s/%s_%s' % (
                args.out_dir, file_name, args.validation_file
            )
            write_pickle(result_file, item)
        return

    training_pickle = _get_multi_sensory_paths(
        args.data_dir, args.train_file
    )
    train_dataset = geetup_db.get_train_dataset(
        training_pickle, args.target_size, mean_std, args.crop_scale,
        args.gaussian_sigma
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    # optimiser
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    epochs(model, train_loader, validation_loader, optimizer, args)


if __name__ == "__main__":
    parser = geetup_opts.argument_parser()
    args = geetup_opts.check_args(parser, sys.argv[1:])
    main(args)
