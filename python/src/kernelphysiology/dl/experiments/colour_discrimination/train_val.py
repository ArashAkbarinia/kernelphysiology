"""

"""

import os
import numpy as np
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from .datasets import dataloader, colour_spaces
from .models import model as networks, model_utils
from .utils import report_utils, system_utils, argument_handler


def main(argv):
    args = argument_handler.train_arg_parser(argv)
    system_utils.set_random_environment(args.random_seed)

    # NOTE: a hack to handle taskonomy preprocessing
    if 'taskonomy' in args.architecture:
        args.colour_space = 'taskonomy_rgb'

    # it's a 4AFC
    args.num_classes = 4

    # preparing the output folder
    args.output_dir = '%s/networks/t%.3d/%s%s/%s/' % (
        args.output_dir, args.target_size, args.architecture, args.arch_suf, args.experiment_name
    )
    system_utils.create_dir(args.output_dir)

    # this is just a hack for when the training script has crashed
    filename = 'e%.3d_%s' % (8, 'checkpoint.pth.tar')
    file_path = os.path.join(args.output_dir, filename)
    if os.path.exists(file_path):
        return

    # dumping all passed arguments to a json file
    if not args.test_net:
        system_utils.save_arguments(args)

    _main_worker(args)


def _main_worker(args):
    mean, std = model_utils.get_mean_std(args.colour_space, args.vision_type)
    args.preprocess = (mean, std)

    if args.mac_adam:
        net_class = networks.ColourDiscrimination2AFC
        task = '2afc'
    else:
        net_class = networks.ColourDiscriminationOddOneOut
        task = 'odd4'

    if args.test_net:
        model = net_class(args.test_net, args.target_size)
    else:
        model = net_class(args.architecture, args.target_size, args.transfer_weights)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # setting the quadrant points
    if args.pts_path is None:
        args.pts_path = args.val_dir + '/rgb_points.csv'
    test_pts = np.loadtxt(args.pts_path, delimiter=',', dtype=str)

    args.test_pts = _organise_test_points(test_pts)

    # defining validation set here so if only test don't do the rest
    if args.val_dir is None:
        args.val_dir = args.data_dir + '/validation_set/'

    if args.test_net:
        if args.test_attempts > 0:
            _sensitivity_test_points(args, model)
        else:
            _accuracy_test_points(args, model)
        return

    # if transfer_weights, only train the fc layer, otherwise all parameters
    if args.transfer_weights is None or '_scratch' in args.architecture:
        params_to_optimize = [{'params': [p for p in model.parameters()]}]
    else:
        for p in model.features.parameters():
            p.requires_grad = False
        params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
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
                "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])
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

    # loading the validation set
    val_dataset = []
    for ref_pts in args.test_pts.values():
        others_colour = ref_pts['ffun'](np.expand_dims(ref_pts['ref'][:3], axis=(0, 1)))
        for ext_pts in ref_pts['ext']:
            target_colour = ref_pts['bfun'](np.expand_dims(ext_pts[:3], axis=(0, 1)))
            val_colours = {'target_colour': target_colour, 'others_colour': others_colour}
            val_dataset.append(dataloader.val_set(
                args.val_dir, args.target_size, args.preprocess, task=task, **val_colours
            ))
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.train_dir is None:
        args.train_dir = args.data_dir + '/training_set/'

    # loading the training set
    train_kwargs = {'colour_dist': args.train_colours, **_common_db_params(args)}
    train_dataset = dataloader.train_set(
        args.train_dir, args.target_size, args.preprocess, task=task, **train_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        _adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = _train_val(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        validation_log = _train_val(val_loader, model, None, epoch, args)

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
            model_progress_path, np.array(model_progress), delimiter=',', header=header
        )


def _organise_test_points(test_pts):
    out_test_pts = dict()
    for test_pt in test_pts:
        pt_val = test_pt[:3].astype('float')
        test_pt_name = test_pt[-2]
        if 'ref_' == test_pt_name[:4]:
            test_pt_name = test_pt_name[4:]
            if test_pt[-1] == 'dkl':
                ffun = colour_spaces.dkl2rgb01
                bfun = colour_spaces.rgb012dkl
                chns_name = ['D', 'K', 'L']
            elif test_pt[-1] == 'hsv':
                ffun = colour_spaces.hsv012rgb01
                bfun = colour_spaces.rgb2hsv01
                chns_name = ['H', 'S', 'V']
            elif test_pt[-1] == 'rgb':
                ffun = lambda x: x
                bfun = lambda x: x
                chns_name = ['R', 'G', 'B']
            out_test_pts[test_pt_name] = {
                'ref': pt_val, 'ffun': ffun, 'bfun': bfun, 'space': chns_name, 'ext': [], 'chns': []
            }
        else:
            out_test_pts[test_pt_name]['ext'].append(pt_val)
            out_test_pts[test_pt_name]['chns'].append(test_pt[-1])
    return out_test_pts


def _adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // (args.epochs / 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _train_val(db_loader, model, optimizer, epoch, args, print_test=True):
    batch_time = report_utils.AverageMeter()
    data_time = report_utils.AverageMeter()
    losses = report_utils.AverageMeter()
    top1 = report_utils.AverageMeter()

    is_train = optimizer is not None
    is_test = epoch == -1

    if is_train:
        model.train()
        num_samples = args.train_samples
    else:
        model.eval()
        num_samples = args.val_samples

    all_predictions = []
    epoch_str = 'train' if is_train else 'val'
    end = time.time()
    with torch.set_grad_enabled(is_train):
        for i, cu_batch in enumerate(db_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.mac_adam:
                (img0, img1, target) = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                target = target.unsqueeze(dim=1).float()
            else:
                (img0, img1, img2, img3, odd_ind) = cu_batch
                img0 = img0.cuda(args.gpu, non_blocking=True)
                img1 = img1.cuda(args.gpu, non_blocking=True)
                img2 = img2.cuda(args.gpu, non_blocking=True)
                img3 = img3.cuda(args.gpu, non_blocking=True)

                # preparing the target
                target = torch.zeros(odd_ind.shape[0], 4)
                target[torch.arange(odd_ind.shape[0]), odd_ind] = 1
                odd_ind = odd_ind.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.mac_adam:
                output = model(img0, img1)
                odd_ind = target
            else:
                output = model(img0, img1, img2, img3)
            loss = model.loss_function(output, target)

            # measure accuracy and record loss
            # FIXME
            acc1 = report_utils.accuracy(output, odd_ind)
            losses.update(loss.item(), img0.size(0))
            top1.update(acc1[0].cpu().numpy()[0], img0.size(0))

            if is_train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # to use for correlations
            if args.mac_adam:
                pred_outs = np.concatenate(
                    [output.detach().cpu().numpy(), odd_ind.cpu().numpy()],
                    axis=1
                )
            else:
                pred_outs = np.concatenate(
                    [output.detach().cpu().numpy(), odd_ind.unsqueeze(dim=1).cpu().numpy()],
                    axis=1
                )
            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(pred_outs, list):
                pred_outs = [pred_outs]

            if not all_predictions:
                for _ in pred_outs:
                    all_predictions.append([])

            for j, out in enumerate(pred_outs):
                all_predictions[j].append(out)

            # printing the accuracy at certain intervals
            if is_test and print_test:
                print('Testing: [{0}/{1}]'.format(i, len(db_loader)))
            elif i % args.print_freq == 0:
                print(
                    '{0}: [{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch_str, epoch, i, len(db_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1
                    )
                )
            if num_samples is not None and i * len(img0) > num_samples:
                break
        if not is_train:
            # printing the accuracy of the epoch
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print()

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    if is_test:
        accuracy = top1.avg if top1.avg <= 1.0 else top1.avg / 100
        return prediction_output, accuracy
    return [epoch, batch_time.avg, losses.avg, top1.avg]


def _common_db_params(args):
    return {'background': args.background, 'same_rotation': args.same_rotation}


def _sensitivity_test_points(args, model):
    for qname, qval in args.test_pts.items():
        for pt_ind in range(0, len(qval['ext'])):
            _sensitivity_test_point(args, model, qname, pt_ind)


def _accuracy_test_points(args, model):
    for qname, qval in args.test_pts.items():
        tosave = []
        for pt_ind in range(0, len(qval['ext'])):
            acc = _accuracy_test_point(args, model, qname, pt_ind)
            tosave.append([acc, *qval['ext'][pt_ind], qval['chns'][pt_ind]])
        output_file = os.path.join(args.output_dir, 'accuracy_%s.csv' % (qname))
        chns_name = qval['space']
        header = 'acc,%s,%s,%s,chn' % (chns_name[0], chns_name[1], chns_name[2])
        np.savetxt(output_file, np.array(tosave), delimiter=',', fmt='%s', header=header)


def _make_test_loader(args, target_colour, others_colour):
    task = '2afc' if args.mac_adam else 'odd4'
    kwargs = {'target_colour': target_colour, 'others_colour': others_colour,
              **_common_db_params(args)}
    db = dataloader.val_set(args.val_dir, args.target_size, args.preprocess, task=task, **kwargs)

    return torch.utils.data.DataLoader(
        db, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def _accuracy_test_point(args, model, qname, pt_ind):
    qval = args.test_pts[qname]

    low = np.expand_dims(qval['ref'][:3], axis=(0, 1))
    high = np.expand_dims(qval['ext'][pt_ind][:3], axis=(0, 1))

    others_colour = qval['ffun'](low)
    target_colour = qval['ffun'](high)
    db_loader = _make_test_loader(args, target_colour, others_colour)

    _, accuracy = _train_val(db_loader, model, None, -1, args, print_test=False)
    print(qname, pt_ind, accuracy, low.squeeze(), high.squeeze())
    return accuracy


def _sensitivity_test_point(args, model, qname, pt_ind):
    qval = args.test_pts[qname]
    chns_name = qval['space']
    circ_chns = [0] if chns_name[0] == 'H' else []
    output_file = os.path.join(args.output_dir, 'evolutoin_%s_%d.csv' % (qname, pt_ind))
    if os.path.exists((output_file)):
        return

    low = np.expand_dims(qval['ref'][:3], axis=(0, 1))
    high = np.expand_dims(qval['ext'][pt_ind][:3], axis=(0, 1))
    mid = _compute_mean(low, high, circ_chns)

    others_colour = qval['ffun'](low)

    all_results = []
    j = 0
    header = 'acc,%s,%s,%s,R,G,B' % (chns_name[0], chns_name[1], chns_name[2])

    th = 0.75 if args.mac_adam else 0.625
    while True:
        target_colour = qval['ffun'](mid)
        db_loader = _make_test_loader(args, target_colour, others_colour)

        _, accuracy = _train_val(db_loader, model, None, -1, args, print_test=False)
        print(qname, pt_ind, accuracy, j, low.squeeze(), mid.squeeze(), high.squeeze())

        all_results.append(np.array([accuracy, *mid.squeeze(), *target_colour.squeeze()]))
        np.savetxt(output_file, np.array(all_results), delimiter=',', fmt='%f', header=header)

        new_low, new_mid, new_high = _midpoint_colour(accuracy, low, mid, high, th, circ_chns)

        if new_low is None or j == args.test_attempts:
            print('had to skip')
            break
        else:
            low, mid, high = new_low, new_mid, new_high
        j += 1


def _midpoint_colour(accuracy, low, mid, high, th, circ_chns=None):
    diff_acc = accuracy - th
    if abs(diff_acc) < 0.005:
        return None, None, None
    elif diff_acc > 0:
        new_mid = _compute_mean(low, mid, circ_chns)
        return low, new_mid, mid
    else:
        new_mid = _compute_mean(high, mid, circ_chns)
        return mid, new_mid, high


def _compute_mean(a, b, circ_chns):
    c = (a + b) / 2
    for i in circ_chns:
        c[0, 0, i] = _circular_mean(a[0, 0, i], b[0, 0, i])
    return c


def _circular_mean(a, b):
    if abs(a - b) > 0.5:
        mu = (a + b + 1) / 2
    else:
        mu = (a + b) / 2
    if mu >= 1:
        mu = mu - 1
    return mu
