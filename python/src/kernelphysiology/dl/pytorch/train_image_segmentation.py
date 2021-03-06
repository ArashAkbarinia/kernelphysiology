"""
PyTorch training script for various segmentation datasets.
"""

import datetime
import os
import sys
import time
import numpy as np
import json
from collections import OrderedDict

import torch
from torch.utils import data as torch_data
from torch.utils.data import distributed as torch_dist
from torch.nn import functional as nnf
from torchvision.models import segmentation as seg_models

from kernelphysiology.dl.pytorch.models import model_utils as model_utils
from kernelphysiology.dl.pytorch.models import segmentation as custom_models
from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler


def cross_entropy_criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nnf.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def bce_criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nnf.binary_cross_entropy_with_logits(x.squeeze(), target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def select_criterion(dataset_name):
    if 'shadow' in dataset_name:
        criterion = bce_criterion
    else:
        criterion = cross_entropy_criterion
    return criterion


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                    device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}')
    )
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq,
                                                 header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]['lr']
        )
    return metric_logger.meters


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.gpus)

    in_chns = 3
    if args.colour_space in ['lab', 'dkl']:
        if args.vision_type == 'monochromat':
            in_chns = 1
        elif 'dichromat' in args.vision_type:
            in_chns = 2
    data_reading_kwargs = {
        'target_size': args.target_size,
        'colour_vision': args.vision_type,
        'colour_space': args.colour_space
    }
    dataset, num_classes = utils.get_dataset(
        args.dataset, args.data_dir, 'train', **data_reading_kwargs
    )

    json_file_name = os.path.join(args.out_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)

    dataset_test, _ = utils.get_dataset(
        args.dataset, args.data_dir, 'val', **data_reading_kwargs
    )

    if args.distributed:
        train_sampler = torch_dist.DistributedSampler(dataset)
        test_sampler = torch_dist.DistributedSampler(dataset_test)
    else:
        train_sampler = torch_data.RandomSampler(dataset)
        test_sampler = torch_data.SequentialSampler(dataset_test)

    data_loader = torch_data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True
    )

    data_loader_test = torch_data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn
    )

    if args.custom_arch:
        print('Custom model!')
        backbone_name, customs = model_utils.create_custom_resnet(
            args.backbone, None
        )
        if customs is not None:
            args.backbone = {'arch': backbone_name, 'customs': customs}

        model = custom_models.__dict__[args.network_name](
            args.backbone, num_classes=num_classes, aux_loss=args.aux_loss
        )

        if args.pretrained:
            print('Loading %s' % args.pretrained)
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            num_all_keys = len(checkpoint['state_dict'].keys())
            remove_keys = []
            for key_ind, key in enumerate(checkpoint['state_dict'].keys()):
                if key_ind > (num_all_keys - 3):
                    remove_keys.append(key)
            for key in remove_keys:
                del checkpoint['state_dict'][key]
            pretrained_weights = OrderedDict(
                (k.replace('segmentation_model.', ''), v) for k, v in
                checkpoint['state_dict'].items()
            )
            model.load_state_dict(pretrained_weights, strict=False)
    else:
        model = seg_models.__dict__[args.network_name](
            num_classes=num_classes,
            aux_loss=args.aux_loss,
            pretrained=args.pretrained
        )
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    master_model = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpus]
        )
        master_model = model.module

    params_to_optimize = [
        {'params': [p for p in master_model.backbone.parameters() if
                    p.requires_grad]},
        {'params': [p for p in master_model.classifier.parameters() if
                    p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in master_model.aux_classifier.parameters() if
                  p.requires_grad]
        params_to_optimize.append({'params': params, 'lr': args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_lambda = lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_iou = 0
    model_progress = []
    model_progress_path = os.path.join(args.out_dir, 'model_progress.csv')
    # loading the model if to eb resumed
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_iou = checkpoint['best_iou']
        # if model progress exists, load it
        if os.path.exists(model_progress_path):
            model_progress = np.loadtxt(model_progress_path, delimiter=',')
            model_progress = model_progress.tolist()

    criterion = select_criterion(args.dataset)

    start_time = time.time()
    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_log = train_one_epoch(
            model, criterion, optimizer, data_loader, lr_scheduler,
            device, epoch, args.print_freq
        )
        val_confmat = utils.evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )
        val_log = val_confmat.get_log_dict()
        is_best = val_log['iou'] > best_iou
        best_iou = max(best_iou, val_log['iou'])
        model_data = {
            'epoch': epoch + 1,
            'arch': args.network_name,
            'customs': {
                'aux_loss': args.aux_loss,
                'pooling_type': args.pooling_type,
                'in_chns': in_chns,
                'num_classes': num_classes,
                'backbone': args.backbone
            },
            'state_dict': master_model.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'target_size': args.target_size,
            'args': args,
            'best_iou': best_iou,
        }
        if args.save_all:
            checkpoint_path = os.path.join(args.out_dir, 'epoch%3d.pth' % epoch)
        else:
            checkpoint_path = os.path.join(args.out_dir, 'checkpoint.pth')
        utils.save_on_master(model_data, checkpoint_path)
        if is_best:
            utils.save_on_master(
                model_data, os.path.join(args.out_dir, 'model_best.pth')
            )

        epoch_prog, header = add_to_progress(train_log, [], '')
        epoch_prog, header = add_to_progress(
            val_log, epoch_prog, header, prefix='v_'
        )
        model_progress.append(epoch_prog)
        np.savetxt(
            model_progress_path, np.array(model_progress),
            delimiter=';', header=header, fmt='%s'
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def add_to_progress(log, progress, header, prefix=None):
    for key, val in log.items():
        progress.append(str(val))
        if header != '':
            header += ','
        if prefix is not None:
            key = prefix + key
        header = header + key
    return progress, header


if __name__ == '__main__':
    parsed_args = argument_handler.parse_train_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)
