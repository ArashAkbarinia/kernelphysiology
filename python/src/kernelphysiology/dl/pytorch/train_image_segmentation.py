"""
PyTorch training script for various segmentation datasets.
"""

import datetime
import os
import sys
import time
import numpy as np

import torch
import torch.utils.data
from torch import nn
import torchvision

from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


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
    print(args)

    device = torch.device(args.gpus)

    dataset, num_classes = utils.get_dataset(
        args.dataset, args.data_dir, 'train', args.target_size
    )
    dataset_test, _ = utils.get_dataset(
        args.dataset, args.data_dir, 'val', args.target_size
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn
    )

    model = torchvision.models.segmentation.__dict__[args.network_name](
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        pretrained=args.pretrained
    )
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_progress = []
    model_progress_path = os.path.join(args.out_dir, 'model_progress.csv')
    # loading the model if to eb resumed
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # if model progress exists, load it
        if os.path.exists(model_progress_path):
            model_progress = np.loadtxt(model_progress_path, delimiter=',')
            model_progress = model_progress.tolist()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpus]
        )
        model_without_ddp = model.module

    params_to_optimize = [
        {'params': [p for p in model_without_ddp.backbone.parameters() if
                    p.requires_grad]},
        {'params': [p for p in model_without_ddp.classifier.parameters() if
                    p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if
                  p.requires_grad]
        params_to_optimize.append({'params': params, 'lr': args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_lambda = lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_time = time.time()
    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_log = train_one_epoch(
            model, criterion, optimizer, data_loader, lr_scheduler,
            device, epoch, args.print_freq
        )
        validation_log = utils.evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )
        utils.save_on_master(
            {
                'epoch': epoch + 1,
                'arch': args.network_name,
                'customs': {
                    'pooling_type': args.pooling_type,
                    'in_chns': 3,  # len(mean), #TODO
                    'num_classes': 21,  # args.num_classes,
                    # 'blocks': args.blocks,
                    # 'num_kernels': args.num_kernels
                },
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'target_size': args.target_size,
                'args': args
            },
            os.path.join(args.out_dir, 'model_{}.pth'.format(epoch))
        )

        epoch_prog, header = add_to_progress(train_log, [], '')
        epoch_prog, header = add_to_progress(
            validation_log.get_log_dict(), epoch_prog, header, prefix='v_'
        )
        model_progress.append(epoch_prog)
        np.savetxt(
            model_progress_path, np.array(model_progress),
            delimiter=',', header=header
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def add_to_progress(log, progress, header, prefix=None):
    for key, val in log.items():
        progress.append(val)
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
