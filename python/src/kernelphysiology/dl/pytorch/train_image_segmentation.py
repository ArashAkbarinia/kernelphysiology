"""
PyTorch training script for various segmentation datasets.
"""

import datetime
import os
import sys
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

from kernelphysiology.dl.pytorch.datasets.segmentations_db import get_voc_coco
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.dl.pytorch.utils import transforms as T
from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler


def get_dataset(name, data_dir, image_set, target_size):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(
            *args, mode='segmentation', **kwargs
        )

    paths = {
        'voc_org': (torchvision.datasets.VOCSegmentation, 21),
        'voc_sbd': (sbd, 21),
        'voc_coco': (get_voc_coco, 21)
    }
    ds_fn, num_classes = paths[name]

    transform = get_transform(image_set == 'train', target_size)
    ds = ds_fn(data_dir, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, crop_size=480):
    base_size = 520

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = [T.RandomResize(min_size, max_size)]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())

    mean, std = get_preprocessing_function('rgb', None)
    transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


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


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.gpus)

    dataset, num_classes = get_dataset(
        args.dataset, args.data_dir, 'train', args.target_size
    )
    dataset_test, _ = get_dataset(
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

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpus]
        )
        model_without_ddp = model.module

    if args.test_only:
        confmat = evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )
        print(confmat)
        return

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

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model, criterion, optimizer, data_loader, lr_scheduler,
            device, epoch, args.print_freq
        )
        confmat = evaluate(
            model, data_loader_test, device=device, num_classes=num_classes
        )
        print(confmat)
        utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.out_dir, 'model_{}.pth'.format(epoch))
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parsed_args = argument_handler.parse_train_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)
