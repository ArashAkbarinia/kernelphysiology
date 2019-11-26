"""
PyTorch predicting script for various segmentation datasets.
"""

import sys

import torch
import torch.utils.data
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


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.gpus)

    dataset_test, num_classes = get_dataset(
        args.dataset, args.data_dir, 'val', args.target_size
    )

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test
        )
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn
    )

    network_name = args.network_names[0]
    model = torchvision.models.segmentation.__dict__[network_name](
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        pretrained=args.pretrained
    )
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(args.network_files[0], map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpus]
        )

    confmat = evaluate(
        model, data_loader_test, device=device, num_classes=num_classes
    )
    print(confmat)


if __name__ == '__main__':
    parsed_args = argument_handler.parse_predict_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)
