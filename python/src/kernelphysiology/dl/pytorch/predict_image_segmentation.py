"""
PyTorch predicting script for various segmentation datasets.
"""

import sys

import torch
import torch.utils.data
import torchvision

from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.gpus)

    dataset_test, num_classes = utils.get_dataset(
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

    confmat = utils.evaluate(
        model, data_loader_test, device=device, num_classes=num_classes
    )
    print(confmat)


if __name__ == '__main__':
    parsed_args = argument_handler.parse_predict_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)
