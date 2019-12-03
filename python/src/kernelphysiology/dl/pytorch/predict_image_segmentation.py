"""
PyTorch predicting script for various segmentation datasets.
"""

import sys

import torch
import torch.utils.data

from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.pytorch.utils.misc import generic_evaluation


def print_results(current_results, *_argv):
    print(current_results)


def main(args):
    print(args)

    args.device = torch.device(args.gpus)

    torch.cuda.set_device(args.device)
    fn = utils.predict_segmentation

    args.sampler = torch.utils.data.SequentialSampler
    args.collate_fn = utils.collate_fn

    kwargs = {'num_classes': 21, 'device': args.device}
    save_fn = print_results
    generic_evaluation(args, fn, save_fn, **kwargs)


if __name__ == '__main__':
    parsed_args = argument_handler.parse_predict_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)