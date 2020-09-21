"""
PyTorch predicting script for various segmentation datasets.
"""

import sys

import torch
import torch.utils.data

from kernelphysiology.dl.pytorch.utils import segmentation_utils as utils
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.pytorch.utils.misc import generic_evaluation
from kernelphysiology.dl.utils.prepapre_testing import save_segmentation_results


def main(args):
    args.device = torch.device(args.gpus)

    torch.cuda.set_device(args.device)
    fn = utils.predict_segmentation

    args.sampler = torch.utils.data.SequentialSampler
    args.collate_fn = utils.collate_fn

    kwargs = {
        'num_classes': args.num_classes,
        'device': args.device,
        'save_pred': args.save_pred,
        'print_freq': args.print_freq
    }
    save_fn = save_segmentation_results
    generic_evaluation(args, fn, save_fn, **kwargs)


if __name__ == '__main__':
    parsed_args = argument_handler.parse_predict_segmentation_arguments(
        sys.argv[1:]
    )
    main(parsed_args)
