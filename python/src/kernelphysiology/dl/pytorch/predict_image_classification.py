"""
Pytorch prediction script for various datasets and image manipulations.
"""

import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy_preds
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.utils.imutils import simulate_distance
from kernelphysiology.dl.utils import argument_handler
from kernelphysiology.dl.utils import prepapre_testing
from kernelphysiology.dl.pytorch.models.utils import which_network
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.utils.preprocessing import which_preprocessing


def main(argv):
    args = argument_handler.test_arg_parser(argv)
    (args.networks,
     args.network_names,
     args.preprocessings) = prepapre_testing.test_prominent_prepares(
        args.network_name,
        args.preprocessing)

    # FIXME: cant take more than one GPU
    gpu = args.gpus[0]
    torch.cuda.set_device(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    cudnn.benchmark = True

    (image_manipulation_type,
     image_manipulation_values,
     image_manipulation_function) = which_preprocessing(args)

    other_transformations = []
    # TODO: better modelling the distance
    if args.distance > 1:
        other_transformations.append(
            preprocessing.ImageTransformation(
                simulate_distance,
                args.distance,
                args.mask_radius))
    for j, current_network in enumerate(args.networks):
        # which architecture
        (model, target_size) = which_network(current_network,
                                             args.task_type,
                                             args.dataset)
        model = model.cuda(gpu)
        mean, std = get_preprocessing_function(args.colour_space,
                                               args.colour_transformation)
        normalize = transforms.Normalize(mean=mean, std=std)

        # FIXME: for now it only supports classification
        for i, manipulation_value in enumerate(image_manipulation_values):
            current_preprocessing = preprocessing.PreprocessingTransformation(
                image_manipulation_function,
                manipulation_value,
                args.mask_radius)
            # TODO: change args.preprocessings[j] to colour_transformation
            # TODO: perhaps for inverting chromaticity and luminance as well
            # FIXME: for less than 3 channels in lab it wont work
            if (image_manipulation_type == 'original_rgb' or
                    (image_manipulation_type == 'red_green'
                     and args.preprocessings[j] == 'dichromat_rg') or
                    (image_manipulation_type == 'yellow_blue'
                     and args.preprocessings[j] == 'dichromat_yb') or
                    (image_manipulation_type == 'chromaticity'
                     and args.preprocessings[j] == 'monochromat') or
                    (image_manipulation_type == 'lightness'
                     and args.preprocessings[j] == 'lightness')
            ):
                cts = []
            else:
                cts = preprocessing.colour_transformation(
                    args.preprocessings[j],
                    args.colour_space
                )

            # whether should be 1, 2, or 3 channels
            chns_transformation = preprocessing.channel_transformation(
                args.preprocessings[j],
                args.colour_space
            )

            transformations = [*other_transformations, *cts,
                               current_preprocessing]

            print('Processing network %s and %s %f' %
                  (current_network,
                   image_manipulation_type,
                   manipulation_value))

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            # Data loading code
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.validation_dir, transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.CenterCrop(target_size),
                    *transformations,
                    transforms.ToTensor(),
                    *chns_transformation,
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            (_, _, current_results) = validate(val_loader, model, criterion)
            prepapre_testing.save_predictions(current_results,
                                              args.experiment_name,
                                              args.network_names[j],
                                              args.dataset,
                                              image_manipulation_type,
                                              manipulation_value)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_outs = []
    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            if 0 is not None:
                input_imgs = input_imgs.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(input_imgs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ((acc1, acc5), (corrects1, corrects5)) = accuracy_preds(
                output, target, topk=(1, 5))
            corrects1 = corrects1.cpu().numpy()
            corrects5 = corrects5.cpu().numpy().sum(axis=0)

            outs = np.zeros((corrects1.shape[1], 3))
            outs[:, 0] = corrects1
            outs[:, 1] = corrects5
            outs[:, 2] = output.cpu().numpy().argmax(axis=1)

            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(outs, list):
                outs = [outs]

            if not all_outs:
                for _ in outs:
                    all_outs.append([])

            for j, out in enumerate(outs):
                all_outs[j].append(out)

            losses.update(loss.item(), input_imgs.size(0))
            top1.update(acc1[0], input_imgs.size(0))
            top5.update(acc5[0], input_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5))

        print(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

    if len(all_outs) == 1:
        prediction_output = np.concatenate(all_outs[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_outs]
    return top1.avg, top5.avg, prediction_output


if __name__ == '__main__':
    main(sys.argv[1:])
