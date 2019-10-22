"""
Pytorch prediction script for various datasets and image manipulations.
"""

import time
import sys
import numpy as np

import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy_preds
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils.transformations import NormalizeInverse
from kernelphysiology.dl.pytorch.models.utils import which_network
from kernelphysiology.dl.pytorch.models.utils import LayerActivation
from kernelphysiology.dl.pytorch.models.utils import get_preprocessing_function
from kernelphysiology.dl.pytorch.datasets.utils import get_validation_dataset
from kernelphysiology.dl.pytorch.datasets.utils import is_dataset_pil_image
from kernelphysiology.dl.utils.default_configs import get_default_target_size
from kernelphysiology.dl.utils import argument_handler
from kernelphysiology.dl.utils import prepapre_testing


# FIXME: just a hack, if it's already in the desired colour space,
#  don't change it
def tmp_c_space(manipulation_name):
    if manipulation_name in ['chromaticity', 'red_green', 'yellow_blue',
                             'lightness', 'invert_chromaticity',
                             'invert_opponency', 'invert_lightness']:
        return True
    return False


def main(argv):
    args = argument_handler.pytorch_test_arg_parser(argv)
    (network_files,
     network_names,
     network_chromaticities) = prepapre_testing.prepare_networks_testting(
        args.network_name, args.colour_transformation
    )

    # FIXME: cant take more than one GPU
    gpu = args.gpus[0]
    torch.cuda.set_device(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    cudnn.benchmark = True

    manipulation_values = args.parameters['kwargs'][args.manipulation]
    manipulation_name = args.parameters['f_name']
    for j, current_network in enumerate(network_files):
        # which architecture
        (model, target_size) = which_network(
            current_network, args.task_type, args.num_classes,
            args.kill_kernels, args.kill_planes, args.kill_lines
        )
        model = model.cuda(gpu)
        mean, std = get_preprocessing_function(
            args.colour_space, network_chromaticities[j]
        )
        normalize = transforms.Normalize(mean=mean, std=std)

        for i, manipulation_value in enumerate(manipulation_values):
            args.parameters['kwargs'][args.manipulation] = manipulation_value
            prediction_transformation = preprocessing.PredictionTransformation(
                args.parameters, is_dataset_pil_image(args.dataset),
                args.colour_space, tmp_c_space(manipulation_name)
            )
            # TODO: perhaps for inverting chromaticity and luminance as well
            # FIXME: for less than 3 channels in lab it wont work
            if (
                    manipulation_name == 'original_rgb' or
                    (manipulation_name == 'red_green'
                     and network_chromaticities[j] == 'dichromat_rg') or
                    (manipulation_name == 'yellow_blue'
                     and network_chromaticities[j] == 'dichromat_yb') or
                    (manipulation_name == 'chromaticity'
                     and network_chromaticities[j] == 'monochromat') or
                    (manipulation_name == 'lightness'
                     and network_chromaticities[j] == 'lightness')
            ):
                colour_transformations = []
            else:
                colour_transformations = preprocessing.colour_transformation(
                    network_chromaticities[j], args.colour_space
                )

            # whether should be 1, 2, or 3 channels
            chns_transformation = preprocessing.channel_transformation(
                network_chromaticities[j], args.colour_space
            )

            other_transformations = [prediction_transformation]

            print(
                'Processing network %s and %s %f' %
                (current_network, manipulation_name, manipulation_value)
            )

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            # loading validation set
            target_size = get_default_target_size(args.dataset)

            validation_dataset = get_validation_dataset(
                args.dataset, args.validation_dir, colour_transformations,
                other_transformations, chns_transformation, normalize,
                target_size
            )

            val_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )

            if args.random_images is not None:
                out_folder = prepapre_testing.prepare_saving_dir(
                    args.experiment_name, network_names[j],
                    args.dataset, manipulation_name
                )
                normalize_inverse = NormalizeInverse(mean, std)
                visualise_input(
                    val_loader, out_folder, normalize_inverse,
                    manipulation_value, args.print_freq
                )
            elif args.activation_map is not None:
                model = LayerActivation(model, args.activation_map)
                current_results = compute_activation(
                    val_loader, model, gpu, args.print_freq
                )
                prepapre_testing.save_activation(
                    current_results, args.experiment_name, network_names[j],
                    args.dataset, manipulation_name, manipulation_value
                )
            else:
                (_, _, current_results) = predict(
                    val_loader, model, criterion, gpu, args.print_freq
                )
                prepapre_testing.save_predictions(
                    current_results, args.experiment_name, network_names[j],
                    args.dataset, manipulation_name, manipulation_value
                )


def compute_activation(val_loader, model, gpu_num, print_freq=100):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_predictions = []
    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            if gpu_num is not None:
                input_imgs = input_imgs.cuda(gpu_num, non_blocking=True)

            # compute output
            pred_outs = model(input_imgs).cpu().numpy()

            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(pred_outs, list):
                pred_outs = [pred_outs]

            if not all_predictions:
                for _ in pred_outs:
                    all_predictions.append([])

            for j, out in enumerate(pred_outs):
                all_predictions[j].append(out)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                    )
                )

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    return prediction_output


def predict(val_loader, model, criterion, gpu_num, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_predictions = []
    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            if gpu_num is not None:
                input_imgs = input_imgs.cuda(gpu_num, non_blocking=True)
            target = target.cuda(gpu_num, non_blocking=True)

            # compute output
            output = model(input_imgs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ((acc1, acc5), (corrects1, corrects5)) = accuracy_preds(
                output, target, topk=(1, 5)
            )
            corrects1 = corrects1.cpu().numpy()
            corrects5 = corrects5.cpu().numpy().sum(axis=0)

            pred_outs = np.zeros((corrects1.shape[1], 3))
            pred_outs[:, 0] = corrects1
            pred_outs[:, 1] = corrects5
            pred_outs[:, 2] = output.cpu().numpy().argmax(axis=1)

            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(pred_outs, list):
                pred_outs = [pred_outs]

            if not all_predictions:
                for _ in pred_outs:
                    all_predictions.append([])

            for j, out in enumerate(pred_outs):
                all_predictions[j].append(out)

            losses.update(loss.item(), input_imgs.size(0))
            top1.update(acc1[0], input_imgs.size(0))
            top5.update(acc5[0], input_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
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
                        top5=top5
                    )
                )

        print(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5
            )
        )

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    return top1.avg, top5.avg, prediction_output


def visualise_input(val_loader, out_folder, normalize_inverse,
                    manipulation_value, print_freq=100):
    with torch.no_grad():
        for i, (input_imgs, _) in enumerate(val_loader):
            # TODO: make it according to colour space
            for b in range(input_imgs.shape[0]):
                current_im = normalize_inverse(input_imgs[b])
                for c in range(current_im.shape[0]):
                    current_channel = current_im[c].squeeze().numpy()
                    current_channel = (current_channel * 255).astype('uint8')
                    file_name = '%s/image_%d_%d_%s.jpg' % (
                        out_folder, b, c, str(manipulation_value)
                    )
                    cv2.imwrite(file_name, current_channel)

            # TODO: make it nicer
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'.format(i, len(val_loader)))
                break


if __name__ == '__main__':
    main(sys.argv[1:])
