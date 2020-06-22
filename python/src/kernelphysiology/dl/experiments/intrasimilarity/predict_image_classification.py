"""
PyTorch prediction training script for various datasets.
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

from kernelphysiology.dl.pytorch.utils.misc import AverageMeter
from kernelphysiology.dl.pytorch.utils.misc import accuracy_preds
from kernelphysiology.dl.pytorch.utils.misc import generic_evaluation
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.utils import prepapre_testing
from kernelphysiology.dl.experiments.intrasimilarity import model as gen_model


def main(argv):
    args = argument_handler.test_arg_parser(argv, extra_args_fun)
    (args.network_files,
     args.network_names,
     args.network_chromaticities) = prepapre_testing.prepare_networks_testting(
        args.network_name, args.vision_type
    )

    # FIXME: cant take more than one GPU
    gpu = args.gpus[0]

    args.device = torch.device(gpu)
    torch.cuda.set_device(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    cudnn.benchmark = True
    gen_net = None
    if args.gen_net is not None:
        gen_net = gen_model.VQ_CVAE(128, k=512, num_channels=3)
        weights = torch.load(args.gen_net, map_location='cpu')
        gen_net.load_state_dict(weights)
        gen_net.to(args.device)
        gen_net.eval()
    kwargs = {
        'print_freq': args.print_freq,
        'gen_net': gen_net
    }
    if args.random_images is not None:
        fn = visualise_input
        save_fn = None
    else:
        kwargs['device'] = args.device
        if args.activation_map is not None:
            fn = compute_activation
            save_fn = prepapre_testing.save_activation
        else:
            fn = predict
            save_fn = prepapre_testing.save_predictions
            kwargs['criterion'] = criterion
    generic_evaluation(args, fn, save_fn, **kwargs)


def compute_activation(val_loader, model, device, print_freq=100):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_predictions = []
    with torch.no_grad():
        end = time.time()
        for i, (input_imgs, target) in enumerate(val_loader):
            input_imgs = input_imgs.to(device)

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


def predict(val_loader, model, criterion, device, print_freq=100,
            gen_net=None):
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
            input_imgs = input_imgs.to(device)
            target = target.to(device)

            if gen_net is not None:
                gen_out = gen_net(input_imgs)
                input_imgs = gen_out[0]

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
                    file_name = '%s/image_%s_%d_%d.jpg' % (
                        out_folder, str(manipulation_value), b, c
                    )
                    cv2.imwrite(file_name, current_channel)

            # TODO: make it nicer
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'.format(i, len(val_loader)))
                break


def extra_args_fun(parser):
    specific_group = parser.add_argument_group('experiment')

    specific_group.add_argument(
        '--gen_net',
        default=None,
        type=str,
        help='Path to network weights (default: None)'
    )


if __name__ == '__main__':
    main(sys.argv[1:])
