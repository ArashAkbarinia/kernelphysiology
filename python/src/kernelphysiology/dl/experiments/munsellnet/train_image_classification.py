"""
PyTorch training script for various datasets and image manipulations.
"""

import os
import sys
import random
import warnings
import numpy as np

import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from kernelphysiology.dl.experiments.munsellnet import resnet
from kernelphysiology.dl.pytorch.utils.misc import accuracy_preds
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils import argument_handler
from kernelphysiology.dl.pytorch.utils.misc import AverageMeter, accuracy
from kernelphysiology.dl.pytorch.utils.misc import adjust_learning_rate
from kernelphysiology.dl.pytorch.utils.misc import save_checkpoint
from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.utils.default_configs import get_default_target_size
from kernelphysiology.dl.utils import prepare_training
from kernelphysiology.utils.path_utils import create_dir

from kernelphysiology.dl.experiments.munsellnet.dataset import \
    get_train_val_dataset
from kernelphysiology.dl.pytorch.utils import transformations

best_acc1 = 0


def predict(val_loader, model, criterion, device, print_freq=100):
    losses = AverageMeter()
    batch_time = AverageMeter()

    losses_obj = AverageMeter()
    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    losses_mun = AverageMeter()
    top1_mun = AverageMeter()
    top5_mun = AverageMeter()
    losses_ill = AverageMeter()
    top1_ill = AverageMeter()
    top5_ill = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_predictions = []
    with torch.no_grad():
        end = time.time()
        for i, (input_image, targets) in enumerate(val_loader):
            input_image = input_image.to(device)
            targets = targets.to(device)

            # compute output
            out_obj, out_mun, out_ill = model(input_image)

            if out_obj is None:
                loss_obj = 0
                corr1_obj = 0
                corr5_obj = 0
                out_obj = 0
            else:
                loss_obj = criterion(out_obj, targets[:, 0])
                ((acc1_obj, acc5_obj), (corr1_obj, corr5_obj)) = accuracy_preds(
                    out_obj, targets[:, 0], topk=(1, 5)
                )
                corr1_obj = corr1_obj.cpu().numpy().sum(axis=0)
                corr5_obj = corr5_obj.cpu().numpy().sum(axis=0)
                losses_obj.update(loss_obj.item(), input_image.size(0))
                top1_obj.update(acc1_obj[0], input_image.size(0))
                top5_obj.update(acc5_obj[0], input_image.size(0))
                out_obj = out_obj.cpu().numpy().argmax(axis=1)
            if out_mun is None:
                loss_mun = 0
                corr1_mun = 0
                corr5_mun = 0
                out_mun = 0
            else:
                loss_mun = criterion(out_mun, targets[:, 1])
                ((acc1_mun, acc5_mun), (corr1_mun, corr5_mun)) = accuracy_preds(
                    out_mun, targets[:, 1], topk=(1, 5)
                )
                corr1_mun = corr1_mun.cpu().numpy().sum(axis=0)
                corr5_mun = corr5_mun.cpu().numpy().sum(axis=0)
                losses_mun.update(loss_mun.item(), input_image.size(0))
                top1_mun.update(acc1_mun[0], input_image.size(0))
                top5_mun.update(acc5_mun[0], input_image.size(0))
                out_mun = out_mun.cpu().numpy().argmax(axis=1)
            if out_ill is None:
                loss_ill = 0
                corr1_ill = 0
                corr5_ill = 0
                out_ill = 0
            else:
                loss_ill = criterion(out_ill, targets[:, 2])
                ((acc1_ill, acc5_ill), (corr1_ill, corr5_ill)) = accuracy_preds(
                    out_ill, targets[:, 2], topk=(1, 5)
                )
                corr1_ill = corr1_ill.cpu().numpy().sum(axis=0)
                corr5_ill = corr5_ill.cpu().numpy().sum(axis=0)
                losses_ill.update(loss_ill.item(), input_image.size(0))
                top1_ill.update(acc1_ill[0], input_image.size(0))
                top5_ill.update(acc5_ill[0], input_image.size(0))
                out_ill = out_ill.cpu().numpy().argmax(axis=1)

            loss = loss_obj + loss_mun + loss_ill

            pred_outs = np.zeros((input_image.shape[0], 9))
            pred_outs[:, 0] = corr1_obj
            pred_outs[:, 1] = corr5_obj
            pred_outs[:, 2] = out_obj
            pred_outs[:, 3] = corr1_mun
            pred_outs[:, 4] = corr5_mun
            pred_outs[:, 5] = out_mun
            pred_outs[:, 6] = corr1_ill
            pred_outs[:, 7] = corr5_ill
            pred_outs[:, 8] = out_ill

            # I'm not sure if this is all necessary, copied from keras
            if not isinstance(pred_outs, list):
                pred_outs = [pred_outs]

            if not all_predictions:
                for _ in pred_outs:
                    all_predictions.append([])

            for j, out in enumerate(pred_outs):
                all_predictions[j].append(out)

            losses.update(loss.item(), input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                    'LO {obj_loss.val:.2f} ({obj_loss.avg:.2f})\t'
                    'LM {mun_loss.val:.2f} ({mun_loss.avg:.2f})\t'
                    'LI {ill_loss.val:.2f} ({ill_loss.avg:.2f})\t'
                    'Ao {obj_acc.val:.2f} ({obj_acc.avg:.2f})\t'
                    'AM {mun_acc.val:.2f} ({mun_acc.avg:.2f})\t'
                    'AI {ill_acc.val:.2f} ({ill_acc.avg:.2f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, obj_loss=losses_obj,
                        mun_loss=losses_mun, ill_loss=losses_ill,
                        obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
                    )
                )

        print(
            ' * AccObj {obj_acc.avg:.2f} AccMun {mun_acc.avg:.2f}'
            ' AccIll {ill_acc.avg:.2f}'.format(
                obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
            )
        )

    if len(all_predictions) == 1:
        prediction_output = np.concatenate(all_predictions[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_predictions]
    return prediction_output


def validate_on_data(val_loader, model, criterion, args):
    losses = AverageMeter()
    batch_time = AverageMeter()

    losses_obj = AverageMeter()
    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    losses_mun = AverageMeter()
    top1_mun = AverageMeter()
    top5_mun = AverageMeter()
    losses_ill = AverageMeter()
    top1_ill = AverageMeter()
    top5_ill = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_image, targets) in enumerate(val_loader):
            if args.gpus is not None:
                input_image = input_image.cuda(args.gpus, non_blocking=True)
            targets = targets.cuda(args.gpus, non_blocking=True)

            # compute output
            out_obj, out_mun, out_ill = model(input_image)
            if out_obj is None:
                loss_obj = 0
            else:
                loss_obj = criterion(out_obj, targets[:, 0])
                acc1_obj, acc5_obj = accuracy(out_obj, targets[:, 0],
                                              topk=topks)
                losses_obj.update(loss_obj.item(), input_image.size(0))
                top1_obj.update(acc1_obj[0], input_image.size(0))
                top5_obj.update(acc5_obj[0], input_image.size(0))
            if out_mun is None:
                loss_mun = 0
            else:
                loss_mun = criterion(out_mun, targets[:, 1])
                acc1_mun, acc5_mun = accuracy(out_mun, targets[:, 1],
                                              topk=topks)
                losses_mun.update(loss_mun.item(), input_image.size(0))
                top1_mun.update(acc1_mun[0], input_image.size(0))
                top5_mun.update(acc5_mun[0], input_image.size(0))
            if out_ill is None:
                loss_ill = 0
            else:
                loss_ill = criterion(out_ill, targets[:, 2])
                acc1_ill, acc5_ill = accuracy(out_ill, targets[:, 2],
                                              topk=topks)
                losses_ill.update(loss_ill.item(), input_image.size(0))
                top1_ill.update(acc1_ill[0], input_image.size(0))
                top5_ill.update(acc5_ill[0], input_image.size(0))

            loss = loss_obj + loss_mun + loss_ill
            losses.update(loss.item(), input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                    'LO {obj_loss.val:.2f} ({obj_loss.avg:.2f})\t'
                    'LM {mun_loss.val:.2f} ({mun_loss.avg:.2f})\t'
                    'LI {ill_loss.val:.2f} ({ill_loss.avg:.2f})\t'
                    'Ao {obj_acc.val:.2f} ({obj_acc.avg:.2f})\t'
                    'AM {mun_acc.val:.2f} ({mun_acc.avg:.2f})\t'
                    'AI {ill_acc.val:.2f} ({ill_acc.avg:.2f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, obj_loss=losses_obj,
                        mun_loss=losses_mun, ill_loss=losses_ill,
                        obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
                    )
                )
        # printing the accuracy of the epoch
        print(
            ' * AccObj {obj_acc.avg:.2f} AccMun {mun_acc.avg:.2f}'
            ' AccIll {ill_acc.avg:.2f}'.format(
                obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
            )
        )

    return [batch_time.avg, losses.avg, losses_obj.avg, losses_mun.avg,
            losses_ill.avg, top1_obj.avg, top1_mun.avg, top1_ill.avg]


def correct_image(normalise_inverse, normalise_back, input_image, out_ill,
                  ill_colours):
    corrected_images = input_image.clone()
    for i in range(out_ill.shape[0]):
        current_image = corrected_images[i].squeeze()
        current_image = normalise_inverse.__call__(current_image)
        ill_ind = out_ill[i, :].argmax()
        illuminant = ill_colours[ill_ind]
        # from skimage import io
        # im_save = current_image.cpu().numpy().transpose([1, 2, 0])
        # io.imsave('/home/arash/Desktop/before.png', im_save / im_save.max())
        for j in range(3):
            current_image[j,] /= illuminant[j]
        # im_save = current_image.cpu().numpy().transpose([1, 2, 0])
        # io.imsave('/home/arash/Desktop/after.png', im_save / im_save.max())
        # import pdb
        # pdb.set_trace()
        corrected_images[i] = normalise_back.__call__(current_image)
    return corrected_images


def train_on_data(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_obj = AverageMeter()
    top1_obj = AverageMeter()
    top5_obj = AverageMeter()
    losses_mun = AverageMeter()
    top1_mun = AverageMeter()
    top5_mun = AverageMeter()
    losses_ill = AverageMeter()
    top1_ill = AverageMeter()
    top5_ill = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to train mode
    model.train()

    mean, std = model_utils.get_preprocessing_function(
        args.colour_space, args.colour_transformation
    )
    normalise_inverse = transformations.NormalizeInverse(mean, std)
    normalise_back = transforms.Normalize(mean=mean, std=std)

    end = time.time()
    for i, (input_image, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            input_image = input_image.cuda(args.gpus, non_blocking=True)
        targets = targets.cuda(args.gpus, non_blocking=True)

        # compute output
        out_obj, out_mun, out_ill = model(input_image)

        if out_obj is None:
            loss_obj = 0
        else:
            loss_obj = criterion(out_obj, targets[:, 0])
            acc1_obj, acc5_obj = accuracy(out_obj, targets[:, 0], topk=topks)
            losses_obj.update(loss_obj.item(), input_image.size(0))
            top1_obj.update(acc1_obj[0], input_image.size(0))
            top5_obj.update(acc5_obj[0], input_image.size(0))
        if out_mun is None:
            loss_mun = 0
        else:
            loss_mun = criterion(out_mun, targets[:, 1])
            acc1_mun, acc5_mun = accuracy(out_mun, targets[:, 1], topk=topks)
            losses_mun.update(loss_mun.item(), input_image.size(0))
            top1_mun.update(acc1_mun[0], input_image.size(0))
            top5_mun.update(acc5_mun[0], input_image.size(0))
        if out_ill is None:
            loss_ill = 0
        else:
            loss_ill = criterion(out_ill, targets[:, 2])
            acc1_ill, acc5_ill = accuracy(out_ill, targets[:, 2], topk=topks)
            losses_ill.update(loss_ill.item(), input_image.size(0))
            top1_ill.update(acc1_ill[0], input_image.size(0))
            top5_ill.update(acc5_ill[0], input_image.size(0))

        loss = loss_obj + loss_mun + loss_ill
        losses.update(loss.item(), input_image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if out_mun is None and args.ill_colour is not None:
            input_image2 = correct_image(
                normalise_inverse, normalise_back, input_image, out_ill,
                args.ill_colour
            )
            out_obj2, out_mun2, _ = model(input_image2)
            loss_mun2 = 0
            loss_obj2 = 0
            if out_mun2 is not None:
                loss_mun2 = criterion(out_mun2, targets[:, 1])
            if out_obj2 is not None:
                loss_obj2 = criterion(out_obj2, targets[:, 0])
            loss2 = loss_obj2 + loss_mun2
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # printing the accuracy at certain intervals
        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                'LO {obj_loss.val:.2f} ({obj_loss.avg:.2f})\t'
                'LM {mun_loss.val:.2f} ({mun_loss.avg:.2f})\t'
                'LI {ill_loss.val:.2f} ({ill_loss.avg:.2f})\t'
                'Ao {obj_acc.val:.2f} ({obj_acc.avg:.2f})\t'
                'AM {mun_acc.val:.2f} ({mun_acc.avg:.2f})\t'
                'AI {ill_acc.val:.2f} ({ill_acc.avg:.2f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, obj_loss=losses_obj,
                    mun_loss=losses_mun, ill_loss=losses_ill,
                    obj_acc=top1_obj, mun_acc=top1_mun, ill_acc=top1_ill
                )
            )
    return [epoch, batch_time.avg, losses.avg, losses_obj.avg, losses_mun.avg,
            losses_ill.avg, top1_obj.avg, top1_mun.avg, top1_ill.avg]


def extra_args_fun(parser):
    specific_group = parser.add_argument_group('Munsell specific')

    specific_group.add_argument(
        '--prediction',
        action='store_true'
    )
    specific_group.add_argument(
        '-nw', '--network_weights',
        default=None,
        type=str,
        help='Path to network weights (default: None)'
    )
    specific_group.add_argument(
        '-pn', '--pred_name',
        default=None,
        type=str,
        help='Network name (default: None)'
    )
    specific_group.add_argument(
        '-oa', '--object_area',
        default=None,
        type=str,
        help='Area for object classification (default: None)'
    )
    specific_group.add_argument(
        '-ma', '--munsell_area',
        default=None,
        type=str,
        help='Area for Munsell classification (default: None)'
    )
    specific_group.add_argument(
        '-ia', '--illuminant_area',
        default=None,
        type=str,
        help='Area for illuminant classification (default: None)'
    )
    specific_group.add_argument(
        '--manipulation',
        type=str,
        default=None,
        help='Image manipulation type to be evaluated (default: None)'
    )

    specific_group.add_argument(
        '--parameters',
        nargs='+',
        type=str,
        default=None,
        help='Parameters passed to the evaluation function (default: None)'
    )

    specific_group.add_argument(
        '--imagenet_weights',
        type=str,
        default=None,
        help='ImageNet wieghts (default: None)'
    )

    specific_group.add_argument(
        '--ill_colour',
        type=str,
        default=None,
        help='Colour of illuminants (default: None)'
    )


def main(argv):
    args = argument_handler.train_arg_parser(argv, extra_args_fun)
    if args.prediction:
        from kernelphysiology.dl.utils import augmentation
        from kernelphysiology.dl.utils import arguments as ah
        args.manipulation, args.parameters = ah.create_manipulation_list(
            args.manipulation, args.parameters,
            augmentation.get_testing_augmentations()
        )

    if args.lr is None:
        args.lr = 0.1
    if args.weight_decay is None:
        args.weight_decay = 1e-4
    # FIXME: cant take more than one GPU
    args.gpus = args.gpus[0]

    # TODO: why load weights is False?
    args.out_dir = prepare_training.prepare_output_directories(
        dataset_name=args.dataset, network_name=args.network_name,
        optimiser='sgd', load_weights=False,
        experiment_name=args.experiment_name, framework='pytorch'
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting from checkpoints.'
        )

    if args.gpus is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely '
            'disable data parallelism.'
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)
        )
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)


# FIXME: just a hack, if it's already in the desired colour space,
#  don't change it
def tmp_c_space(manipulation_name):
    if manipulation_name in ['chromaticity', 'red_green', 'yellow_blue',
                             'lightness', 'invert_chromaticity',
                             'invert_opponency', 'invert_lightness']:
        return True
    return False


def main_worker(ngpus_per_node, args):
    global best_acc1

    is_pill_img = 'wcs_xyz_png_1600' in args.data_dir

    mean, std = model_utils.get_preprocessing_function(
        args.colour_space, args.colour_transformation
    )

    # preparing the output folder
    create_dir(args.out_dir)

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpus
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    # create model
    if args.prediction:
        checkpoint = torch.load(args.network_weights, map_location='cpu')
        blocks = checkpoint['customs']['blocks']
        pooling_type = checkpoint['customs']['pooling_type']
        num_kernels = checkpoint['customs']['num_kernels']
        outputs = checkpoint['customs']['outputs']
        for key, val in outputs.items():
            if 'area' not in val:
                outputs[key] = None
        model = resnet.__dict__[args.network_name](
            blocks, pooling_type=pooling_type,
            in_chns=len(mean), inplanes=num_kernels,
            outputs=outputs
        )
        model.load_state_dict(checkpoint['state_dict'])
    elif args.transfer_weights is not None:
        print('Transferred model!')
        (model, _) = model_utils.which_network(
            args.transfer_weights, args.task_type, num_classes=args.old_classes
        )
        model = model_utils.NewClassificationModel(model, args.num_classes)
    elif args.custom_arch:
        print('Custom model!')
        if (args.network_name == 'resnet_basic_custom' or
                args.network_name == 'resnet_bottleneck_custom'):
            outputs = {'objects': None, 'munsells': None, 'illuminants': None}
            imagenet_weights = args.imagenet_weights
            if args.object_area is not None:
                outputs['objects'] = {
                    'num_classes': 2100, 'area': args.object_area
                }
            if args.munsell_area is not None:
                outputs['munsells'] = {
                    'num_classes': 1600, 'area': args.munsell_area
                }
            if args.illuminant_area is not None:
                outputs['illuminants'] = {
                    'num_classes': 280, 'area': args.illuminant_area
                }

            model = resnet.__dict__[args.network_name](
                args.blocks, pooling_type=args.pooling_type,
                in_chns=len(mean), inplanes=args.num_kernels,
                outputs=outputs, imagenet_weights=imagenet_weights
            )
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.network_name))
        model = models.__dict__[args.network_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.network_name))
        model = models.__dict__[args.network_name]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpus is not None:
            torch.cuda.set_device(args.gpus)
            model.cuda(args.gpus)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpus]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpus is not None:
        torch.cuda.set_device(args.gpus)
        model = model.cuda(args.gpus)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if (args.network_name.startswith('alexnet') or
                args.network_name.startswith('vgg')):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpus)

    # optimiser
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    model_progress = []
    model_progress_path = os.path.join(args.out_dir, 'model_progress.csv')
    # optionally resume from a checkpoint
    # TODO: it would be best if resume load the architecture from this file
    # TODO: merge with which_architecture
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.initial_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpus)
                model = model.cuda(args.gpus)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']
                )
            )
            if os.path.exists(model_progress_path):
                model_progress = np.loadtxt(model_progress_path, delimiter=',')
                model_progress = model_progress.tolist()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=mean, std=std)

    other_transformations = []
    if args.num_augmentations != 0:
        augmentations = preprocessing.RandomAugmentationTransformation(
            args.augmentation_settings, args.num_augmentations, is_pill_img
        )
        other_transformations.append(augmentations)

    target_size = get_default_target_size(args.dataset, args.target_size)

    train_dataset, validation_dataset = get_train_val_dataset(
        args.data_dir, other_transformations, [], normalize,
        args.imagenet_weights
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = None

    if args.prediction:
        manipulation_values = args.parameters['kwargs'][args.manipulation]
        manipulation_name = args.parameters['f_name']

        for j, manipulation_value in enumerate(manipulation_values):
            args.parameters['kwargs'][args.manipulation] = manipulation_value
            prediction_transformation = preprocessing.PredictionTransformation(
                args.parameters, is_pill_img,
                args.colour_space, tmp_c_space(manipulation_name)
            )
            other_transformations = [prediction_transformation]
            _, validation_dataset = get_train_val_dataset(
                args.data_dir, other_transformations, other_transformations,
                normalize, args.imagenet_weights
            )

            val_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True
            )

            pred_log = predict(
                val_loader, model, criterion, torch.device(args.gpus)
            )
            from kernelphysiology.dl.utils import prepapre_testing
            prepapre_testing.save_predictions(
                pred_log, args.experiment_name, args.pred_name, args.dataset,
                manipulation_name, manipulation_value
            )
        return

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler
    )

    if args.ill_colour is not None:
        print('Performing with illuminant correction')
        args.ill_colour = np.loadtxt(args.ill_colour, delimiter=',')

    # training on epoch
    for epoch in range(args.initial_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.imagenet_weights is None:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_log = train_on_data(
            train_loader, model, criterion, optimizer, epoch, args
        )

        # evaluate on validation set
        validation_log = validate_on_data(
            val_loader, model, criterion, args
        )

        model_progress.append([*train_log, *validation_log])

        # remember best acc@1 and save checkpoint
        acc1 = validation_log[2]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.network_name,
                    'customs': {
                        'pooling_type': args.pooling_type,
                        'in_chns': len(mean),
                        'num_classes': args.num_classes,
                        'blocks': args.blocks,
                        'num_kernels': args.num_kernels,
                        'outputs': outputs
                    },
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'target_size': target_size,
                },
                is_best, out_folder=args.out_dir
            )
        # TODO: get this header directly as a dictionary keys
        header = 'epoch,t_time,t_loss,t_lo,t_lm,t_li,t_ao,t_am,t_ai,' \
                 'v_time,v_loss,v_lo,v_lm,v_li,v_ao,v_am,v_ai'
        np.savetxt(
            model_progress_path, np.array(model_progress),
            delimiter=',', header=header
        )


if __name__ == '__main__':
    main(sys.argv[1:])
