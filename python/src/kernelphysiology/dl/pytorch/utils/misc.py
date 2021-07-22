"""
Miscellaneous utility functions and classes.
"""

import os
import time
import shutil

import torch
import torchvision.transforms as transforms

from kernelphysiology.dl.pytorch.datasets import utils_db
from kernelphysiology.dl.pytorch.models import model_utils
from kernelphysiology.dl.pytorch.utils import preprocessing
from kernelphysiology.dl.pytorch.utils.cv2_transforms import NormalizeInverse
from kernelphysiology.dl.utils import prepapre_testing
from kernelphysiology.dl.utils.default_configs import get_default_target_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
                    out_folder='', save_all=False):
    filename = os.path.join(out_folder, filename)
    torch.save(state, filename)
    if is_best:
        model_best_path = os.path.join(out_folder, 'model_best.pth.tar')
        shutil.copyfile(filename, model_best_path)
    if save_all:
        shutil.copyfile(filename, os.path.join(
            out_folder, 'checkpoint_epoch_%.3d.pth.tar' % state['epoch']))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs / 3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_preds(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        corrects = []
        for k in topk:
            corrects.append(correct[:k])
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, corrects


def accuracy(output, target, topk=(1,)):
    res, _ = accuracy_preds(output, target, topk=topk)
    return res


def train_on_data(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            input_image = input_image.cuda(args.gpus, non_blocking=True)
        target = target.cuda(args.gpus, non_blocking=True)

        # compute output
        output = model(input_image)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=topks)
        losses.update(loss.item(), input_image.size(0))
        top1.update(acc1[0], input_image.size(0))
        top5.update(acc5[0], input_image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # printing the accuracy at certain intervals
        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5
                )
            )
        if (args.train_samples is not None and
                (i * len(input_image) > args.train_samples)):
            break
    return [epoch, batch_time.avg, losses.avg, top1.avg, top5.avg]


def validate_on_data(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.top_k is None:
        topks = (1,)
    else:
        topks = (1, args.top_k)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_image, target) in enumerate(val_loader):
            if args.gpus is not None:
                input_image = input_image.cuda(args.gpus, non_blocking=True)
            target = target.cuda(args.gpus, non_blocking=True)

            # compute output
            output = model(input_image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=topks)
            losses.update(loss.item(), input_image.size(0))
            top1.update(acc1[0], input_image.size(0))
            top5.update(acc5[0], input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # printing the accuracy at certain intervals
            if i % args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5
                    )
                )
            if (args.validation_samples is not None and
                    (i * len(input_image) > args.validation_samples)):
                break
        # printing the accuracy of the epoch
        print(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5
            )
        )

    return [batch_time.avg, losses.avg, top1.avg, top5.avg]


def prepare_device(gpus):
    if gpus is None or len(gpus) == 0 or gpus[0] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    return device


# FIXME: just a hack, if it's already in the desired colour space,
#  don't change it
def tmp_c_space(manipulation_name):
    if manipulation_name in ['chromaticity', 'red_green', 'yellow_blue',
                             'lightness', 'invert_chromaticity',
                             'invert_opponency', 'invert_lightness']:
        return True
    return False


# TODO: perhaps for inverting chromaticity and luminance as well
# FIXME: for less than 3 channels in lab it wont work
def _requires_colour_transform(exp, chromaticity):
    if (
            exp == 'original_rgb' or
            (exp == 'red_green' and chromaticity == 'dichromat_rg') or
            (exp == 'yellow_blue' and chromaticity == 'dichromat_yb') or
            (exp == 'chromaticity' and chromaticity == 'monochromat') or
            (exp == 'lightness' and chromaticity == 'lightness')
    ):
        return False
    return True


def generic_evaluation(args, fn, save_fn=None, **kwargs):
    manipulation_values = args.parameters['kwargs'][args.manipulation]
    manipulation_name = args.parameters['f_name']
    for j, current_network in enumerate(args.network_files):
        # which architecture
        (model, target_size) = model_utils.which_network(
            current_network, args.task_type, num_classes=args.num_classes,
            kill_kernels=args.kill_kernels, kill_planes=args.kill_planes,
            kill_lines=args.kill_lines
        )
        model.to(args.device)
        mean, std = model_utils.get_preprocessing_function(
            args.colour_space, args.network_chromaticities[j]
        )
        normalize = transforms.Normalize(mean=mean, std=std)

        for i, manipulation_value in enumerate(manipulation_values):
            args.parameters['kwargs'][args.manipulation] = manipulation_value

            output_file = prepapre_testing._prepare_saving_file(
                args.experiment_name, args.network_names[j],
                args.dataset, manipulation_name, manipulation_value,
                extension='csv'
            )
            if os.path.exists(output_file):
                continue

            if args.task_type == 'segmentation' or 'voc' in args.dataset:
                prediction_transformation = preprocessing.prediction_transformation_seg(
                    args.parameters, args.colour_space,
                    tmp_c_space(manipulation_name)
                )
            else:
                prediction_transformation = preprocessing.prediction_transformation(
                    args.parameters, args.colour_space,
                    tmp_c_space(manipulation_name)
                )
            colour_vision = 'trichromat'
            if _requires_colour_transform(
                    manipulation_name, args.network_chromaticities[j]
            ):
                colour_vision = args.network_chromaticities[j]

            other_transformations = []
            if args.mosaic_pattern is not None:
                other_transformations.append(
                    preprocessing.mosaic_transformation(args.mosaic_pattern)
                )
            if args.sf_filter is not None:
                other_transformations.append(
                    preprocessing.sf_transformation(args.sf_filter)
                )
            other_transformations.append(prediction_transformation)

            print(
                'Processing network %s and %s %f' %
                (current_network, manipulation_name, manipulation_value)
            )

            # which dataset
            # reading it after the model, because each might have their own
            # specific size
            # loading validation set
            target_size = get_default_target_size(
                args.dataset, args.target_size
            )

            target_transform = utils_db.ImagenetCategoryTransform(
                args.categories, args.cat_dir
            )

            validation_dataset = utils_db.get_validation_dataset(
                args.dataset, args.validation_dir, colour_vision,
                args.colour_space, other_transformations, normalize,
                target_size, task=args.task_type,
                target_transform=target_transform
            )

            # TODO: nicer solution:
            if 'sampler' not in args:
                sampler = None
            else:
                sampler = args.sampler(validation_dataset)
            if 'collate_fn' not in args:
                args.collate_fn = None

            # FIXME: add segmentation datasests
            val_loader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=sampler,
                collate_fn=args.collate_fn
            )

            if args.random_images is not None:
                out_folder = prepapre_testing.prepare_saving_dir(
                    args.experiment_name, args.network_names[j],
                    args.dataset, manipulation_name
                )
                normalize_inverse = NormalizeInverse(mean, std)
                fn(
                    val_loader, out_folder, normalize_inverse,
                    manipulation_value, **kwargs
                )
            elif args.activation_map is not None:
                model = model_utils.LayerActivation(model, args.activation_map)
                current_results = fn(
                    val_loader, model, **kwargs
                )
                save_fn(
                    current_results, args.experiment_name,
                    args.network_names[j],
                    args.dataset, manipulation_name, manipulation_value
                )
            else:
                (_, _, current_results) = fn(
                    val_loader, model, **kwargs
                )
                save_fn(
                    current_results, args.experiment_name,
                    args.network_names[j],
                    args.dataset, manipulation_name, manipulation_value
                )


def is_saving_node(distributed, rank, ngpus_per_node):
    if not distributed or (distributed and rank % ngpus_per_node == 0):
        return True
    return False
