'''
Prediction script for image classification task.
'''

import time
import sys
import numpy as np

from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kernelphysiology.utils.imutils import simulate_distance
from kernelphysiology.dl.utils import argument_handler
from kernelphysiology.dl.pytorch.models.utils import which_network, get_preprocessing_function
from kernelphysiology.utils.preprocessing import which_preprocessing


class PreprocessingTransformation(object):

    def __init__(self, manipulation_function, manipulation_value, manipulation_radius):
        self.manipulation_function = manipulation_function
        self.manipulation_value = manipulation_value
        self.manipulation_radius = manipulation_radius

    def __call__(self, x):
        x = np.asarray(x, dtype='uint8')
        x = self.manipulation_function(x, self.manipulation_value,
                                       mask_radius=self.manipulation_radius,
                                       preprocessing_function=None)
        x = pil_image.fromarray(x.astype('uint8'), 'RGB')
        return x


def main(argv):
    args = argument_handler.test_arg_parser(argv)
    (args.networks, args.network_names, args.preprocessings, args.output_file) = argument_handler.test_prominent_prepares(args.experiment_name, args.network_name, args.preprocessing)

    # FIXME: cant take more than one GPU
    gpu = args.gpus[0]
    torch.cuda.set_device(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    cudnn.benchmark = True

    (image_manipulation_type, image_manipulation_values, image_manipulation_function) = which_preprocessing(args)

    distance_transformation = []
    if args.distance > 1:
        distance_transformation.append(PreprocessingTransformation(simulate_distance, args.distance, args.mask_radius))
    for j, network_name in enumerate(args.networks):
        # which architecture
        (model, target_size) = which_network(network_name, args.task_type)
        model = model.cuda(gpu)
        normalize = get_preprocessing_function(args.preprocessing)

        # FIXME: for now it only supprts classiication
        # TODO: merge code with evaluation
        for i, manipulation_value in enumerate(image_manipulation_values):
            current_manipulation_preprocessing = PreprocessingTransformation(image_manipulation_function, manipulation_value, args.mask_radius)
            transformations = [*distance_transformation, current_manipulation_preprocessing]

            print('Processing network %s and %s %f' % (network_name, image_manipulation_type, manipulation_value))

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
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            (_, _, current_results) = validate(val_loader, model, criterion)
            np.savetxt('%s_%s_%s_%s.csv' % (args.output_file, args.network_names[j], image_manipulation_type, str(manipulation_value)), current_results, delimiter=',', fmt='%i')


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
        for i, (input, target) in enumerate(val_loader):
            if 0 is not None:
                input = input.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ((acc1, acc5), (corrects1, corrects5)) = accuracy(output, target, topk=(1, 5))
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
                for out in outs:
                    all_outs.append([])

            for j, out in enumerate(outs):
                all_outs[j].append(out)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    if len(all_outs) == 1:
        prediction_output = np.concatenate(all_outs[0])
    else:
        prediction_output = [np.concatenate(out) for out in all_outs]
    return (top1.avg, top5.avg, prediction_output)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return (res, corrects)


if __name__ == '__main__':
    main(sys.argv[1:])