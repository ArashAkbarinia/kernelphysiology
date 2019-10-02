"""
Miscellaneous utility functions and classes.
"""

import os
import time
import shutil

import torch


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
                    out_folder=''):
    filename = os.path.join(out_folder, filename)
    torch.save(state, filename)
    if is_best:
        model_best_path = os.path.join(out_folder, 'model_best.pth.tar')
        shutil.copyfile(filename, model_best_path)


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
        # printing the accuracy of the epoch
        print(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5
            )
        )

    return [batch_time.avg, losses.avg, top1.avg, top5.avg]
