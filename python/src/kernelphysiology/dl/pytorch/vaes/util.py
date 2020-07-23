import numpy as np
import shutil
import os
import logging.config
from datetime import datetime
import json

from skimage import io

import torch
from torchvision.utils import save_image, make_grid
from torch.nn import functional as F

from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor
from kernelphysiology.transformations import labels


def setup_logging_from_args(args):
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """

    def set_args_default(field_name, value):
        if hasattr(args, field_name):
            return eval('args.' + field_name)
        else:
            return value

    # Set default args in case they don't exist in args
    resume = set_args_default('resume', False)
    save_name = set_args_default('save_name', '')
    results_dir = set_args_default('results_dir', './results')

    if save_name is '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')

    setup_logging(log_file, resume)
    export_args(args, save_path)
    return save_path


def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def export_args(args, save_path):
    """
    args: argparse.Namespace
        arguments to save
    save_path: string
        path to directory to save at
    """
    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def write_images(data, outputs, writer, suffix, mean, std, inv_func=None):
    original = inv_normalise_tensor(data, mean, std)
    if inv_func is not None:
        original = inv_func(original)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = inv_normalise_tensor(outputs[0], mean, std)
    if inv_func is not None:
        reconstructed = inv_func(reconstructed)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


def save_checkpoint(model, save_path):
    epoch = model['epoch']
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    weights_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model['state_dict'], weights_path)
    resume_path = os.path.join(save_path, 'checkpoints', 'last_epoch.pth')
    torch.save(model, resume_path)


def tensor_tosave(tensor, inv_func=None):
    imgs = []
    for i in range(tensor.shape[0]):
        img = tensor[i].cpu().numpy().transpose((1, 2, 0))
        if inv_func is not None:
            img = inv_func(img)
        else:
            img *= 255
            img = np.uint8(img)
        imgs.append(img)
    return imgs


def tensor_colourlabel(tensor, do_argmax=False):
    imgs = []
    for i in range(tensor.shape[0]):
        img = tensor[i].detach().cpu().numpy()
        if do_argmax:
            img = img.argmax(axis=0)
        img = np.uint8(img)
        # FIXME: hardcoded dataset
        img = labels.colour_label(img, dataset='voc')
        imgs.append(img)
    return imgs


def grid_save_reconstructed_images(data, outputs, mean, std, epoch, save_path,
                                   name, inv_func=None):
    # FIXME this is not a solution!!
    if outputs[0].shape[1] < 4:
        original = inv_normalise_tensor(data, mean, std).detach()
        original = tensor_tosave(original, inv_func)
        reconstructed = inv_normalise_tensor(outputs[0], mean, std).detach()
        reconstructed = tensor_tosave(reconstructed, inv_func)
    else:
        original = tensor_colourlabel(data)
        reconstructed = tensor_colourlabel(outputs[0], True)

    original = np.concatenate(original, axis=1)
    reconstructed = np.concatenate(reconstructed, axis=1)
    both_together = np.concatenate([original, reconstructed], axis=0)
    io.imsave(
        os.path.join(save_path, name + '_' + str(epoch) + '.png'),
        both_together
    )


def grid_save_reconstructed_labhue(data, outputs, mean, std, epoch, save_path,
                                   name, inv_func=None):
    # FIXME this is not a solution!!
    original = inv_normalise_tensor(data, mean, std).detach()
    original_lab = tensor_tosave(original[:, :3], inv_func)
    original_hue = tensor_tosave(original[:, 3:], inv_func)
    reconstructed = inv_normalise_tensor(outputs[0], mean, std).detach()
    reconstructed_lab = tensor_tosave(reconstructed[:, :3], inv_func)
    reconstructed_hue = tensor_tosave(reconstructed[:, 3:], inv_func)

    original_lab = np.concatenate(original_lab, axis=1)
    reconstructed_lab = np.concatenate(reconstructed_lab, axis=1)
    original_hue = np.concatenate(original_hue, axis=1)
    original_hue = np.repeat(original_hue, 3, axis=2)
    reconstructed_hue = np.concatenate(reconstructed_hue, axis=1)
    reconstructed_hue = np.repeat(reconstructed_hue, 3, axis=2)
    both_together = np.concatenate([
        original_lab, reconstructed_lab,
        original_hue, reconstructed_hue
    ], axis=0)
    io.imsave(
        os.path.join(save_path, name + '_' + str(epoch) + '.png'),
        both_together
    )


def grid_save_reconstructed_bsds(data, outputs, mean, std, epoch, save_path,
                                 name, inv_func=None):
    original = []
    data = data.detach().cpu().numpy()
    reconstructed = []
    outputs = F.sigmoid(outputs[0]).detach().cpu().numpy().squeeze()
    for i in range(data.shape[0]):
        original.append(data[i])
        reconstructed.append(outputs[i])

    original = np.concatenate(original, axis=1)
    reconstructed = np.concatenate(reconstructed, axis=1)
    both_together = np.concatenate([original, reconstructed], axis=0)
    io.imsave(
        os.path.join(save_path, name + '_' + str(epoch) + '.png'),
        both_together
    )


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat(
        [data[:n], outputs.view(batch_size, size[1], size[2], size[3])[:n]]
    )
    save_image(
        comparison.cpu(),
        os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n,
        normalize=True
    )
