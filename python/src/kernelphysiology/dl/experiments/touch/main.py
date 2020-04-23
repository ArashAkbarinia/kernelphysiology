import os
import sys
import time
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from skimage import io

import util as ex_util
from model import *
from dataset import get_train_dataset, get_val_dataset

models = {
    'custom': {'vqvae': VQ_CVAE, 'vqvae2': VQ_CVAE},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
}
dataset_test_args = {
    'custom': {},
}
dataset_n_channels = {
    'custom': 1,
}
default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128},
}


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument('data_dir', help='path to training data')
    parser.add_argument(
        '--test_inds', type=int, nargs='+', help='inds test participants'
    )
    parser.add_argument('--target-size', default=100, type=int)
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='number of data loading workers (default: 4)'
    )
    parser.add_argument('--pred', type=str, default=None,
                        help='Only prediction')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vqvae',
                              choices=['vae', 'vqvae'],
                              help='autoencoder variant to use: vae | vqvae')
    model_parser.add_argument('--batch-size', type=int, default=4,
                              metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k',
                              metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')
    parser.add_argument('--resume', type=str, default=None,
                        help='The path to resume.')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument(
        '--dataset', default='custom',
        choices=['mnist', 'cifar10', 'imagenet', 'coco', 'custom'],
        help='dataset to use: mnist | cifar10 | imagenet | coco | custom'
    )
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true',
                                 default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10,
                                metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR',
                                default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')

    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]

    save_path = ex_util.setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    args.inv_func = None

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](hidden, k=k,
                                             num_channels=num_channels)
    if args.resume is not None:
        weights = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(weights)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, int(args.epochs / 3), 0.5
    )

    # NOTE: right now there's no additional transformaiton function
    trans_funcs = []
    # normlisation
    args.mean = [0.5, 0.5, 0.5]
    args.std = [0.5, 0.5, 0.5]

    train_dataset = get_train_dataset(
        args.data_dir + '/img/', args.data_dir + '/gt/',
        args.data_dir + '/all_imgs.txt', args.test_inds,
        trans_funcs, args.mean, args.std, args.target_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    val_dataset = get_val_dataset(
        args.data_dir + '/img/', args.data_dir + '/gt/',
        args.data_dir + '/all_imgs.txt', args.test_inds,
        trans_funcs, args.mean, args.std, args.target_size
    )

    # NOTE: shuffle is False
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    if args.pred is not None:
        weights = torch.load(args.pred, map_location='cpu')
        model.load_state_dict(weights)
        model.cuda()
        predict_net(model, val_loader, save_path, args)
        return

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(1, args.epochs + 1):
        train_losses = train(
            epoch, model, train_loader, optimizer, args.cuda, args.log_interval,
            save_path, args, writer
        )
        test_losses = test_net(epoch, model, val_loader, args.cuda, save_path,
                               args, writer)
        ex_util.save_checkpoint(model, epoch, save_path)

        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(
                name, {'train': train_losses[train_name],
                       'test': test_losses[test_name], }
            )
        scheduler.step()


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path,
          args, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, loader_data in enumerate(train_loader):
        data = loader_data[0]
        target = loader_data[1]

        max_len = len(train_loader)
        target = target.cuda()

        data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)

        loss = model.loss_function(target, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(
                ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info(
                'Train Epoch: {epoch} [{batch:5d}/{total_batch} '
                '({percent:2d}%)]   time: {time:3.2f}   {loss}'
                    .format(epoch=epoch, batch=batch_idx * len(data),
                            total_batch=max_len * len(data),
                            percent=int(100. * batch_idx / max_len),
                            time=time.time() - start_time, loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx in [18, 180, 1650, max_len - 1]:
            ex_util.grid_save_reconstructed_images(
                target, outputs, args.mean, args.std, epoch, save_path,
                'reconstruction_train%.5d' % batch_idx, args.inv_func
            )

        if args.dataset in ['imagenet', 'coco', 'custom'] and batch_idx * len(
                data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (max_len / data.shape[0])
        else:
            epoch_losses[key] /= (
                    len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    # writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
    # model.print_atom_hist(outputs[3])
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, loader_data in enumerate(test_loader):
            data = loader_data[0]
            target = loader_data[1]
            target = target.cuda()
            data = data.cuda()

            outputs = model(data)
            model.loss_function(target, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i in [0, 100, 200, 300, 400]:
                ex_util.grid_save_reconstructed_images(
                    target, outputs, args.mean, args.std, epoch, save_path,
                    'reconstruction_test%.5d' % i, args.inv_func
                )
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(
        ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def predict_net(model, test_loader, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        j = 0
        for i, loader_data in enumerate(test_loader):
            data = loader_data[0]
            target = loader_data[1]
            target = target.cuda()
            data = data.cuda()

            outputs = model(data)
            out_ready = ex_util.inv_normalise_tensor(
                outputs[0], args.mean, args.std).detach()
            for img_ind in range(out_ready.shape[0]):
                cur_out_ready = out_ready[img_ind]
                cur_out_ready = cur_out_ready.cpu().numpy().squeeze()
                cur_out_ready *= 255
                cur_out_ready = cur_out_ready.astype('uint8')
                fname = os.path.join(save_path, 'pred_%.5d.png' % j)
                io.imsave(fname, cur_out_ready)
                j += 1


if __name__ == "__main__":
    main(sys.argv[1:])
