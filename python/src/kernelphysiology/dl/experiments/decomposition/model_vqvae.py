"""
VQ-VAE for image decomposition.
"""

import abc
import numpy as np
import logging

import torch
from torch import nn
import torch.utils.data

from kernelphysiology.dl.experiments.decomposition import nearest_embed
from kernelphysiology.dl.pytorch.optimisations import losses


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z, **kwargs):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned
        from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class ResBlock(nn.Module):
    def __init__(self, in_chns, out_chns, mid_chns=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_chns is None:
            mid_chns = out_chns

        self.in_chns = in_chns
        self.out_chns = out_chns
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_chns, mid_chns, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_chns, out_chns, kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_chns))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        if self.in_chns == self.out_chns:
            return x + self.convs(x)
        else:
            return self.convs(x)


class DecomposeNet(AbstractAutoEncoder):
    def __init__(self, hidden, k, d, vq_coef=1, commit_coef=0.5,
                 in_chns=3, out_chns=3, tanh=False):
        super(DecomposeNet, self).__init__()

        self.hidden = hidden
        self.k = k
        self.d = d
        self.emb = nearest_embed.NearestEmbed(k, d)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chns, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            ResBlock(hidden, hidden, bn=True),
            nn.BatchNorm2d(hidden),
            ResBlock(hidden, d, bn=True),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, hidden),
            nn.BatchNorm2d(hidden),
            ResBlock(hidden, hidden),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden, out_chns, kernel_size=4, stride=2, padding=1
            )
        )
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        self.tanh = tanh

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, insize=None):
        x = self.decoder(x)

        target_size = [insize[0], insize[1]]
        out_img = nn.functional.interpolate(x, size=target_size)
        if self.tanh:
            out_img = torch.tanh(out_img)
        return out_img

    def forward(self, x):
        insize = x.shape[2:]
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q, insize=insize), z_e, emb, argmin

    def sample(self, size):
        if self.cuda():
            sample = torch.tensor(
                torch.randn(size, self.d, self.f, self.f), requires_grad=False
            ).cuda()
        else:
            sample = torch.tensor(
                torch.randn(size, self.d, self.f, self.f), requires_grad=False
            )
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def sample_inds(self, inds):
        assert len(inds.shape) == 2
        rows = inds.shape[0]
        cols = inds.shape[1]
        inds = inds.reshape(rows * cols)
        weights = self.emb.weight.detach().cpu().numpy()
        sample = np.zeros((self.d, rows, cols))
        sample = sample.reshape(self.d, rows * cols)
        for i in range(self.k):
            which_inds = inds == i
            sample[:, which_inds] = np.broadcast_to(
                weights[:, i], (which_inds.sum(), self.d)
            ).T
        sample = sample.reshape(self.d, rows, cols)
        emb = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        return self.decode(emb).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = losses.decomposition_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1)
        )

        return (
                self.mse +
                self.vq_coef * self.vq_loss +
                self.commit_coef * self.commit_loss
        )

    def latest_losses(self):
        return {
            'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss
        }

    def print_atom_hist(self, argmin):
        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
