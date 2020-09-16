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
                 in_chns=3, outs_dict=None):
        super(DecomposeNet, self).__init__()

        if outs_dict is None:
            # if nothing specified like a normal VAE, input is the output
            outs_dict = {'input': {'shape': (None, None, in_chns)}}
        self.outs_dict = outs_dict

        self.hidden = hidden
        self.k = k
        self.d = d
        self.embs = [
            nearest_embed.NearestEmbed(k, hidden),
            nearest_embed.NearestEmbed(k, d)
        ]

        self.encoder_a = nn.Sequential(
            nn.Conv2d(in_chns, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            ResBlock(hidden, hidden, bn=True),
            nn.BatchNorm2d(hidden),
        )
        self.encoder_b = nn.Sequential(
            ResBlock(hidden, d, bn=True),
            nn.BatchNorm2d(d),
        )

        self.decoder_a = nn.Sequential(
            ResBlock(d, hidden),
            nn.BatchNorm2d(hidden),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            ResBlock(hidden, d),
        )
        self.decoder_a0 = nn.Sequential(
            ResBlock(d, hidden),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.decoder_b = nn.Sequential(
            nn.ConvTranspose2d(d, hidden, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.out_layers = dict()
        for key, val in outs_dict.items():
            self.out_layers[key] = nn.ConvTranspose2d(
                hidden, val['shape'][-1], kernel_size=4, stride=2, padding=1
            )
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder_a[-1].weight.detach().fill_(1 / 40)

        for i in range(len(self.embs)):
            self.embs[i].weight.detach().normal_(0, 0.02)
            torch.fmod(self.embs[i].weight, 0.04)

    def encode(self, x):
        ze = self.encoder_a(x)
        zq, argmin = self.embs[0](ze, weight_sg=True)
        emb, _ = self.embs[0](ze.detach())
        return self.encoder_b(zq), (ze, emb)

    def decode(self, x, insize=None):
        x_a0 = self.decoder_a0(x)
        x_b = self.decoder_b(x)
        out_imgs = dict()
        for key, val in self.out_layers.items():
            target_size = [insize[0], insize[1]]
            for i in range(2):
                if self.outs_dict[key]['shape'][i] is not None:
                    # we pass the output as a scale of input
                    target_size[i] *= self.outs_dict[key]['shape'][i]
                    # in case the scale has caused a floating point
                    target_size[i] = int(target_size[i])
                    if self.outs_dict[key]['shape'][i] == 0.5:
                        xi = x_a0
                    else:
                        xi = x_b
            out_imgs[key] = torch.tanh(torch.nn.functional.upsample_bilinear(
                val(xi), size=target_size
            ))
        return out_imgs

    def forward(self, x):
        insize = x.shape[2:]
        ze, (zea, emba) = self.encode(x)
        self.f = ze.shape[-1]
        zq, argmin = self.embs[1](self.decoder_a(ze), weight_sg=True)
        emb, _ = self.embs[1](ze.detach())
        return self.decode(zq, insize=insize), (ze, emb), (zea, emba), argmin

    def sample(self, size):
        # FIXME!
        if self.cuda():
            sample = torch.tensor(
                torch.randn(size, self.d, self.f, self.f), requires_grad=False
            ).cuda()
        else:
            sample = torch.tensor(
                torch.randn(size, self.d, self.f, self.f), requires_grad=False
            )
        emb, _ = self.embs[0](sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def sample_inds(self, inds):
        assert len(inds.shape) == 2
        rows = inds.shape[0]
        cols = inds.shape[1]
        inds = inds.reshape(rows * cols)
        weights = self.embs.weight.detach().cpu().numpy()
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

    def loss_function(self, x, recon_x, emb0, emb1, argmin):
        self.mse = losses.decomposition_loss(recon_x, x)

        self.vq_loss = torch.mean(
            torch.norm((emb0[0] - emb0[1].detach()) ** 2, 2, 1)
        ) + torch.mean(
            torch.norm((emb1[0] - emb1[1].detach()) ** 2, 2, 1)
        )
        self.commit_loss = torch.mean(
            torch.norm((emb0[0].detach() - emb0[1]) ** 2, 2, 1)
        ) + torch.mean(
            torch.norm((emb1[0].detach() - emb1[1]) ** 2, 2, 1)
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

    def cuda(self, device=None):
        for key in self.out_layers.keys():
            self.out_layers[key] = self.out_layers[key].cuda()
        for i in range(len(self.embs)):
            self.embs[i] = self.embs[i].cuda()
        return super().cuda(device=device)


class HueLoss(torch.nn.Module):
    def forward(self, recon_x, x):
        ret = recon_x - x
        ret[ret > 1] -= 2
        ret[ret < -1] += 2
        ret = ret ** 2
        return torch.mean(ret)
