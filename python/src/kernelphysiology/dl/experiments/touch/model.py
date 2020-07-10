import numpy as np
import logging

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from nearest_embed import NearestEmbed


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1,
                      padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            return x + self.convs(x)
        else:
            return self.convs(x)


class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, kl=None, bn=True, vq_coef=1, commit_coef=0.5,
                 num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.d = d
        self.k = k
        if kl is None:
            kl = d
        self.kl = kl
        self.emb = NearestEmbed(k, kl)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, kl, bn=True),
            nn.BatchNorm2d(kl),
        )
        self.decoder = nn.Sequential(
            ResBlock(kl, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2,
                               padding=1),
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

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, insize):
        x = torch.tanh(self.decoder(x))
        return torch.nn.functional.upsample_bilinear(x, size=insize)

    def forward(self, x):
        insize = x.shape[2:]
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q, insize), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
