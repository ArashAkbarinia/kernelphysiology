import numpy as np
import logging

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from nearest_embed import NearestEmbed
from gabor_layers import GaborLayer


class VAE(nn.Module):

    def __init__(self, latent_dim, in_channels=1, hidden_dims=None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.loss = 0
        self.recons_loss = 0
        self.kld_loss = 0
        self.kld_weight = 0.005

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims = [16, 32, 64, 128]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = result.view(result.size(0), -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 16, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = torch.nn.functional.upsample_bilinear(result, size=(260, 260))
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, input, recons, _, mu, log_var, **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        self.recons_loss = F.mse_loss(recons, input)

        self.kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0)

        self.loss = self.recons_loss + self.kld_weight * self.kld_loss
        return self.loss

    def latest_losses(self):
        return {'loss': self.loss, 'Reconstruction_Loss': self.recons_loss,
                'KLD': -self.kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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
                 num_channels=3, gabor_layer=False, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.d = d
        self.k = k
        if kl is None:
            kl = d
        self.kl = kl
        self.emb = NearestEmbed(k, kl)

        if gabor_layer:
            first_layer = GaborLayer(num_channels, d, kernel_size=5, stride=2,
                                     padding=1, kernels=1)
        else:
            first_layer = nn.Conv2d(num_channels, d, kernel_size=4, stride=2,
                                    padding=1)
        self.encoder = nn.Sequential(
            first_layer,
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
        self.classification_branch = self._make_classification_layer(
            Bottleneck, d, d, 3, stride=2
        )
        num_participants = 12
        self.fc = nn.Linear(d * 1 * Bottleneck.expansion, num_participants)

        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.classification = 0
        self.class_coef = 1
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def _make_classification_layer(self, block, planes, inplanes, blocks,
                                   stride=1):
        layers = []
        # if None, nothing in this layer
        if blocks > 0:
            norm_layer = nn.BatchNorm2d
            downsample = None
            previous_dilation = 1
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers.append(
                block(
                    inplanes, planes, stride, downsample, 1,
                    64, previous_dilation, norm_layer
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        inplanes, planes, groups=1,
                        base_width=64, dilation=1,
                        norm_layer=norm_layer
                    )
                )
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

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
        x_class = self.classification_branch(z_e)
        x_class = x_class.view(x_class.size(0), -1)
        participant_class = self.fc(x_class)
        return self.decode(z_q, insize), z_e, emb, argmin, participant_class

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, participant, recon_x, z_e, emb, argmin,
                      participant_class):
        self.mse = F.mse_loss(recon_x, x)
        self.classification = F.cross_entropy(participant_class, participant)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        return self.mse + self.class_coef * self.classification + \
               self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {
            'mse': self.mse, 'class': self.classification,
            'vq': self.vq_loss, 'commitment': self.commit_loss
        }

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
