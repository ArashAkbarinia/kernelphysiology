"""
WaveNet
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as torch_funcs

from kernelphysiology.dl.pytorch.vaes.nearest_embed import NearestEmbed

__all__ = [
    'WaveNet', 'wavenet_bottleneck'
]


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=True
    )


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=True
    )


def conv1x1_transpose(in_channels, out_channels, stride=1):
    """1x1 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=True
    )


def conv3x3_transpose(in_channels, out_channels, padding=1, stride=1,
                      output_padding=0):
    """3x3 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, padding=padding,
        stride=stride, output_padding=output_padding, bias=True
    )


class BottleneckBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        out_planes_expanded = out_planes * self.expansion

        self.conv1c = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2c = conv3x3(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3c = conv1x1(out_planes, out_planes_expanded)
        self.bn3 = nn.BatchNorm2d(out_planes_expanded)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1c(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2c(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3c(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlockTranspose(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, upsample=None):
        super(BottleneckBlockTranspose, self).__init__()
        out_planes_expanded = out_planes * self.expansion
        # TODO: not a nice solution!
        if upsample is not None:
            out_planes_expanded = round(out_planes_expanded / 2)

        self.conv1t = conv1x1_transpose(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2t = conv3x3_transpose(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3t = conv1x1_transpose(out_planes, out_planes_expanded)
        self.bn3 = nn.BatchNorm2d(out_planes_expanded)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1t(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2t(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3t(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class WaveNet(nn.Module):

    def __init__(self, block_type, num_blocks, latent_dim=512,
                 in_channels=3, out_channels=None, num_kernels=64):
        super(WaveNet, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.current_planes = num_kernels
        self.resolution = 128

        self.loss = 0
        self.recons_loss = 0
        self.kld_loss = 0

        # prior to the systematic layers
        self.preprocess = self._preprocess_layer()

        # start of the systematic convolutional layers
        self.layer1c = self._make_layer(
            block_type, num_kernels, 1 * num_blocks[0], stride=2
        )
        self.layer2c = self._make_layer(
            block_type, num_kernels * 2, num_blocks[1], stride=2
        )
        self.layer3c = self._make_layer(
            block_type, num_kernels * 4, num_blocks[2], stride=2
        )
        self.layer4c = self._make_layer(
            block_type, num_kernels * 8, num_blocks[3], stride=2
        )
        self.encoder = nn.Sequential(
            self.layer1c, self.layer2c, self.layer3c, self.layer4c
        )

        latent_spatial = int(self.resolution / 16)
        latent_in_size = self.current_planes * (latent_spatial ** 2)
        # the latent space
        # self.fc_mu = nn.Linear(latent_in_size, latent_dim)
        # self.fc_var = nn.Linear(latent_in_size, latent_dim)

        self.emb = NearestEmbed(latent_dim, self.current_planes)
        self.vq_coef = 1
        self.commit_coef = 0.5
        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        transpose_block_type = BottleneckBlockTranspose

        self.latent_kernels = int(latent_dim / 64)
        # self.current_planes = self.latent_kernels

        # start of the systematic transpose layers
        self.layer1t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 8, num_blocks[3], stride=2
        )
        self.layer2t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 4, num_blocks[2], stride=2
        )
        self.layer3t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 2, num_blocks[1], stride=2
        )
        self.layer4t = self._make_layer_transpose(
            transpose_block_type, num_kernels * 1, num_blocks[0], stride=2
        )

        self.decoder = nn.Sequential(
            self.layer1t, self.layer2t, self.layer3t, self.layer4t
        )

        # posterior to systematic layers
        self.postprocess = self._postprocess_layer()

        # initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.detach().normal_(0, 0.02)
                torch.fmod(m.weight, 0.04)
                nn.init.constant_(m.bias, 0)

    def _preprocess_layer(self):
        layers = [
            conv3x3(self.in_channels, self.current_planes)
        ]
        return nn.Sequential(*layers)

    def _postprocess_layer(self):
        layers = [
            conv3x3_transpose(
                self.current_planes, self.out_channels, padding=1, stride=1,
                output_padding=0
            )
        ]
        return nn.Sequential(*layers)

    def _make_layer_transpose(self, block_type, nkernels, num_blocks, stride=1):
        layers = []
        if num_blocks > 0:
            out_channels = round(nkernels * block_type.expansion / 2)
            for _ in range(1, num_blocks):
                layers.append(block_type(self.current_planes, nkernels))

            upsample = None
            if stride != 1 or self.current_planes != out_channels:
                upsample = nn.Sequential(
                    conv1x1_transpose(
                        self.current_planes, out_channels, stride=stride
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(
                block_type(self.current_planes, nkernels, stride, upsample)
            )
            self.current_planes = out_channels

        return nn.Sequential(*layers)

    def _make_layer(self, block_type, nkernels, num_blocks, stride=1):
        layers = []
        if num_blocks > 0:
            out_channels = nkernels * block_type.expansion
            downsample = None
            if stride != 1 or self.current_planes != out_channels:
                downsample = nn.Sequential(
                    conv1x1(
                        self.current_planes, out_channels, stride
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(
                block_type(self.current_planes, nkernels, stride, downsample)
            )

            self.current_planes = out_channels
            for _ in range(1, num_blocks):
                layers.append(block_type(self.current_planes, nkernels))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.preprocess(x)
        x = torch_funcs.interpolate(
            x, size=self.resolution, mode='bilinear', align_corners=False
        )
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        # mu = self.fc_mu(x)
        # var = self.fc_var(x)
        # return mu, var
        return x

    def decode(self, x, input_shape):
        # x = x.view(-1, self.latent_kernels, 8, 8)
        x = self.decoder(x)
        x = torch_funcs.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )
        x = self.postprocess(x)
        return torch.tanh(x)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # mu, var = self.encode(x)
        # z = self.reparameterise(mu, var)
        # x = self.decode(z, input_shape)
        #
        # return x, mu, var
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q, input_shape), z_e, emb, argmin

    def reparameterise(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = torch_funcs.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        return (self.mse + self.vq_coef * self.vq_loss +
                self.commit_coef * self.commit_loss)

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    # def loss_function(self, target, recons, mu, var):
    #     self.recons_loss = torch_funcs.mse_loss(recons, target)
    #
    #     self.kld_loss = torch.mean(
    #         -0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0
    #     )
    #
    #     self.loss = self.recons_loss + self.kld_loss
    #     return self.loss
    #
    # def latest_losses(self):
    #     return {
    #         'loss': self.loss,
    #         'Reconstruction_Loss': self.recons_loss,
    #         'KLD': -self.kld_loss
    #     }


def wavenet_bottleneck(num_blocks=None, **kwargs):
    """Constructs a WaveNet-Bottleneck-Custom model."""
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    model = WaveNet(BottleneckBlock, num_blocks, **kwargs)
    return model
