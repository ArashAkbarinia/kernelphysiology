"""
https://github.com/Linardos/SalEMA
original target size 192x256
"""

import torch
from torch import nn
# from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.nn.functional import dropout2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.activation import ReLU

from torchvision.models import vgg16


class Upsample(nn.Module):
    # Upsample has been deprecated, this workaround allows us to still use the
    # function within sequential.
    # https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Salema(nn.Module):

    def __init__(self):
        super(Salema, self).__init__()
        self.model = SalEMA()

    def forward(self, x):
        state = None
        for i in range(x.shape[1]):
            x_i = x[:, i].squeeze()
            state, saliency_map = self.model(input_=x_i, prev_state=state)

        saliency_map = saliency_map.squeeze()
        return saliency_map


class SalEMA(nn.Module):
    """
    In this model, we pick a Convolutional layer from the bottleneck and apply
    EMA as a simple temporal regularizer. The smaller the alpha, the less each
    newly added frame will impact the outcome. This way the temporal information
    becomes most relevant.
    """

    def __init__(self, alpha=0.1, ema_loc=30, residual=False, dropout=True):
        super(SalEMA, self).__init__()

        self.dropout = dropout
        self.residual = residual
        if alpha is None:
            self.alpha = nn.Parameter(torch.Tensor([0.25]))
            print("Initial alpha set to: {}".format(self.alpha))
        else:
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
        assert (1 >= self.alpha >= 0)
        self.ema_loc = ema_loc  # 30 = bottleneck
        self.sigmoid = Sigmoid()

        # Create encoder based on VGG16 architecture
        original_vgg16 = vgg16()

        # select only convolutional layers
        encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])

        # define decoder based on VGG16 (inverse order and Upsampling layers)
        decoder_list = [
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        decoder = torch.nn.Sequential(*decoder_list)

        # assamble the full architecture encoder-decoder
        self.salgan = torch.nn.Sequential(
            *(list(encoder.children()) + list(decoder.children()))
        )

        print('Model initialized, EMA located at {}'.format(
            self.salgan[self.ema_loc])
        )

    def forward(self, input_, prev_state=None):
        x = self.salgan[:self.ema_loc](input_)
        residual = x

        if self.dropout is True:
            x = dropout2d(x)
        # salgan[self.ema_loc] will act as the temporal state
        if prev_state is None:
            # Initially don't apply alpha as there is no prev state we will
            # consistently have bad saliency maps at the start if we were to
            # do so.
            current_state = self.salgan[self.ema_loc](x)
        else:
            current_state = self.sigmoid(self.alpha) * self.salgan[
                self.ema_loc](x) + (1 - self.sigmoid(self.alpha)) * prev_state

        if self.residual is True:
            x = current_state + residual
        else:
            x = current_state

        if self.ema_loc < len(self.salgan) - 1:
            x = self.salgan[self.ema_loc + 1:](x)

        # x is a saliency map at this point
        return current_state, x
