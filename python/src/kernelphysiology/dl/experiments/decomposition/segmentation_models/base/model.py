import torch
from . import initialization as init
from torch import nn

from kernelphysiology.dl.pytorch.optimisations import losses


class SegmentationModel(torch.nn.Module):

    def __init__(self, outs_dict=None):
        super(SegmentationModel, self).__init__()
        self.mse = 0
        self.outs_dict = outs_dict

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        decoder_output = nn.functional.interpolate(
            decoder_output, size=x.shape[2:], mode='bilinear'
        )

        masks = self.segmentation_head(decoder_output)

        if self.outs_dict is not None:
            out_imgs = dict()
            for key in self.outs_dict.keys():
                out_imgs[key] = torch.tanh(masks)
            return out_imgs,
        else:
            out_imgs = {'out': masks}
            if self.classification_head is not None:
                labels = self.classification_head(features[-1])
                out_imgs['aux'] = labels
                return masks, labels
            return out_imgs

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def loss_function(self, x, recon_x):
        self.mse = losses.decomposition_loss_dict(recon_x, x)
        return self.mse

    def latest_losses(self):
        return {'mse': self.mse}
