import torch
from . import initialization as init
from torch.nn import functional as F


class SegmentationModel(torch.nn.Module):

    def __init__(self, outs_dict):
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

        masks = self.segmentation_head(decoder_output)

        # if self.classification_head is not None:
        #     labels = self.classification_head(features[-1])
        #     return masks, labels
        #
        # return masks

        out_imgs = dict()
        for key in self.outs_dict.keys():
            out_imgs[key] = torch.tanh(masks)
        return out_imgs,

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
        self.mse = 0
        for key in x.keys():
            self.mse += F.mse_loss(recon_x[key], x[key])
        return self.mse

    def latest_losses(self):
        return {'mse': self.mse}
