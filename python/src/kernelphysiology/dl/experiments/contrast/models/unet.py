import torch

from kernelphysiology.dl.experiments.decomposition.segmentation_models import \
    unet


def unet_model(weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu')
    encoder_name = checkpoint['arch_params']['encoder_name']
    model = unet.model.Unet(
        in_channels=3, encoder_name=encoder_name,
        encoder_weights=None, classes=3
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model
