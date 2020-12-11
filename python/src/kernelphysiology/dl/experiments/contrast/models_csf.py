import torch
import torch.nn as nn

from kernelphysiology.dl.experiments.contrast import pretrained_models


class ContrastDiscrimination(nn.Module):
    def __init__(self, network_name, transfer_weights=None, grey_width=True,
                 num_classes=2, scale_factor=1):
        super(ContrastDiscrimination, self).__init__()

        checkpoint = None
        # assuming network_name is path
        if transfer_weights is None:
            checkpoint = torch.load(network_name, map_location='cpu')
            network_name = checkpoint['arch']
            transfer_weights = checkpoint['transfer_weights']

        model = pretrained_models.get_pretrained_model(
            network_name, transfer_weights
        )
        if '_scratch' in network_name:
            network_name = network_name.replace('_scratch', '')
        model = pretrained_models.get_backbones(network_name, model)

        # print(model)
        layer = -1
        if len(transfer_weights) >= 2:
            layer = transfer_weights[1]

        if ('maskrcnn_' in network_name or 'fasterrcnn_' in network_name
                or 'keypointrcnn_' in network_name
                or 'deeplabv3_' in network_name or 'fcn_' in network_name
                or network_name == 'transparency' or network_name == 'simclr'
                or 'resnet' in network_name or 'resnext' in network_name
        ):
            features, org_classes = pretrained_models._resnet_features(
                model, network_name, layer, grey_width
            )
        elif network_name == 'cityscape':
            features, org_classes = pretrained_models._cityscape_features(
                model, network_name, layer, grey_width
            )
        elif 'vgg' in network_name:
            features, org_classes = pretrained_models._vgg_features(
                model, network_name, layer, grey_width
            )
        elif 'mobilenet_v2' in network_name:
            features, org_classes = pretrained_models._mobilenet_v2_features(
                model, network_name, layer, grey_width
            )
        self.features = features
        # 0.25 because the units in pretrained mdoels were computed
        # according to size 256x512.
        self.fc = nn.Linear(
            int(org_classes * scale_factor * 0.25), num_classes
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x0, x1):
        x0 = self.features(x0)
        x0 = x0.view(x0.size(0), -1)
        x1 = self.features(x1)
        x1 = x0.view(x1.size(0), -1)
        x = torch.cat([x0, x1], dim=1)
        x = self.fc(x)
        return x
