"""
models.py

The from-scratch CNN used in phases 1, 2, and 3, plus the ShuffleNet builder
used in phase 4.
"""

import torch.nn as nn
import torchvision

from common import NUM_CLASSES


class FoodCNN(nn.Module):
    """
    A small CNN. Each convolutional block is a 3x3 convolution with padding 1
    (so it keeps the spatial size) followed by ReLU.

    channels      output channels of each conv block
    pool_after    one bool per conv block, True adds a 2x2 max pool after it
    fc_units      sizes of the hidden fully connected layers
    fc_dropout    dropout probability applied after each hidden FC layer
    conv_dropout  channel-wise dropout probability applied after each conv block
    """

    def __init__(self, input_size, channels=(16, 32, 64), pool_after=(False, False, False),
                 fc_units=(128, 64), fc_dropout=0.0, conv_dropout=0.0,
                 num_classes=NUM_CLASSES):
        super().__init__()
        assert len(channels) == len(pool_after)

        layers = []
        in_ch, size = 3, input_size
        for ch, pool in zip(channels, pool_after):
            layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
                size //= 2
            if conv_dropout > 0:
                layers.append(nn.Dropout2d(conv_dropout))
            in_ch = ch
        self.features = nn.Sequential(*layers)

        self.flat_features = in_ch * size * size
        fc, prev = [nn.Flatten()], self.flat_features
        for units in fc_units:
            fc += [nn.Linear(prev, units), nn.ReLU(inplace=True)]
            if fc_dropout > 0:
                fc.append(nn.Dropout(fc_dropout))
            prev = units
        fc.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_shufflenet(num_classes=NUM_CLASSES):
    """
    ShuffleNet V2 (1.0x) pretrained on ImageNet, set up for transfer learning.
    """
    weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    model = torchvision.models.shufflenet_v2_x1_0(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def shufflenet_train_mode(model):
    """
    Put the model in training mode but keep the frozen backbone in eval mode.
    """
    model.eval()
    model.fc.train()
