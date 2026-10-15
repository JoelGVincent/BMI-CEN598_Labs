"""
models.py

The from-scratch CNN used in phases 1, 2, and 3, plus the ShuffleNet builder
used in phase 4.

FoodCNN is one configurable class on purpose. The three from-scratch phases
are meant to be a controlled experiment where each phase adds exactly one
thing to the previous one.
  Phase 1  convolutional layers followed by fully connected layers
  Phase 2  the same network with max pooling added after the conv layers
  Phase 3  the phase 2 network with dropout added

Because the convolutional widths and fully connected widths stay identical,
any change in the results can be traced back to the one thing that changed.
"""

import torch.nn as nn
import torchvision

from common import NUM_CLASSES


class FoodCNN(nn.Module):
    """
    A small CNN. Each convolutional block is a 3x3 convolution with padding 1
    (so it keeps the spatial size) followed by ReLU. After the blocks the
    feature maps are flattened and passed through fully connected layers.

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

    The original fully connected layer maps 1024 features to 1000 ImageNet
    classes. It is replaced by a new layer that maps the same 1024 features to
    the 11 food classes. Every pretrained parameter is frozen so only the new
    layer is trained.
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

    Freezing parameters stops their gradients, but batch normalization layers
    would still update their running mean and variance in training mode. That
    would quietly change the supposedly frozen backbone. Keeping the backbone
    in eval mode leaves it exactly as pretrained, and only the new fc layer
    is trained.
    """
    model.eval()
    model.fc.train()
