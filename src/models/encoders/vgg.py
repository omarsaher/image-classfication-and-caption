"""Pretrained VGG encoder (torchvision) behind the common ``Encoder`` interface.

Official VGG = ``features`` (conv blocks) -> ``avgpool`` -> ``classifier``, and
the classifier is Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear(4096, 1000).

This encoder keeps everything up to the last 4096-d activation as its output.
``ImageClassifier`` then adds Dropout + Linear(4096, num_classes), so the whole
model is exactly the official VGG with its last layer replaced.

Plain VGG has no BatchNorm, so a frozen ``features`` block behaves the same in
``train()`` and ``eval()`` mode: there are no running statistics to protect.
"""
import torch
from torch import nn
from torchvision import models

from src.models.encoders.base import Encoder

# torchvision stores downloads in its cache (~/.cache/torch/hub/checkpoints)
# and reuses them, so each file is only downloaded once.
VGG_WEIGHTS = {
    "vgg13": models.VGG13_Weights.IMAGENET1K_V1,
    "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
    "vgg19": models.VGG19_Weights.IMAGENET1K_V1,
}


class VGGEncoder(Encoder):
    def __init__(self, arch="vgg16", pretrained=True, fc_dropout=0.5):
        super().__init__()
        if arch not in VGG_WEIGHTS:
            raise ValueError(f"Unknown VGG '{arch}'. Available: {sorted(VGG_WEIGHTS)}")
        weights = VGG_WEIGHTS[arch] if pretrained else None
        vgg = getattr(models, arch)(weights=weights, dropout=fc_dropout)

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        # Up to and including the last ReLU; drops the final Dropout + Linear(4096, 1000)
        self.fc = vgg.classifier[:5]

        self.out_channels = [64, 128, 256, 512, 512]
        self.feature_dim = 4096

    @property
    def gradcam_layer(self):
        # Last ReLU of block 5, just before the final MaxPool (14x14 for a 224 input)
        return self.features[-2]

    def forward_features(self, x):
        # Each block ends in a MaxPool that halves the size: strides 2, 4, 8, 16, 32
        features = {}
        stage = 1
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features[f"c{stage}"] = x
                stage += 1
        features["pooled"] = self.fc(torch.flatten(self.avgpool(x), 1))
        return features
