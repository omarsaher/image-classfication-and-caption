"""ResNet-18-style CNN encoder trained from scratch (no pretrained weights).

Default is half width (channels 32-64-128-256, ~2.8M parameters); with
``width_mult=1.0`` it has the standard ResNet-18 widths (64-128-256-512).

Differences from the original ResNet-18:
- the 7x7 stem convolution is replaced by three 3x3 convolutions
  (same receptive field, fewer parameters, more non-linearity);
- the last BatchNorm of every residual block starts at zero, so each block
  begins as an identity mapping, which makes training from scratch easier.
"""
from torch import nn

from src.models.encoders.base import Encoder

BASE_WIDTHS = (64, 128, 256, 512)


def _scaled(channels, width_mult):
    return max(8, int(round(channels * width_mult)))


def conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """Two 3x3 convs plus a shortcut; the shortcut is a 1x1 conv when the shape changes."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ScratchResNet(Encoder):
    def __init__(self, width_mult=0.5, blocks_per_stage=(2, 2, 2, 2), in_channels=3):
        super().__init__()
        widths = [_scaled(c, width_mult) for c in BASE_WIDTHS]
        stem_mid = _scaled(32, width_mult)

        # 224 -> 112
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, stem_mid, stride=2),
            conv_bn_relu(stem_mid, stem_mid),
            conv_bn_relu(stem_mid, widths[0]),
        )
        # 112 -> 56
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 56 -> 56 -> 28 -> 14 -> 7
        stages = []
        in_ch = widths[0]
        for i, (out_ch, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride)]
            blocks += [BasicBlock(out_ch, out_ch) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = [widths[0]] + widths
        self.feature_dim = widths[-1]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    @property
    def gradcam_layer(self):
        return self.stages[-1]

    def forward_features(self, x):
        features = {"c1": self.stem(x)}
        x = self.maxpool(features["c1"])
        for i, stage in enumerate(self.stages, start=2):
            x = stage(x)
            features[f"c{i}"] = x
        features["pooled"] = self.pool(x).flatten(1)
        return features
