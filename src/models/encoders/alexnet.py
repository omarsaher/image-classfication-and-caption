"""AlexNet encoder trained from scratch (no pretrained weights).

Same layout as AlexNet (Krizhevsky et al., 2012): five convolutions
(11x11/4, 5x5, 3x3, 3x3, 3x3 with 96-256-384-384-256 channels), three
overlapping 3x3/2 max-pools, then two fully connected layers of 4096 units
with dropout. The last layer of the original network (FC8) is the
``ImageClassifier`` head, so the encoder's ``pooled`` vector is FC7.

Differences from the original AlexNet:
- BatchNorm after every convolution instead of Local Response Normalisation
  (``batch_norm=False`` gives the original conv + ReLU, without LRN), which
  makes training from scratch on a few thousand images much more stable;
- a single tower instead of the two-GPU split;
- adaptive average pooling to 6x6 before FC6, so other input sizes still work.

Feature maps for a 224x224 input: c1 55x55 (stride 4), c2 27x27 (stride 8),
c3/c4/c5 13x13 (stride 16). AlexNet has no stride-2 or stride-32 stage, so
these strides differ from the other encoders.
"""
from torch import nn

from src.models.encoders.base import Encoder

BASE_WIDTHS = (96, 256, 384, 384, 256)


def _scaled(channels, width_mult):
    return max(8, int(round(channels * width_mult)))


def conv_block(in_ch, out_ch, kernel_size, stride=1, padding=0, batch_norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                        bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class AlexNet(Encoder):
    def __init__(self, width_mult=1.0, fc_dim=4096, fc_dropout=0.5, batch_norm=True,
                 in_channels=3):
        super().__init__()
        w = [_scaled(c, width_mult) for c in BASE_WIDTHS]

        # 224 -> 55
        self.conv1 = conv_block(in_channels, w[0], 11, stride=4, padding=2, batch_norm=batch_norm)
        # 55 -> 27 -> 27
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2),
            conv_block(w[0], w[1], 5, padding=2, batch_norm=batch_norm),
        )
        # 27 -> 13 -> 13
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2),
            conv_block(w[1], w[2], 3, padding=1, batch_norm=batch_norm),
        )
        self.conv4 = conv_block(w[2], w[3], 3, padding=1, batch_norm=batch_norm)
        self.conv5 = conv_block(w[3], w[4], 3, padding=1, batch_norm=batch_norm)
        # 13 -> 6
        self.pool = nn.Sequential(nn.MaxPool2d(3, stride=2), nn.AdaptiveAvgPool2d(6))

        # FC6 and FC7
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(w[4] * 6 * 6, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
        )

        self.out_channels = w
        self.feature_dim = fc_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @property
    def gradcam_layer(self):
        return self.conv5

    def forward_features(self, x):
        features = {}
        for i, layer in enumerate((self.conv1, self.conv2, self.conv3, self.conv4, self.conv5),
                                  start=1):
            x = layer(x)
            features[f"c{i}"] = x
        features["pooled"] = self.fc(self.pool(x).flatten(1))
        return features
