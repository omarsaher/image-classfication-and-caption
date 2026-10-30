"""Common interface for every visual encoder (from scratch or pretrained).

The same encoder is the backbone of classification, colorization and
captioning, so each encoder exposes:

- ``forward_features(x)``: dict with the multi-scale maps ``c1``..``c5``
  (strides 2, 4, 8, 16, 32) and the global ``pooled`` vector.
- ``forward(x)``: the pooled vector only.
- ``out_channels``: channels of ``c1``..``c5``; ``feature_dim``: size of ``pooled``.
- ``gradcam_layer``: the module whose output Grad-CAM explains.
"""
from abc import ABC, abstractmethod

from torch import nn


class Encoder(nn.Module, ABC):
    out_channels: list
    feature_dim: int

    @abstractmethod
    def forward_features(self, x):
        ...

    @property
    @abstractmethod
    def gradcam_layer(self):
        ...

    def forward(self, x):
        return self.forward_features(x)["pooled"]
