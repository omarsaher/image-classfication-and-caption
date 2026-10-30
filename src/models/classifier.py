from torch import nn


class ImageClassifier(nn.Module):
    """Any ``Encoder`` followed by Dropout and a linear layer over its pooled vector."""

    def __init__(self, encoder, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.feature_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    @property
    def gradcam_layer(self):
        return self.encoder.gradcam_layer
