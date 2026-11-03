"""Build VGG13/16/19 classifiers and control which of their layers train.

Importing this module registers the VGG names with the team's model registry,
so ``build_classifier``, ``load_classifier`` and ``Trainer`` work with them.
"""
from functools import partial
from pathlib import Path

from src.models.checkpoint import load_classifier
from src.models.encoders.vgg import VGG_WEIGHTS, VGGEncoder
from src.models.registry import build_classifier, register_encoder

VGG_ARCHITECTURES = tuple(VGG_WEIGHTS)

for _arch in VGG_ARCHITECTURES:
    register_encoder(_arch, partial(VGGEncoder, arch=_arch))


def vgg_model_config(arch, num_classes=200, dropout=0.5, pretrained=True):
    """The ``model`` config section of one VGG, in the registry's format."""
    if arch not in VGG_ARCHITECTURES:
        raise ValueError(f"Unknown VGG '{arch}'. Available: {list(VGG_ARCHITECTURES)}")
    return {
        "name": arch,
        "num_classes": num_classes,
        "dropout": dropout,      # before the new final layer (read by ImageClassifier)
        "fc_dropout": dropout,   # inside the official 4096-d layers (read by VGGEncoder)
        "pretrained": pretrained,
    }


def build_vgg_classifier(arch, num_classes=200, dropout=0.5, pretrained=True):
    """VGG with ImageNet weights and a new final layer; outputs ``[batch, num_classes]`` logits."""
    return build_classifier(vgg_model_config(arch, num_classes, dropout, pretrained))


def set_feature_extractor_trainable(model, trainable):
    """Freeze (``False``) or unfreeze (``True``) the convolutional blocks.

    A frozen parameter gets no gradient, so the optimizer never changes it. The
    4096-d layers and the new final layer always stay trainable.
    """
    for param in model.encoder.features.parameters():
        param.requires_grad = trainable


def freeze_feature_extractor(model):
    set_feature_extractor_trainable(model, False)


def unfreeze_feature_extractor(model):
    set_feature_extractor_trainable(model, True)


def parameter_counts(model):
    """Return ``{"trainable": n, "total": n}``."""
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }


def load_vgg_checkpoint(path, device):
    """Return ``(model, checkpoint_dict)``, with a clear error if the file is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at '{path}'. Train this model first, or check that "
            "you are running from the repository root."
        )
    return load_classifier(path, device)
