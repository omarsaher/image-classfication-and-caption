"""Build models by name from the ``model`` section of a config.

To add another encoder (e.g. a pretrained one for transfer learning), write a
class implementing ``Encoder`` and call ``register_encoder`` with its name;
training, evaluation and the CLI then work with it unchanged.
"""
from src.models.classifier import ImageClassifier
from src.models.encoders.alexnet import AlexNet
from src.models.encoders.scratch_resnet import ScratchResNet

ENCODERS = {
    "scratch_resnet": ScratchResNet,
    "alexnet": AlexNet,
}

# Keys of the model config that belong to the classifier, not the encoder
_CLASSIFIER_KEYS = {"name", "num_classes", "dropout"}


def register_encoder(name, encoder_cls):
    ENCODERS[name] = encoder_cls


def build_encoder(model_cfg):
    name = model_cfg["name"]
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    kwargs = {k: v for k, v in model_cfg.items() if k not in _CLASSIFIER_KEYS}
    return ENCODERS[name](**kwargs)


def build_classifier(model_cfg):
    return ImageClassifier(
        build_encoder(model_cfg),
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.0),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
