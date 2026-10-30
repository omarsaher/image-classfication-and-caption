"""Loading YAML run configs and applying command line overrides.

Configs are plain nested dicts. Overrides use dotted keys, e.g.
``train.lr=0.003`` or ``data.use_offline_augmented=true``; the value is
parsed with YAML so numbers, booleans, lists and null work as expected.
"""
import copy
from pathlib import Path

import yaml


def load_config(path, overrides=None):
    """Read a YAML config file and apply ``key=value`` overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides or [])


def apply_overrides(cfg, overrides):
    """Return a copy of ``cfg`` with each ``a.b.c=value`` override set."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        set_by_path(cfg, key.strip(), parse_value(raw_value))
    return cfg


def parse_value(raw_value):
    """Parse an override value; YAML reads '3e-3' as text, so retry as float."""
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def set_by_path(cfg, dotted_key, value):
    """Set ``cfg[a][b][c] = value`` for ``dotted_key='a.b.c'``."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_by_path(cfg, dotted_key, default=None):
    node = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
