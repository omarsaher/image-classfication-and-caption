import random

import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so runs are repeatable.

    cuDNN autotuning stays on for speed, so GPU results can still differ in
    the last decimals between runs; data splits and initial weights are exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker init: give each worker its own reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
