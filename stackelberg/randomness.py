"""Centralized random-state configuration for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42
_GLOBAL_SEED = DEFAULT_SEED


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and Torch and return the normalized integer seed."""
    global _GLOBAL_SEED
    normalized_seed = int(seed)
    _GLOBAL_SEED = normalized_seed
    os.environ["PYTHONHASHSEED"] = str(normalized_seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(normalized_seed)
        torch.cuda.manual_seed_all(normalized_seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return normalized_seed


def get_global_seed() -> int:
    """Return the most recently configured process-wide seed."""
    return _GLOBAL_SEED


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Create an independent NumPy generator for deterministic local sampling."""
    return np.random.default_rng(int(seed))

