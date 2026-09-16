"""Day-ahead electricity-price sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "DA_RT_data.csv"
RandomSource = Union[np.random.Generator, np.random.RandomState]


def sample_electricity_price(
    T: int,
    rng: Optional[RandomSource] = None,
    *,
    require_24_hours: bool = False,
    price_path: Optional[Path] = None,
) -> np.ndarray:
    """Sample one perturbed daily profile while preserving legacy RNG behavior."""
    if T <= 0:
        raise ValueError("T must be positive.")
    if require_24_hours and T != 24:
        raise ValueError("electricity_price only supports 24-hour horizons.")

    path = DEFAULT_PRICE_PATH if price_path is None else Path(price_path)
    prices = pd.read_csv(path)["DA"].to_numpy(dtype=float) * 7.3 / 1000
    if prices.size % T:
        raise ValueError(f"{path} contains {prices.size} prices, not a multiple of T={T}.")
    profiles = prices.reshape(-1, T)

    source = np.random if rng is None else rng
    if hasattr(source, "integers"):
        index = int(source.integers(len(profiles)))
    else:
        index = int(source.randint(0, len(profiles)))
    return profiles[index] * source.uniform(0.5, 1.5, size=T)

