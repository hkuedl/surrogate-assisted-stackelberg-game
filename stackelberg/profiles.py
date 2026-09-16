"""Demand profiles shared by follower and leader models."""

from __future__ import annotations

import numpy as np


def daily_load_curve(T: int, scenario: str = "SB") -> np.ndarray:
    """Return the normalized demand profile used by the existing solvers."""
    if T <= 0:
        raise ValueError("T must be positive.")
    scenario = scenario.upper()
    scenario = "IP" if scenario == "IE" else scenario
    if scenario not in {"SB", "BG", "IP"}:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from SB, BG, or IP.")

    hour = np.linspace(0.0, 24.0, T, endpoint=False)
    if scenario == "IP":
        curve = 0.82 + 0.30 * np.exp(-0.5 * ((hour - 13.0) / 4.0) ** 2)
    else:
        morning = 0.35 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
        evening = 0.55 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
        curve = 0.75 + 0.10 * np.cos(2.0 * np.pi * (hour - 14.0) / 24.0)
        curve = curve + morning + evening
    return np.clip(curve, 0.5, 1.5)

