"""Loading utilities for trained learning-to-optimize surrogate models."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from L2Otraining import L2OSolver, MODEL_DIR


def load_l2o_model(
    model_type: str = "SB",
    device: Optional[torch.device] = None,
    *,
    model_dir: Path = MODEL_DIR,
    model_tag: Optional[str] = None,
) -> nn.Module:
    """Load an instance-specific L2O model and its architecture metadata.

    Pass ``model_tag="tr"`` to load a checkpoint retrained on trust-region
    samples, such as ``L2O_SB_tr.pth``.
    """
    scenario = model_type.upper()
    scenario = "IP" if scenario == "IE" else scenario
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stem = f"L2O_{scenario}" if not model_tag else f"L2O_{scenario}_{model_tag}"
    checkpoint_path = Path(model_dir) / f"{stem}.pth"
    metadata_path = Path(model_dir) / f"{stem}.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing L2O checkpoint: {checkpoint_path}")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    model = L2OSolver(
        input_dim=int(metadata.get("input_dim", 24)),
        hidden_dim=int(metadata.get("hidden_dim", 128)),
        output_dim=int(metadata.get("output_dim", 24)),
        residual_blocks=int(metadata.get("residual_blocks", 3)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_l2o_models(
    scenarios: Sequence[str],
    device: Optional[torch.device] = None,
    *,
    model_dir: Path = MODEL_DIR,
    model_tag: Optional[str] = None,
) -> list[nn.Module]:
    """Load one conventionally named L2O checkpoint per follower ID."""
    return [
        load_l2o_model(
            scenario, device=device, model_dir=model_dir, model_tag=model_tag
        )
        for scenario in scenarios
    ]
