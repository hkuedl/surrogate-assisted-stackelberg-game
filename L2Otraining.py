"""
    This script is used to train the surrogate model for the lower-solver problem.
"""


from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from stackelberg.randomness import set_global_seed
from stackelberg.instances import (
    BASE_SCENARIOS,
    DEFAULT_INSTANCE_IDS,
    normalize_follower_name,
    scenario_type,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "NNmodel"
SUPPORTED_SCENARIOS = BASE_SCENARIOS
SUPPORTED_INSTANCES = DEFAULT_INSTANCE_IDS
SUPPORTED_PROBLEMS = (*SUPPORTED_SCENARIOS, *SUPPORTED_INSTANCES)
PRICE_MIN = 0.0
PRICE_MAX = 1.5
POWER_MIN_MW = 0.0
POWER_MAX_MW = 10.0


def normalize_price(x: torch.Tensor) -> torch.Tensor:
    """Map raw electricity prices from [0, 1.5] to [-1, 1]."""
    return 2.0 * (x - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) - 1.0


def normalize_power(y: torch.Tensor) -> torch.Tensor:
    """Map physical microgrid power from [0, 10] MW to [0, 1]."""
    return (y - POWER_MIN_MW) / (POWER_MAX_MW - POWER_MIN_MW)


def denormalize_power(y: torch.Tensor) -> torch.Tensor:
    """Map normalized microgrid power back to MW."""
    return y * (POWER_MAX_MW - POWER_MIN_MW) + POWER_MIN_MW


@dataclass(frozen=True)
class TrainingConfig:
    """ Tranining parameters """
    scenario: str
    hidden_dim: int = 128
    residual_blocks: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 500
    validation_fraction: float = 0.2
    seed: int = 42


def _scenario_name(scenario: str) -> str:
    name = normalize_follower_name(scenario)
    if name not in SUPPORTED_PROBLEMS:
        raise ValueError(
            f"Unknown follower {name!r}; choose from {SUPPORTED_PROBLEMS}."
        )
    return name


def _tagged_stem(prefix: str, scenario: str, tag: str | None = None) -> str:
    return f"{prefix}_{scenario}" if not tag else f"{prefix}_{scenario}_{tag}"


def _load_arrays(scenario: str, data_tag: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    stem = _tagged_stem("X", scenario, data_tag), _tagged_stem("Y", scenario, data_tag)
    x_path, y_path = DATA_DIR / f"{stem[0]}.npy", DATA_DIR / f"{stem[1]}.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Generate {x_path.name} and {y_path.name} before training {scenario}.")
    x_data, y_data = np.load(x_path).astype(np.float32), np.load(y_path).astype(np.float32)
    if x_data.ndim != 2 or y_data.ndim != 2 or x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"Expected aligned 2-D arrays, got X{x_data.shape} and Y{y_data.shape}.")
    return x_data, y_data


def load_data(
    scenario: str,
    data_tag: str | None = None,
    *,
    include_original: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load matching generated data and validate its sample dimensions.

    Pass ``data_tag="tr"`` to load trust-region samples such as ``X_SB_tr.npy``.
    Set ``include_original=True`` with a ``data_tag`` to concatenate the original
    ``X_{scenario}.npy`` / ``Y_{scenario}.npy`` files with the tagged samples.
    """
    scenario = _scenario_name(scenario)
    tags: list[str | None] = []
    if include_original and data_tag is not None:
        tags.append(None)
    tags.append(data_tag)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for tag in tags:
        x_data, y_data = _load_arrays(scenario, tag)
        if x_parts and x_data.shape[1:] != x_parts[0].shape[1:]:
            raise ValueError(
                f"Feature mismatch while concatenating {scenario} data: "
                f"{x_parts[0].shape} vs {x_data.shape}."
            )
        if y_parts and y_data.shape[1:] != y_parts[0].shape[1:]:
            raise ValueError(
                f"Target mismatch while concatenating {scenario} data: "
                f"{y_parts[0].shape} vs {y_data.shape}."
            )
        x_parts.append(x_data)
        y_parts.append(y_data)

    x_data = np.concatenate(x_parts, axis=0) if len(x_parts) > 1 else x_parts[0]
    y_data = np.concatenate(y_parts, axis=0) if len(y_parts) > 1 else y_parts[0]
    if len(x_parts) > 1:
        counts = " + ".join(
            f"{'original' if tag is None else tag} {part.shape[0]}"
            for tag, part in zip(tags, x_parts)
        )
        print(f"{scenario}: concatenated {counts} samples -> {x_data.shape[0]}")
    return torch.from_numpy(x_data), torch.from_numpy(y_data)


class ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU(0.2),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class L2OSolver(nn.Module):
    """Residual neural map from raw prices to a bounded response in MW."""

    def __init__(
        self, input_dim: int = 24, hidden_dim: int = 128, output_dim: int = 24, residual_blocks: int = 3
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.res_layers = nn.Sequential(*(ResBlock(hidden_dim) for _ in range(residual_blocks)))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized power from prices in physical units."""
        normalized_x = normalize_price(x)
        return self.final_layer(
            self.res_layers(self.first_layer(normalized_x))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return bounded microgrid power in [0, 10] MW."""
        physical_power = denormalize_power(self.forward_normalized(x))
        return torch.clamp(physical_power, min=POWER_MIN_MW, max=POWER_MAX_MW)


def _data_loaders(
    x_tensor: torch.Tensor, y_tensor: torch.Tensor, config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int((1.0 - config.validation_fraction) * len(dataset))
    if not 0 < train_size < len(dataset):
        raise ValueError("validation_fraction must leave training and validation samples.")
    train_set, validation_set = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    return (
        DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        ),
        DataLoader(validation_set, batch_size=config.batch_size, shuffle=False),
    )


def _mean_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            total += criterion(model(x_batch.to(device)), y_batch.to(device)).item()
    return total / len(loader)


def train(
    scenario: str = "SB",
    *,
    config: Optional[TrainingConfig] = None,
    type: Optional[str] = None,
    data_tag: Optional[str] = None,
    model_tag: Optional[str] = None,
    include_original: bool = False,
) -> Dict[str, float]:
    """Train and save one scenario model, for example ``train("BG")``.

    Pass ``data_tag="tr"`` to train from trust-region samples. Set
    ``include_original=True`` to concatenate ``X_{scenario}.npy`` with
    ``X_{scenario}_tr.npy`` (and the matching ``Y`` files). ``model_tag``
    defaults to ``data_tag`` so ``train("SB", data_tag="tr")`` writes
    ``L2O_SB_tr.pth`` instead of overwriting the original checkpoint.
    """
    scenario = _scenario_name(type or scenario)
    config = config or TrainingConfig(scenario=scenario)
    if _scenario_name(config.scenario) != scenario:
        raise ValueError("config.scenario must match the requested scenario.")
    if model_tag is None:
        model_tag = data_tag

    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor, y_tensor = load_data(
        scenario, data_tag=data_tag, include_original=include_original
    )
    train_loader, validation_loader = _data_loaders(x_tensor, y_tensor, config)
    model = L2OSolver(
        input_dim=x_tensor.shape[1], output_dim=y_tensor.shape[1],
        hidden_dim=config.hidden_dim, residual_blocks=config.residual_blocks,
    ).to(device)
    optimizer, criterion = optim.Adam(model.parameters(), lr=config.learning_rate), nn.MSELoss()
    train_loss = validation_loss = float("nan")

    for epoch in range(config.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            prediction = model.forward_normalized(x_batch.to(device))
            target = normalize_power(y_batch.to(device)).clamp(0.0, 1.0)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            train_loss = _mean_loss(model, train_loader, criterion, device)
            validation_loss = _mean_loss(model, validation_loader, criterion, device)
            print(f"{scenario} epoch {epoch + 1:>4}/{config.epochs}: "
                  f"train={train_loss:.6f}, validation={validation_loss:.6f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_stem = _tagged_stem("L2O", scenario, model_tag)
    torch.save(model.state_dict(), MODEL_DIR / f"{model_stem}.pth")
    metadata = {
        **asdict(config),
        "scenario": scenario_type(scenario),
        "follower_id": scenario,
        "data_tag": data_tag,
        "model_tag": model_tag,
        "include_original": include_original,
        "input_dim": x_tensor.shape[1],
        "output_dim": y_tensor.shape[1],
        "price_min": PRICE_MIN,
        "price_max": PRICE_MAX,
        "power_min_mw": POWER_MIN_MW,
        "power_max_mw": POWER_MAX_MW,
    }
    (MODEL_DIR / f"{model_stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"train_mse": train_loss, "validation_mse": validation_loss}


def evaluate(
    scenario: str = "SB",
    *,
    type: Optional[str] = None,
    data_tag: Optional[str] = None,
    model_tag: Optional[str] = None,
    include_original: bool = False,
) -> Dict[str, float]:
    """Evaluate the saved model on all generated samples for one scenario."""
    scenario = _scenario_name(type or scenario)
    if model_tag is None:
        model_tag = data_tag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor, y_tensor = load_data(
        scenario, data_tag=data_tag, include_original=include_original
    )
    model_stem = _tagged_stem("L2O", scenario, model_tag)
    model_path, metadata_path = MODEL_DIR / f"{model_stem}.pth", MODEL_DIR / f"{model_stem}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    model = L2OSolver(
        input_dim=x_tensor.shape[1], output_dim=y_tensor.shape[1],
        hidden_dim=int(metadata.get("hidden_dim", 128)),
        residual_blocks=int(metadata.get("residual_blocks", 3)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor.to(device)).cpu().numpy()
    target = y_tensor.numpy()
    metrics = {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "max_ae": float(np.max(np.abs(prediction - target))),
        "mean_relative_error": float(np.mean(np.abs(prediction - target) / (np.abs(target) + 1e-7))),
    }
    print(f"{scenario} evaluation: " + ", ".join(f"{name}={value:.6g}" for name, value in metrics.items()))
    return metrics


# Compatibility with existing scripts that call ``eval(type="SB")``.
eval = evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("scenario", choices=SUPPORTED_PROBLEMS)
    args = parser.parse_args()
    train(args.scenario) if args.action == "train" else evaluate(args.scenario)
