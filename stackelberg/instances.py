"""Named microgrid-instance configuration shared by experiments and training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


BASE_SCENARIOS = ("SB", "BG", "IP")


@dataclass(frozen=True)
class FollowerInstanceSpec:
    """Map a stable experiment label to a base follower and its parameters."""

    scenario: str
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)


# This is the single edit point for the standard five-microgrid experiment.
# BG2's date can be changed here without touching solver, training, or runner code.
DEFAULT_INSTANCE_SPECS: dict[str, FollowerInstanceSpec] = {
    "SB1": FollowerInstanceSpec("SB"),
    "SB2": FollowerInstanceSpec("SB", {"load_scale": 1.10}),
    "BG1": FollowerInstanceSpec("BG", {"date": "2023-01-01"}),
    "BG2": FollowerInstanceSpec("BG", {"date": "2023-07-01"}),
    "IP1": FollowerInstanceSpec("IP", {"date": "2023-01-01"}),
}
DEFAULT_INSTANCE_IDS = tuple(DEFAULT_INSTANCE_SPECS)


def normalize_follower_name(name: str) -> str:
    """Normalize labels while retaining legacy SB/BG/IP compatibility."""
    value = str(name).strip().upper()
    return "IP" if value == "IE" else value


def scenario_type(name: str) -> str:
    """Return the registered base solver name for an instance label."""
    value = normalize_follower_name(name)
    spec = DEFAULT_INSTANCE_SPECS.get(value)
    return spec.scenario if spec is not None else value


def default_solver_kwargs(name: str) -> dict[str, Any]:
    """Return a mutable copy of the named instance's default solver settings."""
    value = normalize_follower_name(name)
    spec = DEFAULT_INSTANCE_SPECS.get(value)
    return {} if spec is None else deepcopy(dict(spec.solver_kwargs))


def resolve_follower_configuration(
    names: Sequence[str],
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Resolve labels, registered scenarios, and per-instance solver kwargs."""
    labels = [normalize_follower_name(name) for name in names]
    if follower_kwargs is None:
        overrides: list[Mapping[str, Any]] = [{} for _ in labels]
    else:
        overrides = list(follower_kwargs)
        if len(overrides) != len(labels):
            raise ValueError(
                "follower_kwargs and follower names must have the same length; "
                f"got {len(overrides)} and {len(labels)}."
            )

    scenario_types: list[str] = []
    resolved_kwargs: list[dict[str, Any]] = []
    for label, override in zip(labels, overrides):
        settings = default_solver_kwargs(label)
        settings.update(dict(override))
        scenario_types.append(scenario_type(label))
        resolved_kwargs.append(settings)
    return labels, scenario_types, resolved_kwargs


__all__ = [
    "BASE_SCENARIOS",
    "DEFAULT_INSTANCE_IDS",
    "DEFAULT_INSTANCE_SPECS",
    "FollowerInstanceSpec",
    "default_solver_kwargs",
    "normalize_follower_name",
    "resolve_follower_configuration",
    "scenario_type",
]
