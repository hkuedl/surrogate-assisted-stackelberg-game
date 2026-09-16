"""Scenario-independent follower interface and plugin registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
import pandapower as pp

FollowerFactory = Callable[..., "FollowerProblem"]
_FOLLOWER_FACTORIES: Dict[str, FollowerFactory] = {}
_RESERVED_SCENARIOS = {"SB", "BG", "IP"}


class FollowerProblem(ABC):
    """Contract shared by otherwise independent follower formulations."""

    scenario: str

    @abstractmethod
    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any] | None:
        """Solve this scenario's optimization problem for a price trajectory."""


def register_follower_scenario(
    name: str,
    factory: FollowerFactory,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    """Register an independent scenario factory."""
    for scenario_name in (name, *aliases):
        key = scenario_name.upper()
        if key in _FOLLOWER_FACTORIES and not replace:
            raise ValueError(f"Follower scenario {key!r} is already registered.")
        _FOLLOWER_FACTORIES[key] = factory


def build_registered_follower(
    scenario: str,
    T: int,
    net: pp.pandapowerNet | None = None,
    **kwargs: Any,
) -> FollowerProblem:
    """Build a scenario-specific follower through its registered factory."""
    key = scenario.upper()
    factory = _FOLLOWER_FACTORIES.get(key)
    if factory is not None:
        return factory(T=T, net=net, **kwargs)
    if key in _RESERVED_SCENARIOS or key == "IE":
        canonical = "IP" if key == "IE" else key
        raise NotImplementedError(
            f"The {canonical} follower optimization problem is not implemented. "
            f"Implement an independent {canonical} FollowerProblem and register it "
            "with register_follower_scenario()."
        )
    raise ValueError(f"Unknown follower scenario {scenario!r}.")


def registered_follower_scenarios() -> tuple[str, ...]:
    """Return the scenario names that currently have implementations."""
    return tuple(sorted(_FOLLOWER_FACTORIES))


def unregister_follower_scenario(name: str) -> None:
    """Remove a scenario registration."""
    _FOLLOWER_FACTORIES.pop(name.upper(), None)

