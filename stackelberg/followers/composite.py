"""
    Composite follower for multiple independently priced lower networks.
    When encountering a composite follower, the upper solver will solve for the optimal prices for all followers,
    and then pass the prices to the lower solver for each follower.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Mapping

import numpy as np
import pandapower as pp

from .base import FollowerProblem, build_registered_follower
from ..instances import resolve_follower_configuration


class CompositeFollowerProblem(FollowerProblem):
    """Evaluate one independent follower per microgrid connection."""

    scenario = "COMPOSITE"

    def __init__(
        self,
        followers: Sequence[FollowerProblem],
        T: int,
        *,
        follower_ids: Sequence[str] | None = None,
    ) -> None:
        self.followers = list(followers)
        self.T = int(T)
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if not self.followers:
            raise ValueError("At least one follower is required.")
        for index, follower in enumerate(self.followers):
            follower_T = getattr(follower, "T", self.T)
            if int(follower_T) != self.T:
                raise ValueError(
                    f"Follower {index} uses T={follower_T}, expected T={self.T}."
                )
        self.n_followers = len(self.followers)
        self.follower_ids = (
            [str(value).upper() for value in follower_ids]
            if follower_ids is not None
            else [str(getattr(follower, "scenario", index)).upper()
                  for index, follower in enumerate(self.followers)]
        )
        if len(self.follower_ids) != self.n_followers:
            raise ValueError("follower_ids and followers must have the same length.")
        self.individual_follower_solve_count = 0

    def _reshape_prices(self, pi: np.ndarray) -> np.ndarray:
        prices = np.asarray(pi, dtype=float)
        if prices.shape == (self.n_followers, self.T):
            return prices
        if prices.shape == (self.n_followers * self.T,):
            return prices.reshape(self.n_followers, self.T)
        if self.n_followers == 1 and prices.shape == (self.T,):
            return prices.reshape(1, self.T)
        raise ValueError(
            f"Expected prices with shape ({self.n_followers}, {self.T}) or "
            f"({self.n_followers * self.T},), got {prices.shape}."
        )

    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any] | None:
        prices = self._reshape_prices(pi)
        results = []
        for index, follower in enumerate(self.followers):
            self.individual_follower_solve_count += 1
            result = follower.lower_solve(prices[index])
            if result is not None:
                result = dict(result)
                result["follower_id"] = self.follower_ids[index]
            results.append(result)
        if any(result is None for result in results):
            return None
        responses = np.stack(
            [np.asarray(result["p_mg"], dtype=float).reshape(self.T) for result in results]
        )
        return {
            "p_mg": responses,
            "scenario": self.scenario,
            "follower_results": results,
        }


def build_composite_follower(
    scenarios: Sequence[str],
    lower_nets: Sequence[pp.pandapowerNet],
    T: int,
    *,
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> CompositeFollowerProblem:
    """Build a one-to-one ordered collection of registered followers."""
    follower_ids, scenario_list, instance_kwargs = resolve_follower_configuration(
        scenarios, follower_kwargs
    )
    network_list = list(lower_nets)
    if len(scenario_list) != len(network_list):
        raise ValueError(
            "scenarios and lower_nets must have the same length; "
            f"got {len(scenario_list)} and {len(network_list)}."
        )
    followers = []
    for scenario, net, specific_kwargs in zip(
        scenario_list, network_list, instance_kwargs
    ):
        combined_kwargs = {**kwargs, **specific_kwargs}
        follower = build_registered_follower(
            scenario, T=T, net=net, **combined_kwargs
        )
        follower.follower_id = follower_ids[len(followers)]
        followers.append(follower)
    return CompositeFollowerProblem(followers, T=T, follower_ids=follower_ids)
