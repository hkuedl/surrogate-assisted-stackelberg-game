"""Reusable lower-level follower solvers for surrogate-data generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from stackelberg.follower import (
    BuildingGroupFollowerProblem,
    CompositeFollowerProblem,
    FollowerProblem,
    CoptFollowerSolver,
    IndustrialParkFollowerProblem,
    build_composite_follower,
    build_registered_follower,
    extract_follower_constants,
    register_follower_scenario,
    registered_follower_scenarios,
    unregister_follower_scenario,
)
from stackelberg.profiles import daily_load_curve
from stackelberg.instances import (
    default_solver_kwargs,
    normalize_follower_name,
    scenario_type,
)


def _load_curve(T: int, scenario: str) -> np.ndarray:
    return daily_load_curve(T, scenario)


def extract_constants(net: pp.pandapowerNet, T: int, scenario: str = "SB") -> Dict[str, Any]:
    if scenario.upper() != "SB":
        raise NotImplementedError(
            f"The {scenario.upper()} follower constants are not implemented."
        )
    constants = extract_follower_constants(
        net, T, scenario="SB", require_positive_load=True
    )
    constants.pop("load_curve")
    constants.pop("verbose")
    constants.pop("random_seed")
    return constants


class LowerSolver(CoptFollowerSolver):
    """Scenario-aware follower with the data-generation compatibility API."""

    scenario = "SB"

    def __init__(
        self,
        T: int = 24,
        net: Optional[pp.pandapowerNet] = None,
        verbose: bool = False,
        *,
        failure_mode: str = "raise",
        include_metadata: bool = True,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        lower_constants: Dict[str, Any] | None = None,
        load_scale: float = 1.0,
    ) -> None:
        network = self.default_network() if net is None else net
        constants = (
            extract_follower_constants(
                network,
                T,
                scenario=self.scenario,
                verbose=verbose,
                require_positive_load=True,
                load_scale=load_scale,
            )
            if lower_constants is None
            else dict(lower_constants)
        )
        super().__init__(
            constants,
            failure_mode=failure_mode,
            include_metadata=include_metadata,
            scenario=self.scenario,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        self.net = network
        self.constants.pop("load_curve", None)
        self.constants.pop("verbose", None)
        self.constants.pop("random_seed", None)

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        return pn.create_cigre_network_lv()


class SingleBusSolver(LowerSolver):
    scenario = "SB"


class BuildingGroupSolver(BuildingGroupFollowerProblem):
    """Independent BG extension point; it does not inherit the SB solver."""
    scenario = "BG"

class IndustrialParkSolver(IndustrialParkFollowerProblem):
    """Independent IP extension point; it does not inherit the SB solver."""
    scenario = "IP"


SOLVER_CLASSES: Dict[str, Type[FollowerProblem]] = {
    "SB": SingleBusSolver,
    "BG": BuildingGroupSolver,
    "IP": IndustrialParkSolver,
    "IE": IndustrialParkSolver,
}

register_follower_scenario("SB", SingleBusSolver, replace=True)
register_follower_scenario("BG", BuildingGroupSolver, replace=True)
register_follower_scenario(
    "IP", IndustrialParkSolver, aliases=("IE",), replace=True
)


def build_solver(
    problem_type: str,
    T: int = 24,
    net: Optional[pp.pandapowerNet] = None,
    **kwargs: Any,
) -> FollowerProblem:
    resolved_kwargs = default_solver_kwargs(problem_type)
    resolved_kwargs.update(kwargs)
    follower = build_registered_follower(
        scenario_type(problem_type), T=T, net=net, **resolved_kwargs
    )
    follower.follower_id = normalize_follower_name(problem_type)
    return follower


def solver(
    problem_type: str, input: Dict[str, Any], net: Optional[pp.pandapowerNet] = None
) -> Dict[str, Any]:
    pi = np.asarray(input["pi"], dtype=float)
    return build_solver(
        problem_type, T=pi.size, net=net, verbose=bool(input.get("verbose", False))
    ).lower_solve(pi)
