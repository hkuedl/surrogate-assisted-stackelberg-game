"""Compatibility exports for independent follower scenario modules."""

from __future__ import annotations

from typing import Any, Dict

import pandapower as pp

from .followers import (
    BuildingGroupFollowerProblem,
    CompositeFollowerProblem,
    FollowerProblem,
    IndustrialParkFollowerProblem,
    SingleBusFollowerProblem,
    build_composite_follower,
    build_registered_follower,
    create_industrial_park_network,
    extract_sb_constants,
    register_follower_scenario,
    registered_follower_scenarios,
    unregister_follower_scenario,
)


def extract_follower_constants(
    net: pp.pandapowerNet,
    T: int,
    *,
    scenario: str = "SB",
    verbose: bool = False,
    require_positive_load: bool = False,
    seed: int | None = None,
    load_scale: float = 1.0,
) -> Dict[str, Any]:
    """Compatibility wrapper for the SB-only constants extractor."""
    if scenario.upper() != "SB":
        raise NotImplementedError(
            f"The {scenario.upper()} follower constants require an independent extractor."
        )
    return extract_sb_constants(
        net,
        T,
        scenario=scenario,
        verbose=verbose,
        require_positive_load=require_positive_load,
        seed=seed,
        load_scale=load_scale,
    )


class CoptFollowerSolver(SingleBusFollowerProblem):
    """Compatibility constructor for the COPT-backed SB problem."""

    def __init__(
        self,
        lower_constants: Dict[str, Any],
        *,
        failure_mode: str = "none",
        include_metadata: bool = False,
        scenario: str = "SB",
        time_limit: float | None = None,
        mip_gap: float | None = None,
    ) -> None:
        if scenario.upper() not in {"SB", "LOWER_LEVEL"}:
            raise ValueError(
                "CoptFollowerSolver is the SB compatibility class and cannot "
                f"represent scenario {scenario!r}."
            )
        super().__init__(
            T=int(lower_constants.get("T", 24)),
            lower_constants=lower_constants,
            failure_mode=failure_mode,
            include_metadata=include_metadata,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )


# Preserve the old public import while downstream callers migrate names.
GurobiFollowerSolver = CoptFollowerSolver


__all__ = [
    "FollowerProblem",
    "SingleBusFollowerProblem",
    "BuildingGroupFollowerProblem",
    "IndustrialParkFollowerProblem",
    "create_industrial_park_network",
    "CompositeFollowerProblem",
    "CoptFollowerSolver",
    "GurobiFollowerSolver",
    "extract_sb_constants",
    "extract_follower_constants",
    "build_registered_follower",
    "build_composite_follower",
    "register_follower_scenario",
    "registered_follower_scenarios",
    "unregister_follower_scenario",
]
