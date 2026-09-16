"""Independent follower formulations and registration API."""

from .base import (
    FollowerProblem,
    build_registered_follower,
    register_follower_scenario,
    registered_follower_scenarios,
    unregister_follower_scenario,
)
from .bg import BuildingGroupFollowerProblem
from .composite import CompositeFollowerProblem, build_composite_follower
from .ip import (
    IndustrialParkFollowerProblem,
    create_industrial_park_network,
    extract_ip_constants,
)
from .sb import SingleBusFollowerProblem, extract_sb_constants

register_follower_scenario("SB", SingleBusFollowerProblem, replace=True)
register_follower_scenario(
    "IP", IndustrialParkFollowerProblem, aliases=("IE",), replace=True
)

__all__ = [
    "FollowerProblem",
    "SingleBusFollowerProblem",
    "BuildingGroupFollowerProblem",
    "IndustrialParkFollowerProblem",
    "CompositeFollowerProblem",
    "extract_sb_constants",
    "extract_ip_constants",
    "create_industrial_park_network",
    "build_registered_follower",
    "build_composite_follower",
    "register_follower_scenario",
    "registered_follower_scenarios",
    "unregister_follower_scenario",
]
