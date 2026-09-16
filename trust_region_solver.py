"""Compatibility entry point for the trust-region solver."""

from stackelberg.trust_region import (
    CompositeGlobalSurrogate, DistributionNetworkLeader, GlobalSurrogate,
    LowerSolver, ResidualModel, SurrogateModel, TRResidualController,
    TrustRegion, TrustRegionExperimentConfig,
    _LeaderSubproblem, _TempCorrectedSurrogate, _build_passive_profiles,
    _default_load_curve, _to_jsonable, _unique_rows, compute_ratio, decide_step,
    build_trust_region_experiment, electricity_price, extract_constants,
    extract_lower_constants, load_l2o_map_net, main,
    run_trust_region_experiment, sample_in_trust_region,
    save_trust_region_result, save_trust_region_true_responses,
)

__all__ = [
    "electricity_price", "extract_lower_constants", "LowerSolver",
    "extract_constants", "load_l2o_map_net", "SurrogateModel",
    "GlobalSurrogate", "CompositeGlobalSurrogate", "ResidualModel", "TrustRegion",
    "sample_in_trust_region", "compute_ratio", "decide_step",
    "DistributionNetworkLeader", "TRResidualController",
    "TrustRegionExperimentConfig", "build_trust_region_experiment",
    "run_trust_region_experiment", "save_trust_region_result",
    "save_trust_region_true_responses", "main",
]

if __name__ == "__main__":
    main()
