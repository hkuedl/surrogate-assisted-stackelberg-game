"""Compatibility entry point for the ground-truth solver."""

from stackelberg.zero_order import (
    LowerSolver, UpperSolver, ZeroOrderExperimentConfig,
    _build_passive_profiles, _default_load_curve, build_zero_order_follower,
    electricity_price, extract_constants, extract_lower_constants, main,
    run_zero_order_experiment, save_zero_solver_result, solve_gt_optimal,
    test_gt_solver_smoke,
)

__all__ = [
    "electricity_price", "extract_lower_constants", "extract_constants",
    "UpperSolver", "LowerSolver", "solve_gt_optimal",
    "ZeroOrderExperimentConfig", "run_zero_order_experiment",
    "build_zero_order_follower", "save_zero_solver_result",
    "test_gt_solver_smoke", "main",
]

if __name__ == "__main__":
    main()
