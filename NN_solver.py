"""Compatibility entry point for the neural-surrogate solver."""

import time

from stackelberg.randomness import DEFAULT_SEED, set_global_seed
from stackelberg.nn_solver import (
    CompositeMapNet, NNSolverExperimentConfig, TorchDistFlowOptimizer,
    _build_passive_profiles, _default_load_curve, electricity_price,
    evaluate_true_objective, extract_constants, load_l2o_map_net,
    run_nn_solver_experiment, solve_with_torch, test_extract_constants,
    test_torch_distflow_smoke,
)

__all__ = [
    "load_l2o_map_net", "electricity_price", "extract_constants",
    "CompositeMapNet", "TorchDistFlowOptimizer", "solve_with_torch",
    "evaluate_true_objective", "NNSolverExperimentConfig",
    "run_nn_solver_experiment", "test_extract_constants",
    "test_torch_distflow_smoke",
]

if __name__ == "__main__":
    set_global_seed(DEFAULT_SEED)
    start_time = time.time()
    test_torch_distflow_smoke(steps=5000, lr=1e-2, seed=DEFAULT_SEED)
    print(f"Time: {time.time() - start_time:.1f}s")
