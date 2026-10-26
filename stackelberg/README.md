# Stackelberg Solver Implementation

This directory contains the reusable implementation of the surrogate-based Stackelberg coordination framework.

For the paper overview, installation instructions, and saved-result plotting workflow, see the [repository README](../README.md).

The package implements follower-response models, neural-surrogate optimization, CMA-ES optimization, local residual correction, and experiment utilities.

## Module Guide

| Module | Responsibility |
|---|---|
| `trust_region.py` | Surrogate models, residual fitting, trust-region updates, and the main experiment API |
| `nn_solver.py` | Leader optimization using neural response surrogates |
| `zero_order.py` | CMA-ES search with actual follower-response evaluations |
| `distflow.py` | COPT-based network evaluation and differentiable DistFlow penalties |
| `network.py` | Network construction, topology, and conversion to solver constants |
| `instances.py` | Named MG instances and their parameter overrides |
| `l2o.py` | Loading trained neural surrogates and architecture metadata |
| `follower.py` | Compatibility exports for follower implementations |
| `followers/` | Independent follower formulations and their composition |
| `tr_ablation.py` | Four-group trust-region mechanism experiments and analysis |
| `mv_oberrhein_scaling.py` | Construction and execution of the 108-bus scalability study |
| `trust_region_b.py` | Alternative leader implementation with COPT network-state initialization |
| `pricing.py` | Electricity-price profiles and sampling |
| `profiles.py` | Shared demand profiles |
| `randomness.py` | Shared random-seed utilities |
| `__init__.py` | Package initialization and exported random-state utilities |

The standard trust-region workflow uses `trust_region.py`.

## Follower Models

| Module | Model |
|---|---|
| `followers/base.py` | `FollowerProblem` interface and scenario registry |
| `followers/sb.py` | Single-bus MG |
| `followers/bg.py` | Building-group MG with an internal residential feeder |
| `followers/ip.py` | Industrial-park MG with process scheduling and an internal feeder |
| `followers/composite.py` | Ordered collection of independently priced followers |
| `followers/__init__.py` | Public follower exports and registration |

Each follower implements:

```python
lower_solve(pi)
```

A successful call returns a dictionary containing the active-power response under the key `p_mg`. Additional entries depend on the follower model.

The standard experiment instances are defined in `instances.py`:

| Instance | Base model | Instance setting |
|---|---|---|
| `SB1` | `SB` | Base single-bus configuration |
| `SB2` | `SB` | Load scaled by `1.10` |
| `BG1` | `BG` | Building profiles for `2023-01-01` |
| `BG2` | `BG` | Building profiles for `2023-07-01` |
| `IP1` | `IP` | Industrial-park configuration for `2023-01-01` |

Instance labels also determine model filenames, such as `NNmodel/L2O_BG1.pth`.

## Data Shapes and Ordering

For a horizon of `T` periods:

- A single follower receives a price vector with shape `(T,)`.
- A composite follower accepts prices with shape `(N, T)` or a flattened vector with shape `(N * T,)`.
- Composite follower responses under `p_mg` have shape `(N, T)`.
- Flattening uses follower-major order: all periods for the first follower, then all periods for the second follower, and so on.

The main experiments use `T = 24`. Five followers therefore produce a 120-dimensional leader price vector.

The entries of `scenarios`, `lower_nets`, `mg_buses`, and any supplied per-follower models must refer to the same followers in the same order.

## Trust-Region Components

The main implementation is in `trust_region.py`.

| Component | Purpose |
|---|---|
| `SurrogateModel` | Common surrogate interface |
| `GlobalSurrogate` | Neural response model with residual corrections |
| `CompositeGlobalSurrogate` | Collection of follower-specific surrogates |
| `ResidualModel` | Local residual fitting, including RBF models |
| `DistributionNetworkLeader` | Surrogate leader optimization and true-objective evaluation |
| `TRResidualController` | Sampling, correction, candidate evaluation, acceptance, and radius updates |
| `TrustRegionExperimentConfig` | Experiment configuration |
| `build_trust_region_experiment` | Construction of a controller and its metadata |
| `run_trust_region_experiment` | Execution and optional result export |

A standard iteration proceeds as follows:

1. Construct local sampling designs around the current prices.
2. Obtain follower responses and form prediction residuals.
3. Fit one local residual model per follower.
4. Solve the corrected surrogate leader problem within the trust region.
5. Evaluate the candidate using actual follower responses and the network evaluator.
6. Apply the acceptance policy and update the trust-region radius.

The neural-network weights remain fixed during local residual correction. Surrogate retraining is handled separately by the root-level training code.

Surrogate leader optimization uses differentiable network penalties. True-objective evaluation uses the COPT-based network formulation in `distflow.py`.

## Example: Run the Standard Five-Follower Setup

Run this example from the repository root after installing the experiment dependencies and training the five surrogate models.

`build_config` supplies the standard network, follower mapping, and solver settings used by the mechanism-ablation workflow.

```python
from pathlib import Path

from stackelberg.tr_ablation import build_config, load_warm_start
from stackelberg.trust_region import run_trust_region_experiment

warm_start, nn_objective = load_warm_start(
    Path("results/rng42/NN_solver_history_20260914-120155.csv")
)

config = build_config(
    warm_start,
    problem_seed=42,
    trial_seed=42,
    radius_policy="adaptive",
    acceptance_policy="filter",
    max_iterations=3,
    steps_warm=100,
    output_dir=Path("results/example_tr"),
)

result = run_trust_region_experiment(config)

print("Initial NN objective:", nn_objective)
print("Price-vector shape:", result["x_opt"].shape)
print("Completed iterations:", result["n_iter"])
print("Stopping reason:", result["stop_reason"])
print("Saved files:", result.get("saved_paths"))
```

This example uses a short iteration budget. Use the experiment notebooks and archived parameter records for the settings of a particular study.

For custom systems, construct a `TrustRegionExperimentConfig` with the desired `upper_net`, `lower_nets`, `mg_buses`, and `scenarios`.

## Configuration

| Setting | Meaning |
|---|---|
| `T` | Number of time periods |
| `seed` | Experiment random seed |
| `x0` | Initial leader price vector |
| `tr_params` | Initial and maximum radius, radius-update factors, and acceptance thresholds |
| `algo_params["n_samples"]` | Local design size per follower |
| `algo_params["K_max"]` | Maximum number of trust-region iterations |
| `algo_params["solve_params"]` | Initial/local optimization steps, learning rate, and penalty weights |
| `residual_model_params` | Residual-model type and fitting parameters |
| `radius_policy` | `"adaptive"` or `"fixed"` |
| `acceptance_policy` | `"filter"` or `"always_accept"` |
| `fixed_radius` | Radius used by the fixed-radius policy |
| `max_consecutive_rejections` | Rejection-based stopping limit |
| `copt_initialize_network_state` | Optional COPT initialization of local network variables |
| `save_results`, `output_dir` | Result-export settings |

The mechanism-ablation groups are:

| Group | Radius | Acceptance |
|---|---|---|
| `adaptive_filter` | Adaptive | Filtered |
| `adaptive_always_accept` | Adaptive | Unconditional |
| `fixed_filter` | Fixed | Filtered |
| `fixed_always_accept` | Fixed | Unconditional |

Use a new output directory when changing the configuration of an ablation experiment. Completed-group manifests are used to check configuration consistency during resumed runs.

## Result Format

`run_trust_region_experiment` returns a dictionary that includes:

| Key | Content |
|---|---|
| `x_init` | Prices after the initial surrogate solve |
| `x_opt` | Returned leader price vector |
| `history` | Per-iteration records |
| `n_iter` | Number of completed iterations |
| `converged` | Convergence flag |
| `stop_reason` | Reason for termination |
| `params` | Experiment metadata |
| `elapsed_seconds` | Measured controller runtime |
| `saved_paths` | Export paths, when files are saved |

When result saving is enabled, the runner produces:

```text
trust_region_result_<timestamp>.json
trust_region_history_<timestamp>.csv
```

The JSON stores decisions, configuration metadata, and detailed history. The CSV stores iteration metrics for analysis.

Important history fields include:

- `F_current`, `F_candidate`, and `F_next`: true objective values.
- `F_tilde_current` and `F_tilde_candidate`: surrogate objective values.
- `rho`: agreement between actual and predicted improvement.
- `decision`: update decision.
- `R_before` and `R_after`: trust-region radii.
- `composite_queries` and `cumulative_composite_queries`: communication accounting.

The code minimizes `F`; profit is represented by `-F`. The best observed candidate and the returned solution can differ, particularly under unconditional acceptance.

## Dependencies on Root-Level Modules

This directory is part of the repository and relies on root-level modules, including:

- `L2Otraining.py` for neural-network definitions and model locations.
- `follower_solver.py` for follower construction interfaces.
- `buildings.py` for building-profile utilities.
- `zero_solver.py` for compatibility imports used by the neural solver.

Keep these modules alongside `stackelberg/` and execute scripts or notebooks from the repository root.

Trained models are loaded from `NNmodel/L2O_<instance>.pth`, with corresponding `.json` metadata. Missing checkpoint files should be generated using the training workflow documented in the [main README](../README.md).
