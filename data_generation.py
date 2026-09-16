import argparse
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union
import time
from stackelberg.randomness import DEFAULT_SEED, make_rng, set_global_seed
from stackelberg.instances import (
    BASE_SCENARIOS,
    DEFAULT_INSTANCE_IDS,
    default_solver_kwargs,
    normalize_follower_name,
    scenario_type,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPORTED_SCENARIOS = BASE_SCENARIOS
SUPPORTED_INSTANCES = DEFAULT_INSTANCE_IDS
SUPPORTED_PROBLEMS = (*SUPPORTED_SCENARIOS, *SUPPORTED_INSTANCES)


def electricity_price(T: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one perturbed day-ahead price trajectory from the local data."""
    price_path = DATA_DIR / "DA_RT_data.csv"
    day_ahead_price = pd.read_csv(price_path)["DA"].to_numpy(dtype=float) * 7.3 / 1000
    if day_ahead_price.size % T:
        raise ValueError(f"{price_path} contains {day_ahead_price.size} prices, not a multiple of T={T}.")
    daily_profiles = day_ahead_price.reshape(-1, T)
    profile = daily_profiles[rng.integers(len(daily_profiles))]
    return profile * rng.uniform(0.5, 1.5, size=T)


def _resolve_module(solver_module: Union[str, ModuleType]) -> ModuleType:
    return importlib.import_module(solver_module) if isinstance(solver_module, str) else solver_module


def _build_lower_solver(
    solver_module: ModuleType,
    problem_type: str,
    T: int,
    net: Any,
    solver_kwargs: Dict[str, Any],
) -> Any:
    """Build a solver module's lower solver once, before the sampling loop.

    Solver modules used here must expose ``build_solver(problem_type, T, ...)``
    and return an object with ``lower_solve(pi)``.  This mirrors
    ``zero_solver.LowerSolver.lower_solve`` while allowing each scenario to use
    a different solver module.
    """
    factory = getattr(solver_module, "build_solver", None)
    if factory is None:
        raise TypeError(
            f"{solver_module.__name__} must define build_solver(problem_type, T, ...). "
            "See follower_solver.build_solver for the required interface."
        )
    return factory(problem_type=problem_type, T=T, net=net, **solver_kwargs)


def data_generation(
    T: int,
    Nsample: int,
    problem_type: str = "SB",
    solver_module: Union[str, ModuleType] = "follower_solver",
    *,
    net: Any = None,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = DEFAULT_SEED,
    type: Optional[str] = None,
    checkpoint_interval: int = 100,
    max_consecutive_failures: int = 20,
    reuse_existing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``price -> aggregate grid import`` labels for one follower.

    ``type=`` is retained as a deprecated alias for compatibility with the
    previous script.  Pass a module name or module object through
    ``solver_module`` to switch to another lower-level implementation.
    """
    if type is not None:
        problem_type = type
    follower_id = normalize_follower_name(problem_type)
    if follower_id not in SUPPORTED_PROBLEMS:
        raise ValueError(
            f"Unknown follower {follower_id!r}; choose from {SUPPORTED_PROBLEMS}."
        )
    scenario = scenario_type(follower_id)
    if Nsample <= 0:
        raise ValueError("Nsample must be positive.")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be positive.")

    seed = set_global_seed(seed)
    rng = make_rng(seed)
    module = _resolve_module(solver_module)
    resolved_solver_kwargs = default_solver_kwargs(follower_id)
    resolved_solver_kwargs.update(solver_kwargs or {})
    X = np.empty((Nsample, T), dtype=float)
    Y = np.empty((Nsample, T), dtype=float)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    x_path = DATA_DIR / f"X_{follower_id}.npy"
    y_path = DATA_DIR / f"Y_{follower_id}.npy"
    completed = 0
    if reuse_existing:
        if x_path.exists() != y_path.exists():
            raise FileNotFoundError(
                f"Existing data must contain both {x_path.name} and {y_path.name}."
            )
        if x_path.is_file():
            existing_x = np.load(x_path).astype(float)
            existing_y = np.load(y_path).astype(float)
            if (
                existing_x.ndim != 2
                or existing_y.ndim != 2
                or existing_x.shape != existing_y.shape
                or existing_x.shape[1] != T
            ):
                raise ValueError(
                    f"Invalid existing data shapes X{existing_x.shape}, "
                    f"Y{existing_y.shape} for T={T}."
                )
            completed = min(len(existing_x), Nsample)
            X[:completed] = existing_x[:completed]
            Y[:completed] = existing_y[:completed]
            print(
                f"Reused {completed}/{Nsample} existing samples for {follower_id}."
            )
            if completed == Nsample:
                return X, Y
            # Continue the deterministic price stream instead of repeating its
            # first ``completed`` samples when extending a partial dataset.
            rng.uniform(0.0, 1.5, size=(completed, T))

    lower_solver = _build_lower_solver(
        module, scenario, T, net, resolved_solver_kwargs
    )

    print(
        f"Generating {Nsample - completed} new samples for {follower_id} "
        f"({scenario} solver) with {module.__name__}..."
    )
    begin_time = time.time()
    attempted = 0
    failed = 0
    consecutive_failures = 0
    while completed < Nsample:
        attempted += 1
        # pi = electricity_price(T, rng)
        pi = rng.uniform(0, 1.5, size=T)
        result = lower_solver.lower_solve(pi)
        if result is None:
            failed += 1
            consecutive_failures += 1
            print(
                f"Skipped failed attempt {attempted}; "
                f"collected {completed}/{Nsample} samples"
            )
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Stopped after {consecutive_failures} consecutive failed samples. "
                    f"Collected {completed}/{Nsample}; the latest checkpoint is preserved."
                )
            continue

        consecutive_failures = 0
        X[completed] = pi
        Y[completed] = np.asarray(result["p_mg"], dtype=float).reshape(T)
        completed += 1
        if completed % checkpoint_interval == 0 or completed == Nsample:
            np.save(x_path, X[:completed])
            np.save(y_path, Y[:completed])
            current_time = time.time()
            print(
                f"Collected and saved {completed}/{Nsample} samples "
                f"({attempted} attempts, {failed} skipped). "
                f"Time taken: {current_time - begin_time} seconds"
            )
            begin_time = current_time

    return X, Y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate price-response samples for one or more follower scenarios."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SUPPORTED_PROBLEMS,
        default=list(SUPPORTED_INSTANCES),
        help="Follower instances to generate. Default: SB1 SB2 BG1 BG2 IP1.",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument(
        "--date",
        default=None,
        help="Legacy date override for BG/IP. Prefer the per-instance options.",
    )
    parser.add_argument("--bg1-date", default=None)
    parser.add_argument("--bg2-date", default=None)
    parser.add_argument("--ip1-date", default=None)
    parser.add_argument("--sb2-load-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Regenerate each requested dataset from the first sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.time()
    for scenario in args.scenarios:
        scenario_start = time.time()
        solver_kwargs: Dict[str, Any] = {
            "time_limit": args.time_limit,
            "mip_gap": args.mip_gap,
            "failure_mode": "none",
        }
        date_overrides = {
            "BG1": args.bg1_date,
            "BG2": args.bg2_date,
            "IP1": args.ip1_date,
        }
        normalized = normalize_follower_name(scenario)
        selected_date = date_overrides.get(normalized)
        if selected_date is not None:
            solver_kwargs["date"] = selected_date
        elif args.date is not None and scenario_type(normalized) in {"BG", "IP"}:
            solver_kwargs["date"] = args.date
        if normalized == "SB2" and args.sb2_load_scale is not None:
            solver_kwargs["load_scale"] = args.sb2_load_scale

        data_generation(
            T=args.periods,
            Nsample=args.samples,
            problem_type=scenario,
            solver_kwargs=solver_kwargs,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
            reuse_existing=not args.no_reuse_existing,
        )
        print(
            f"Completed {scenario} in {time.time() - scenario_start:.2f} seconds."
        )

    print(
        f"Completed {', '.join(args.scenarios)} in "
        f"{time.time() - overall_start:.2f} seconds."
    )


if __name__ == "__main__":
    main()
