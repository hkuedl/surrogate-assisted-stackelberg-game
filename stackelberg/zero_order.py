"""Ground-truth zero-order Stackelberg solver."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from follower_solver import build_solver as build_scenario_follower
from follower_solver import build_composite_follower

from .distflow import CoptDistFlowEvaluator, build_microgrid_profile
from .follower import CoptFollowerSolver, extract_follower_constants
from .network import (
    build_passive_profiles,
    create_case33bw_20kv,
    extract_upper_constants,
)
from .instances import resolve_follower_configuration
from .pricing import sample_electricity_price
from .profiles import daily_load_curve
from .randomness import DEFAULT_SEED, set_global_seed


def electricity_price(T: int) -> np.ndarray:
    return sample_electricity_price(T, require_24_hours=True)


def _default_load_curve(T: int) -> np.ndarray:
    return daily_load_curve(T)


def extract_lower_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    scenario = str(input.get("scenario", "SB")).upper()
    if scenario != "SB":
        raise NotImplementedError(
            f"The {scenario} follower optimization problem is not implemented."
        )
    constants = extract_follower_constants(net, T, verbose=False)
    constants.pop("verbose")
    constants.pop("random_seed")
    return constants


def _build_passive_profiles(net: pp.pandapowerNet, T: int, load_curve: np.ndarray, s_base_mva: float) -> Dict[str, np.ndarray]:
    return build_passive_profiles(net, T, load_curve, s_base_mva)


def extract_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    return extract_upper_constants(
        input, net, T, squared_voltage_limits=True, price_sampler=electricity_price
    )


class UpperSolver:
    """
    Solver for upper-level optimization in bilevel Stackelberg game.
    """
    def __init__(self, constants: Dict[str, Any], lower_solver: Optional["LowerSolver"] = None):
        self.constants = constants
        self.lower_solver = lower_solver

        self.T = int(constants["T"])
        self.n_bus = int(constants["n_bus"])
        self.n_line = int(constants["n_line"])
        self.slack = int(constants["slack"])
        self.from_bus = [int(v) for v in constants["from_bus"]]
        self.to_bus = [int(v) for v in constants["to_bus"]]
        self.r = np.asarray(constants["r"], dtype=float)
        self.x = np.asarray(constants["x"], dtype=float)
        self.l_max = np.asarray(constants["l_max"], dtype=float)
        self.load_curve = np.asarray(constants["load_curve"], dtype=float)
        self.v_base_kv = float(constants["v_base_kv"])
        self.s_base_mva = float(constants["s_base_mva"])
        self.lambda_g = np.asarray(
            constants["lambda_g"] if "lambda_g" in constants else constants["lamda_g"],
            dtype=float,
        )
        self.pi_min = float(constants["pi_min"])
        self.pi_max = float(constants["pi_max"])
        self.p0_max = float(constants["p0_max"])
        self.q0_max = float(constants["q0_max"])
        self.v_ref = float(constants["v_ref"])
        self.v_min = float(constants["v_min"])
        self.v_max = float(constants["v_max"])
        self.mg_buses = [int(b) for b in constants.get("mg_buses", [])]
        self.independent_prices = bool(constants.get("independent_prices", False))
        self.n_followers = len(self.mg_buses) if self.independent_prices else 1
        self.decision_size = self.n_followers * self.T
        self.p_passive = np.asarray(constants["p_passive"], dtype=float)
        self.q_passive = np.asarray(constants["q_passive"], dtype=float)
        self.verbose = bool(constants.get("verbose", False))
        self.random_seed = int(constants.get("random_seed", DEFAULT_SEED))
        self.interaction_count = 0
        self.interaction_history = []
        self.distflow_evaluator = CoptDistFlowEvaluator(
            constants,
            failure="penalty",
            revenue="aggregate",
            accept_suboptimal=True,
            verbose=self.verbose,
        )

        self.children = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = self.from_bus[e]
            j = self.to_bus[e]
            self.children[i].append(e)
            self.parent_edge[j] = e

    def _reshape_price(self, pi: np.ndarray) -> np.ndarray:
        price = np.asarray(pi, dtype=float)
        if not self.independent_prices:
            return price.reshape(self.T)
        return price.reshape(self.n_followers, self.T)

    def distflow(self, pi: np.ndarray, p_mg: np.ndarray) -> float:
        """Evaluate fixed leader/follower decisions with hard DistFlow constraints."""
        return self.distflow_evaluator.evaluate(self._reshape_price(pi), p_mg)

    def _build_mg_profile(self, p_mg: np.ndarray) -> np.ndarray:
        return build_microgrid_profile(self.constants, p_mg)

    def evaluate(self, pi: np.ndarray) -> float:
        """
        Evaluate the objective function.
        """
        prices = self._reshape_price(pi)
        lower_result = self.lower_solver.lower_solve(prices)

        if lower_result is None:
            eval_obj = 1e10 # large penalty for infeasible solution
        else:
            eval_obj = self.distflow(prices, lower_result["p_mg"])
        self.interaction_count += 1
        self.interaction_history.append({
            "interaction": self.interaction_count,
            "upper_objective": float(eval_obj),
        })
        return eval_obj


    def upper_solve(
        self,
        pi_0: np.ndarray,
        maxiter: int = 1000,
        cma_verbose: int = 1,
        cma_sigma0: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Solve the upper-level optimization problem.
        Using the CMA-ES algorithm.
        """
        try:
            import cma
        except ImportError as error:
            raise ImportError(
                "zero_solver.upper_solve requires the optional 'cma' package."
            ) from error
        cma_seed = self.random_seed if seed is None else int(seed)
        x0 = np.asarray(pi_0, dtype=float).reshape(-1)
        if x0.size != self.decision_size:
            raise ValueError(
                f"pi_0 must contain {self.decision_size} prices, got {x0.size}."
            )
        sigma0 = (
            (self.pi_max - self.pi_min) / 4
            if cma_sigma0 is None
            else float(cma_sigma0)
        )
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError(f"cma_sigma0 must be finite and positive, got {sigma0}.")
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'bounds': [self.pi_min, self.pi_max],
            'seed': cma_seed,
            'verbose': cma_verbose, # 0: no output, 1: output best solution, 2: output all solutions
            'maxiter': maxiter,
            'CMA_elitist': True,
        })
        while not es.stop():
            ys = es.ask()
            fits = [self.evaluate(y) for y in ys]
            es.tell(ys, fits)
            es.disp()
        x_best = np.asarray(es.result.xbest, dtype=float)
        if self.independent_prices:
            x_best = x_best.reshape(self.n_followers, self.T)
        return {
            "optimal pi": x_best,
            "optimal value": es.result.fbest,
            "cma_sigma0": sigma0,
            "cma_elitist": True,
        }

    def get_interaction_history(self) -> pd.DataFrame:
        """
        Return the upper objective value after each lower-solver call.
        """
        return pd.DataFrame(self.interaction_history)

    


class LowerSolver(CoptFollowerSolver):
    """Compatibility wrapper preserving the zero solver's None-on-failure policy."""

    def __init__(self, lower_constants: Dict[str, Any]):
        super().__init__(lower_constants, failure_mode="none", scenario="lower_level")


def _normalize_follower_mapping(
    *,
    scenarios: Sequence[str] | None = None,
    lower_nets: Sequence[pp.pandapowerNet] | None = None,
    mg_buses: Sequence[int] | int | None = None,
    scenario: str | None = None,
    mg_net: pp.pandapowerNet | None = None,
) -> tuple[list[str], list[pp.pandapowerNet], list[int]]:
    """Accept either the multi-follower lists or the legacy single-follower args."""
    multi_requested = scenarios is not None or lower_nets is not None
    if multi_requested:
        if scenarios is None or lower_nets is None or mg_buses is None:
            raise ValueError(
                "Multi-follower mode requires scenarios, lower_nets, and mg_buses."
            )
        scenario_list = [str(value).upper() for value in scenarios]
        network_list = list(lower_nets)
        if isinstance(mg_buses, (int, np.integer)):
            bus_list = [int(mg_buses)]
        else:
            bus_list = [int(value) for value in mg_buses]
    else:
        scenario_list = [str(scenario or "SB").upper()]
        network_list = [
            mg_net if mg_net is not None else pn.create_cigre_network_lv()
        ]
        if mg_buses is None:
            bus_list = [18]
        elif isinstance(mg_buses, (int, np.integer)):
            bus_list = [int(mg_buses)]
        else:
            bus_list = [int(value) for value in mg_buses]

    n_followers = len(scenario_list)
    if n_followers == 0:
        raise ValueError("At least one follower scenario is required.")
    if len(network_list) != n_followers or len(bus_list) != n_followers:
        raise ValueError(
            "scenarios, lower_nets, and mg_buses must have the same nonzero length."
        )
    return scenario_list, network_list, bus_list


def build_zero_order_follower(
    scenarios: Sequence[str],
    lower_nets: Sequence[pp.pandapowerNet],
    T: int,
    *,
    verbose: bool = False,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None,
):
    """Build a single follower or a composite follower matching the TR mapping."""
    follower_ids, scenario_types, resolved_kwargs = resolve_follower_configuration(
        scenarios, follower_kwargs
    )
    network_list = list(lower_nets)
    if len(follower_ids) != len(network_list):
        raise ValueError(
            "scenarios and lower_nets must have the same length; "
            f"got {len(follower_ids)} and {len(network_list)}."
        )
    common_kwargs = {
        "verbose": verbose,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
    }
    if len(follower_ids) == 1:
        single_kwargs = {**common_kwargs, **resolved_kwargs[0]}
        follower = build_scenario_follower(
            scenario_types[0],
            T=T,
            net=network_list[0],
            **single_kwargs,
        )
        follower.follower_id = follower_ids[0]
        return follower
    return build_composite_follower(
        follower_ids,
        network_list,
        T=T,
        follower_kwargs=resolved_kwargs,
        **common_kwargs,
    )


def solve_gt_optimal(
    input: Optional[Dict[str, Any]] = None,
    upper_net: Optional[pp.pandapowerNet] = None,
    mg_net: Optional[pp.pandapowerNet] = None,
    *,
    scenarios: Sequence[str] | None = None,
    lower_nets: Sequence[pp.pandapowerNet] | None = None,
    mg_buses: Sequence[int] | int | None = None,
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Solve the ground-truth Stackelberg game.

    Default setup:
        upper distribution network: 20 kV case33bw
        microgrid network: pn.create_cigre_network_lv()
        horizon: 24 hours

    Multi-follower usage mirrors the trust-region notebook:
        scenarios / lower_nets / mg_buses are aligned one-to-one.
        Each follower receives an independent 24-hour price trajectory when
        more than one follower is configured.
    """
    if input is None:
        input = {}

    random_seed = set_global_seed(input.get("seed", DEFAULT_SEED))
    T = int(input.get("T", 24))
    if T != 24:
        raise ValueError("solve_gt_optimal only supports the 24-hour setup.")

    if upper_net is None:
        upper_net = create_case33bw_20kv()

    scenario_list, network_list, bus_list = _normalize_follower_mapping(
        scenarios=scenarios if scenarios is not None else input.get("scenarios"),
        lower_nets=lower_nets if lower_nets is not None else input.get("lower_nets"),
        mg_buses=(
            mg_buses
            if mg_buses is not None
            else input.get("mg_buses", input.get("mg_bus"))
        ),
        scenario=input.get("scenario"),
        mg_net=mg_net,
    )

    n_followers = len(scenario_list)
    independent_prices = n_followers > 1
    upper_constants = extract_constants(
        {
            "mg_buses": bus_list,
            "independent_prices": independent_prices,
            "seed": random_seed,
        },
        net=upper_net,
        T=T,
    )
    upper_constants["random_seed"] = random_seed
    upper_constants["v_min"] = float(input.get("v_min", 0.80**2))
    verbose = bool(input.get("verbose", False))
    if "verbose" in input:
        upper_constants["verbose"] = verbose

    requested_follower_kwargs = (
        follower_kwargs
        if follower_kwargs is not None
        else input.get("follower_kwargs")
    )
    _, scenario_types, resolved_follower_kwargs = resolve_follower_configuration(
        scenario_list, requested_follower_kwargs
    )
    lower_solver = build_zero_order_follower(
        scenario_list,
        network_list,
        T=T,
        verbose=verbose,
        time_limit=input.get("lower_time_limit"),
        mip_gap=input.get("lower_mip_gap"),
        follower_kwargs=resolved_follower_kwargs,
    )
    upper_solver = UpperSolver(upper_constants, lower_solver=lower_solver)

    decision_size = n_followers * T
    if "pi_0" in input:
        pi_0 = np.asarray(input["pi_0"], dtype=float).reshape(-1)
    else:
        pi_0 = np.full(
            decision_size,
            0.5 * (upper_solver.pi_min + upper_solver.pi_max),
            dtype=float,
        )
    if pi_0.size != decision_size:
        raise ValueError(
            f"pi_0 must contain {decision_size} prices, got {pi_0.size}."
        )

    time_start = time.time()
    result = upper_solver.upper_solve(
        pi_0,
        maxiter=int(input.get("maxiter", 1000)),
        cma_verbose=int(input.get("cma_verbose", 1)),
        cma_sigma0=input.get("cma_sigma0"),
        seed=random_seed,
    )
    interaction_history = upper_solver.get_interaction_history()
    result["interaction_history"] = interaction_history
    if "interaction_history_path" in input:
        interaction_history.to_csv(input["interaction_history_path"], index=False)

    optimal_pi = upper_solver._reshape_price(result["optimal pi"])
    lower_result = lower_solver.lower_solve(optimal_pi)
    result["optimal pi"] = optimal_pi
    result["p_mg"] = None if lower_result is None else lower_result["p_mg"]
    result["mg_buses"] = bus_list
    result["scenarios"] = scenario_list
    result["scenario_types"] = scenario_types
    result["follower_kwargs"] = resolved_follower_kwargs
    result["scenario"] = scenario_list[0] if n_followers == 1 else "COMPOSITE"
    result["independent_prices"] = independent_prices
    result["lower_time_limit"] = input.get("lower_time_limit")
    result["lower_mip_gap"] = input.get("lower_mip_gap")
    result["runtime_seconds"] = time.time() - time_start
    result["n_interactions"] = upper_solver.interaction_count
    return result


@dataclass
class ZeroOrderExperimentConfig:
    """Configuration shared by scripts and notebooks (aligned with TR mapping)."""

    upper_net: pp.pandapowerNet
    lower_nets: Sequence[pp.pandapowerNet]
    mg_buses: Sequence[int]
    scenarios: Sequence[str]
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None
    T: int = 24
    seed: int = DEFAULT_SEED
    pi_0: np.ndarray | None = None
    maxiter: int = 1000
    cma_verbose: int = 1
    cma_sigma0: float | None = None
    verbose: bool = False
    lower_time_limit: float | None = None
    lower_mip_gap: float | None = None
    v_min: float = 0.80**2
    save_results: bool = True
    output_dir: str = "results"
    interaction_history_path: str | None = None
    extra_input: Dict[str, Any] = field(default_factory=dict)


def run_zero_order_experiment(config: ZeroOrderExperimentConfig) -> Dict[str, Any]:
    """Build and run a multi-follower zero-order CMA-ES experiment."""
    payload = {
        "T": config.T,
        "seed": config.seed,
        "maxiter": config.maxiter,
        "cma_verbose": config.cma_verbose,
        "cma_sigma0": config.cma_sigma0,
        "verbose": config.verbose,
        "lower_time_limit": config.lower_time_limit,
        "lower_mip_gap": config.lower_mip_gap,
        "v_min": config.v_min,
        **config.extra_input,
    }
    if config.pi_0 is not None:
        payload["pi_0"] = config.pi_0
    if config.interaction_history_path is not None:
        payload["interaction_history_path"] = config.interaction_history_path

    result = solve_gt_optimal(
        payload,
        upper_net=config.upper_net,
        scenarios=config.scenarios,
        lower_nets=config.lower_nets,
        mg_buses=config.mg_buses,
        follower_kwargs=config.follower_kwargs,
    )
    if config.save_results:
        result["saved_paths"] = save_zero_solver_result(
            result, output_dir=config.output_dir
        )
    return result


def save_zero_solver_result(result: Dict[str, Any], output_dir: str = "results") -> Dict[str, str]:
    """
    Save zero-solver outputs beyond the interaction history.

    The JSON keeps scalar metadata and arrays; the solution CSV is convenient
    for plotting or comparing the optimal hourly profiles.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    history_path = output_path / f"zero_solver_history_{timestamp}.csv"
    solution_path = output_path / f"zero_solver_solution_{timestamp}.csv"
    json_path = output_path / f"zero_solver_result_{timestamp}.json"

    interaction_history = result.get("interaction_history")
    if isinstance(interaction_history, pd.DataFrame):
        interaction_history.to_csv(history_path, index=False)

    optimal_pi = np.asarray(result["optimal pi"], dtype=float)
    p_mg = result.get("p_mg")
    p_mg_array = None if p_mg is None else np.asarray(p_mg, dtype=float)
    scenarios = [str(value) for value in result.get("scenarios", [result.get("scenario", "SB")])]
    n_followers = len(scenarios)
    T = optimal_pi.size // n_followers if n_followers > 0 else optimal_pi.size
    pi_matrix = optimal_pi.reshape(n_followers, T)
    p_mg_matrix = (
        None
        if p_mg_array is None
        else np.asarray(p_mg_array, dtype=float).reshape(n_followers, T)
    )

    solution_data: Dict[str, Any] = {"t": np.arange(T)}
    for index, scenario in enumerate(scenarios):
        solution_data[f"pi_{scenario}_{index}"] = pi_matrix[index]
        if p_mg_matrix is not None:
            solution_data[f"p_mg_{scenario}_{index}"] = p_mg_matrix[index]
    pd.DataFrame(solution_data).to_csv(solution_path, index=False)

    result_to_save = {
        "optimal_pi": pi_matrix.tolist(),
        "optimal_value": float(result["optimal value"]),
        "p_mg": None if p_mg_matrix is None else p_mg_matrix.tolist(),
        "mg_buses": result.get("mg_buses"),
        "scenarios": scenarios,
        "scenario_types": result.get("scenario_types"),
        "follower_kwargs": result.get("follower_kwargs"),
        "scenario_types": result.get("scenario_types"),
        "independent_prices": bool(result.get("independent_prices", n_followers > 1)),
        "cma_sigma0": result.get("cma_sigma0"),
        "cma_elitist": bool(result.get("cma_elitist", False)),
        "lower_time_limit": result.get("lower_time_limit"),
        "lower_mip_gap": result.get("lower_mip_gap"),
        "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
        "n_interactions": int(result.get("n_interactions", 0)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2)

    saved_paths = {"json": str(json_path), "solution_csv": str(solution_path)}
    if isinstance(interaction_history, pd.DataFrame):
        saved_paths["history_csv"] = str(history_path)
    return saved_paths


def test_gt_solver_smoke():
    """
    Smoke test for the ground-truth bilevel solver components.

    Run manually with:
        python -c "import gt_solver; gt_solver.test_gt_solver_smoke()"
    """
    T = 24
    upper_net = create_case33bw_20kv()
    mg_net = pn.create_cigre_network_lv()
    mg_bus = 5

    upper_constants = extract_constants({"mg_buses": [mg_bus]}, net=upper_net, T=T)
    upper_constants["v_min"] = 0.80**2
    lower_constants = extract_lower_constants({}, net=mg_net, T=T)
    lower_solver = LowerSolver(lower_constants)
    upper_solver = UpperSolver(upper_constants, lower_solver=lower_solver)

    pi = np.full(T, 0.5, dtype=float)
    lower_result = lower_solver.lower_solve(pi)
    assert lower_result is not None
    assert lower_result["p_mg"].shape == (T,)
    assert np.all(lower_result["p_mg"] >= -1e-8)

    obj = upper_solver.evaluate(pi)
    assert np.isfinite(obj)
    assert obj < 1e10

    print("test_gt_solver_smoke passed.")


def main():
    result = solve_gt_optimal({
        "maxiter": 1000,
        "cma_verbose": 1,
        "interaction_history_path": "interaction_history.csv",
    })
    saved_paths = save_zero_solver_result(result)
    print(f"optimal pi: {result['optimal pi']}")
    print(f"optimal value: {result['optimal value']}")
    print(f"p_mg: {result['p_mg']}")
    print(f"runtime seconds: {result['runtime_seconds']:.4f}")
    print("interaction history saved to interaction_history.csv")
    print(f"Saved full result to: {saved_paths['json']}")
    print(f"Saved optimal solution to: {saved_paths['solution_csv']}")
    if "history_csv" in saved_paths:
        print(f"Saved timestamped interaction history to: {saved_paths['history_csv']}")
