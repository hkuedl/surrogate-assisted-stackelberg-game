"""Neural-surrogate upper-level solver implementation."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import torch
import torch.nn as nn

from follower_solver import build_solver as build_scenario_follower
from zero_solver import (
    UpperSolver as ZeroUpperSolver,
)
from .distflow import torch_penalty_distflow
from .l2o import load_l2o_model, load_l2o_models
from .network import (
    build_passive_profiles,
    create_case33bw_20kv,
    extract_upper_constants,
)
from .pricing import sample_electricity_price
from .profiles import daily_load_curve
from .randomness import DEFAULT_SEED, set_global_seed
from .zero_order import build_zero_order_follower
from .instances import resolve_follower_configuration


def load_l2o_map_net(model_type: str = "SB", device: Optional[torch.device] = None) -> nn.Module:
    return load_l2o_model(model_type, device)


def electricity_price(T: int) -> np.ndarray:
    return sample_electricity_price(T)


def _default_load_curve(T: int) -> np.ndarray:
    return daily_load_curve(T)


def _build_passive_profiles(net: pp.pandapowerNet, T: int, load_curve: np.ndarray, s_base_mva: float) -> Dict[str, np.ndarray]:
    return build_passive_profiles(net, T, load_curve, s_base_mva)


def extract_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    return extract_upper_constants(
        input, net, T, squared_voltage_limits=False, price_sampler=electricity_price
    )


def test_extract_constants():
    T = 24
    net = create_case33bw_20kv()
    constants = extract_constants({}, net=net, T=T)

    assert constants["n_bus"] == len(net.bus)
    assert constants["n_line"] == int(net.line["in_service"].sum())
    assert constants["load_curve"].shape == (T,)
    assert constants["l_max"].shape == (constants["n_line"],)
    assert np.all(constants["load_curve"] >= 0.5)
    assert np.all(constants["load_curve"] <= 1.5)
    assert constants["v_ref"] == 1.0
    assert constants["z_base_ohm"] > 0.0
    assert constants["i_base_ka"] > 0.0

    print("test_extract_constants passed.")
    print(
        f"n_bus={constants['n_bus']}, n_line={constants['n_line']}, "
        f"v_base={constants['v_base_kv']}kV, s_base={constants['s_base_mva']}MVA"
    )


class CompositeMapNet(nn.Module):
    """One independent L2O map per follower, aligned with mg_buses order."""

    def __init__(self, map_nets: Sequence[nn.Module], T: int):
        super().__init__()
        nets = list(map_nets)
        if not nets:
            raise ValueError("At least one map_net is required.")
        self.map_nets = nn.ModuleList(nets)
        self.T = int(T)
        self.n_followers = len(self.map_nets)

    def forward(self, pi: torch.Tensor) -> torch.Tensor:
        if pi.ndim == 1:
            if self.n_followers != 1:
                raise ValueError(
                    f"Flat price vector requires n_followers=1, got {self.n_followers}."
                )
            return self.map_nets[0](pi.unsqueeze(0)).squeeze(0)

        if tuple(pi.shape) == (self.n_followers, self.T):
            outputs = [
                self.map_nets[index](pi[index].unsqueeze(0)).squeeze(0)
                for index in range(self.n_followers)
            ]
            return torch.stack(outputs, dim=0)

        if pi.ndim == 2 and pi.shape[1] == self.T and self.n_followers == 1:
            return self.map_nets[0](pi)

        if pi.ndim == 3 and pi.shape[1:] == (self.n_followers, self.T):
            outputs = [
                self.map_nets[index](pi[:, index, :])
                for index in range(self.n_followers)
            ]
            return torch.stack(outputs, dim=1)

        raise ValueError(
            f"Unexpected price tensor shape {tuple(pi.shape)}; expected "
            f"({self.T},), ({self.n_followers}, {self.T}), or "
            f"(batch, {self.n_followers}, {self.T})."
        )


def _as_map_module(
    map_net: Union[nn.Module, Sequence[nn.Module]],
    *,
    T: int,
    n_followers: int,
) -> nn.Module:
    if isinstance(map_net, (list, tuple)):
        if len(map_net) != n_followers:
            raise ValueError(
                f"Expected {n_followers} map_nets, got {len(map_net)}."
            )
        if n_followers == 1:
            return map_net[0]
        return CompositeMapNet(map_net, T=T)
    if n_followers > 1 and not isinstance(map_net, CompositeMapNet):
        raise ValueError(
            "Independent multi-follower prices require a CompositeMapNet "
            "or a sequence of map_nets."
        )
    return map_net


class TorchDistFlowOptimizer(nn.Module):
    def __init__(
        self,
        constants: Dict[str, Any],
        map_net: Union[nn.Module, Sequence[nn.Module]],
        device="cpu",
        x0: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.device = device
        # Constants
        self.T = constants["T"]
        self.n_bus = constants["n_bus"]
        self.n_line = constants["n_line"]
        self.slack = constants["slack"]
        self.from_bus = torch.tensor(constants["from_bus"], dtype=torch.long, device=device)
        self.to_bus   = torch.tensor(constants["to_bus"], dtype=torch.long, device=device)
        self.r = torch.tensor(constants["r"], dtype=torch.float32, device=device)
        self.x = torch.tensor(constants["x"], dtype=torch.float32, device=device)
        self.l_max = torch.tensor(constants["l_max"], dtype=torch.float32, device=device)
        self.lambda_g = torch.tensor(constants["lambda_g"], dtype=torch.float32, device=device)
        self.s_base_mva = float(constants["s_base_mva"])
        self.pi_min = float(constants["pi_min"])
        self.pi_max = float(constants["pi_max"])
        self.p0_max = float(constants["p0_max"])
        self.q0_max = float(constants["q0_max"])
        self.v_ref = float(constants["v_ref"])
        self.v_min = float(constants["v_min"])
        self.v_max = float(constants["v_max"])
        self.mg_buses = constants["mg_buses"]
        self.mg_index = {b: i for i, b in enumerate(self.mg_buses)}
        self.independent_prices = bool(constants.get("independent_prices", False))
        self.n_followers = len(self.mg_buses) if self.independent_prices else 1
        self.map_net = _as_map_module(
            map_net, T=self.T, n_followers=self.n_followers
        ).to(device)
        self.p_passive = torch.as_tensor(constants["p_passive"], dtype=torch.float32, device=device)  # [n_bus, T]
        # Keep the passive reactive-power profile as immutable reference data.
        self.register_buffer(
            "q_passive",
            torch.as_tensor(
                constants["q_passive"], dtype=torch.float32, device=device
            ).clone(),
        )
        # ========== Variables to optimize ==========
        # Network states are optimized as auxiliary variables to satisfy constraints.
        pi_shape = (self.n_followers, self.T) if self.independent_prices else (self.T,)
        if x0 is None:
            pi_raw_init = torch.zeros(pi_shape, dtype=torch.float32, device=device)
        else:
            x0_array = np.asarray(x0, dtype=float)
            expected_size = int(np.prod(pi_shape))
            if x0_array.size != expected_size:
                raise ValueError(
                    f"x0 must contain {expected_size} prices, got {x0_array.size}."
                )
            x0_array = x0_array.reshape(pi_shape)
            if not np.all(np.isfinite(x0_array)):
                raise ValueError("x0 must contain only finite prices.")
            if np.any(x0_array < self.pi_min) or np.any(x0_array > self.pi_max):
                raise ValueError(
                    f"x0 prices must lie in [{self.pi_min}, {self.pi_max}]."
                )
            if self.pi_max <= self.pi_min:
                raise ValueError("pi_max must be greater than pi_min.")

            # The optimized parameter lives in unconstrained logit space. Values
            # exactly on a bound are represented by the nearest finite float so
            # Adam receives finite parameters and gradients.
            bounded_x0 = torch.as_tensor(
                x0_array, dtype=torch.float32, device=device
            )
            scaled_x0 = (bounded_x0 - self.pi_min) / (self.pi_max - self.pi_min)
            eps = torch.finfo(scaled_x0.dtype).eps
            scaled_x0 = torch.clamp(scaled_x0, min=eps, max=1.0 - eps)
            pi_raw_init = torch.logit(scaled_x0)
        self.pi_raw = nn.Parameter(pi_raw_init)
        self.p0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.q0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.v  = nn.Parameter(torch.ones(self.n_bus, self.T, device=device) * (self.v_ref**2))
        self.l  = nn.Parameter(torch.ones(self.n_line, self.T, device=device) * 1e-3)
        self.p  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        self.q  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        # Explicit auxiliary variable from the DN formulation.  The shared
        # penalty model enforces q_injection == q_passive.
        self.q_injection = nn.Parameter(self.q_passive.detach().clone())
        # Build children / parent_edge
        child_edges = {i: [] for i in range(self.n_bus)}
        parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            child_edges[i].append(e)
            parent_edge[j] = e
        self.child_edges = child_edges
        self.parent_edge = parent_edge

    def bounded_pi(self):
        # Map to [pi_min, pi_max]
        return self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(self.pi_raw)

    def forward(self, penalty_weights=None):
        if penalty_weights is None:
            penalty_weights = {
                "eq": 1000.0, "ineq": 100.0, "soc": 100.0, "bound": 10.0
            }
        pi = self.bounded_pi()
        if self.independent_prices:
            p_mg_prediction = self.map_net(pi)
        else:
            p_mg_prediction = self.map_net(pi.unsqueeze(0)).squeeze(0)
        return torch_penalty_distflow(
            self,
            pi,
            p_mg_prediction,
            penalty_weights,
            square_voltage_limits=True,
        )


def solve_with_torch(
    model,
    steps=5000,
    lr=1e-2,
    history_path="results/NN_solver_history.csv",
    save_every=100,
    true_eval_constants: Optional[Dict[str, Any]] = None,
    true_eval_lower_nets: Optional[Sequence[pp.pandapowerNet]] = None,
    true_eval_scenarios: Optional[Sequence[str]] = None,
    true_eval_follower_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
    true_eval_mg_net: Optional[pp.pandapowerNet] = None,
    true_eval_scenario: str = "SB",
    seed: Optional[int] = None,
):
    if seed is not None:
        set_global_seed(seed)
    opt = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    history = []
    for k in range(steps):
        opt.zero_grad()
        loss, aux = model()
        loss.backward()
        opt.step()
        if save_every > 0 and k % save_every == 0:
            with torch.no_grad():
                loss_log, aux_log = model()
            pi_np = aux_log["pi"].detach().cpu().numpy()
            record = {
                "iteration": k,
                "loss": loss_log.item(),
                "obj": aux_log["obj"].item(),
                "eq_pen": aux_log["eq_pen"].item(),
                "ineq_pen": aux_log["ineq_pen"].item(),
                "soc_pen": aux_log["soc_pen"].item(),
                "bound_pen": aux_log["bound_pen"].item(),
            }
            log_msg = (
                f"[{k}] loss={record['loss']:.4f}, obj={record['obj']:.4f}, "
                f"eq={record['eq_pen']:.4f}, soc={record['soc_pen']:.4f}, "
                f"bound={record['bound_pen']:.4f}"
            )
            if true_eval_constants is not None:
                true_eval = evaluate_true_objective(
                    pi_np,
                    true_eval_constants,
                    lower_nets=true_eval_lower_nets,
                    scenarios=true_eval_scenarios,
                    follower_kwargs=true_eval_follower_kwargs,
                    mg_net=true_eval_mg_net,
                    scenario=true_eval_scenario,
                )
                record["true_loss"] = true_eval["true_obj"]
                log_msg += f", true_loss={record['true_loss']:.4f}"
            print(log_msg)
            flat_pi = np.asarray(pi_np, dtype=float).reshape(-1)
            record.update({f"pi_{i}": value for i, value in enumerate(flat_pi)})
            history.append(record)
    with torch.no_grad():
        _, aux = model()
    if history_path is not None and history:
        history_dir = os.path.dirname(history_path)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
    return aux["pi"].detach().cpu().numpy()


def evaluate_true_objective(
    pi: np.ndarray,
    upper_constants: Dict[str, Any],
    mg_net: Optional[pp.pandapowerNet] = None,
    verbose: bool = False,
    scenario: str = "SB",
    *,
    scenarios: Optional[Sequence[str]] = None,
    lower_nets: Optional[Sequence[pp.pandapowerNet]] = None,
    follower_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate NN-produced pi with the true lower-level response and hard DistFlow model.
    """
    T = int(upper_constants["T"])
    independent_prices = bool(upper_constants.get("independent_prices", False))
    n_followers = (
        len(upper_constants["mg_buses"]) if independent_prices else 1
    )

    if scenarios is not None or lower_nets is not None:
        if scenarios is None or lower_nets is None:
            raise ValueError("scenarios and lower_nets must be provided together.")
        scenario_list = [str(value).upper() for value in scenarios]
        network_list = list(lower_nets)
        if len(scenario_list) != n_followers or len(network_list) != n_followers:
            raise ValueError(
                "scenarios/lower_nets must match the number of microgrid buses."
            )
        lower_solver = build_zero_order_follower(
            scenario_list,
            network_list,
            T=T,
            verbose=verbose,
            follower_kwargs=follower_kwargs,
        )
    else:
        if mg_net is None:
            mg_net = pn.create_cigre_network_lv()
        if n_followers != 1:
            raise ValueError(
                "Multi-follower true evaluation requires scenarios and lower_nets."
            )
        lower_solver = build_zero_order_follower(
            [scenario],
            [mg_net],
            T=T,
            verbose=verbose,
            follower_kwargs=follower_kwargs,
        )

    eval_constants = dict(upper_constants)
    eval_constants["verbose"] = verbose
    # zero_solver.UpperSolver expects squared-voltage limits.
    eval_constants["v_min"] = upper_constants["v_min"] ** 2
    eval_constants["v_max"] = upper_constants["v_max"] ** 2

    upper_solver = ZeroUpperSolver(eval_constants, lower_solver=lower_solver)
    prices = upper_solver._reshape_price(pi)
    lower_result = lower_solver.lower_solve(prices)
    if lower_result is None:
        return {"true_obj": 1e10, "p_mg_true": None}

    p_mg_true = lower_result["p_mg"]
    true_obj = upper_solver.distflow(prices, p_mg_true)
    return {"true_obj": true_obj, "p_mg_true": p_mg_true}


@dataclass
class NNSolverExperimentConfig:
    """Configuration shared by scripts and notebooks (aligned with TR mapping)."""

    upper_net: pp.pandapowerNet
    lower_nets: Sequence[pp.pandapowerNet]
    mg_buses: Sequence[int]
    scenarios: Sequence[str]
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None
    T: int = 24
    seed: int = DEFAULT_SEED
    device: str = "cpu"
    x0: np.ndarray | None = None
    steps: int = 500
    lr: float = 1e-2
    save_every: int = 50
    history_path: str = "results/NN_solver_history.csv"
    map_nets: Sequence[nn.Module] | None = None
    evaluate_true: bool = True
    save_results: bool = True
    model_tag: str | None = None


def _timestamped_history_path(history_path: str) -> str:
    """Add a run timestamp and avoid replacing an existing history file."""
    path_root, extension = os.path.splitext(history_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = f"{path_root}_{timestamp}{extension}"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path_root}_{timestamp}_{suffix}{extension}"
        suffix += 1
    return candidate


def run_nn_solver_experiment(config: NNSolverExperimentConfig) -> Dict[str, Any]:
    """Build and run a multi-follower neural-surrogate experiment."""
    scenarios = [str(value).upper() for value in config.scenarios]
    lower_nets = list(config.lower_nets)
    mg_buses = [int(value) for value in config.mg_buses]
    n_followers = len(scenarios)
    if n_followers == 0:
        raise ValueError("At least one follower scenario is required.")
    if len(lower_nets) != n_followers or len(mg_buses) != n_followers:
        raise ValueError(
            "scenarios, lower_nets, and mg_buses must have the same nonzero length."
        )
    _, scenario_types, follower_kwargs = resolve_follower_configuration(
        scenarios, config.follower_kwargs
    )

    random_seed = set_global_seed(config.seed)
    device = torch.device(config.device)
    independent_prices = n_followers > 1
    constants = extract_constants(
        {
            "mg_buses": mg_buses,
            "independent_prices": independent_prices,
            "seed": random_seed,
        },
        net=config.upper_net,
        T=config.T,
    )

    if config.map_nets is None:
        map_nets = load_l2o_models(
            scenarios, device=device, model_tag=config.model_tag
        )
    else:
        map_nets = list(config.map_nets)
        if len(map_nets) != n_followers:
            raise ValueError(
                f"Expected {n_followers} map_nets, got {len(map_nets)}."
            )

    model = TorchDistFlowOptimizer(
        constants, map_net=map_nets, device=device, x0=config.x0
    ).to(device)
    for parameter in model.map_net.parameters():
        parameter.requires_grad = False
    history_path = (
        _timestamped_history_path(config.history_path)
        if config.save_results
        else None
    )
    time_start = time.time()
    pi = solve_with_torch(
        model,
        steps=config.steps,
        lr=config.lr,
        history_path=history_path,
        save_every=config.save_every,
        true_eval_constants=constants if config.evaluate_true else None,
        true_eval_lower_nets=lower_nets if config.evaluate_true else None,
        true_eval_scenarios=scenarios if config.evaluate_true else None,
        true_eval_follower_kwargs=(
            follower_kwargs if config.evaluate_true else None
        ),
        seed=random_seed,
    )
    true_eval = (
        evaluate_true_objective(
            pi,
            constants,
            scenarios=scenarios,
            lower_nets=lower_nets,
            follower_kwargs=follower_kwargs,
        )
        if config.evaluate_true
        else {"true_obj": None, "p_mg_true": None}
    )
    return {
        "optimal pi": pi,
        "true_obj": true_eval["true_obj"],
        "p_mg_true": true_eval["p_mg_true"],
        "mg_buses": mg_buses,
        "scenarios": scenarios,
        "scenario_types": scenario_types,
        "follower_kwargs": follower_kwargs,
        "independent_prices": independent_prices,
        "runtime_seconds": time.time() - time_start,
        "history_path": history_path,
    }


def test_torch_distflow_smoke(
    steps: int = 100,
    lr: float = 1e-2,
    model_type: str = "SB",
    mg_bus: int = 18,
    seed: int = DEFAULT_SEED,
    *,
    scenarios: Optional[Sequence[str]] = None,
    lower_nets: Optional[Sequence[pp.pandapowerNet]] = None,
    mg_buses: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    One-shot smoke test: build case33bw + TorchDistFlowOptimizer, run a few Adam steps.
    Loads L2O_{scenario}.pth for each follower via run_nn_solver_experiment.
    Run from project root so data/DA_RT_data.csv resolves.

    Multi-follower usage mirrors the trust-region notebook via scenarios/lower_nets/mg_buses.
    """
    set_global_seed(seed)
    T = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_case33bw_20kv()

    if scenarios is None:
        scenarios = [model_type]
        lower_nets = [pn.create_cigre_network_lv()] if lower_nets is None else list(lower_nets)
        mg_buses = [mg_bus] if mg_buses is None else list(mg_buses)
    else:
        scenarios = [str(value).upper() for value in scenarios]
        if lower_nets is None or mg_buses is None:
            raise ValueError("Multi-follower smoke test requires lower_nets and mg_buses.")
        lower_nets = list(lower_nets)
        mg_buses = [int(value) for value in mg_buses]

    if len(scenarios) != len(lower_nets) or len(scenarios) != len(mg_buses):
        raise ValueError(
            "scenarios, lower_nets, and mg_buses must have the same nonzero length."
        )

    result = run_nn_solver_experiment(
        NNSolverExperimentConfig(
            upper_net=net,
            lower_nets=lower_nets,
            mg_buses=mg_buses,
            scenarios=scenarios,
            T=T,
            seed=seed,
            device=str(device),
            steps=steps,
            lr=lr,
            save_every=max(1, steps // 5),
            history_path="results/NN_solver_smoke_history.csv",
            evaluate_true=True,
            save_results=True,
        )
    )
    print("pi:", result["optimal pi"])
    print(f"true objective: {result['true_obj']:.4f}")
    print("true p_mg:", result["p_mg_true"])
    return result["optimal pi"]
