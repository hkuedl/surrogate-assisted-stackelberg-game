"""Isolated trust-region variant with opt-in COPT network-state warm start.

This module deliberately leaves :mod:`stackelberg.trust_region` unchanged.
It reuses the original controller, residual models, and experiment builder,
but replaces the leader with a variant that can initialize the differentiable
DistFlow variables from one hard COPT solve at the local trust-region center.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from . import trust_region as base


@dataclass
class TrustRegionExperimentConfig(base.TrustRegionExperimentConfig):
    """Experiment configuration for the isolated warm-start variant."""

    copt_initialize_network_state: bool = False


class DistributionNetworkLeader(base.DistributionNetworkLeader):
    """Distribution-network leader with optional COPT state initialization."""

    def _copt_network_state(
        self,
        x_center: np.ndarray,
        surrogate: base.SurrogateModel,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Solve hard DistFlow at ``x_center`` and extract its network state."""
        price = np.asarray(x_center, dtype=float).reshape(-1)
        response = np.asarray(surrogate.predict(price), dtype=float)
        if self.independent_prices:
            response = response.reshape(self.n_followers, self.T)
        objective = self.distflow_evaluator.evaluate(
            self._reshape_price(price), response
        )
        evaluator = self.distflow_evaluator

        def one_dimensional(variables: Any, size: int) -> np.ndarray:
            return np.asarray(
                [variables[index].x for index in range(size)], dtype=float
            )

        def two_dimensional(
            variables: Any,
            first: int,
            second: int,
        ) -> np.ndarray:
            return np.asarray(
                [
                    [variables[i, t].x for t in range(second)]
                    for i in range(first)
                ],
                dtype=float,
            )

        return {
            "p0": one_dimensional(evaluator.p0, self.T),
            "q0": one_dimensional(evaluator.q0, self.T),
            "v": two_dimensional(evaluator.voltage, self.n_bus, self.T),
            "l": two_dimensional(evaluator.current, self.n_line, self.T),
            "p": two_dimensional(evaluator.p_flow, self.n_line, self.T),
            "q": two_dimensional(evaluator.q_flow, self.n_line, self.T),
            "q_injection": two_dimensional(
                evaluator.q_injection, self.n_bus, self.T
            ),
        }, float(objective)

    def _initialize_torch_state(
        self,
        model: base._LeaderSubproblem,
        state: Mapping[str, np.ndarray],
    ) -> None:
        """Validate and copy the COPT state into Torch parameters."""
        with torch.no_grad():
            for name in ("p0", "q0", "v", "l", "p", "q", "q_injection"):
                parameter = getattr(model, name)
                if name not in state:
                    raise RuntimeError(
                        f"COPT warm-start state is missing parameter {name}."
                    )
                value = torch.as_tensor(
                    state[name], dtype=parameter.dtype, device=parameter.device
                )
                if value.shape != parameter.shape:
                    raise RuntimeError(
                        f"COPT warm-start shape mismatch for {name}: "
                        f"expected {tuple(parameter.shape)}, got {tuple(value.shape)}."
                    )
                if value.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"COPT warm-start dtype mismatch for {name}: "
                        f"expected {parameter.dtype}, got {value.dtype}."
                    )
                if value.device != parameter.device:
                    raise RuntimeError(
                        f"COPT warm-start device mismatch for {name}: "
                        f"expected {parameter.device}, got {value.device}."
                    )
                parameter.copy_(value)

    def solve(
        self,
        surrogate: base.SurrogateModel,
        trust_region: base.TrustRegion | None = None,
        x_center: np.ndarray | None = None,
        solve_params: dict | None = None,
    ) -> tuple:
        """Solve the original Torch subproblem with an optional COPT state."""
        if solve_params is None:
            solve_params = {"steps": 5000, "lr": 1e-2}
        model = base._LeaderSubproblem(self.constants, surrogate, self.device)
        uses_torch_surrogate = hasattr(surrogate, "predict_torch")
        if x_center is not None:
            with torch.no_grad():
                model.pi_raw.copy_(
                    torch.tensor(
                        self._inverse_sigmoid(self._reshape_price(x_center)),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        copt_warm_start_seconds = 0.0
        copt_warm_start_objective = None
        if (
            trust_region is not None
            and x_center is not None
            and bool(solve_params.get("copt_initialize_network_state", False))
        ):
            start = time.perf_counter()
            with torch.no_grad():
                state, copt_warm_start_objective = self._copt_network_state(
                    x_center, surrogate
                )
                self._initialize_torch_state(model, state)
            copt_warm_start_seconds = time.perf_counter() - start

        optimizer = torch.optim.Adam(model.parameters(), lr=solve_params["lr"])
        penalty_weights = solve_params.get(
            "penalty_weights",
            {"eq": 1000.0, "ineq": 100.0, "soc": 100.0, "bound": 10.0},
        )
        tr_weight = solve_params.get("tr_weight", 500.0)
        diagnostic_trace_every = int(
            solve_params.get("diagnostic_trace_every", 0) or 0
        )
        if diagnostic_trace_every < 0:
            raise ValueError("diagnostic_trace_every must be nonnegative.")
        collect_diagnostics = diagnostic_trace_every > 0
        diagnostic_trace: list[dict[str, float]] = []
        best_diagnostics = None
        self.last_solve_diagnostics = None

        tr_lb = tr_ub = None
        if trust_region is not None and x_center is not None:
            center = torch.tensor(
                self._reshape_price(x_center),
                dtype=torch.float32,
                device=self.device,
            )
            radius = trust_region.get_radius()
            tr_lb = torch.clamp(
                center - radius, min=model.pi_min, max=model.pi_max
            )
            tr_ub = torch.clamp(
                center + radius, min=model.pi_min, max=model.pi_max
            )

        best_merit = np.inf
        best_x = None
        best_objective = None

        def diagnostic_snapshot(
            iteration: int,
            loss: torch.Tensor,
            aux: dict,
            tr_penalty: torch.Tensor,
            gradient_available: bool,
        ) -> dict[str, float]:
            pi_value = aux["pi"].detach().cpu().numpy().reshape(-1)
            eq_pen = float(aux["eq_pen"].detach().cpu())
            ineq_pen = float(aux["ineq_pen"].detach().cpu())
            soc_pen = float(aux["soc_pen"].detach().cpu())
            bound_pen = float(aux["bound_pen"].detach().cpu())
            gradients = (
                [
                    parameter.grad.detach().reshape(-1)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if gradient_available
                else []
            )
            if gradients:
                gradient = torch.cat(gradients)
                gradient_l2 = float(torch.linalg.vector_norm(gradient).cpu())
                gradient_inf = float(torch.max(torch.abs(gradient)).cpu())
            else:
                gradient_l2 = float("nan")
                gradient_inf = float("nan")
            distance_inf = (
                float(
                    np.max(
                        np.abs(pi_value - np.asarray(x_center).reshape(-1))
                    )
                )
                if x_center is not None
                else float("nan")
            )
            return {
                "iteration": int(iteration),
                "merit": float(loss.detach().cpu()),
                "raw_surrogate_objective": float(aux["obj"].detach().cpu()),
                "eq_pen": eq_pen,
                "ineq_pen": ineq_pen,
                "soc_pen": soc_pen,
                "bound_pen": bound_pen,
                "weighted_penalty": float(
                    penalty_weights["eq"] * eq_pen
                    + penalty_weights["ineq"] * ineq_pen
                    + penalty_weights["soc"] * soc_pen
                    + penalty_weights["bound"] * bound_pen
                ),
                "tr_penalty": float(tr_penalty.detach().cpu()),
                "gradient_l2": gradient_l2,
                "gradient_inf": gradient_inf,
                "distance_inf": distance_inf,
                "distance_over_R": (
                    distance_inf / trust_region.get_radius()
                    if trust_region is not None and np.isfinite(distance_inf)
                    else float("nan")
                ),
            }

        def record_best(
            loss: torch.Tensor,
            aux: dict,
            iteration: int,
            tr_penalty: torch.Tensor,
        ) -> None:
            nonlocal best_merit, best_x, best_objective, best_diagnostics
            merit = float(loss.detach().cpu())
            if np.isfinite(merit) and merit < best_merit:
                best_merit = merit
                best_x = aux["pi"].detach().cpu().numpy().copy()
                best_objective = float(aux["obj"].detach().cpu())
                if collect_diagnostics:
                    best_diagnostics = diagnostic_snapshot(
                        iteration,
                        loss,
                        aux,
                        tr_penalty,
                        gradient_available=False,
                    )

        for iteration in range(solve_params["steps"]):
            optimizer.zero_grad()
            loss, aux = model.forward(penalty_weights)
            tr_penalty = loss.new_zeros(())
            if tr_lb is not None and tr_ub is not None:
                pi_current = model.bounded_pi()
                tr_violation = torch.maximum(
                    torch.relu(tr_lb - pi_current).max(),
                    torch.relu(pi_current - tr_ub).max(),
                )
                tr_penalty = tr_weight * tr_violation**2
                loss = loss + tr_penalty
            record_best(loss, aux, iteration, tr_penalty)
            loss.backward()
            if (not uses_torch_surrogate) and getattr(
                surrogate, "_residuals", []
            ):
                self._add_residual_gradient(model, surrogate)
            if collect_diagnostics and (
                iteration == 0
                or (iteration + 1) % diagnostic_trace_every == 0
            ):
                diagnostic_trace.append(
                    diagnostic_snapshot(
                        iteration,
                        loss,
                        aux,
                        tr_penalty,
                        gradient_available=True,
                    )
                )
            optimizer.step()
            if tr_lb is not None and tr_ub is not None:
                with torch.no_grad():
                    pi_projected = torch.clamp(
                        model.bounded_pi(), min=tr_lb, max=tr_ub
                    )
                    scaled = (pi_projected - model.pi_min) / (
                        model.pi_max - model.pi_min
                    )
                    scaled = torch.clamp(scaled, 1e-6, 1 - 1e-6)
                    model.pi_raw.copy_(torch.log(scaled / (1 - scaled)))

        with torch.no_grad():
            final_loss, final_aux = model.forward(penalty_weights)
            final_tr_penalty = final_loss.new_zeros(())
            if tr_lb is not None and tr_ub is not None:
                pi_current = model.bounded_pi()
                tr_violation = torch.maximum(
                    torch.relu(tr_lb - pi_current).max(),
                    torch.relu(pi_current - tr_ub).max(),
                )
                final_tr_penalty = tr_weight * tr_violation**2
                final_loss = final_loss + final_tr_penalty
            record_best(
                final_loss,
                final_aux,
                int(solve_params["steps"]),
                final_tr_penalty,
            )
            if collect_diagnostics:
                diagnostic_trace.append(
                    diagnostic_snapshot(
                        int(solve_params["steps"]),
                        final_loss,
                        final_aux,
                        final_tr_penalty,
                        gradient_available=False,
                    )
                )

        if best_x is None or best_objective is None:
            raise RuntimeError(
                "Leader surrogate optimization produced no finite iterate."
            )
        x_opt = best_x.reshape(-1) if self.independent_prices else best_x
        if collect_diagnostics:
            self.last_solve_diagnostics = {
                "steps": int(solve_params["steps"]),
                "lr": float(solve_params["lr"]),
                "trace_every": diagnostic_trace_every,
                "copt_warm_start_seconds": copt_warm_start_seconds,
                "copt_warm_start_objective": copt_warm_start_objective,
                "best": best_diagnostics,
                "initial": diagnostic_trace[0] if diagnostic_trace else None,
                "final": diagnostic_trace[-1] if diagnostic_trace else None,
                "trace": diagnostic_trace,
            }
        return x_opt, best_objective


def build_trust_region_experiment(
    config: TrustRegionExperimentConfig,
) -> tuple[base.TRResidualController, dict]:
    """Build the original experiment and replace only its leader instance."""
    controller, metadata = base.build_trust_region_experiment(config)
    original = controller.leader
    controller.leader = DistributionNetworkLeader(
        original.constants, device=original.device
    )
    solve_params = dict(controller.solve_params or {})
    solve_params["copt_initialize_network_state"] = bool(
        config.copt_initialize_network_state
    )
    controller.solve_params = solve_params

    optimizer_diagnostics_history: list[dict[str, Any]] = []
    original_solve = controller.leader.solve

    def solve_with_diagnostics(*args: Any, **kwargs: Any) -> tuple:
        solved = original_solve(*args, **kwargs)
        diagnostics = controller.leader.last_solve_diagnostics
        if (
            diagnostics is not None
            and diagnostics.get("copt_warm_start_objective") is not None
        ):
            optimizer_diagnostics_history.append(diagnostics)
        return solved

    controller.leader.solve = solve_with_diagnostics
    controller.optimizer_diagnostics_history = optimizer_diagnostics_history

    metadata["variant"] = "trust_region_b"
    metadata["copt_initialize_network_state"] = bool(
        config.copt_initialize_network_state
    )
    metadata["algo_params"] = dict(metadata["algo_params"])
    metadata["algo_params"]["solve_params"] = dict(solve_params)
    return controller, metadata


def run_trust_region_experiment(config: TrustRegionExperimentConfig) -> dict:
    """Build, run, and save one isolated trust-region-B experiment."""
    controller, metadata = build_trust_region_experiment(config)
    start = time.time()
    result = controller.run()
    result["elapsed_seconds"] = time.time() - start
    result["params"] = metadata
    result["optimizer_diagnostics"] = controller.leader.last_solve_diagnostics
    result["optimizer_diagnostics_history"] = (
        controller.optimizer_diagnostics_history
    )
    saved_paths: dict[str, Any] = {}
    if config.save_results:
        saved_paths.update(
            base.save_trust_region_result(result, output_dir=config.output_dir)
        )
    if config.save_true_responses:
        saved_paths["true_responses"] = base.save_trust_region_true_responses(
            result.get("true_responses", {}),
            output_dir=config.true_response_dir,
            tag=config.true_response_tag,
        )
    if saved_paths:
        result["saved_paths"] = saved_paths
    return result


__all__ = [
    "DistributionNetworkLeader",
    "TrustRegionExperimentConfig",
    "build_trust_region_experiment",
    "run_trust_region_experiment",
]
