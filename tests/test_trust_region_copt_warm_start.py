"""Focused tests for the opt-in COPT network-state warm start."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import stackelberg.trust_region as trust_region


class _Surrogate:
    def __init__(self) -> None:
        self.predict_calls: list[np.ndarray] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.predict_calls.append(np.asarray(x, dtype=float).copy())
        return np.zeros(2, dtype=float)

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(2, dtype=x.dtype, device=x.device)


class _Evaluator:
    def __init__(self, state: dict[str, np.ndarray]) -> None:
        self.calls = 0

        def one_dimensional(values: np.ndarray) -> dict:
            return {
                index: SimpleNamespace(x=float(value))
                for index, value in enumerate(values)
            }

        def two_dimensional(values: np.ndarray) -> dict:
            return {
                (i, t): SimpleNamespace(x=float(values[i, t]))
                for i in range(values.shape[0])
                for t in range(values.shape[1])
            }

        self.p0 = one_dimensional(state["p0"])
        self.q0 = one_dimensional(state["q0"])
        self.voltage = two_dimensional(state["v"])
        self.current = two_dimensional(state["l"])
        self.p_flow = two_dimensional(state["p"])
        self.q_flow = two_dimensional(state["q"])
        self.q_injection = two_dimensional(state["q_injection"])

    def evaluate(self, pi: np.ndarray, p_mg: np.ndarray) -> float:
        self.calls += 1
        return 7.5


def _state() -> dict[str, np.ndarray]:
    return {
        "p0": np.array([0.11, 0.12]),
        "q0": np.array([0.21, 0.22]),
        "v": np.array([[1.01, 1.02], [1.03, 1.04]]),
        "l": np.array([[0.31, 0.32]]),
        "p": np.array([[0.41, 0.42]]),
        "q": np.array([[0.51, 0.52]]),
        "q_injection": np.array([[0.61, 0.62], [0.63, 0.64]]),
    }


def _leader() -> trust_region.DistributionNetworkLeader:
    constants = {
        "T": 2,
        "n_bus": 2,
        "n_line": 1,
        "slack": 0,
        "from_bus": np.array([0]),
        "to_bus": np.array([1]),
        "r": np.array([0.01]),
        "x": np.array([0.02]),
        "l_max": np.array([10.0]),
        "lambda_g": np.zeros(2),
        "s_base_mva": 1.0,
        "pi_min": 0.0,
        "pi_max": 1.5,
        "p0_max": 10.0,
        "q0_max": 10.0,
        "p_net_min": -10.0,
        "p_net_max": 10.0,
        "q_net_min": -10.0,
        "q_net_max": 10.0,
        "v_ref": 1.0,
        "v_min": 0.8,
        "v_max": 1.2,
        "mg_buses": [1],
        "independent_prices": False,
        "p_passive": np.zeros((2, 2)),
        "q_passive": np.zeros((2, 2)),
    }
    leader = object.__new__(trust_region.DistributionNetworkLeader)
    leader.constants = constants
    leader.device = "cpu"
    leader.T = 2
    leader.n_bus = 2
    leader.n_line = 1
    leader.independent_prices = False
    leader.n_followers = 1
    leader.decision_size = 2
    leader.last_solve_diagnostics = None
    return leader


def _trust_region() -> trust_region.TrustRegion:
    return trust_region.TrustRegion(0.1, 1.0, 0.7, 1.5)


def test_default_and_steps_init_do_not_call_copt() -> None:
    center = np.array([0.4, 0.6])
    for region, enabled in ((_trust_region(), False), (None, True)):
        leader = _leader()
        evaluator = _Evaluator(_state())
        leader.distflow_evaluator = evaluator
        leader.solve(
            _Surrogate(),
            trust_region=region,
            x_center=center,
            solve_params={
                "steps": 0,
                "lr": 1e-2,
                "copt_initialize_network_state": enabled,
            },
        )
        assert evaluator.calls == 0


def test_local_solve_calls_copt_once_and_copies_all_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = _leader()
    state = _state()
    evaluator = _Evaluator(state)
    leader.distflow_evaluator = evaluator
    surrogate = _Surrogate()
    created_models = []
    original_subproblem = trust_region._LeaderSubproblem

    def capture_subproblem(*args, **kwargs):
        model = original_subproblem(*args, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(trust_region, "_LeaderSubproblem", capture_subproblem)
    solve_kwargs = {
        "trust_region": _trust_region(),
        "x_center": np.array([0.4, 0.6]),
        "solve_params": {
            "steps": 0,
            "lr": 1e-2,
            "copt_initialize_network_state": True,
            "diagnostic_trace_every": 1,
        },
    }
    result = leader.solve(surrogate, **solve_kwargs)

    assert len(result) == 2
    assert evaluator.calls == 1
    assert len(surrogate.predict_calls) == 1
    model = created_models[0]
    for name, expected in state.items():
        parameter = getattr(model, name)
        assert parameter.dtype == torch.float32
        assert parameter.device == torch.device("cpu")
        assert parameter.grad is None
        np.testing.assert_allclose(parameter.detach().numpy(), expected)
    assert leader.last_solve_diagnostics["copt_warm_start_objective"] == 7.5
    assert leader.last_solve_diagnostics["copt_warm_start_seconds"] >= 0.0

    leader.solve(surrogate, **solve_kwargs)
    assert evaluator.calls == 2
    assert len(surrogate.predict_calls) == 2


def test_copt_state_shape_mismatch_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = _leader()
    state = _state()
    state["p0"] = np.zeros(3)
    monkeypatch.setattr(
        leader,
        "_copt_network_state",
        lambda x_center, surrogate: (state, 0.0),
    )

    with pytest.raises(
        RuntimeError,
        match=r"COPT warm-start shape mismatch for p0: expected \(2,\), got \(3,\)",
    ):
        leader.solve(
            _Surrogate(),
            trust_region=_trust_region(),
            x_center=np.array([0.4, 0.6]),
            solve_params={
                "steps": 0,
                "lr": 1e-2,
                "copt_initialize_network_state": True,
            },
        )


def test_experiment_config_default_is_disabled() -> None:
    field = trust_region.TrustRegionExperimentConfig.__dataclass_fields__[
        "copt_initialize_network_state"
    ]
    assert field.default is False
