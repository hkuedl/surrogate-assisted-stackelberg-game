"""Tests for the isolated 108-bus MV Oberrhein scalability experiment."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import pytest

import stackelberg.mv_oberrhein_scaling as scaling
from stackelberg.network import extract_upper_constants


def test_modified_mv_oberrhein_is_a_108_bus_20kv_tree() -> None:
    net = scaling.create_mv_oberrhein_108()
    graph = scaling._active_line_graph(net)

    assert len(net.bus) == 108
    assert np.allclose(net.bus["vn_kv"], 20.0)
    assert len(net.ext_grid) == 1
    assert len(net.trafo) == 0
    assert graph.number_of_nodes() == 108
    assert graph.number_of_edges() == 107
    assert nx.is_tree(graph)
    assert net.sn_mva == pytest.approx(10.0)
    assert net.load["p_mw"].sum() == pytest.approx(20.274)
    assert len(net.sgen) == 0
    assert "original_max_i_ka" in net.line
    assert np.allclose(net.line["max_i_ka"], 99_999.0)

    pp.runpp(net)
    assert net.converged
    assert np.isfinite(net.res_bus["vm_pu"]).all()


def test_modified_network_uses_existing_upper_constant_adapter() -> None:
    net = scaling.create_mv_oberrhein_108()
    placements = scaling.build_microgrid_placements(net)[:5]
    constants = extract_upper_constants(
        {"mg_buses": [item.bus_id for item in placements]},
        net,
        24,
        squared_voltage_limits=True,
    )

    assert constants["n_bus"] == 108
    assert constants["n_line"] == 107
    assert constants["slack_bus_id"] == int(net.ext_grid.iloc[0]["bus"])
    assert constants["s_base_mva"] == pytest.approx(10.0)
    assert constants["p_passive"].shape == (108, 24)
    assert constants["q_passive"].shape == (108, 24)
    assert constants["mg_buses_original"] == [
        item.bus_id for item in placements
    ]


def test_twenty_placements_are_unique_balanced_and_nested() -> None:
    net = scaling.create_mv_oberrhein_108()
    placements = scaling.build_microgrid_placements(net)

    assert len(placements) == 20
    assert len({item.bus_id for item in placements}) == 20
    assert [item.scenario for item in placements] == list(
        scaling.SCENARIO_BLOCK
    ) * 4
    assert [item.block for item in placements] == [
        block for block in range(1, 5) for _ in range(5)
    ]
    feeder_counts = {
        feeder: sum(item.feeder_root_bus == feeder for item in placements)
        for feeder in {item.feeder_root_bus for item in placements}
    }
    assert sorted(feeder_counts.values()) == [10, 10]
    ip_feeder_counts = {
        feeder: sum(
            item.feeder_root_bus == feeder and item.scenario == "IP1"
            for item in placements
        )
        for feeder in feeder_counts
    }
    assert sorted(ip_feeder_counts.values()) == [2, 2]

    for count in scaling.MICROGRID_COUNTS:
        prefix = placements[:count]
        assert [item.scenario for item in prefix] == list(
            scaling.SCENARIO_BLOCK
        ) * (count // 5)
        assert set(item.bus_id for item in prefix) <= set(
            item.bus_id for item in placements
        )


def test_repeated_scenarios_build_fresh_lower_networks() -> None:
    scenarios = list(scaling.SCENARIO_BLOCK) * 2
    networks = scaling.build_lower_networks(scenarios)

    assert len(networks) == len(scenarios)
    assert len({id(net) for net in networks}) == len(networks)
    assert networks[0].bus is not networks[5].bus
    assert np.allclose(networks[4].bus["vn_kv"], 20.0)
    assert np.allclose(networks[9].bus["vn_kv"], 20.0)


def test_controller_timing_is_attached_only_to_instrumented_instance() -> None:
    class FakeLeader:
        def solve(self, value: int) -> int:
            return value + 1

    class FakeController:
        def __init__(self) -> None:
            self.leader = FakeLeader()

        def _collect_residual_fit_data(self, value: int) -> int:
            return value * 2

        def _one_iteration(self, k: int, x_k: np.ndarray):
            sampled = self._collect_residual_fit_data(k)
            solved = self.leader.solve(sampled)
            return {"k": k, "value": solved}, x_k, False

    untouched = FakeController()
    controller = FakeController()
    scaling._instrument_controller(controller)

    record, _, _ = controller._one_iteration(0, np.zeros(1))

    assert record["iteration_seconds"] >= 0.0
    assert record["residual_sampling_seconds"] >= 0.0
    assert record["leader_surrogate_solve_seconds"] >= 0.0
    assert record["other_iteration_seconds"] >= 0.0
    untouched_record, _, _ = untouched._one_iteration(0, np.zeros(1))
    assert "iteration_seconds" not in untouched_record


def test_controller_records_common_copt_warm_start_diagnostics() -> None:
    class FakeLeader:
        last_solve_diagnostics = None

        def solve(self, value: int) -> int:
            self.last_solve_diagnostics = {
                "copt_warm_start_seconds": 0.125,
                "copt_warm_start_objective": -4.5,
            }
            return value + 1

        def evaluate_objective(
            self, x_value: np.ndarray, y_value: np.ndarray
        ) -> float:
            return float(np.sum(x_value) + np.sum(y_value))

    class FakeController:
        def __init__(self) -> None:
            self.leader = FakeLeader()
            self.sample_tolerance = 1e-9
            self.eps_f = 1e-9

        def _collect_residual_fit_data(self, k: int, x_k: np.ndarray):
            X_fit = np.asarray([x_k, x_k + 0.1], dtype=float)
            Y_true = np.zeros((2, 1), dtype=float)
            return k, [(X_fit, Y_true)]

        def _one_iteration(self, k: int, x_k: np.ndarray):
            sampled, _ = self._collect_residual_fit_data(k, x_k)
            self.leader.solve(sampled)
            record = {
                "F_tilde_current": 1.0,
                "F_tilde_candidate": 0.5,
                "F_current": 1.0,
                "F_candidate": 0.25,
                "x_cand": x_k + 0.05,
                "x_k": x_k,
                "R_before": 0.1,
            }
            return record, x_k, False

    controller = FakeController()
    scaling._instrument_controller(controller, collect_diagnostics=True)

    record, _, _ = controller._one_iteration(0, np.zeros(1))

    assert record["copt_warm_start_seconds"] == pytest.approx(0.125)
    assert record["copt_warm_start_objective"] == pytest.approx(-4.5)


def test_mv108_config_passes_common_copt_switch_to_tr_config() -> None:
    config = scaling.MVScalingExperimentConfig(
        copt_initialize_network_state=True
    )

    tr_config = scaling._build_tr_config(
        config,
        upper_net=object(),
        lower_nets=[],
        mg_buses=[],
        scenarios=[],
        warm_start=np.zeros(config.T),
        output_dir=Path("unused"),
    )

    assert tr_config.copt_initialize_network_state is True


def test_default_scaling_configuration_preserves_agreed_budgets() -> None:
    config = scaling.MVScalingExperimentConfig()

    assert tuple(config.microgrid_counts) == (5, 10, 15, 20)
    assert config.seed == 42
    assert config.nn_steps == 10_000
    assert config.tr_max_iterations == 10
    assert config.tr_n_samples == 60
    assert config.tr_steps_warm == 1_000
    assert config.copt_initialize_network_state is False
    assert config.output_dir == "results/mv_oberrhein_scaling"


def test_case_digest_allows_expanding_a_resumed_batch() -> None:
    first = scaling.MVScalingExperimentConfig(
        microgrid_counts=(5,),
        output_dir="first",
        resume=False,
    )
    expanded = scaling.MVScalingExperimentConfig(
        microgrid_counts=(5, 10, 15, 20),
        output_dir="expanded",
        resume=True,
    )

    assert scaling._settings_digest(first, 5) == scaling._settings_digest(
        expanded, 5
    )
    assert scaling._settings_digest(first, 5) != scaling._settings_digest(
        expanded, 10
    )


def test_analysis_compares_exponential_fit(tmp_path) -> None:
    for count, seconds in zip((5, 10, 15, 20), (1.0, 2.0, 4.0, 8.0)):
        run_dir = tmp_path / f"mg_{count:02d}"
        run_dir.mkdir()
        manifest = {
            "status": "complete",
            "n_microgrids": count,
            "n_bus": 108,
            "n_line": 107,
            "decision_size": count * 24,
            "n_iterations": 10,
            "stop_reason": "max_iterations",
            "tr_elapsed_seconds": seconds * 10,
            "seconds_per_iteration": seconds,
            "median_seconds_per_iteration": seconds,
            "std_seconds_per_iteration": 0.1,
            "mean_residual_sampling_seconds": seconds * 0.4,
            "mean_leader_surrogate_solve_seconds": seconds * 0.5,
            "mean_other_iteration_seconds": seconds * 0.1,
            "queries_per_iteration": 61,
            "composite_queries": 610,
            "individual_follower_solves": 610 * count,
            "seconds_per_composite_query": seconds / 61,
            "equivalent_seconds_per_follower_solve": seconds / (61 * count),
            "warm_start_true_objective": 1.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

    summary = scaling.analyze_scaling_results(tmp_path)
    fit = json.loads((tmp_path / "scaling_fit.json").read_text())

    assert summary["n_microgrids"].tolist() == [5, 10, 15, 20]
    assert fit["best_descriptive_fit_by_r_squared"] == "exponential"
    assert fit["exponential_seconds_per_iteration"][
        "multiplicative_factor_per_microgrid"
    ] == pytest.approx(2.0 ** (1.0 / 5.0))
    assert (tmp_path / "scaling_runtime.pdf").is_file()
    assert (tmp_path / "scaling_runtime_loglog.pdf").is_file()
