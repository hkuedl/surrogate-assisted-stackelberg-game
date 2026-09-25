"""Isolated 108-bus MV Oberrhein scalability experiment.

This module deliberately builds on the existing public experiment APIs without
changing their defaults.  It creates a line-only 20 kV version of the larger
MV Oberrhein feeder, maps repeated copies of the five existing microgrid
instances to nested bus sets, and runs NN-warm-started trust-region experiments
for 5, 10, 15, and 20 followers.

All outputs are written below ``results/mv_oberrhein_scaling`` by default, so
the existing notebooks and result files are never replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

from .followers.bg import create_cigre_residential_network
from .followers.ip import create_industrial_park_network
from .instances import scenario_type
from .nn_solver import NNSolverExperimentConfig, run_nn_solver_experiment
from .randomness import DEFAULT_SEED, set_global_seed
from .trust_region import (
    TrustRegionExperimentConfig,
    build_trust_region_experiment,
    save_trust_region_result,
)


SCENARIO_BLOCK = ("SB1", "SB2", "BG1", "BG2", "IP1")
MICROGRID_COUNTS = (5, 10, 15, 20)
MV_OBERRHEIN_BASE_MVA = 10.0
RELAXED_LINE_CURRENT_KA = 99_999.0


@dataclass(frozen=True)
class MicrogridPlacement:
    """One named follower and its deterministic 20 kV connection bus."""

    instance_id: str
    scenario: str
    block: int
    bus_id: int
    bus_name: str
    feeder_root_bus: int
    resistance_distance: float


@dataclass
class MVScalingExperimentConfig:
    """Configuration for the isolated MV Oberrhein scalability runner."""

    microgrid_counts: Sequence[int] = MICROGRID_COUNTS
    T: int = 24
    seed: int = DEFAULT_SEED
    device: str = "cpu"
    nn_steps: int = 10_000
    nn_lr: float = 1e-2
    tr_max_iterations: int = 10
    tr_n_samples: int = 60
    tr_steps_warm: int = 1_000
    tr_lr: float = 1e-2
    lower_time_limit: float | None = 30.0
    lower_mip_gap: float | None = 1e-3
    output_dir: str = "results/mv_oberrhein_scaling"
    warm_start_path: str | None = None
    collect_diagnostics: bool = False
    diagnostic_trace_every: int = 25
    copt_initialize_network_state: bool = False
    save_results: bool = True
    resume: bool = True
    run_feasibility_check: bool = True


def _active_line_graph(net: pp.pandapowerNet) -> nx.Graph:
    """Return the closed, in-service line graph with impedance weights."""
    graph = nx.Graph()
    graph.add_nodes_from(
        int(bus)
        for bus, row in net.bus.iterrows()
        if bool(row.get("in_service", True))
    )
    open_line_ids = set()
    if len(net.switch):
        open_line_ids = {
            int(value)
            for value in net.switch.loc[
                (net.switch["et"] == "l")
                & (~net.switch["closed"].astype(bool)),
                "element",
            ]
        }
    for line_id, row in net.line.iterrows():
        if not bool(row.get("in_service", True)) or int(line_id) in open_line_ids:
            continue
        source, target = int(row["from_bus"]), int(row["to_bus"])
        if source not in graph or target not in graph:
            continue
        resistance = float(row["r_ohm_per_km"]) * float(row["length_km"])
        reactance = float(row["x_ohm_per_km"]) * float(row["length_km"])
        impedance = math.hypot(resistance, reactance)
        graph.add_edge(
            source,
            target,
            line_id=int(line_id),
            resistance_ohm=resistance,
            impedance_ohm=max(impedance, 1e-12),
        )
    return graph


def _validate_mv_oberrhein_108(net: pp.pandapowerNet) -> None:
    """Fail loudly if the modified network no longer matches its contract."""
    active_buses = net.bus[net.bus["in_service"].astype(bool)]
    active_ext_grids = net.ext_grid[net.ext_grid["in_service"].astype(bool)]
    active_trafos = net.trafo[net.trafo["in_service"].astype(bool)]
    graph = _active_line_graph(net)
    if len(active_buses) != 108:
        raise ValueError(f"Expected 108 in-service buses, got {len(active_buses)}.")
    if not np.allclose(active_buses["vn_kv"].to_numpy(dtype=float), 20.0):
        raise ValueError("Every MV Oberrhein scalability bus must be at 20 kV.")
    if len(active_ext_grids) != 1:
        raise ValueError("The modified network requires exactly one ext_grid.")
    if len(active_trafos):
        raise ValueError("The modified network must be transformer-free.")
    if graph.number_of_nodes() != 108 or graph.number_of_edges() != 107:
        raise ValueError(
            "The closed-line topology must contain 108 nodes and 107 edges; "
            f"got {graph.number_of_nodes()} and {graph.number_of_edges()}."
        )
    if not nx.is_tree(graph):
        raise ValueError("The modified MV Oberrhein network must be a connected tree.")
    root_bus = int(active_ext_grids.iloc[0]["bus"])
    if root_bus not in graph:
        raise ValueError("The 20 kV ext_grid bus must belong to the active tree.")
    if not np.isclose(float(net.sn_mva), MV_OBERRHEIN_BASE_MVA):
        raise ValueError(
            f"Expected sn_mva={MV_OBERRHEIN_BASE_MVA}, got {net.sn_mva}."
        )


def create_mv_oberrhein_108() -> pp.pandapowerNet:
    """Create the isolated 108-bus, 20 kV, line-only scalability network.

    The larger of the two ``mv_oberrhein`` substation areas is selected.  Its
    upstream 110/20 kV transformer is represented by an ideal 20 kV source at
    the transformer secondary.  The built-in ``scenario='load'`` scaling is
    materialized into the load values before the generic Stackelberg profile
    builder is used.  Static generators have zero scaling in that scenario and
    are removed explicitly.

    Original line ratings are preserved in ``original_max_i_ka`` while the
    optimization rating is relaxed to match the non-binding thermal-limit
    convention of the existing converted case33 experiment.
    """
    separated = pn.mv_oberrhein(
        scenario="load",
        include_substations=False,
        separation_by_sub=True,
    )
    if not isinstance(separated, tuple) or len(separated) != 2:
        raise RuntimeError("mv_oberrhein must return two separated subnetworks.")
    net = deepcopy(separated[1])

    active_trafos = net.trafo[net.trafo["in_service"].astype(bool)]
    if len(active_trafos) != 1:
        raise ValueError("The selected Oberrhein area must contain one transformer.")
    transformer = active_trafos.iloc[0]
    hv_bus = int(transformer["hv_bus"])
    root_bus = int(transformer["lv_bus"])
    if not np.isclose(float(net.bus.loc[root_bus, "vn_kv"]), 20.0):
        raise ValueError("The selected transformer secondary must be a 20 kV bus.")

    # Materialize the scenario scaling because the shared passive-profile
    # adapter intentionally reads physical p_mw/q_mvar values only.
    load_scaling = (
        net.load["scaling"].astype(float)
        if "scaling" in net.load
        else pd.Series(1.0, index=net.load.index)
    )
    net.load.loc[:, "p_mw"] = net.load["p_mw"].astype(float) * load_scaling
    net.load.loc[:, "q_mvar"] = net.load["q_mvar"].astype(float) * load_scaling
    if "scaling" in net.load:
        net.load.loc[:, "scaling"] = 1.0

    # In the built-in load scenario every sgen has scaling=0.  Check that
    # invariant before removing the disabled rows instead of silently ignoring
    # active generation.
    if len(net.sgen):
        sgen_scaling = (
            net.sgen["scaling"].astype(float)
            if "scaling" in net.sgen
            else pd.Series(1.0, index=net.sgen.index)
        )
        active_generation = np.asarray(net.sgen["p_mw"], dtype=float) * np.asarray(
            sgen_scaling, dtype=float
        )
        if not np.allclose(active_generation, 0.0):
            raise ValueError(
                "scenario='load' unexpectedly contains active static generation."
            )
        net.sgen.drop(index=net.sgen.index, inplace=True)

    # Replace the upstream transformer and 110 kV grid by an ideal 20 kV PCC.
    net.ext_grid.drop(index=net.ext_grid.index, inplace=True)
    net.trafo.drop(index=net.trafo.index, inplace=True)
    net.bus.drop(index=hv_bus, inplace=True)
    pp.create_ext_grid(
        net,
        bus=root_bus,
        vm_pu=1.0,
        name="MV Oberrhein 20 kV PCC",
    )

    net.sn_mva = MV_OBERRHEIN_BASE_MVA
    net.line.loc[:, "original_max_i_ka"] = net.line["max_i_ka"].astype(float)
    net.line.loc[:, "max_i_ka"] = RELAXED_LINE_CURRENT_KA
    net.name = "Modified 108-bus 20 kV MV Oberrhein scalability network"
    net["scaling_metadata"] = {
        "source": "pandapower.networks.mv_oberrhein",
        "source_scenario": "load",
        "separated_subnetwork": 1,
        "root_bus": root_bus,
        "removed_hv_bus": hv_bus,
        "base_mva": MV_OBERRHEIN_BASE_MVA,
        "thermal_limits_relaxed": True,
        "relaxed_line_current_ka": RELAXED_LINE_CURRENT_KA,
        "active_sgen_removed": 0,
    }
    _validate_mv_oberrhein_108(net)
    return net


def _farthest_point_subset(
    graph: nx.Graph,
    candidates: Sequence[int],
    root_distances: Mapping[int, float],
    count: int,
) -> list[int]:
    """Select an impedance-spread deterministic subset from one feeder."""
    available = sorted({int(bus) for bus in candidates})
    if count < 0 or count > len(available):
        raise ValueError(
            f"Cannot select {count} buses from {len(available)} candidates."
        )
    if count == 0:
        return []
    selected = [
        max(available, key=lambda bus: (float(root_distances[bus]), -bus))
    ]
    while len(selected) < count:
        remaining = [bus for bus in available if bus not in selected]

        def separation_key(bus: int) -> tuple[float, float, int]:
            nearest = min(
                nx.shortest_path_length(
                    graph, bus, chosen, weight="impedance_ohm"
                )
                for chosen in selected
            )
            return float(nearest), float(root_distances[bus]), -bus

        selected.append(max(remaining, key=separation_key))
    return selected


def build_microgrid_placements(
    net: pp.pandapowerNet,
) -> list[MicrogridPlacement]:
    """Return the fixed nested placement order for twenty microgrids."""
    _validate_mv_oberrhein_108(net)
    root_bus = int(net.ext_grid[net.ext_grid["in_service"].astype(bool)].iloc[0]["bus"])
    graph = _active_line_graph(net)
    feeder_roots = sorted(int(bus) for bus in graph.neighbors(root_bus))
    if len(feeder_roots) != 2:
        raise ValueError(
            f"Expected two root feeders, found {len(feeder_roots)}."
        )
    root_distances = nx.single_source_dijkstra_path_length(
        graph, root_bus, weight="impedance_ohm"
    )
    load_buses = sorted(
        {
            int(bus)
            for bus in net.load.loc[
                net.load["p_mw"].astype(float) > 0.0, "bus"
            ]
            if int(bus) != root_bus
        }
    )
    feeder_candidates: dict[int, list[int]] = {
        feeder_root: [] for feeder_root in feeder_roots
    }
    for bus in load_buses:
        path = nx.shortest_path(graph, root_bus, bus)
        if len(path) < 2 or int(path[1]) not in feeder_candidates:
            raise ValueError(f"Load bus {bus} is not assigned to a root feeder.")
        feeder_candidates[int(path[1])].append(bus)

    ip_sites: dict[int, list[int]] = {}
    other_sites: dict[int, list[int]] = {}
    for feeder_root in feeder_roots:
        candidates = feeder_candidates[feeder_root]
        if len(candidates) < 10:
            raise ValueError(
                f"Feeder {feeder_root} needs at least ten load-bus candidates."
            )
        ip_sites[feeder_root] = sorted(
            candidates,
            key=lambda bus: (float(root_distances[bus]), bus),
        )[:2]
        remaining = [
            bus for bus in candidates if bus not in ip_sites[feeder_root]
        ]
        other_sites[feeder_root] = _farthest_point_subset(
            graph, remaining, root_distances, count=8
        )

    # Each block contains one copy of the five standard instances.  Alternating
    # 2+3 and 3+2 allocations leaves ten connections on each root feeder and
    # two IP instances close to the PCC on each feeder.
    block_counts = (
        (2, 3),
        (3, 2),
        (2, 3),
        (3, 2),
    )
    ip_feeder_by_block = (
        feeder_roots[0],
        feeder_roots[1],
        feeder_roots[0],
        feeder_roots[1],
    )
    ip_offsets = {feeder_root: 0 for feeder_root in feeder_roots}
    other_offsets = {feeder_root: 0 for feeder_root in feeder_roots}
    placements: list[MicrogridPlacement] = []
    for block_index, counts in enumerate(block_counts, start=1):
        ip_feeder = ip_feeder_by_block[block_index - 1]
        ip_offset = ip_offsets[ip_feeder]
        ip_bus = ip_sites[ip_feeder][ip_offset]
        ip_offsets[ip_feeder] += 1

        non_ip_buses: list[tuple[int, int]] = []
        for feeder_root, requested_count in zip(feeder_roots, counts):
            non_ip_count = requested_count - int(feeder_root == ip_feeder)
            start = other_offsets[feeder_root]
            stop = start + non_ip_count
            selected = other_sites[feeder_root][start:stop]
            if len(selected) != non_ip_count:
                raise RuntimeError("The deterministic placement pool was exhausted.")
            other_offsets[feeder_root] = stop
            non_ip_buses.extend((feeder_root, bus) for bus in selected)
        non_ip_buses.sort(
            key=lambda item: (
                feeder_roots.index(item[0]),
                float(root_distances[item[1]]),
                item[1],
            )
        )
        scenario_to_location = {
            scenario: location
            for scenario, location in zip(SCENARIO_BLOCK[:-1], non_ip_buses)
        }
        scenario_to_location["IP1"] = (ip_feeder, ip_bus)

        for scenario in SCENARIO_BLOCK:
            feeder_root, bus = scenario_to_location[scenario]
            placements.append(
                MicrogridPlacement(
                    instance_id=f"{scenario}_copy{block_index}",
                    scenario=scenario,
                    block=block_index,
                    bus_id=int(bus),
                    bus_name=str(net.bus.loc[bus, "name"]),
                    feeder_root_bus=int(feeder_root),
                    resistance_distance=float(root_distances[bus]),
                )
            )

    if len(placements) != 20 or len({item.bus_id for item in placements}) != 20:
        raise RuntimeError("The twenty microgrid placements must be unique.")
    if [item.scenario for item in placements] != list(SCENARIO_BLOCK) * 4:
        raise RuntimeError("Every placement block must retain the standard order.")
    feeder_totals = {
        feeder_root: sum(
            item.feeder_root_bus == feeder_root for item in placements
        )
        for feeder_root in feeder_roots
    }
    if set(feeder_totals.values()) != {10}:
        raise RuntimeError(
            f"Expected ten placements per root feeder, got {feeder_totals}."
        )
    return placements


def build_lower_networks(scenarios: Sequence[str]) -> list[pp.pandapowerNet]:
    """Build a fresh lower network for every repeated follower instance."""
    factories: dict[str, Callable[[], pp.pandapowerNet]] = {
        "SB": pn.create_cigre_network_lv,
        "BG": create_cigre_residential_network,
        "IP": create_industrial_park_network,
    }
    networks = []
    for scenario in scenarios:
        base = scenario_type(scenario)
        if base not in factories:
            raise ValueError(f"No lower-network factory for {scenario!r}.")
        networks.append(factories[base]())
    return networks


def build_scaling_case(
    n_microgrids: int,
) -> tuple[
    pp.pandapowerNet,
    list[pp.pandapowerNet],
    list[int],
    list[str],
    list[MicrogridPlacement],
]:
    """Construct one nested 5/10/15/20-follower experiment case."""
    count = int(n_microgrids)
    if count not in MICROGRID_COUNTS:
        raise ValueError(
            f"n_microgrids must be one of {MICROGRID_COUNTS}, got {count}."
        )
    upper_net = create_mv_oberrhein_108()
    placements = build_microgrid_placements(upper_net)[:count]
    scenarios = [item.scenario for item in placements]
    mg_buses = [item.bus_id for item in placements]
    lower_nets = build_lower_networks(scenarios)
    return upper_net, lower_nets, mg_buses, scenarios, placements


def _instrument_controller(
    controller: Any,
    *,
    collect_diagnostics: bool = False,
    progress_path: Path | None = None,
) -> None:
    """Attach per-iteration timers to one controller instance only."""
    progress_records: list[dict[str, Any]] = []
    stage = {
        "sampling": 0.0,
        "leader_solve": 0.0,
        "diagnostics": 0.0,
        "optimizer_diagnostics": None,
    }
    original_sampling = controller._collect_residual_fit_data
    original_leader_solve = controller.leader.solve
    original_iteration = controller._one_iteration

    def timed_sampling(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        result = original_sampling(*args, **kwargs)
        stage["sampling"] += time.perf_counter() - start
        if collect_diagnostics:
            diagnostic_start = time.perf_counter()
            _, fit_data = result
            x_center = np.asarray(
                args[1] if len(args) > 1 else kwargs["x_k"], dtype=float
            ).reshape(-1)
            X_blocks = [
                np.asarray(X_fit, dtype=float) for X_fit, _ in fit_data
            ]
            Y_blocks = [
                np.asarray(Y_true, dtype=float) for _, Y_true in fit_data
            ]
            row_counts = {len(values) for values in X_blocks + Y_blocks}
            if len(row_counts) != 1:
                raise RuntimeError(
                    "Follower-local diagnostic designs do not share row counts."
                )
            joint_X = np.concatenate(X_blocks, axis=1)
            joint_Y = np.stack(Y_blocks, axis=1)
            true_objectives = np.asarray([
                controller.leader.evaluate_objective(joint_X[index], joint_Y[index])
                for index in range(len(joint_X))
            ], dtype=float)
            distances = np.max(np.abs(joint_X - x_center), axis=1)
            center_matches = distances <= controller.sample_tolerance
            if not np.any(center_matches):
                raise RuntimeError("The diagnostic local design has no center row.")
            center_index = int(np.flatnonzero(center_matches)[0])
            F_current = float(true_objectives[center_index])
            improvements = F_current - true_objectives
            improving = improvements > float(controller.eps_f)
            noncenter = ~center_matches
            best_index = int(np.argmin(true_objectives))
            best_noncenter_index = (
                int(np.flatnonzero(noncenter)[np.argmin(true_objectives[noncenter])])
                if np.any(noncenter)
                else center_index
            )
            controller._mv_diagnostic_artifacts = {
                "sample_X": joint_X,
                "sample_Y": joint_Y,
                "sample_true_objectives": true_objectives,
                "sample_improvements": improvements,
                "sample_distances_inf": distances,
                "sample_center_mask": center_matches,
                "sample_improving_mask": improving,
                "sample_summary": {
                    "F_current": F_current,
                    "best_sample_true_objective": float(
                        true_objectives[best_index]
                    ),
                    "best_sample_improvement": float(improvements[best_index]),
                    "number_of_improving_samples": int(np.count_nonzero(improving)),
                    "best_sample_index": best_index,
                    "best_sample_distance_inf": float(distances[best_index]),
                    "best_noncenter_sample_true_objective": float(
                        true_objectives[best_noncenter_index]
                    ),
                    "best_noncenter_sample_improvement": float(
                        improvements[best_noncenter_index]
                    ),
                    "improvement_tolerance": float(controller.eps_f),
                    "n_samples": int(len(joint_X)),
                },
            }
            stage["diagnostics"] += time.perf_counter() - diagnostic_start
        return result

    def timed_leader_solve(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            result = original_leader_solve(*args, **kwargs)
            if collect_diagnostics:
                stage["optimizer_diagnostics"] = getattr(
                    controller.leader, "last_solve_diagnostics", None
                )
            return result
        finally:
            stage["leader_solve"] += time.perf_counter() - start

    def timed_iteration(k: int, x_k: np.ndarray):
        stage["sampling"] = 0.0
        stage["leader_solve"] = 0.0
        stage["diagnostics"] = 0.0
        stage["optimizer_diagnostics"] = None
        start = time.perf_counter()
        record, x_next, converged = original_iteration(k, x_k)
        elapsed = time.perf_counter() - start
        record["iteration_seconds"] = elapsed
        record["residual_sampling_seconds"] = stage["sampling"]
        record["leader_surrogate_solve_seconds"] = stage["leader_solve"]
        record["diagnostic_hard_distflow_seconds"] = stage["diagnostics"]
        record["other_iteration_seconds"] = max(
            elapsed
            - stage["sampling"]
            - stage["leader_solve"]
            - stage["diagnostics"],
            0.0,
        )
        if collect_diagnostics:
            artifacts = getattr(controller, "_mv_diagnostic_artifacts", {})
            record.update(artifacts.get("sample_summary", {}))
            record["Pred"] = float(
                record["F_tilde_current"] - record["F_tilde_candidate"]
            )
            record["Ared"] = float(
                record["F_current"] - record["F_candidate"]
            )
            candidate_distance = float(
                np.max(
                    np.abs(
                        np.asarray(record["x_cand"], dtype=float)
                        - np.asarray(record["x_k"], dtype=float)
                    )
                )
            )
            record["candidate_distance_inf"] = candidate_distance
            record["candidate_distance_over_R"] = (
                candidate_distance / float(record["R_before"])
            )
            optimizer = stage["optimizer_diagnostics"]
            if optimizer is not None:
                artifacts["optimizer_diagnostics"] = optimizer
                record["copt_warm_start_seconds"] = optimizer.get(
                    "copt_warm_start_seconds", 0.0
                )
                record["copt_warm_start_objective"] = optimizer.get(
                    "copt_warm_start_objective"
                )
                for label in ("initial", "final", "best"):
                    snapshot = optimizer.get(label)
                    if snapshot is None:
                        continue
                    for key, value in snapshot.items():
                        record[f"optimizer_{label}_{key}"] = value
        if progress_path is not None:
            progress_records.append(record.copy())
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            _scalar_history_frame(progress_records).to_csv(temporary, index=False)
            temporary.replace(progress_path)
        return record, x_next, converged

    controller._collect_residual_fit_data = timed_sampling
    controller.leader.solve = timed_leader_solve
    controller._one_iteration = timed_iteration


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _settings_payload(config: MVScalingExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["microgrid_counts"] = [int(value) for value in config.microgrid_counts]
    return _jsonable(payload)


def _case_settings_payload(
    config: MVScalingExperimentConfig,
    n_microgrids: int,
) -> dict[str, Any]:
    """Return settings that determine one case's numerical result.

    Batch selection, output location, and the resume switch are deliberately
    excluded.  This lets a completed five-microgrid case be reused when a
    later invocation expands the requested batch to 5/10/15/20 cases.
    """
    payload = _settings_payload(config)
    for key in ("microgrid_counts", "output_dir", "resume"):
        payload.pop(key, None)
    payload["n_microgrids"] = int(n_microgrids)
    return payload


def _settings_digest(
    config: MVScalingExperimentConfig,
    n_microgrids: int,
) -> str:
    encoded = json.dumps(
        _case_settings_payload(config, n_microgrids),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _scalar_history_frame(history: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in history:
        rows.append(
            {
                key: value
                for key, value in record.items()
                if value is None
                or isinstance(value, (str, bool, int, float, np.generic))
            }
        )
    return pd.DataFrame(rows)


def _build_tr_config(
    config: MVScalingExperimentConfig,
    *,
    upper_net: pp.pandapowerNet,
    lower_nets: Sequence[pp.pandapowerNet],
    mg_buses: Sequence[int],
    scenarios: Sequence[str],
    warm_start: np.ndarray,
    output_dir: Path,
) -> TrustRegionExperimentConfig:
    return TrustRegionExperimentConfig(
        upper_net=upper_net,
        lower_nets=lower_nets,
        mg_buses=mg_buses,
        scenarios=scenarios,
        T=config.T,
        seed=config.seed,
        device=config.device,
        x0=warm_start,
        lower_time_limit=config.lower_time_limit,
        lower_mip_gap=config.lower_mip_gap,
        copt_initialize_network_state=config.copt_initialize_network_state,
        tr_params={
            "R_0": 0.05,
            "R_max": 1.0,
            "gamma_dec": 0.7,
            "gamma_inc": 1.5,
            "eta_1": 0.1,
            "eta_2": 0.75,
        },
        algo_params={
            "n_samples": config.tr_n_samples,
            "K_max": config.tr_max_iterations,
            "sampling_method": "lhs",
            "residual_update_mode": "accumulate",
            "rbf_geometry_condition_max": 1e12,
            "solve_params": {
                "steps_init": 0,
                "steps_warm": config.tr_steps_warm,
                "lr": config.tr_lr,
                "penalty_weights": {
                    "eq": 1000.0,
                    "ineq": 100.0,
                    "soc": 100.0,
                    "bound": 10.0,
                },
                "tr_weight": 1e6,
                "verbose": False,
                "diagnostic_trace_every": (
                    config.diagnostic_trace_every
                    if config.collect_diagnostics
                    else 0
                ),
            },
        },
        residual_model_params={
            "model_type": "rbf",
            "rbf_kernel": "cubic",
            "rbf_smoothing": 1e-3,
            "rbf_degree": 1,
            "rbf_scale": "trust_radius",
            "rbf_epsilon": None,
        },
        save_results=config.save_results,
        output_dir=str(output_dir),
        save_true_responses=False,
    )


def _run_nn_warm_start(
    config: MVScalingExperimentConfig,
    *,
    upper_net: pp.pandapowerNet,
    lower_nets: Sequence[pp.pandapowerNet],
    mg_buses: Sequence[int],
    scenarios: Sequence[str],
) -> tuple[np.ndarray, float]:
    x0 = np.full((len(scenarios), config.T), 0.75, dtype=float)
    start = time.perf_counter()
    result = run_nn_solver_experiment(
        NNSolverExperimentConfig(
            upper_net=upper_net,
            lower_nets=lower_nets,
            mg_buses=mg_buses,
            scenarios=scenarios,
            T=config.T,
            seed=config.seed,
            device=config.device,
            x0=x0,
            steps=config.nn_steps,
            lr=config.nn_lr,
            save_every=0,
            evaluate_true=False,
            save_results=False,
        )
    )
    elapsed = time.perf_counter() - start
    prices = np.asarray(result["optimal pi"], dtype=float).reshape(
        len(scenarios), config.T
    )
    return prices, elapsed


def _check_warm_start_feasibility(
    tr_config: TrustRegionExperimentConfig,
    warm_start: np.ndarray,
) -> float:
    """Evaluate one true response and hard DistFlow objective before timing."""
    controller, _ = build_trust_region_experiment(tr_config)
    prices = np.asarray(warm_start, dtype=float).reshape(
        len(tr_config.scenarios), tr_config.T
    )
    lower_result = controller.follower.lower_solve(prices)
    if lower_result is None:
        raise RuntimeError("The NN warm start is infeasible for a lower problem.")
    objective = controller.leader.evaluate_objective(
        prices.reshape(-1), lower_result["p_mg"]
    )
    if not np.isfinite(objective):
        raise RuntimeError("The NN warm-start hard DistFlow objective is not finite.")
    return float(objective)


def run_single_scaling_case(
    config: MVScalingExperimentConfig,
    n_microgrids: int,
) -> dict[str, Any]:
    """Run one NN-warm-started TR case and persist it in an isolated folder."""
    count = int(n_microgrids)
    output_root = Path(config.output_dir)
    run_dir = output_root / f"mg_{count:02d}"
    manifest_path = run_dir / "manifest.json"
    digest = _settings_digest(config, count)
    if config.resume and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("settings_digest") != digest:
            raise ValueError(
                f"Existing run uses different settings: {manifest_path}"
            )
        if saved.get("status") == "complete":
            print(f"Skipping completed MV scaling run with {count} microgrids.")
            return saved

    run_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    upper_net, lower_nets, mg_buses, scenarios, placements = build_scaling_case(
        count
    )
    warm_start_source_path = None
    if config.warm_start_path is None:
        warm_start, nn_seconds = _run_nn_warm_start(
            config,
            upper_net=upper_net,
            lower_nets=lower_nets,
            mg_buses=mg_buses,
            scenarios=scenarios,
        )
    else:
        source = Path(config.warm_start_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Warm-start file does not exist: {source}")
        warm_start = np.asarray(np.load(source), dtype=float)
        expected_shape = (count, config.T)
        if warm_start.shape != expected_shape:
            raise ValueError(
                f"Warm start must have shape {expected_shape}, got "
                f"{warm_start.shape}."
            )
        if not np.isfinite(warm_start).all():
            raise ValueError("Warm start contains non-finite values.")
        warm_start_source_path = str(source.resolve())
        nn_seconds = 0.0
    warm_start_path = run_dir / "nn_warm_start.npy"
    np.save(warm_start_path, warm_start)
    placement_path = run_dir / "placements.csv"
    pd.DataFrame([asdict(item) for item in placements]).to_csv(
        placement_path, index=False
    )

    tr_config = _build_tr_config(
        config,
        upper_net=upper_net,
        lower_nets=lower_nets,
        mg_buses=mg_buses,
        scenarios=scenarios,
        warm_start=warm_start,
        output_dir=run_dir,
    )
    warm_start_objective = None
    if config.run_feasibility_check:
        warm_start_objective = _check_warm_start_feasibility(
            tr_config, warm_start
        )

    # Rebuild after the untimed feasibility check so no COPT/model state or
    # follower warm start leaks into the measured trust-region run.
    controller, metadata = build_trust_region_experiment(tr_config)
    metadata["mv108_copt_initialize_network_state"] = bool(
        config.copt_initialize_network_state
    )
    _instrument_controller(
        controller,
        collect_diagnostics=config.collect_diagnostics,
        progress_path=run_dir / "timed_history.partial.csv",
    )
    start = time.perf_counter()
    result = controller.run()
    tr_seconds = time.perf_counter() - start
    result["elapsed_seconds"] = tr_seconds
    result["params"] = metadata

    saved_paths: dict[str, str] = {}
    if config.save_results:
        saved_paths.update(save_trust_region_result(result, output_dir=str(run_dir)))
    timed_history_path = run_dir / "timed_history.csv"
    _scalar_history_frame(result["history"]).to_csv(timed_history_path, index=False)
    saved_paths["timed_history_csv"] = str(timed_history_path)

    if config.collect_diagnostics:
        artifacts = getattr(controller, "_mv_diagnostic_artifacts", {})
        if not artifacts:
            raise RuntimeError("Diagnostic run produced no local-sample artifacts.")
        sample_npz_path = run_dir / "local_samples.npz"
        np.savez_compressed(
            sample_npz_path,
            X=artifacts["sample_X"],
            Y=artifacts["sample_Y"],
            true_objective=artifacts["sample_true_objectives"],
            improvement=artifacts["sample_improvements"],
            distance_inf=artifacts["sample_distances_inf"],
            is_center=artifacts["sample_center_mask"],
            is_improving=artifacts["sample_improving_mask"],
        )
        sample_csv_path = run_dir / "local_sample_objectives.csv"
        pd.DataFrame({
            "sample_index": np.arange(len(artifacts["sample_X"])),
            "is_center": artifacts["sample_center_mask"],
            "true_objective": artifacts["sample_true_objectives"],
            "improvement": artifacts["sample_improvements"],
            "is_improving": artifacts["sample_improving_mask"],
            "distance_inf": artifacts["sample_distances_inf"],
        }).to_csv(sample_csv_path, index=False)
        sample_summary_path = run_dir / "local_sample_summary.json"
        _write_json(sample_summary_path, artifacts["sample_summary"])
        optimizer_path = run_dir / "optimizer_trace.csv"
        optimizer = artifacts.get("optimizer_diagnostics")
        if optimizer is None:
            raise RuntimeError("Diagnostic run produced no optimizer trace.")
        pd.DataFrame(optimizer["trace"]).to_csv(optimizer_path, index=False)
        saved_paths.update({
            "local_samples_npz": str(sample_npz_path),
            "local_sample_objectives_csv": str(sample_csv_path),
            "local_sample_summary_json": str(sample_summary_path),
            "optimizer_trace_csv": str(optimizer_path),
        })

    n_iterations = int(result["n_iter"])
    queries_per_iteration = config.tr_n_samples + 1
    composite_queries = n_iterations * queries_per_iteration
    individual_follower_solves = composite_queries * count
    iteration_seconds = [
        float(row["iteration_seconds"]) for row in result["history"]
    ]
    residual_sampling_seconds = [
        float(row["residual_sampling_seconds"]) for row in result["history"]
    ]
    leader_solve_seconds = [
        float(row["leader_surrogate_solve_seconds"])
        for row in result["history"]
    ]
    other_iteration_seconds = [
        float(row["other_iteration_seconds"]) for row in result["history"]
    ]
    manifest = {
        "status": "complete",
        "settings_digest": digest,
        "settings": _settings_payload(config),
        "n_microgrids": count,
        "n_bus": 108,
        "n_line": 107,
        "decision_size": count * config.T,
        "seed": config.seed,
        "scenarios": scenarios,
        "mg_buses": mg_buses,
        "nn_warm_start_seconds_excluded": nn_seconds,
        "nn_warm_start_path": str(warm_start_path),
        "nn_warm_start_source_path": warm_start_source_path,
        "warm_start_true_objective": warm_start_objective,
        "tr_elapsed_seconds": tr_seconds,
        "n_iterations": n_iterations,
        "stop_reason": result.get("stop_reason"),
        "seconds_per_iteration": (
            float(np.mean(iteration_seconds)) if iteration_seconds else math.nan
        ),
        "median_seconds_per_iteration": (
            float(np.median(iteration_seconds))
            if iteration_seconds
            else math.nan
        ),
        "std_seconds_per_iteration": (
            float(np.std(iteration_seconds, ddof=1))
            if len(iteration_seconds) > 1
            else 0.0
        ),
        "mean_residual_sampling_seconds": (
            float(np.mean(residual_sampling_seconds))
            if residual_sampling_seconds
            else math.nan
        ),
        "mean_leader_surrogate_solve_seconds": (
            float(np.mean(leader_solve_seconds))
            if leader_solve_seconds
            else math.nan
        ),
        "mean_other_iteration_seconds": (
            float(np.mean(other_iteration_seconds))
            if other_iteration_seconds
            else math.nan
        ),
        "composite_queries": composite_queries,
        "queries_per_iteration": queries_per_iteration,
        "individual_follower_solves": individual_follower_solves,
        "seconds_per_composite_query": (
            tr_seconds / composite_queries if composite_queries else math.nan
        ),
        "equivalent_seconds_per_follower_solve": (
            tr_seconds / individual_follower_solves
            if individual_follower_solves
            else math.nan
        ),
        "placement_path": str(placement_path),
        "saved_paths": saved_paths,
    }
    _write_json(manifest_path, manifest)
    return manifest


def analyze_scaling_results(output_dir: str | Path) -> pd.DataFrame:
    """Create CSV/PDF summaries from completed isolated scaling runs."""
    output_path = Path(output_dir)
    manifests = []
    for path in sorted(output_path.glob("mg_*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            manifests.append(payload)
    if not manifests:
        raise ValueError(f"No completed scaling runs found in {output_path}.")
    summary_columns = [
        "n_microgrids",
        "n_bus",
        "n_line",
        "decision_size",
        "n_iterations",
        "stop_reason",
        "tr_elapsed_seconds",
        "seconds_per_iteration",
        "median_seconds_per_iteration",
        "std_seconds_per_iteration",
        "mean_residual_sampling_seconds",
        "mean_leader_surrogate_solve_seconds",
        "mean_other_iteration_seconds",
        "queries_per_iteration",
        "composite_queries",
        "individual_follower_solves",
        "seconds_per_composite_query",
        "equivalent_seconds_per_follower_solve",
        "warm_start_true_objective",
    ]
    summary = pd.DataFrame(manifests).sort_values("n_microgrids")
    summary = summary.loc[:, summary_columns]
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / "scaling_summary.csv", index=False)

    if len(summary) >= 2:
        counts = summary["n_microgrids"].to_numpy(dtype=float)
        seconds_per_iteration = summary["seconds_per_iteration"].to_numpy(
            dtype=float
        )
        finite = np.isfinite(seconds_per_iteration) & (seconds_per_iteration > 0)
        fit_metadata: dict[str, Any] = {
            "interpretation": (
                "Descriptive only: one run per size, four sizes at most, and "
                "a fixed small TR iteration budget. Compare fitted shapes rather than "
                "treating the best fit as proof of asymptotic complexity."
            )
        }
        fitted_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if finite.sum() >= 2:
            fit_counts = counts[finite]
            fit_seconds = seconds_per_iteration[finite]
            dense_counts = np.linspace(fit_counts.min(), fit_counts.max(), 200)

            def r_squared(prediction: np.ndarray) -> float:
                residual_sum = float(np.sum((fit_seconds - prediction) ** 2))
                total_sum = float(
                    np.sum((fit_seconds - np.mean(fit_seconds)) ** 2)
                )
                return (
                    1.0 - residual_sum / total_sum
                    if total_sum > 0.0
                    else math.nan
                )

            linear = np.polyfit(fit_counts, fit_seconds, 1)
            power = np.polyfit(
                np.log(fit_counts), np.log(fit_seconds), 1
            )
            exponential = np.polyfit(fit_counts, np.log(fit_seconds), 1)
            linear_prediction = np.polyval(linear, fit_counts)
            power_prediction = np.exp(
                np.polyval(power, np.log(fit_counts))
            )
            exponential_prediction = np.exp(
                np.polyval(exponential, fit_counts)
            )
            model_scores = {
                "linear": r_squared(linear_prediction),
                "power_law": r_squared(power_prediction),
                "exponential": r_squared(exponential_prediction),
            }
            fit_metadata.update(
                {
                    "linear_seconds_per_iteration": {
                        "slope": float(linear[0]),
                        "intercept": float(linear[1]),
                        "r_squared_original_scale": model_scores["linear"],
                    },
                    "power_law_seconds_per_iteration": {
                        "exponent": float(power[0]),
                        "coefficient": float(np.exp(power[1])),
                        "r_squared_original_scale": model_scores["power_law"],
                    },
                    "exponential_seconds_per_iteration": {
                        "growth_rate_per_microgrid": float(exponential[0]),
                        "coefficient": float(np.exp(exponential[1])),
                        "multiplicative_factor_per_microgrid": float(
                            np.exp(exponential[0])
                        ),
                        "doubling_microgrids": (
                            float(np.log(2.0) / exponential[0])
                            if exponential[0] > 0.0
                            else None
                        ),
                        "r_squared_original_scale": model_scores["exponential"],
                    },
                    "best_descriptive_fit_by_r_squared": max(
                        model_scores, key=model_scores.get
                    ),
                }
            )
            fitted_curves = {
                "linear fit": (dense_counts, np.polyval(linear, dense_counts)),
                "power-law fit": (
                    dense_counts,
                    np.exp(np.polyval(power, np.log(dense_counts))),
                ),
                "exponential fit": (
                    dense_counts,
                    np.exp(np.polyval(exponential, dense_counts)),
                ),
            }
        _write_json(output_path / "scaling_fit.json", fit_metadata)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            counts,
            summary["tr_elapsed_seconds"].to_numpy(dtype=float),
            marker="o",
        )
        axes[0].set(
            xlabel="Number of microgrids",
            ylabel="TR elapsed time (s)",
            title="End-to-end TR time",
        )
        axes[1].plot(counts, seconds_per_iteration, marker="o")
        for label, (fit_x, fit_y) in fitted_curves.items():
            axes[1].plot(fit_x, fit_y, linestyle="--", linewidth=1, label=label)
        axes[1].set(
            xlabel="Number of microgrids",
            ylabel="Seconds per TR iteration",
            title="Iteration-normalized time",
        )
        if fitted_curves:
            axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path / "scaling_runtime.pdf", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(counts, seconds_per_iteration, marker="o")
        ax.set(
            xlabel="Number of microgrids",
            ylabel="Seconds per TR iteration",
            title="Descriptive log-log scaling",
        )
        fig.tight_layout()
        fig.savefig(output_path / "scaling_runtime_loglog.pdf", dpi=180)
        plt.close(fig)
    return summary


def run_mv_oberrhein_scaling_experiment(
    config: MVScalingExperimentConfig | None = None,
) -> pd.DataFrame:
    """Run all requested sizes sequentially and generate their summary."""
    selected = MVScalingExperimentConfig() if config is None else config
    counts = [int(value) for value in selected.microgrid_counts]
    if not counts or any(value not in MICROGRID_COUNTS for value in counts):
        raise ValueError(
            f"microgrid_counts must be a non-empty subset of {MICROGRID_COUNTS}."
        )
    if selected.T != 24:
        raise ValueError("The existing follower surrogates require T=24.")
    if (
        selected.nn_steps <= 0
        or selected.tr_max_iterations <= 0
        or selected.tr_n_samples <= 0
        or selected.tr_steps_warm <= 0
    ):
        raise ValueError("NN, TR, sampling, and warm-step budgets must be positive.")
    for count in counts:
        run_single_scaling_case(selected, count)
    return analyze_scaling_results(selected.output_dir)


__all__ = [
    "MICROGRID_COUNTS",
    "MVScalingExperimentConfig",
    "MicrogridPlacement",
    "SCENARIO_BLOCK",
    "analyze_scaling_results",
    "build_lower_networks",
    "build_microgrid_placements",
    "build_scaling_case",
    "create_mv_oberrhein_108",
    "run_mv_oberrhein_scaling_experiment",
    "run_single_scaling_case",
]
