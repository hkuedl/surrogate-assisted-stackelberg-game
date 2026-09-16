"""Network constants and topology shared by upper-level solvers."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from .pricing import sample_electricity_price
from .profiles import daily_load_curve
from .randomness import get_global_seed


CASE33BW_TARGET_VOLTAGE_KV = 20.0


def create_case33bw_20kv() -> pp.pandapowerNet:
    """Create case33bw at 20 kV while preserving its per-unit line impedances."""
    net = pn.case33bw()
    original_voltage_kv = np.asarray(net.bus["vn_kv"], dtype=float)
    if original_voltage_kv.size == 0 or not np.allclose(
        original_voltage_kv, original_voltage_kv[0]
    ):
        raise ValueError("case33bw must have one common, positive nominal voltage.")
    source_voltage_kv = float(original_voltage_kv[0])
    if source_voltage_kv <= 0.0:
        raise ValueError("case33bw nominal voltage must be positive.")

    impedance_scale = (CASE33BW_TARGET_VOLTAGE_KV / source_voltage_kv) ** 2
    net.bus.loc[:, "vn_kv"] = CASE33BW_TARGET_VOLTAGE_KV
    for column in ("r_ohm_per_km", "x_ohm_per_km", "r0_ohm_per_km", "x0_ohm_per_km"):
        if column in net.line:
            net.line.loc[:, column] = net.line[column].astype(float) * impedance_scale
    return net


def build_topology(
    n_bus: int, from_bus: list[int], to_bus: list[int]
) -> Tuple[Dict[int, list[int]], Dict[int, int | None]]:
    """Build outgoing-edge and parent-edge lookups for a radial network."""
    children: Dict[int, list[int]] = {i: [] for i in range(n_bus)}
    parent_edge: Dict[int, int | None] = {i: None for i in range(n_bus)}
    for edge, (source, target) in enumerate(zip(from_bus, to_bus)):
        source, target = int(source), int(target)
        if source not in children or target not in parent_edge:
            raise ValueError(f"Branch {edge} references a bus outside [0, {n_bus}).")
        if parent_edge[target] is not None:
            raise ValueError(f"Bus {target} has more than one parent; the network is not radial.")
        children[source].append(edge)
        parent_edge[target] = edge
    return children, parent_edge


def _orient_radial_lines(
    net: pp.pandapowerNet,
) -> tuple[Any, list[int], list[int], list[int], Dict[int, int], int]:
    """Validate and orient an in-service, line-only radial network from its slack bus."""
    active_bus_ids = [
        int(bus)
        for bus, row in net.bus.iterrows()
        if bool(row.get("in_service", True))
    ]
    if not active_bus_ids:
        raise ValueError("The upper network must contain at least one in-service bus.")
    bus_lookup = {bus: index for index, bus in enumerate(active_bus_ids)}

    ext_grid_mask = (
        net.ext_grid["in_service"].astype(bool)
        if "in_service" in net.ext_grid
        else np.ones(len(net.ext_grid), dtype=bool)
    )
    active_ext_grids = net.ext_grid[ext_grid_mask]
    if len(active_ext_grids) != 1:
        raise ValueError("A radial upper network requires exactly one in-service ext_grid.")
    slack_bus_id = int(active_ext_grids.iloc[0]["bus"])
    if slack_bus_id not in bus_lookup:
        raise ValueError("The ext_grid must be connected to an in-service bus.")
    slack = bus_lookup[slack_bus_id]

    if hasattr(net, "trafo") and len(net.trafo):
        trafo_mask = (
            net.trafo["in_service"].astype(bool)
            if "in_service" in net.trafo
            else np.ones(len(net.trafo), dtype=bool)
        )
        active_trafos = net.trafo[trafo_mask]
        if len(active_trafos):
            raise NotImplementedError(
                "The current DistFlow adapter supports radial networks made of lines only; "
                "in-service transformers require an explicit transformer branch model."
            )

    line_df = net.line[net.line["in_service"].astype(bool)].copy()
    if hasattr(net, "switch") and len(net.switch):
        open_line_switches = net.switch[
            (net.switch["et"] == "l") & (~net.switch["closed"].astype(bool))
        ]
        if len(open_line_switches):
            line_df = line_df.drop(
                index=line_df.index.intersection(open_line_switches["element"])
            )
    line_df = line_df.reset_index().rename(columns={"index": "line_id"})
    if len(line_df) != len(active_bus_ids) - 1:
        raise ValueError(
            "The active upper network must be a connected tree with n_bus - 1 lines."
        )

    adjacency: Dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(active_bus_ids))
    }
    for row_index, row in line_df.iterrows():
        from_bus_id, to_bus_id = int(row["from_bus"]), int(row["to_bus"])
        if from_bus_id not in bus_lookup or to_bus_id not in bus_lookup:
            raise ValueError("Every active line must connect two in-service buses.")
        source, target = bus_lookup[from_bus_id], bus_lookup[to_bus_id]
        adjacency[source].append((target, row_index))
        adjacency[target].append((source, row_index))

    queue = deque([slack])
    visited = {slack}
    oriented_rows: list[int] = []
    from_bus: list[int] = []
    to_bus: list[int] = []
    while queue:
        source = queue.popleft()
        for target, row_index in adjacency[source]:
            if target in visited:
                continue
            visited.add(target)
            queue.append(target)
            oriented_rows.append(row_index)
            from_bus.append(source)
            to_bus.append(target)
    if len(visited) != len(active_bus_ids):
        raise ValueError("The active upper network must be connected to the ext_grid.")

    oriented_lines = line_df.iloc[oriented_rows].reset_index(drop=True)
    return oriented_lines, from_bus, to_bus, active_bus_ids, bus_lookup, slack


def build_passive_profiles(
    net: pp.pandapowerNet,
    T: int,
    load_curve: np.ndarray,
    s_base_mva: float,
    bus_lookup: Dict[int, int] | None = None,
) -> Dict[str, np.ndarray]:
    """Build per-bus passive active/reactive demand in per unit."""
    if bus_lookup is None:
        bus_lookup = {int(bus): index for index, bus in enumerate(net.bus.index)}
    n_bus = len(bus_lookup)
    p_passive = np.zeros((n_bus, T), dtype=float)
    q_passive = np.zeros((n_bus, T), dtype=float)
    for _, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_id = int(row["bus"])
        if bus_id not in bus_lookup:
            continue
        bus = bus_lookup[bus_id]
        p_passive[bus] += float(row["p_mw"]) / s_base_mva * load_curve
        q_passive[bus] += float(row["q_mvar"]) / s_base_mva * load_curve
    return {"p_passive": p_passive, "q_passive": q_passive}


def extract_upper_constants(
    input: Dict[str, Any],
    net: pp.pandapowerNet,
    T: int,
    *,
    squared_voltage_limits: bool,
    price_sampler: Callable[[int], np.ndarray] = sample_electricity_price,
) -> Dict[str, Any]:
    """Extract leader constants from any connected, line-only radial pandapower net."""
    if T <= 0:
        raise ValueError("T must be positive.")
    (
        line_df,
        from_bus,
        to_bus,
        bus_ids,
        bus_lookup,
        slack,
    ) = _orient_radial_lines(net)
    n_bus = len(bus_ids)
    n_line = len(line_df)
    slack_bus_id = bus_ids[slack]
    v_base_kv = float(net.bus.loc[slack_bus_id, "vn_kv"])
    s_base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    z_base_ohm = v_base_kv**2 / s_base_mva
    i_base_ka = s_base_mva / (np.sqrt(3.0) * v_base_kv)
    load_curve = daily_load_curve(T)

    r = (
        np.asarray(line_df["r_ohm_per_km"] * line_df["length_km"], dtype=float)
        / z_base_ohm
    )
    x = (
        np.asarray(line_df["x_ohm_per_km"] * line_df["length_km"], dtype=float)
        / z_base_ohm
    )
    if "max_i_ka" in line_df:
        i_max_ka = np.asarray(line_df["max_i_ka"], dtype=float)
    elif "i_max_ka" in line_df:
        i_max_ka = np.asarray(line_df["i_max_ka"], dtype=float)
    else:
        i_max_ka = np.full(n_line, 10.0, dtype=float)
    l_max = (i_max_ka / i_base_ka) ** 2
    lambda_g = np.asarray(price_sampler(T), dtype=float)
    profiles = build_passive_profiles(
        net, T, load_curve, s_base_mva, bus_lookup=bus_lookup
    )
    default_v_min, default_v_max = (
        (0.80**2, 1.05**2) if squared_voltage_limits else (0.80, 1.05)
    )
    v_min = float(input.get("v_min", default_v_min))
    v_max = float(input.get("v_max", default_v_max))
    original_mg_buses = [int(bus) for bus in input.get("mg_buses", [])]
    unknown_mg_buses = [bus for bus in original_mg_buses if bus not in bus_lookup]
    if unknown_mg_buses:
        raise ValueError(
            f"Microgrid buses are not active upper-network buses: {unknown_mg_buses}."
        )
    mg_buses = [bus_lookup[bus] for bus in original_mg_buses]

    return {
        "T": T,
        "n_bus": n_bus,
        "n_line": n_line,
        "slack": slack,
        "slack_bus_id": slack_bus_id,
        "bus_ids": bus_ids,
        "bus_lookup": bus_lookup,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "r": r,
        "x": x,
        "l_max": l_max,
        "load_curve": load_curve,
        "v_base_kv": v_base_kv,
        "s_base_mva": s_base_mva,
        "z_base_ohm": z_base_ohm,
        "i_base_ka": i_base_ka,
        "lamda_g": lambda_g,
        "lambda_g": lambda_g,
        "random_seed": int(input.get("seed", get_global_seed())),
        "pi_min": float(input.get("pi_min", 0.0)),
        "pi_max": float(input.get("pi_max", 1.5)),
        "p0_max": float(input.get("p0_max", 10.0)),
        "q0_max": float(input.get("q0_max", 10.0)),
        "v_ref": float(input.get("v_ref", 1.0)),
        "v_min": v_min,
        "v_max": v_max,
        "mg_buses": mg_buses,
        "mg_buses_original": original_mg_buses,
        "independent_prices": bool(input.get("independent_prices", False)),
        "p_mg": input.get("p_mg", 0),
        "q_mg": input.get("q_mg", 0),
        **profiles,
        "verbose": False,
    }

