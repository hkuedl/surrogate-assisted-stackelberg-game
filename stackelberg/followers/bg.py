"""Building-group follower based on the CIGRE LV residential feeder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import coptpy as cp
from coptpy import COPT
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.toolbox import select_subnet

from buildings import (
    Building,
    LargeCommercialBuilding1,
    LargeCommercialBuilding2,
    ResidentialBuilding1,
    ResidentialBuilding2,
)

from ..network import build_topology
from ..randomness import get_global_seed
from .base import FollowerProblem


def create_cigre_residential_network() -> pp.pandapowerNet:
    """Return the standalone 0.4 kV R1--R18 CIGRE residential feeder.

    The upstream 20 kV bus R0 and the R0--R1 transformer are represented by an
    external grid at R1, which is the building group's point of common coupling.
    Original CIGRE bus indices are intentionally retained for traceability.
    """
    cigre_net = pn.create_cigre_network_lv()
    residential_buses = [
        int(bus)
        for bus, row in cigre_net.bus.iterrows()
        if str(row["name"]).startswith("Bus R") and np.isclose(row["vn_kv"], 0.4)
    ]
    net = select_subnet(
        cigre_net,
        residential_buses,
        include_switch_buses=False,
        include_results=False,
        keep_everything_else=False,
    )
    pcc_matches = net.bus.index[net.bus["name"] == "Bus R1"].tolist()
    if len(pcc_matches) != 1:
        raise ValueError("The CIGRE residential feeder must contain exactly one Bus R1.")
    pp.create_ext_grid(net, bus=int(pcc_matches[0]), vm_pu=1.0, name="Residential PCC")
    net.name = "CIGRE LV residential feeder R1-R18"
    return net


def _extract_residential_topology(
    net: pp.pandapowerNet,
) -> tuple[list[int], Dict[int, int], int, Any, list[int], list[int]]:
    """
        Validate and index a source-oriented, line-only radial feeder.
        Output: results of residential subnetwork topology.
    """
    bus_ids = [
        int(bus)
        for bus, row in net.bus.iterrows()
        if bool(row.get("in_service", True))
    ]
    if not bus_ids:
        raise ValueError("The BG network must contain at least one in-service bus.")
    bus_lookup = {bus: index for index, bus in enumerate(bus_ids)}

    active_ext_grids = net.ext_grid[
        net.ext_grid["in_service"].astype(bool)
    ]
    if len(active_ext_grids) != 1:
        raise ValueError("The BG network requires exactly one in-service ext_grid at its PCC.")
    pcc_bus_id = int(active_ext_grids.iloc[0]["bus"])
    if pcc_bus_id not in bus_lookup:
        raise ValueError("The BG ext_grid must be connected to an in-service bus.")
    pcc = bus_lookup[pcc_bus_id]

    if len(net.trafo) and net.trafo["in_service"].astype(bool).any():
        raise ValueError(
            "The BG feeder must start on the 0.4 kV side; remove the upstream transformer."
        )
    lines = net.line[net.line["in_service"].astype(bool)].copy()
    if len(lines) != len(bus_ids) - 1:
        raise ValueError("The BG network must be a radial tree with n_bus - 1 lines.")

    from_bus: list[int] = []
    to_bus: list[int] = []
    parent_count = np.zeros(len(bus_ids), dtype=int)
    for _, row in lines.iterrows():
        source_id, target_id = int(row["from_bus"]), int(row["to_bus"])
        if source_id not in bus_lookup or target_id not in bus_lookup:
            raise ValueError("Every BG line must connect two in-service BG buses.")
        source, target = bus_lookup[source_id], bus_lookup[target_id]
        from_bus.append(source)
        to_bus.append(target)
        parent_count[target] += 1

    expected_parent_count = np.ones(len(bus_ids), dtype=int)
    expected_parent_count[pcc] = 0
    if not np.array_equal(parent_count, expected_parent_count):
        raise ValueError(
            "BG lines must be directed away from the PCC and give each non-PCC bus "
            "exactly one parent."
        )
    return bus_ids, bus_lookup, pcc, lines, from_bus, to_bus


BUILDING_TYPES: tuple[type[Building], ...] = (
    ResidentialBuilding1,
    ResidentialBuilding2,
    LargeCommercialBuilding1,
    LargeCommercialBuilding2,
)


def default_building_mapping(
    bus_ids: Sequence[int], pcc_bus_id: int
) -> Dict[int, type[Building]]:
    """Cycle the four profile classes over all non-PCC buses.

    This is the single place to edit when a physical building-to-bus mapping is
    available.
    """
    building_bus_ids = [int(bus) for bus in bus_ids if int(bus) != pcc_bus_id]
    return {
        bus: BUILDING_TYPES[index % len(BUILDING_TYPES)]
        for index, bus in enumerate(building_bus_ids)
    }


def representative_storage_bus_ids(
    bus_ids: Sequence[int],
    pcc: int,
    from_bus: Sequence[int],
    to_bus: Sequence[int],
    count: int = 3,
) -> list[int]:
    """Select upstream, midpoint, and terminal buses on the longest feeder path."""
    if count <= 0:
        raise ValueError("Storage bus count must be positive.")
    parent: Dict[int, int] = {
        int(target): int(source) for source, target in zip(from_bus, to_bus)
    }
    non_pcc_buses = [bus for bus in range(len(bus_ids)) if bus != pcc]
    if not non_pcc_buses:
        raise ValueError("The BG network needs a non-PCC bus for storage.")

    def path_from_pcc(bus: int) -> list[int]:
        path: list[int] = []
        current = bus
        while current != pcc:
            path.append(current)
            if current not in parent:
                raise ValueError("Every storage candidate must be connected to the PCC.")
            current = parent[current]
        return list(reversed(path))

    paths = {bus: path_from_pcc(bus) for bus in non_pcc_buses}
    terminal = max(
        non_pcc_buses,
        key=lambda bus: (len(paths[bus]), -int(bus_ids[bus])),
    )
    longest_path = paths[terminal]
    selected_count = min(count, len(longest_path))
    positions = np.linspace(0, len(longest_path) - 1, selected_count)
    selected_buses = [longest_path[int(round(position))] for position in positions]
    return [int(bus_ids[bus]) for bus in dict.fromkeys(selected_buses)]


def extract_bg_constants(
    net: pp.pandapowerNet,
    T: int,
    *,
    scenario: str = "BG",
    date: str = "2023-01-01",
    data_dir: str | Path | None = None,
    building_mapping: Mapping[int, type[Building]] | None = None,
    storage_bus_ids: Sequence[int] | None = None,
    verbose: bool = False,
    require_positive_load: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Extract a scaled 24-hour building fleet and radial feeder constants."""
    if scenario.upper() != "BG":
        raise ValueError("extract_bg_constants only accepts scenario='BG'.")
    if T != 24:
        raise ValueError("BG prepared building profiles require T=24.")
    bus_ids, bus_lookup, pcc, lines, from_bus, to_bus = _extract_residential_topology(net)
    n_bus = len(bus_ids)
    pcc_bus_id = bus_ids[pcc]

    base_p_bus = np.zeros(n_bus, dtype=float)
    base_q_bus = np.zeros(n_bus, dtype=float)
    for _, row in net.load.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_id = int(row["bus"])
        if bus_id not in bus_lookup:
            raise ValueError(f"BG load references an inactive bus: {bus_id}.")
        bus = bus_lookup[bus_id]
        base_p_bus[bus] += float(row["p_mw"])
        base_q_bus[bus] += float(row["q_mvar"])
    base_p = float(base_p_bus.sum())
    base_q = float(base_q_bus.sum())
    if require_positive_load and base_p <= 0.0:
        raise ValueError("The BG network must contain positive active load.")

    mapping = (
        default_building_mapping(bus_ids, pcc_bus_id)
        if building_mapping is None
        else {int(bus): building_type for bus, building_type in building_mapping.items()}
    )
    expected_buses = set(bus_ids) - {pcc_bus_id}
    if set(mapping) != expected_buses:
        missing = sorted(expected_buses - set(mapping))
        extra = sorted(set(mapping) - expected_buses)
        raise ValueError(
            f"Building mapping must cover every non-PCC bus exactly once; "
            f"missing={missing}, extra={extra}."
        )
    if not all(isinstance(cls, type) and issubclass(cls, Building) for cls in mapping.values()):
        raise TypeError("Every building mapping value must be a Building subclass.")

    building_bus_ids = [bus for bus in bus_ids if bus != pcc_bus_id]
    building_buses = [bus_lookup[bus] for bus in building_bus_ids]
    selected_storage_bus_ids = (
        representative_storage_bus_ids(bus_ids, pcc, from_bus, to_bus)
        if storage_bus_ids is None
        else list(dict.fromkeys(int(bus) for bus in storage_bus_ids))
    )
    invalid_storage_buses = sorted(set(selected_storage_bus_ids) - set(building_bus_ids))
    if invalid_storage_buses:
        raise ValueError(
            "Storage buses must be active non-PCC building buses; "
            f"invalid={invalid_storage_buses}."
        )
    if not selected_storage_bus_ids:
        raise ValueError("At least one storage bus must be selected.")
    storage_buildings = [
        building_bus_ids.index(bus) for bus in selected_storage_bus_ids
    ]
    storage_buses = [bus_lookup[bus] for bus in selected_storage_bus_ids]
    buildings = [
        mapping[bus](date) if data_dir is None else mapping[bus](date, data_dir)
        for bus in building_bus_ids
    ]
    raw_load_kw = np.vstack([building.get_load_curve() for building in buildings])
    raw_pv_kw = np.vstack([building.get_pv_curve() for building in buildings])
    outdoor_temperature = np.vstack(
        [building.get_temperature_curve() for building in buildings]
    )
    raw_mean_mw = float(raw_load_kw.mean(axis=1).sum() / 1000.0)
    if raw_mean_mw <= 0.0:
        raise ValueError("Prepared BG building profiles must contain positive load.")
    profile_scale = base_p / raw_mean_mw  # this keep the building load is the same as the original load from pandapower.
    p_load = raw_load_kw / 1000.0 * profile_scale  # the load is in MW
    p_pv = raw_pv_kw / 1000.0 * profile_scale  # the PV is in MW
    q_to_p_ratio = base_q / base_p if base_p > 0.0 else 0.0
    q_load = p_load * q_to_p_ratio

    mean_building_load = p_load.mean(axis=1)
    p_hvac_max = np.maximum(0.3 * mean_building_load, 1e-6)
    p_heat_max = p_hvac_max.copy()
    p_cool_max = p_hvac_max.copy()
    storage_power = np.maximum(0.25 * mean_building_load, 1e-6)
    p_ch_max = np.zeros(len(buildings), dtype=float)
    p_ch_max[storage_buildings] = storage_power.sum() / len(storage_buildings)
    p_dis_max = p_ch_max.copy()
    storage_energy = np.maximum(2.0 * mean_building_load, 1e-6)
    e_max = np.zeros(len(buildings), dtype=float)
    e_max[storage_buildings] = storage_energy.sum() / len(storage_buildings)
    e_min = 0.1 * e_max
    e_init = 0.5 * e_max
    tau_ref = np.full(len(buildings), 25.0)
    tau_min = np.full(len(buildings), 20.0)
    tau_max = np.full(len(buildings), 30.0)
    tau_init = tau_ref.copy()
    alpha = np.full(len(buildings), 0.9)
    gamma = 1.0 - alpha
    beta_hvac = 1.2 / p_hvac_max
    beta_heat = beta_hvac.copy()
    beta_cool = beta_hvac.copy()

    # mapping building load to bus load, because the zeroth node is the PCC, which does not contain a building or load.
    p_load_bus = np.zeros((n_bus, T), dtype=float)
    q_load_bus = np.zeros((n_bus, T), dtype=float)
    p_pv_bus = np.zeros((n_bus, T), dtype=float)
    for building, bus in enumerate(building_buses):
        p_load_bus[bus] = p_load[building]
        q_load_bus[bus] = q_load[building]
        p_pv_bus[bus] = p_pv[building]

    v_base_kv = float(net.bus.loc[pcc_bus_id, "vn_kv"])
    s_base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    z_base_ohm = v_base_kv**2 / s_base_mva
    i_base_ka = s_base_mva / (np.sqrt(3.0) * v_base_kv)
    r_ohm = np.asarray(lines["r_ohm_per_km"] * lines["length_km"], dtype=float)
    x_ohm = np.asarray(lines["x_ohm_per_km"] * lines["length_km"], dtype=float)
    max_i_ka = np.asarray(lines["max_i_ka"], dtype=float)
    children, parent_edge = build_topology(n_bus, from_bus, to_bus)

    constants: Dict[str, Any] = {
        "T": T,
        "n_bus": n_bus,
        "n_line": len(lines),
        "n_building": len(buildings),
        "bus_ids": bus_ids,
        "bus_lookup": bus_lookup,
        "bus_names": [str(net.bus.loc[bus, "name"]) for bus in bus_ids],
        "pcc": pcc,
        "pcc_bus_id": pcc_bus_id,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "children": children,
        "parent_edge": parent_edge,
        "line_ids": [int(line) for line in lines.index],
        "line_names": [str(name) for name in lines["name"]],
        "r_ohm": r_ohm,
        "x_ohm": x_ohm,
        "r": r_ohm / z_base_ohm,
        "x": x_ohm / z_base_ohm,
        "l_max": (max_i_ka / i_base_ka) ** 2,
        "v_base_kv": v_base_kv,
        "s_base_mva": s_base_mva,
        "z_base_ohm": z_base_ohm,
        "i_base_ka": i_base_ka,
        "v_ref": 1.0,
        "v_min": 0.85**2,
        "v_max": 1.05**2,
        "p_mg_max": 3.0 * base_p,
        "q_mg_max": 3.0 * max(base_q, 0.25 * base_p),
        "base_p_bus": base_p_bus,
        "base_q_bus": base_q_bus,
        "profile_scale": profile_scale,
        "building_bus_ids": building_bus_ids,
        "building_buses": building_buses,
        "n_storage": len(storage_buildings),
        "n_storage_binary": len(storage_buildings) * T,
        "n_hvac_binary": 2 * len(buildings) * T,
        "n_binary": (len(storage_buildings) + 2 * len(buildings)) * T,
        "storage_bus_ids": selected_storage_bus_ids,
        "storage_buses": storage_buses,
        "storage_buildings": storage_buildings,
        "building_names": [building.name for building in buildings],
        "building_types": [building.building_type for building in buildings],
        "building_classes": [type(building).__name__ for building in buildings],
        "profile_date": str(buildings[0].date.date()),
        "p_load_building": p_load,
        "q_load_building": q_load,
        "p_pv_building": p_pv,
        "outdoor_temperature": outdoor_temperature,
        "p_load_bus": p_load_bus,
        "q_load_bus": q_load_bus,
        "p_pv_bus": p_pv_bus,
        "p_load": p_load.sum(axis=0),
        "q_load": q_load.sum(axis=0),
        "p_hvac_max": p_hvac_max,
        "p_heat_max": p_heat_max,
        "p_cool_max": p_cool_max,
        "p_ch_max": p_ch_max,
        "p_dis_max": p_dis_max,
        "eta_ch": 0.95,
        "eta_dis": 0.92,
        "e_min": e_min,
        "e_max": e_max,
        "e_init": e_init,
        "alpha": alpha,
        "beta_heat": beta_heat,
        "beta_cool": beta_cool,
        "gamma": gamma,
        "tau_ref": tau_ref,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_init": tau_init,
        "omega": np.ones(len(buildings)),
        "c_cur": np.ones((len(buildings), T)),
        "verbose": bool(verbose),
        "random_seed": get_global_seed() if seed is None else int(seed),
    }
    return constants


class BuildingGroupFollowerProblem(FollowerProblem):
    """Building fleet scheduling with a radial LinDistFlow feeder model."""

    scenario = "BG"

    def __init__(
        self,
        T: int = 24,
        net: pp.pandapowerNet | None = None,
        verbose: bool = False,
        *,
        date: str = "2023-01-01",
        data_dir: str | Path | None = None,
        building_mapping: Mapping[int, type[Building]] | None = None,
        storage_bus_ids: Sequence[int] | None = None,
        lower_constants: Dict[str, Any] | None = None,
        failure_mode: str = "raise",
        include_metadata: bool = True,
        time_limit: float | None = None,
        mip_gap: float | None = None,
    ) -> None:
        if failure_mode not in {"none", "raise"}:
            raise ValueError("failure_mode must be 'none' or 'raise'.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        if mip_gap is not None and mip_gap < 0:
            raise ValueError("mip_gap must be nonnegative.")
        self.net = self.default_network() if net is None else net
        self.constants = (
            extract_bg_constants(
                self.net,
                T,
                date=date,
                data_dir=data_dir,
                building_mapping=building_mapping,
                storage_bus_ids=storage_bus_ids,
                verbose=verbose,
                require_positive_load=True,
            )
            if lower_constants is None
            else lower_constants
        )
        self.T = int(self.constants.get("T", T))
        self.failure_mode = failure_mode
        self.include_metadata = include_metadata
        self.verbose = bool(self.constants.get("verbose", verbose))
        self.random_seed = int(self.constants.get("random_seed", get_global_seed()))
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.last_status: int | None = None
        self._has_mip_start = False
        self._mip_start_values: list[float] = []
        required = {
            "n_bus",
            "n_line",
            "n_building",
            "n_storage",
            "pcc",
            "from_bus",
            "to_bus",
            "building_buses",
            "storage_buildings",
            "p_load_building",
            "q_load_building",
            "p_pv_building",
        }
        missing = sorted(required - self.constants.keys())
        if missing:
            raise ValueError(f"BG constants are missing required keys: {missing}.")
        self._build_model()

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        """Return the standalone CIGRE LV residential feeder."""
        return create_cigre_residential_network()

    def clear_mip_start(self) -> None:
        """Clear the incumbent used to warm-start the next BG solve."""
        self._has_mip_start = False
        self._mip_start_values = []

    def _build_model(self) -> None:
        """Build and cache the price-parametric BG model."""
        c = self.constants
        n_bus = int(c["n_bus"])
        n_line = int(c["n_line"])
        n_building = int(c["n_building"])
        n_storage = int(c["n_storage"])
        pcc = int(c["pcc"])
        from_bus = [int(bus) for bus in c["from_bus"]]
        to_bus = [int(bus) for bus in c["to_bus"]]
        children, parent_edge = build_topology(n_bus, from_bus, to_bus)
        building_at_bus = {
            int(bus): building for building, bus in enumerate(c["building_buses"])
        }
        storage_buildings = [int(building) for building in c["storage_buildings"]]
        storage_at_building = {
            building: storage
            for storage, building in enumerate(storage_buildings)
        }
        s_base = float(c["s_base_mva"])

        p_load = np.asarray(c["p_load_building"], dtype=float)
        q_load = np.asarray(c["q_load_building"], dtype=float)
        p_pv = np.asarray(c["p_pv_building"], dtype=float)
        tau_out = np.asarray(c["outdoor_temperature"], dtype=float)
        c_cur = np.broadcast_to(np.asarray(c["c_cur"], dtype=float), (n_building, self.T))
        expected_profile_shape = (n_building, self.T)
        for name, values in (
            ("p_load_building", p_load),
            ("q_load_building", q_load),
            ("p_pv_building", p_pv),
            ("outdoor_temperature", tau_out),
        ):
            if values.shape != expected_profile_shape:
                raise ValueError(f"{name} must have shape {expected_profile_shape}; got {values.shape}.")

        def vector(name: str) -> np.ndarray:
            values = np.broadcast_to(
                np.asarray(c[name], dtype=float), (n_building,)
            )
            return np.array(values, dtype=float, copy=True)

        p_heat_max = vector("p_heat_max")
        p_cool_max = vector("p_cool_max")
        p_ch_max = vector("p_ch_max")
        p_dis_max = vector("p_dis_max")
        e_min = vector("e_min")
        e_max = vector("e_max")
        e_init = vector("e_init")
        alpha = vector("alpha")
        beta_heat = vector("beta_heat")
        beta_cool = vector("beta_cool")
        gamma = vector("gamma")
        tau_ref = vector("tau_ref")
        tau_min = vector("tau_min")
        tau_max = vector("tau_max")
        tau_init = vector("tau_init")
        omega = vector("omega")

        self._env = cp.Envr()
        model = self._env.createModel("BG_follower")
        model.setParam(COPT.Param.Logging, int(self.verbose))
        model.setParam(COPT.Param.Threads, -1)  # Let COPT choose available threads.
        if self.time_limit is not None:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)
        if self.mip_gap is not None:
            model.setParam(COPT.Param.RelGap, self.mip_gap)

        p_mg = model.addVars(self.T, lb=0.0, ub=float(c["p_mg_max"]), nameprefix="p_mg")
        q_mg = model.addVars(self.T, lb=0.0, ub=float(c["q_mg_max"]), nameprefix="q_mg")
        heat = model.addVars(n_building, self.T, lb=0.0, nameprefix="p_heat")
        cool = model.addVars(n_building, self.T, lb=0.0, nameprefix="p_cool")
        heat_mode = model.addVars(
            n_building, self.T, vtype=COPT.BINARY, nameprefix="heat_mode"
        )
        cool_mode = model.addVars(
            n_building, self.T, vtype=COPT.BINARY, nameprefix="cool_mode"
        )
        curtail = model.addVars(n_building, self.T, lb=0.0, nameprefix="p_cur")
        charge = model.addVars(n_storage, self.T, lb=0.0, nameprefix="p_ch")
        discharge = model.addVars(n_storage, self.T, lb=0.0, nameprefix="p_dis")
        energy = model.addVars(n_storage, self.T, nameprefix="energy")
        indoor_temperature = model.addVars(n_building, self.T, nameprefix="indoor_temperature")
        temperature_deviation = model.addVars(
            n_building,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="temperature_deviation",
        )
        absolute_temperature_deviation = model.addVars(
            n_building,
            self.T,
            lb=0.0,
            nameprefix="absolute_temperature_deviation",
        )
        storage_mode = model.addVars(
            n_storage, self.T, vtype=COPT.BINARY, nameprefix="storage_mode"
        )
        voltage = model.addVars(n_bus, self.T, lb=float(c["v_min"]), ub=float(c["v_max"]), nameprefix="v")
        p_flow = model.addVars(n_line, self.T, lb=-COPT.INFINITY, ub=COPT.INFINITY, nameprefix="P")
        q_flow = model.addVars(n_line, self.T, lb=-COPT.INFINITY, ub=COPT.INFINITY, nameprefix="Q")

        base_objective = cp.quicksum(
            omega[j]
            * absolute_temperature_deviation[j, t]
            + c_cur[j, t] * curtail[j, t]
            for j in range(n_building)
            for t in range(self.T)
        )
        model.setObjective(base_objective, COPT.MINIMIZE)
        # Binary HVAC operating modes prevent simultaneous heating and cooling.
        for j in range(n_building):
            for t in range(self.T):
                model.addConstr(
                    heat[j, t] <= p_heat_max[j] * heat_mode[j, t],
                    name=f"heat_max[{j},{t}]",
                )
                model.addConstr(
                    cool[j, t] <= p_cool_max[j] * cool_mode[j, t],
                    name=f"cool_max[{j},{t}]",
                )
                model.addConstr(
                    heat_mode[j, t] + cool_mode[j, t] <= 1,
                    name=f"hvac_mode[{j},{t}]",
                )
                model.addConstr(curtail[j, t] <= p_pv[j, t], name=f"curtail_max[{j},{t}]")
                previous_temperature = tau_init[j] if t == 0 else indoor_temperature[j, t - 1]
                model.addConstr(indoor_temperature[j, t] == alpha[j] * previous_temperature + beta_heat[j] * heat[j, t] - beta_cool[j] * cool[j, t] + gamma[j] * tau_out[j, t],name=f"thermal_balance[{j},{t}]")
                model.addConstr(
                    temperature_deviation[j, t]
                    == indoor_temperature[j, t] - tau_ref[j],
                    name=f"temperature_deviation[{j},{t}]",
                )
                model.addConstr(
                    absolute_temperature_deviation[j, t]
                    >= temperature_deviation[j, t],
                    name=f"absolute_temperature_deviation_pos[{j},{t}]",
                )
                model.addConstr(
                    absolute_temperature_deviation[j, t]
                    >= -temperature_deviation[j, t],
                    name=f"absolute_temperature_deviation_neg[{j},{t}]",
                )
                model.addConstr(indoor_temperature[j, t] >= tau_min[j],name=f"temperature_min[{j},{t}]")
                model.addConstr(indoor_temperature[j, t] <= tau_max[j],name=f"temperature_max[{j},{t}]")
            model.addConstr(indoor_temperature[j, self.T - 1] == tau_init[j], name=f"terminal_temperature[{j}]")

        # Storage is deployed only at the selected representative buses.
        for storage, building in enumerate(storage_buildings):
            for t in range(self.T):
                model.addConstr(
                    charge[storage, t]
                    <= p_ch_max[building] * storage_mode[storage, t],
                    name=f"charge_max[{storage},{t}]",
                )
                model.addConstr(
                    discharge[storage, t]
                    <= p_dis_max[building] * (1 - storage_mode[storage, t]),
                    name=f"discharge_max[{storage},{t}]",
                )
                previous_energy = (
                    e_init[building] if t == 0 else energy[storage, t - 1]
                )
                model.addConstr(
                    energy[storage, t]
                    == previous_energy
                    + float(c["eta_ch"]) * charge[storage, t]
                    - discharge[storage, t] / float(c["eta_dis"]),
                    name=f"energy_balance[{storage},{t}]",
                )
                model.addConstr(
                    energy[storage, t] >= e_min[building],
                    name=f"energy_min[{storage},{t}]",
                )
                model.addConstr(
                    energy[storage, t] <= e_max[building],
                    name=f"energy_max[{storage},{t}]",
                )
            model.addConstr(
                energy[storage, self.T - 1] == e_init[building],
                name=f"terminal_energy[{storage}]",
            )

        # Lossless LinDistFlow constraints. The original branch-current variables,
        # loss terms, and rotated-SOC constraints are intentionally omitted.
        r = np.asarray(c["r"], dtype=float)
        x = np.asarray(c["x"], dtype=float)
        # LinDistFlow uses the nominal-voltage apparent-power rating.
        line_s_max = (
            np.sqrt(np.asarray(c["l_max"], dtype=float)) * float(c["v_ref"])
        )
        for t in range(self.T):
            model.addConstr(voltage[pcc, t] == float(c["v_ref"]) ** 2,name=f"pcc_voltage[{t}]")
            for edge in range(n_line):
                source, target = from_bus[edge], to_bus[edge]
                model.addConstr(
                    voltage[target, t]
                    == voltage[source, t]
                    - 2 * (r[edge] * p_flow[edge, t] + x[edge] * q_flow[edge, t]),
                    name=f"voltage_drop[{edge},{t}]",
                )
                model.addConstr(
                    p_flow[edge, t] >= -line_s_max[edge],
                    name=f"p_flow_min[{edge},{t}]",
                )
                model.addConstr(
                    p_flow[edge, t] <= line_s_max[edge],
                    name=f"p_flow_max[{edge},{t}]",
                )
                model.addConstr(
                    q_flow[edge, t] >= -line_s_max[edge],
                    name=f"q_flow_min[{edge},{t}]",
                )
                model.addConstr(
                    q_flow[edge, t] <= line_s_max[edge],
                    name=f"q_flow_max[{edge},{t}]",
                )
            for bus in range(n_bus):
                parent = parent_edge[bus]
                if parent is None:
                    continue
                building = building_at_bus[bus]
                storage = storage_at_building.get(building)
                storage_charge = 0.0 if storage is None else charge[storage, t]
                storage_discharge = 0.0 if storage is None else discharge[storage, t]
                p_injection = (
                    p_load[building, t]
                    + heat[building, t]
                    + cool[building, t]
                    + curtail[building, t]
                    - p_pv[building, t]
                    + storage_charge
                    - storage_discharge
                ) / s_base
                q_injection = q_load[building, t] / s_base
                model.addConstr(
                    cp.quicksum(p_flow[edge, t] for edge in children[bus])
                    == p_flow[parent, t] - p_injection,
                    name=f"active_balance[{bus},{t}]",
                )
                    
                model.addConstr(
                    cp.quicksum(q_flow[edge, t] for edge in children[bus])
                    == q_flow[parent, t] - q_injection,
                    name=f"reactive_balance[{bus},{t}]",
                )

            model.addConstr(p_mg[t] / s_base == cp.quicksum(p_flow[edge, t] for edge in children[pcc]),name=f"pcc_active_balance[{t}]")
            model.addConstr(q_mg[t] / s_base == cp.quicksum(q_flow[edge, t] for edge in children[pcc]),name=f"pcc_reactive_balance[{t}]")

        model.update()
        self._model = model
        self._p_mg = p_mg
        self._q_mg = q_mg
        self._heat = heat
        self._cool = cool
        self._heat_mode = heat_mode
        self._cool_mode = cool_mode
        self._curtail = curtail
        self._charge = charge
        self._discharge = discharge
        self._energy = energy
        self._indoor_temperature = indoor_temperature
        self._temperature_deviation = temperature_deviation
        self._absolute_temperature_deviation = absolute_temperature_deviation
        self._storage_mode = storage_mode
        self._voltage = voltage
        self._p_flow = p_flow
        self._q_flow = q_flow
        self._n_bus = n_bus
        self._n_line = n_line
        self._n_building = n_building
        self._storage_buildings = storage_buildings
        self._from_bus = from_bus
        self._base_objective = base_objective

    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any] | None:
        """Solve the cached BG mixed-integer quadratic problem."""
        price = np.asarray(pi, dtype=float).reshape(-1)
        if price.size != self.T:
            raise ValueError(f"Expected a {self.T}-period price vector, got {price.size}.")

        model = self._model
        p_mg = self._p_mg
        model.setObjective(
            self._base_objective
            + cp.quicksum(float(price[t]) * p_mg[t] for t in range(self.T)),
            COPT.MINIMIZE,
        )
        model.update()

        n_bus = self._n_bus
        n_line = self._n_line
        n_building = self._n_building
        storage_buildings = self._storage_buildings
        from_bus = self._from_bus
        heat = self._heat
        cool = self._cool
        heat_mode = self._heat_mode
        cool_mode = self._cool_mode
        curtail = self._curtail
        charge = self._charge
        discharge = self._discharge
        energy = self._energy
        indoor_temperature = self._indoor_temperature
        voltage = self._voltage
        p_flow = self._p_flow
        q_flow = self._q_flow

        mip_start_used = self._has_mip_start
        if mip_start_used:
            model.setMipStart(model.getVars(), self._mip_start_values)
            model.loadMipStart()
        model.solve()
        self.last_status = int(model.status)
        accepted_statuses = (COPT.OPTIMAL, COPT.TIMEOUT)
        if model.status not in accepted_statuses or not model.hasmipsol:
            diagnostic = ""
            if model.status == COPT.INFEASIBLE and self.verbose:
                model.computeIIS()
                linear_iis = [
                    constraint.getName()
                    for constraint in model.getConstrs()
                    if constraint.getLowerIIS() or constraint.getUpperIIS()
                ]
                diagnostic = f" IIS={linear_iis}."
            if self.failure_mode == "raise":
                raise RuntimeError(
                    f"BG follower solve failed with COPT status {model.status}."
                    f"{diagnostic}"
                )
            return None

        self._mip_start_values = [variable.x for variable in model.getVars()]
        self._has_mip_start = True

        charge_result = np.zeros((n_building, self.T), dtype=float)
        discharge_result = np.zeros((n_building, self.T), dtype=float)
        energy_result = np.zeros((n_building, self.T), dtype=float)
        for storage, building in enumerate(storage_buildings):
            charge_result[building] = [
                charge[storage, t].x for t in range(self.T)
            ]
            discharge_result[building] = [
                discharge[storage, t].x for t in range(self.T)
            ]
            energy_result[building] = [
                energy[storage, t].x for t in range(self.T)
            ]

        voltage_result = np.array(
            [
                [voltage[bus, t].x for t in range(self.T)]
                for bus in range(n_bus)
            ]
        )
        p_flow_result = np.array(
            [
                [p_flow[edge, t].x for t in range(self.T)]
                for edge in range(n_line)
            ]
        )
        q_flow_result = np.array(
            [
                [q_flow[edge, t].x for t in range(self.T)]
                for edge in range(n_line)
            ]
        )
        current_result = np.empty((n_line, self.T), dtype=float)
        for edge, source in enumerate(from_bus):
            current_result[edge] = (
                p_flow_result[edge] ** 2 + q_flow_result[edge] ** 2
            ) / np.maximum(voltage_result[source], 1e-12)

        result: Dict[str, Any] = {
            "p_mg": np.array([p_mg[t].x for t in range(self.T)])
        }
        if self.include_metadata:
            result.update(
                objective=float(model.objval),
                solver_status=int(model.status),
                mip_gap=float(model.bestgap),
                scenario=self.scenario,
                mip_start_used=mip_start_used,
                power_flow_model="LinDistFlow",
                p_heat=np.array(
                    [
                        [heat[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                p_cool=np.array(
                    [
                        [cool[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                heat_mode=np.array(
                    [
                        [heat_mode[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                cool_mode=np.array(
                    [
                        [cool_mode[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                p_hvac=np.array(
                    [
                        [
                            heat[j, t].x + cool[j, t].x
                            for t in range(self.T)
                        ]
                        for j in range(n_building)
                    ]
                ),
                p_cur=np.array(
                    [
                        [curtail[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                p_ch=charge_result,
                p_dis=discharge_result,
                energy=energy_result,
                indoor_temperature=np.array(
                    [
                        [indoor_temperature[j, t].x for t in range(self.T)]
                        for j in range(n_building)
                    ]
                ),
                voltage=voltage_result,
                current=current_result,
                p_flow=p_flow_result,
                q_flow=q_flow_result,
            )
        return result
