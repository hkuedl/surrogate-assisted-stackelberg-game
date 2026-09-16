"""Industrial-park follower with process scheduling and an internal feeder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import coptpy as cp
from coptpy import COPT
import numpy as np
import pandapower as pp
import pandas as pd

from buildings import LargeCommercialBuilding2
from ..network import build_topology
from ..randomness import get_global_seed
from .base import FollowerProblem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
PROCESS_NAMES = ("C", "KFP", "CP", "FG")
PLINE_NAMES = ("Production Line 1", "Production Line 2")
PV_PEAK_MW = 2.0
IP_S_BASE_MVA = 10.0
IP_V_BASE_KV = 20.0
IP_PCC_BUS_NAME = "PCC 0"
IP_DER_BUS_NAME = "Bus 1 - PV and BESS"
IP_PLINE_BUS_NAMES = ("Bus 2 - Production Line 1", "Bus 3 - Production Line 2")


def create_industrial_park_network() -> pp.pandapowerNet:
    """Create the four-bus, 20 kV radial feeder used inside the industrial park.

    Bus 0 is the PCC, bus 1 hosts PV and BESS, and buses 2 and 3 host the two
    production lines. Branch ratings are 12.5, 4.5, and 9.0 MVA respectively.
    The branch impedances are benchmark-equivalent values representing short
    internal cables and downstream equipment.
    """
    net = pp.create_empty_network(sn_mva=IP_S_BASE_MVA)
    pcc = pp.create_bus(net, vn_kv=IP_V_BASE_KV, name=IP_PCC_BUS_NAME)
    der_bus = pp.create_bus(net, vn_kv=IP_V_BASE_KV, name=IP_DER_BUS_NAME)
    pline_1_bus = pp.create_bus(net, vn_kv=IP_V_BASE_KV, name=IP_PLINE_BUS_NAMES[0])
    pline_2_bus = pp.create_bus(net, vn_kv=IP_V_BASE_KV, name=IP_PLINE_BUS_NAMES[1])
    pp.create_ext_grid(net, bus=pcc, vm_pu=1.0, name="Industrial park PCC")

    def create_branch(
        source: int,
        target: int,
        *,
        name: str,
        r_ohm: float,
        x_ohm: float,
        s_max_mva: float,
    ) -> None:
        max_i_ka = s_max_mva / (np.sqrt(3.0) * IP_V_BASE_KV)
        pp.create_line_from_parameters(
            net,
            from_bus=source,
            to_bus=target,
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=name,
        )

    create_branch(
        pcc,
        der_bus,
        name="Branch 0-1 - Main feeder",
        r_ohm=0.30,
        x_ohm=0.40,
        s_max_mva=12.5,
    )
    create_branch(
        der_bus,
        pline_1_bus,
        name="Branch 1-2 - Production Line 1",
        r_ohm=0.50,
        x_ohm=0.40,
        s_max_mva=4.5,
    )
    create_branch(
        der_bus,
        pline_2_bus,
        name="Branch 1-3 - Production Line 2",
        r_ohm=0.50,
        x_ohm=0.40,
        s_max_mva=9.0,
    )
    net.name = "Four-bus 20 kV industrial-park feeder"
    return net


def _extract_ip_network_constants(net: pp.pandapowerNet) -> Dict[str, Any]:
    """Validate and convert the dedicated IP feeder to per-unit constants."""
    bus_ids = [
        int(bus) for bus, row in net.bus.iterrows() if bool(row.get("in_service", True))
    ]
    if len(bus_ids) != 4:
        raise ValueError("The IP internal network must contain four in-service buses.")
    bus_lookup = {bus: index for index, bus in enumerate(bus_ids)}

    active_ext_grids = net.ext_grid[net.ext_grid["in_service"].astype(bool)]
    if len(active_ext_grids) != 1:
        raise ValueError("The IP internal network requires exactly one in-service PCC.")
    pcc_bus_id = int(active_ext_grids.iloc[0]["bus"])
    if pcc_bus_id not in bus_lookup:
        raise ValueError("The IP PCC must be connected to an in-service bus.")
    pcc = bus_lookup[pcc_bus_id]

    if len(net.trafo) and net.trafo["in_service"].astype(bool).any():
        raise ValueError(
            "The simplified IP LinDistFlow network must be line-only; represent "
            "internal transformers through equivalent branch parameters."
        )
    if len(net.switch):
        open_switches = net.switch[~net.switch["closed"].astype(bool)]
        if len(open_switches):
            raise ValueError(
                "The simplified IP network does not support open switches."
            )

    lines = net.line[net.line["in_service"].astype(bool)].copy()
    if len(lines) != len(bus_ids) - 1:
        raise ValueError("The IP internal network must be a three-branch radial tree.")
    from_bus: list[int] = []
    to_bus: list[int] = []
    parent_count = np.zeros(len(bus_ids), dtype=int)
    for _, row in lines.iterrows():
        source_id, target_id = int(row["from_bus"]), int(row["to_bus"])
        if source_id not in bus_lookup or target_id not in bus_lookup:
            raise ValueError("Every IP branch must connect two in-service buses.")
        source, target = bus_lookup[source_id], bus_lookup[target_id]
        from_bus.append(source)
        to_bus.append(target)
        parent_count[target] += 1
    expected_parent_count = np.ones(len(bus_ids), dtype=int)
    expected_parent_count[pcc] = 0
    if not np.array_equal(parent_count, expected_parent_count):
        raise ValueError(
            "IP branches must be directed away from PCC 0 and give each non-PCC "
            "bus exactly one parent."
        )
    children, parent_edge = build_topology(len(bus_ids), from_bus, to_bus)

    name_to_bus_id = {
        str(row["name"]): int(bus) for bus, row in net.bus.loc[bus_ids].iterrows()
    }
    required_names = {IP_PCC_BUS_NAME, IP_DER_BUS_NAME, *IP_PLINE_BUS_NAMES}
    missing_names = sorted(required_names - name_to_bus_id.keys())
    if missing_names:
        raise ValueError(f"IP internal-network buses are missing: {missing_names}.")
    if name_to_bus_id[IP_PCC_BUS_NAME] != pcc_bus_id:
        raise ValueError(
            "The external grid must be connected to the bus named 'PCC 0'."
        )

    nominal_voltages = np.asarray(net.bus.loc[bus_ids, "vn_kv"], dtype=float)
    if not np.allclose(nominal_voltages, nominal_voltages[0]):
        raise ValueError("The transformer-free IP feeder requires one voltage level.")
    v_base_kv = float(nominal_voltages[0])
    s_base_mva = float(net.sn_mva)
    if s_base_mva <= 0.0 or v_base_kv <= 0.0:
        raise ValueError("IP voltage and apparent-power bases must be positive.")
    z_base_ohm = v_base_kv**2 / s_base_mva
    r_ohm = np.asarray(lines["r_ohm_per_km"] * lines["length_km"], dtype=float)
    x_ohm = np.asarray(lines["x_ohm_per_km"] * lines["length_km"], dtype=float)
    branch_s_max_mva = np.asarray(
        np.sqrt(3.0)
        * np.asarray(net.bus.loc[lines["from_bus"], "vn_kv"], dtype=float)
        * np.asarray(lines["max_i_ka"], dtype=float),
        dtype=float,
    )
    pline_bus_ids = [name_to_bus_id[name] for name in IP_PLINE_BUS_NAMES]
    der_bus_id = name_to_bus_id[IP_DER_BUS_NAME]
    der_bus = bus_lookup[der_bus_id]
    pline_buses = [bus_lookup[bus] for bus in pline_bus_ids]
    der_parent = parent_edge[der_bus]
    if der_parent is None or from_bus[der_parent] != pcc:
        raise ValueError("Bus 1 must be the only direct child of PCC 0.")
    for pline, pline_bus in enumerate(pline_buses, start=1):
        pline_parent = parent_edge[pline_bus]
        if pline_parent is None or from_bus[pline_parent] != der_bus:
            raise ValueError(
                f"Production line {pline} must be a direct child of Bus 1."
            )
    if len(children[pcc]) != 1:
        raise ValueError("PCC 0 must have exactly one outgoing main feeder.")
    root_branch = int(children[pcc][0])

    return {
        "n_bus": len(bus_ids),
        "n_branch": len(lines),
        "bus_ids": bus_ids,
        "bus_lookup": bus_lookup,
        "bus_names": [str(net.bus.loc[bus, "name"]) for bus in bus_ids],
        "pcc": pcc,
        "pcc_bus_id": pcc_bus_id,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "children": children,
        "parent_edge": parent_edge,
        "root_branch": root_branch,
        "branch_ids": [int(branch) for branch in lines.index],
        "branch_names": [str(name) for name in lines["name"]],
        "r_ohm": r_ohm,
        "x_ohm": x_ohm,
        "r": r_ohm / z_base_ohm,
        "x": x_ohm / z_base_ohm,
        "branch_s_max_mva": branch_s_max_mva,
        "branch_s_max_pu": branch_s_max_mva / s_base_mva,
        "v_base_kv": v_base_kv,
        "s_base_mva": s_base_mva,
        "z_base_ohm": z_base_ohm,
        "v_ref": 1.0,
        "v_min": 0.95**2,
        "v_max": 1.05**2,
        "der_bus": der_bus,
        "der_bus_id": der_bus_id,
        "pv_bus": bus_lookup[der_bus_id],
        "battery_bus": bus_lookup[der_bus_id],
        "pline_buses": pline_buses,
        "pline_bus_ids": pline_bus_ids,
    }


def extract_ip_constants(
    net: pp.pandapowerNet,
    T: int,
    *,
    scenario: str = "IP",
    date: str = "2023-01-01",
    data_dir: str | Path | None = None,
    parameters_path: str | Path | None = None,
    verbose: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Load industrial-process, PV, storage, and internal-network constants."""
    if scenario.upper() not in {"IP", "IE"}:
        raise ValueError("extract_ip_constants only accepts scenario='IP'.")
    if T != 24:
        raise ValueError("IP prepared PV profiles require T=24.")

    network_constants = _extract_ip_network_constants(net)
    selected_data_dir = DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    selected_parameters_path = (
        selected_data_dir / "IP_parameters.csv"
        if parameters_path is None
        else Path(parameters_path)
    )
    parameters = pd.read_csv(selected_parameters_path, index_col=0)
    parameters.index = parameters.index.astype(str).str.strip()
    parameter_columns = ["g_min", "e_min", "g_max", "e_max", "s_max", "s_min"]
    parameters = parameters.loc[:, parameter_columns].apply(pd.to_numeric)

    n_pline = 2
    n_stage = len(PROCESS_NAMES)
    n_storage = n_stage - 1
    process_rows = [
        [f"{process}{pline + 1}" for process in PROCESS_NAMES]
        for pline in range(n_pline)
    ]

    def process_values(column: str) -> np.ndarray:
        return np.asarray(
            [
                [parameters.loc[row, column] for row in pline_rows]
                for pline_rows in process_rows
            ],
            dtype=float,
        )

    varphi_min = process_values("g_min")
    varphi_max = process_values("g_max")
    p_process_min_mw = process_values("e_min") / 1000.0
    p_process_max_mw = process_values("e_max") / 1000.0
    sos_max = process_values("s_max")[:, :n_storage]
    sos_min = process_values("s_min")[:, :n_storage]

    eta = np.broadcast_to(np.array([1.5, 1.2, 1.2, 1.2]), (n_pline, n_stage)).copy()
    p_tilde = p_process_max_mw / varphi_max
    startup_cost = 0.5 * p_process_max_mw
    shutdown_cost = 0.2 * p_process_max_mw
    process_power_factor = np.full((n_pline, n_stage), 0.95)
    process_q_to_p = np.tan(np.arccos(process_power_factor))

    f_max = np.zeros((n_pline, n_pline, n_storage))
    for source in range(n_pline):
        for target in range(n_pline):
            if source != target:
                f_max[source, target] = 0.25 * np.minimum(
                    sos_max[source], sos_max[target]
                )

    raw_pv_profile = LargeCommercialBuilding2(date, selected_data_dir).get_pv_curve()
    pv_profile = raw_pv_profile / raw_pv_profile.max() * PV_PEAK_MW

    return {
        "T": int(T),
        "n_pline": n_pline,
        "n_stage": n_stage,
        "n_storage": n_storage,
        "process_names": PROCESS_NAMES,
        "pline_names": PLINE_NAMES,
        "p_process_min_mw": p_process_min_mw,
        "p_process_max_mw": p_process_max_mw,
        "process_power_factor": process_power_factor,
        "process_q_to_p": process_q_to_p,
        "eta": eta,
        "varphi_min": varphi_min,
        "varphi_max": varphi_max,
        "p_tilde": p_tilde,
        "c_startup": np.broadcast_to(
            startup_cost[:, :, None], (n_pline, n_stage, T)
        ).copy(),
        "c_shutdown": np.broadcast_to(
            shutdown_cost[:, :, None], (n_pline, n_stage, T)
        ).copy(),
        "min_on_time": np.full((n_pline, n_stage), 8, dtype=int),
        "ramp_rate": 0.25 * varphi_max,
        "u_init": np.zeros((n_pline, n_stage)),
        "varphi_init": np.zeros((n_pline, n_stage)),
        "c_transport": 0.1,
        "sos_min": sos_min,
        "sos_max": sos_max,
        "sos_init": 0.5 * sos_max,
        "f_max": f_max,
        "transfer_forward_max": f_max[0, 1].copy(),
        "transfer_reverse_max": f_max[1, 0].copy(),
        "q_pline_min": 0.2 * varphi_max[:, -1] * T,
        "q_required": float(0.5 * varphi_max[:, -1].sum() * T),
        "p_pv_hat": pv_profile,
        "p_pv_peak_mw": PV_PEAK_MW,
        "eta_ch": 0.95,
        "eta_dis": 0.92,
        "e_min": 0.0,
        "e_max": 2.0,
        "e_init": 1.0,
        "p_ch_max": 0.50,
        "p_dis_max": 0.50,
        "p_mg_max": 10.0,
        "q_mg_max": float(
            network_constants["branch_s_max_mva"][network_constants["root_branch"]]
        ),
        "delta_t_hours": 1.0,
        "profile_date": str(date),
        "parameters_path": str(selected_parameters_path),
        "verbose": bool(verbose),
        "random_seed": get_global_seed() if seed is None else int(seed),
        **network_constants,
    }


class IndustrialParkFollowerProblem(FollowerProblem):
    """Industrial scheduling MILP with a four-bus internal LinDistFlow feeder."""

    scenario = "IP"

    def __init__(
        self,
        T: int = 24,
        net: pp.pandapowerNet | None = None,
        verbose: bool = False,
        *,
        date: str = "2023-01-01",
        data_dir: str | Path | None = None,
        parameters_path: str | Path | None = None,
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
            extract_ip_constants(
                self.net,
                T,
                date=date,
                data_dir=data_dir,
                parameters_path=parameters_path,
                verbose=verbose,
            )
            if lower_constants is None
            else lower_constants
        )
        self.T = int(self.constants["T"])
        self.failure_mode = failure_mode
        self.include_metadata = include_metadata
        self.verbose = bool(self.constants.get("verbose", verbose))
        self.random_seed = int(self.constants.get("random_seed", get_global_seed()))
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self._has_mip_start = False
        self._mip_start_values: list[float] = []
        self._build_model()

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        """Return the dedicated four-bus industrial-park feeder."""
        return create_industrial_park_network()

    def clear_mip_start(self) -> None:
        """Clear the incumbent used to warm-start the next IP solve."""
        self._has_mip_start = False
        self._mip_start_values = []

    def _build_model(self) -> None:
        """Build and cache the price-parametric IP LinDistFlow MILP."""
        c = self.constants
        n_pline = int(c["n_pline"])
        n_stage = int(c["n_stage"])
        n_storage = int(c["n_storage"])
        n_bus = int(c["n_bus"])
        n_branch = int(c["n_branch"])
        pcc = int(c["pcc"])
        der_bus = int(c["der_bus"])
        pline_buses = [int(bus) for bus in c["pline_buses"]]
        from_bus = [int(bus) for bus in c["from_bus"]]
        to_bus = [int(bus) for bus in c["to_bus"]]
        children = c["children"]
        parent_edge = c["parent_edge"]
        s_base = float(c["s_base_mva"])

        if n_pline != 2 or len(pline_buses) != 2:
            raise ValueError(
                "The simplified signed-transfer model requires two production lines."
            )

        self._env = cp.Envr()
        model = self._env.createModel("IP_follower")
        model.setParam(COPT.Param.Logging, int(self.verbose))
        model.setParam(COPT.Param.Threads, -1)
        if self.time_limit is not None:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)
        if self.mip_gap is not None:
            model.setParam(COPT.Param.RelGap, self.mip_gap)

        # Process input phi is substituted as varphi / eta.
        varphi = model.addVars(n_pline, n_stage, self.T, lb=0.0, nameprefix="varphi")
        is_on = model.addVars(
            n_pline, n_stage, self.T, vtype=COPT.BINARY, nameprefix="is_on"
        )
        startup = model.addVars(
            n_pline, n_stage, self.T, lb=0.0, ub=1.0, nameprefix="startup"
        )
        shutdown = model.addVars(
            n_pline, n_stage, self.T, lb=0.0, ub=1.0, nameprefix="shutdown"
        )
        sos = model.addVars(
            n_pline,
            n_storage,
            self.T + 1,
            lb=-COPT.INFINITY,
            nameprefix="SoS",
        )

        # Positive transfer is production line 1 -> 2; negative is line 2 -> 1.
        transfer = model.addVars(
            n_storage,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="transfer",
        )
        transfer_abs = model.addVars(
            n_storage, self.T, lb=0.0, nameprefix="transfer_abs"
        )

        p_mg = model.addVars(self.T, lb=0.0, ub=c["p_mg_max"], nameprefix="p_mg")
        q_mg = model.addVars(self.T, lb=0.0, ub=c["q_mg_max"], nameprefix="q_mg")
        p_pv = model.addVars(self.T, lb=0.0, nameprefix="p_pv")
        charge = model.addVars(self.T, lb=0.0, ub=c["p_ch_max"], nameprefix="charge")
        discharge = model.addVars(
            self.T, lb=0.0, ub=c["p_dis_max"], nameprefix="discharge"
        )
        energy = model.addVars(
            self.T + 1, lb=c["e_min"], ub=c["e_max"], nameprefix="energy"
        )
        is_charging = model.addVars(self.T, vtype=COPT.BINARY, nameprefix="is_charging")
        is_discharging = model.addVars(
            self.T, vtype=COPT.BINARY, nameprefix="is_discharging"
        )

        voltage = model.addVars(
            n_bus,
            self.T,
            lb=float(c["v_min"]),
            ub=float(c["v_max"]),
            nameprefix="voltage_squared",
        )
        p_flow = model.addVars(
            n_branch,
            self.T,
            lb=0.0,
            ub=COPT.INFINITY,
            nameprefix="p_flow",
        )
        q_flow = model.addVars(
            n_branch,
            self.T,
            lb=0.0,
            ub=COPT.INFINITY,
            nameprefix="q_flow",
        )
        for branch in range(n_branch):
            for t in range(self.T):
                p_flow[branch, t].ub = float(c["branch_s_max_pu"][branch])
                q_flow[branch, t].ub = float(c["branch_s_max_pu"][branch])

        model.setObjective(
            cp.quicksum(
                c["c_transport"] * transfer_abs[storage, t]
                for storage in range(n_storage)
                for t in range(self.T)
            )
            + cp.quicksum(
                c["c_startup"][pline, stage, t] * startup[pline, stage, t]
                + c["c_shutdown"][pline, stage, t] * shutdown[pline, stage, t]
                for pline in range(n_pline)
                for stage in range(n_stage)
                for t in range(self.T)
            ),
            COPT.MINIMIZE,
        )

        for pline in range(n_pline):
            for stage in range(n_stage):
                for t in range(self.T):
                    model.addConstr(
                        varphi[pline, stage, t]
                        >= c["varphi_min"][pline, stage] * is_on[pline, stage, t],
                        name=f"production_min[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        varphi[pline, stage, t]
                        <= c["varphi_max"][pline, stage] * is_on[pline, stage, t],
                        name=f"production_max[{pline},{stage},{t}]",
                    )

                    previous_on = (
                        c["u_init"][pline, stage]
                        if t == 0
                        else is_on[pline, stage, t - 1]
                    )
                    previous_output = (
                        c["varphi_init"][pline, stage]
                        if t == 0
                        else varphi[pline, stage, t - 1]
                    )
                    model.addConstr(
                        startup[pline, stage, t]
                        >= is_on[pline, stage, t] - previous_on,
                        name=f"startup_lower[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        startup[pline, stage, t] <= is_on[pline, stage, t],
                        name=f"startup_current[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        startup[pline, stage, t] <= 1 - previous_on,
                        name=f"startup_previous[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        shutdown[pline, stage, t]
                        >= previous_on - is_on[pline, stage, t],
                        name=f"shutdown_lower[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        shutdown[pline, stage, t] <= previous_on,
                        name=f"shutdown_previous[{pline},{stage},{t}]",
                    )
                    model.addConstr(
                        shutdown[pline, stage, t] <= 1 - is_on[pline, stage, t],
                        name=f"shutdown_current[{pline},{stage},{t}]",
                    )
                    if str(c["process_names"][stage]).upper() != "FG":
                        model.addConstr(
                            varphi[pline, stage, t] - previous_output
                            <= c["ramp_rate"][pline, stage]
                            + c["varphi_max"][pline, stage]
                            * startup[pline, stage, t],
                            name=f"ramp_up[{pline},{stage},{t}]",
                        )
                    model.addConstr(
                        previous_output - varphi[pline, stage, t]
                        <= c["ramp_rate"][pline, stage]
                        + c["varphi_max"][pline, stage] * shutdown[pline, stage, t],
                        name=f"ramp_down[{pline},{stage},{t}]",
                    )

                minimum_on = int(c["min_on_time"][pline, stage])
                for t in range(max(0, self.T - minimum_on + 1)):
                    model.addConstr(
                        cp.quicksum(
                            is_on[pline, stage, tau] for tau in range(t, t + minimum_on)
                        )
                        >= minimum_on * startup[pline, stage, t],
                        name=f"minimum_on_time[{pline},{stage},{t}]",
                    )

        for storage in range(n_storage):
            for t in range(self.T):
                model.addConstr(
                    transfer[storage, t] <= c["transfer_forward_max"][storage],
                    name=f"transfer_forward_max[{storage},{t}]",
                )
                model.addConstr(
                    transfer[storage, t] >= -c["transfer_reverse_max"][storage],
                    name=f"transfer_reverse_max[{storage},{t}]",
                )
                model.addConstr(
                    transfer_abs[storage, t] >= transfer[storage, t],
                    name=f"transfer_abs_pos[{storage},{t}]",
                )
                model.addConstr(
                    transfer_abs[storage, t] >= -transfer[storage, t],
                    name=f"transfer_abs_neg[{storage},{t}]",
                )
                model.addConstr(
                    transfer[storage, t]
                    <= sos[0, storage, t] - c["sos_min"][0, storage],
                    name=f"transfer_available_pline_1[{storage},{t}]",
                )
                model.addConstr(
                    -transfer[storage, t]
                    <= sos[1, storage, t] - c["sos_min"][1, storage],
                    name=f"transfer_available_pline_2[{storage},{t}]",
                )

        for pline in range(n_pline):
            transfer_sign = -1.0 if pline == 0 else 1.0
            for storage in range(n_storage):
                for boundary in range(self.T + 1):
                    model.addConstr(
                        sos[pline, storage, boundary] >= c["sos_min"][pline, storage],
                        name=f"sos_min[{pline},{storage},{boundary}]",
                    )
                    model.addConstr(
                        sos[pline, storage, boundary] <= c["sos_max"][pline, storage],
                        name=f"sos_max[{pline},{storage},{boundary}]",
                    )
                model.addConstr(
                    sos[pline, storage, 0] == c["sos_init"][pline, storage],
                    name=f"initial_sos[{pline},{storage}]",
                )
                model.addConstr(
                    sos[pline, storage, self.T] == sos[pline, storage, 0],
                    name=f"terminal_sos[{pline},{storage}]",
                )
                for t in range(self.T):
                    model.addConstr(
                        sos[pline, storage, t + 1]
                        == sos[pline, storage, t]
                        + varphi[pline, storage, t]
                        - varphi[pline, storage + 1, t] / c["eta"][pline, storage + 1]
                        + transfer_sign * transfer[storage, t],
                        name=f"sos_dynamics[{pline},{storage},{t}]",
                    )

        last_stage = n_stage - 1
        for pline in range(n_pline):
            model.addConstr(
                cp.quicksum(varphi[pline, last_stage, t] for t in range(self.T))
                >= c["q_pline_min"][pline],
                name=f"pline_production_requirement[{pline}]",
            )
        model.addConstr(
            cp.quicksum(
                varphi[pline, last_stage, t]
                for pline in range(n_pline)
                for t in range(self.T)
            )
            >= c["q_required"],
            name="total_production_requirement",
        )

        model.addConstr(energy[0] == c["e_init"], name="initial_energy")
        for t in range(self.T):
            model.addConstr(p_pv[t] <= c["p_pv_hat"][t], name=f"pv_max[{t}]")
            model.addConstr(
                energy[t + 1]
                == energy[t]
                + c["delta_t_hours"]
                * (c["eta_ch"] * charge[t] - discharge[t] / c["eta_dis"]),
                name=f"energy_dynamics[{t}]",
            )
            model.addConstr(
                charge[t] <= c["p_ch_max"] * is_charging[t],
                name=f"charging_bound[{t}]",
            )
            model.addConstr(
                discharge[t] <= c["p_dis_max"] * is_discharging[t],
                name=f"discharging_bound[{t}]",
            )
            model.addConstr(
                is_charging[t] + is_discharging[t] <= 1,
                name=f"exclusive_storage_mode[{t}]",
            )

        r = np.asarray(c["r"], dtype=float)
        x = np.asarray(c["x"], dtype=float)
        p_tilde = np.asarray(c["p_tilde"], dtype=float)
        q_to_p = np.asarray(c["process_q_to_p"], dtype=float)

        for t in range(self.T):
            p_demand = []
            q_demand = []
            for bus in range(n_bus):
                active_terms = [
                    p_tilde[pline, stage] * varphi[pline, stage, t]
                    for pline in range(n_pline)
                    if pline_buses[pline] == bus
                    for stage in range(n_stage)
                ]
                reactive_terms = [
                    q_to_p[pline, stage]
                    * p_tilde[pline, stage]
                    * varphi[pline, stage, t]
                    for pline in range(n_pline)
                    if pline_buses[pline] == bus
                    for stage in range(n_stage)
                ]
                active_demand = cp.quicksum(active_terms)
                reactive_demand = cp.quicksum(reactive_terms)
                if bus == der_bus:
                    active_demand += charge[t] - discharge[t] - p_pv[t]
                p_demand.append(active_demand)
                q_demand.append(reactive_demand)

            model.addConstr(
                voltage[pcc, t] == float(c["v_ref"]) ** 2,
                name=f"pcc_voltage[{t}]",
            )
            for branch in range(n_branch):
                source, target = from_bus[branch], to_bus[branch]
                model.addConstr(
                    voltage[target, t]
                    == voltage[source, t]
                    - 2.0
                    * (r[branch] * p_flow[branch, t] + x[branch] * q_flow[branch, t]),
                    name=f"voltage_drop[{branch},{t}]",
                )

            for bus in range(n_bus):
                parent = parent_edge[bus]
                if parent is None:
                    continue
                model.addConstr(
                    cp.quicksum(p_flow[branch, t] for branch in children[bus])
                    == p_flow[parent, t] - p_demand[bus] / s_base,
                    name=f"active_balance[{bus},{t}]",
                )
                model.addConstr(
                    cp.quicksum(q_flow[branch, t] for branch in children[bus])
                    == q_flow[parent, t] - q_demand[bus] / s_base,
                    name=f"reactive_balance[{bus},{t}]",
                )

            model.addConstr(
                p_mg[t] / s_base
                == cp.quicksum(p_flow[branch, t] for branch in children[pcc])
                + p_demand[pcc] / s_base,
                name=f"pcc_active_balance[{t}]",
            )
            model.addConstr(
                q_mg[t] / s_base
                == cp.quicksum(q_flow[branch, t] for branch in children[pcc])
                + q_demand[pcc] / s_base,
                name=f"pcc_reactive_balance[{t}]",
            )

        model.update()
        self._model = model
        self._varphi = varphi
        self._is_on = is_on
        self._startup = startup
        self._shutdown = shutdown
        self._sos = sos
        self._transfer = transfer
        self._transfer_abs = transfer_abs
        self._p_mg = p_mg
        self._q_mg = q_mg
        self._p_pv = p_pv
        self._charge = charge
        self._discharge = discharge
        self._energy = energy
        self._is_charging = is_charging
        self._is_discharging = is_discharging
        self._voltage = voltage
        self._p_flow = p_flow
        self._q_flow = q_flow
        self._n_pline = n_pline
        self._n_stage = n_stage
        self._n_storage = n_storage
        self._n_bus = n_bus
        self._n_branch = n_branch

    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any] | None:
        """Solve the cached industrial-park scheduling and LinDistFlow MILP."""
        price = np.asarray(pi, dtype=float).reshape(-1)
        if price.size != self.T:
            raise ValueError(
                f"Expected a {self.T}-period price vector, got {price.size}."
            )

        model = self._model
        for t in range(self.T):
            self._p_mg[t].obj = float(price[t]) * float(self.constants["delta_t_hours"])

        mip_start_used = self._has_mip_start
        if mip_start_used:
            model.setMipStart(model.getVars(), self._mip_start_values)
            model.loadMipStart()
        model.solve()

        accepted_statuses = (COPT.OPTIMAL, COPT.TIMEOUT)
        if model.status not in accepted_statuses or not model.hasmipsol:
            if self.failure_mode == "raise":
                raise RuntimeError(
                    f"IP follower solve failed with COPT status {model.status}."
                )
            return None

        self._mip_start_values = [variable.x for variable in model.getVars()]
        self._has_mip_start = True

        def values_1d(variables: cp.tupledict, size: int) -> np.ndarray:
            return np.array([variables[index].x for index in range(size)])

        def values_2d(variables: cp.tupledict, first: int, second: int) -> np.ndarray:
            return np.array(
                [[variables[i, j].x for j in range(second)] for i in range(first)]
            )

        def values_3d(
            variables: cp.tupledict, first: int, second: int, third: int
        ) -> np.ndarray:
            return np.array(
                [
                    [
                        [variables[i, j, k].x for k in range(third)]
                        for j in range(second)
                    ]
                    for i in range(first)
                ]
            )

        result: Dict[str, Any] = {"p_mg": values_1d(self._p_mg, self.T)}
        if self.include_metadata:
            c = self.constants
            varphi_result = values_3d(
                self._varphi, self._n_pline, self._n_stage, self.T
            )
            phi_result = varphi_result / np.asarray(c["eta"])[:, :, None]
            production_power_result = (
                np.asarray(c["p_tilde"])[:, :, None] * varphi_result
            )
            signed_transfer = values_2d(self._transfer, self._n_storage, self.T)
            net_transfer = np.stack((-signed_transfer, signed_transfer), axis=0)
            result.update(
                objective=float(model.objval),
                solver_status=int(model.status),
                mip_gap=float(model.bestgap),
                scenario=self.scenario,
                mip_start_used=mip_start_used,
                phi=phi_result,
                varphi=varphi_result,
                production_power=production_power_result,
                is_on=values_3d(self._is_on, self._n_pline, self._n_stage, self.T),
                startup=values_3d(self._startup, self._n_pline, self._n_stage, self.T),
                shutdown=values_3d(
                    self._shutdown, self._n_pline, self._n_stage, self.T
                ),
                sos=values_3d(self._sos, self._n_pline, self._n_storage, self.T + 1),
                transfer=signed_transfer,
                net_transfer=net_transfer,
                q_mg=values_1d(self._q_mg, self.T),
                p_pv=values_1d(self._p_pv, self.T),
                p_prod=production_power_result.sum(axis=(0, 1)),
                charge=values_1d(self._charge, self.T),
                discharge=values_1d(self._discharge, self.T),
                energy=values_1d(self._energy, self.T + 1),
                voltage=values_2d(self._voltage, self._n_bus, self.T),
                p_flow=values_2d(self._p_flow, self._n_branch, self.T),
                q_flow=values_2d(self._q_flow, self._n_branch, self.T),
            )
        return result
