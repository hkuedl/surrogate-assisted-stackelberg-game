"""Reusable lower-level follower solvers for surrogate-data generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import gurobipy as gp
import numpy as np
import pandapower as pp
import pandapower.networks as pn


def _load_curve(T: int, scenario: str) -> np.ndarray:
    """Return a scenario-specific normalized daily demand profile."""
    hour = np.linspace(0.0, 24.0, T, endpoint=False)
    if scenario == "IP":
        # Industrial demand is flatter, with a daytime production peak.
        curve = 0.82 + 0.30 * np.exp(-0.5 * ((hour - 13.0) / 4.0) ** 2)
    else:
        morning = 0.35 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
        evening = 0.55 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
        curve = 0.75 + 0.10 * np.cos(2.0 * np.pi * (hour - 14.0) / 24.0) + morning + evening
    return np.clip(curve, 0.5, 1.5)


def extract_constants(net: pp.pandapowerNet, T: int, scenario: str = "SB") -> Dict[str, Any]:
    """Extract the common optimization constants from a pandapower network."""
    if T <= 0:
        raise ValueError("T must be positive.")
    load_curve = _load_curve(T, scenario)
    base_p = float(net.load["p_mw"].sum()) if len(net.load) else 0.0
    base_q = float(net.load["q_mvar"].sum()) if len(net.load) else 0.0
    if base_p <= 0.0:
        raise ValueError("The follower network must contain positive active load.")

    e_max = 0.2 * base_p
    return {
        "T": T,
        "p_load": base_p * load_curve,
        "q_load": base_q * load_curve,
        "c_g": 1.0,
        "q_g": 0.1,
        "p_mg_max": 1.5 * base_p,
        "s_max": base_p,
        "p_ch_max": 0.25 * e_max,
        "p_dis_max": 0.25 * e_max,
        "eta_ch": 0.95,
        "eta_dis": 0.92,
        "e_min": 0.0,
        "e_max": e_max,
        "e_init": 0.5 * e_max,
    }


class LowerSolver:
    """Common Gurobi implementation with the same ``lower_solve(pi)`` API as zero_solver."""

    scenario = "SB"

    def __init__(
        self,
        T: int = 24,
        net: Optional[pp.pandapowerNet] = None,
        verbose: bool = False,
    ) -> None:
        self.net = self.default_network() if net is None else net
        self.constants = extract_constants(self.net, T=T, scenario=self.scenario)
        self.T = T
        self.verbose = verbose

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        return pn.create_cigre_network_lv()

    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any]:
        """Solve the follower problem and return the aggregate grid import profile."""
        pi = np.asarray(pi, dtype=float).reshape(-1)
        if pi.size != self.T:
            raise ValueError(f"Expected a {self.T}-period price vector, got {pi.size}.")

        c = self.constants
        model = gp.Model(f"{self.scenario}_follower")
        model.Params.OutputFlag = int(self.verbose)

        p_mg = model.addVars(self.T, lb=0.0, ub=c["p_mg_max"], name="p_mg")
        shedding = model.addVars(self.T, lb=0.0, ub=c["s_max"], name="shedding")
        charge = model.addVars(self.T, lb=0.0, ub=c["p_ch_max"], name="charge")
        discharge = model.addVars(self.T, lb=0.0, ub=c["p_dis_max"], name="discharge")
        energy = model.addVars(self.T, lb=c["e_min"], ub=c["e_max"], name="energy")
        is_charging = model.addVars(self.T, vtype=gp.GRB.BINARY, name="is_charging")
        is_discharging = model.addVars(self.T, vtype=gp.GRB.BINARY, name="is_discharging")

        model.setObjective(
            gp.quicksum(
                pi[t] * p_mg[t] + c["c_g"] * shedding[t] + c["q_g"] * shedding[t] * shedding[t]
                for t in range(self.T)
            ),
            gp.GRB.MINIMIZE,
        )
        for t in range(self.T):
            model.addConstr(p_mg[t] + shedding[t] + discharge[t] == c["p_load"][t] + charge[t])
            model.addConstr(charge[t] <= c["p_ch_max"] * is_charging[t])
            model.addConstr(discharge[t] <= c["p_dis_max"] * is_discharging[t])
            model.addConstr(is_charging[t] + is_discharging[t] <= 1)
            previous_energy = c["e_init"] if t == 0 else energy[t - 1]
            model.addConstr(
                energy[t] == previous_energy + c["eta_ch"] * charge[t] - discharge[t] / c["eta_dis"]
            )
        model.addConstr(energy[self.T - 1] == c["e_init"], name="terminal_soc")

        model.optimize()
        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"{self.scenario} follower solve failed with Gurobi status {model.Status}.")
        return {
            "p_mg": np.array([p_mg[t].X for t in range(self.T)]),
            "objective": float(model.ObjVal),
            "scenario": self.scenario,
        }


class SingleBusSolver(LowerSolver):
    """Single-bus approximation of the low-voltage network."""

    scenario = "SB"


class BuildingGroupSolver(LowerSolver):
    """Building-group follower, parameterized from the CIGRE LV network."""

    scenario = "BG"


class IndustrialParkSolver(LowerSolver):
    """Industrial-park follower, parameterized from the CIGRE MV network."""

    scenario = "IP"

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        return pn.create_cigre_network_mv()


SOLVER_CLASSES: Dict[str, Type[LowerSolver]] = {
    "SB": SingleBusSolver,
    "BG": BuildingGroupSolver,
    "IP": IndustrialParkSolver,
    "IE": IndustrialParkSolver,  # Legacy name retained for existing scripts.
}


def build_solver(
    problem_type: str,
    T: int = 24,
    net: Optional[pp.pandapowerNet] = None,
    **kwargs: Any,
) -> LowerSolver:
    """Create a scenario solver; this is the entry point used by data_generation."""
    scenario = problem_type.upper()
    try:
        return SOLVER_CLASSES[scenario](T=T, net=net, **kwargs)
    except KeyError as error:
        raise ValueError(f"Unknown problem type {problem_type!r}; choose from SB, BG, or IP.") from error


def solver(problem_type: str, input: Dict[str, Any], net: Optional[pp.pandapowerNet] = None) -> Dict[str, Any]:
    """Backward-compatible one-shot wrapper around :func:`build_solver`."""
    pi = np.asarray(input["pi"], dtype=float)
    return build_solver(problem_type, T=pi.size, net=net, verbose=bool(input.get("verbose", False))).lower_solve(pi)
