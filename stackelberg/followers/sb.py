"""Single-bus follower optimization problem."""

from __future__ import annotations

from typing import Any, Dict

import coptpy as cp
from coptpy import COPT
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from ..profiles import daily_load_curve
from ..randomness import get_global_seed
from .base import FollowerProblem


def extract_sb_constants(
    net: pp.pandapowerNet,
    T: int,
    *,
    scenario: str = "SB",
    verbose: bool = False,
    require_positive_load: bool = False,
    seed: int | None = None,
    load_scale: float = 1.0,
) -> Dict[str, Any]:
    """Extract constants used only by the implemented SB formulation."""
    if scenario.upper() != "SB":
        raise ValueError("extract_sb_constants only accepts scenario='SB'.")
    if T <= 0:
        raise ValueError("T must be positive.")
    load_scale = float(load_scale)
    if not np.isfinite(load_scale) or load_scale <= 0.0:
        raise ValueError("load_scale must be finite and positive.")
    load_curve = daily_load_curve(T, "SB")
    base_p = float(net.load["p_mw"].sum()) if len(net.load) else 0.0
    base_q = float(net.load["q_mvar"].sum()) if len(net.load) else 0.0
    if require_positive_load and base_p <= 0.0:
        raise ValueError("The SB network must contain positive active load.")
    e_max = 0.2 * base_p
    return {
        "T": T,
        "load_curve": load_curve,
        "p_load": load_scale * base_p * load_curve,
        "q_load": load_scale * base_q * load_curve,
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
        "verbose": bool(verbose),
        "random_seed": get_global_seed() if seed is None else int(seed),
    }


class SingleBusFollowerProblem(FollowerProblem):
    """Implemented SB MILP with storage and local generation."""

    scenario = "SB"

    def __init__(
        self,
        T: int = 24,
        net: pp.pandapowerNet | None = None,
        verbose: bool = False,
        *,
        lower_constants: Dict[str, Any] | None = None,
        failure_mode: str = "raise",
        include_metadata: bool = True,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        load_scale: float = 1.0,
    ) -> None:
        if failure_mode not in {"none", "raise"}:
            raise ValueError("failure_mode must be 'none' or 'raise'.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        if mip_gap is not None and mip_gap < 0:
            raise ValueError("mip_gap must be nonnegative.")
        self.net = self.default_network() if net is None else net
        self.constants = (
            extract_sb_constants(
                self.net,
                T,
                verbose=verbose,
                require_positive_load=True,
                load_scale=load_scale,
            )
            if lower_constants is None
            else lower_constants
        )
        self.T = int(
            self.constants.get("T", np.asarray(self.constants["p_load"]).size)
        )
        self.failure_mode = failure_mode
        self.include_metadata = include_metadata
        self.verbose = bool(self.constants.get("verbose", verbose))
        self.random_seed = int(
            self.constants.get("random_seed", get_global_seed())
        )
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.eta_ch = float(self.constants["eta_ch"])
        self.eta_dis = float(self.constants["eta_dis"])
        self.e_min = float(self.constants["e_min"])
        self.e_max = float(self.constants["e_max"])
        self.e_init = float(self.constants["e_init"])
        self.p_ch_max = float(self.constants["p_ch_max"])
        self.p_dis_max = float(self.constants["p_dis_max"])
        self.p_mg_max = float(self.constants["p_mg_max"])
        self.s_max = float(self.constants["s_max"])
        self.c_g = float(self.constants["c_g"])
        self.q_g = float(self.constants["q_g"])
        self.p_load = np.asarray(self.constants["p_load"], dtype=float)
        self._build_model()

    @staticmethod
    def default_network() -> pp.pandapowerNet:
        """Return the current default SB network."""
        return pn.create_cigre_network_lv()

    def _build_model(self) -> None:
        """Build and cache the price-parametric SB model."""
        c = self.constants
        self._env = cp.Envr()
        model = self._env.createModel("SB_follower")
        model.setParam(COPT.Param.Logging, int(self.verbose))
        model.setParam(COPT.Param.Threads, -1)
        if self.time_limit is not None:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)
        if self.mip_gap is not None:
            model.setParam(COPT.Param.RelGap, self.mip_gap)

        p_mg = model.addVars(
            self.T, lb=0.0, ub=c["p_mg_max"], nameprefix="p_mg"
        )
        generation = model.addVars(
            self.T, lb=0.0, ub=c["s_max"], nameprefix="generation"
        )
        charge = model.addVars(
            self.T, lb=0.0, ub=c["p_ch_max"], nameprefix="charge"
        )
        discharge = model.addVars(
            self.T, lb=0.0, ub=c["p_dis_max"], nameprefix="discharge"
        )
        energy = model.addVars(
            self.T, lb=c["e_min"], ub=c["e_max"], nameprefix="energy"
        )
        is_charging = model.addVars(
            self.T, vtype=COPT.BINARY, nameprefix="is_charging"
        )
        is_discharging = model.addVars(
            self.T, vtype=COPT.BINARY, nameprefix="is_discharging"
        )

        model.setObjective(
            cp.quicksum(
                c["c_g"] * generation[t]
                + c["q_g"] * generation[t] * generation[t]
                for t in range(self.T)
            ),
            COPT.MINIMIZE,
        )
        for t in range(self.T):
            model.addConstr(
                p_mg[t] + generation[t] + discharge[t]
                == c["p_load"][t] + charge[t],
                name=f"power_balance[{t}]",
            )
            model.addConstr(charge[t] <= c["p_ch_max"] * is_charging[t])
            model.addConstr(discharge[t] <= c["p_dis_max"] * is_discharging[t])
            model.addConstr(is_charging[t] + is_discharging[t] <= 1)
            previous = c["e_init"] if t == 0 else energy[t - 1]
            model.addConstr(
                energy[t]
                == previous + c["eta_ch"] * charge[t] - discharge[t] / c["eta_dis"]
            )
        model.addConstr(energy[self.T - 1] == c["e_init"], name="terminal_soc")
        model.update()

        self._model = model
        self._p_mg = p_mg
        self._generation = generation
        self._charge = charge
        self._discharge = discharge
        self._energy = energy
        self._is_charging = is_charging
        self._is_discharging = is_discharging

    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any] | None:
        """Solve the cached SB optimization problem."""
        price = np.asarray(pi, dtype=float).reshape(-1)
        if price.size != self.T:
            raise ValueError(f"Expected a {self.T}-period price vector, got {price.size}.")

        model = self._model
        p_mg = self._p_mg
        for t in range(self.T):
            p_mg[t].obj = float(price[t])
        model.solve()

        accepted_statuses = (COPT.OPTIMAL, COPT.TIMEOUT)
        if model.status not in accepted_statuses or not model.hasmipsol:
            if self.failure_mode == "raise":
                raise RuntimeError(
                    f"SB follower solve failed with COPT status {model.status}."
                )
            return None
        result: Dict[str, Any] = {
            "p_mg": np.array([p_mg[t].x for t in range(self.T)])
        }
        if self.include_metadata:
            result.update(
                objective=float(model.objval),
                solver_status=int(model.status),
                mip_gap=float(model.bestgap),
                scenario=self.scenario,
            )
        return result
