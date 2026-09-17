"""Hard-constraint COPT DistFlow evaluation shared by leader solvers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional

import coptpy as cp
from coptpy import COPT
import numpy as np
import torch
import torch.nn.functional as F

from .network import build_topology


def _bus_time_array(
    value: Any,
    *,
    n_bus: int,
    T: int,
    name: str,
) -> np.ndarray:
    """Broadcast a scalar or bus/time profile to shape ``(n_bus, T)``."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full((n_bus, T), float(array), dtype=float)
    if array.shape == (n_bus,):
        return np.repeat(array[:, None], T, axis=1)
    if array.shape == (T,) and T != n_bus:
        return np.repeat(array[None, :], n_bus, axis=0)
    if array.shape == (n_bus, 1):
        return np.repeat(array, T, axis=1)
    if array.shape == (1, T):
        return np.repeat(array, n_bus, axis=0)
    if array.shape == (n_bus, T):
        return array.copy()
    raise ValueError(
        f"{name} must be a scalar or have shape ({n_bus},), ({T},), "
        f"({n_bus}, 1), (1, {T}), or ({n_bus}, {T}); got {array.shape}."
    )


def _constant_bound(
    constants: Dict[str, Any],
    names: Iterable[str],
    *,
    n_bus: int,
    T: int,
    default: float,
) -> np.ndarray:
    """Read the first configured alias for a nodal net-injection bound."""
    for name in names:
        if name in constants:
            return _bus_time_array(
                constants[name], n_bus=n_bus, T=T, name=name
            )
    return np.full((n_bus, T), default, dtype=float)


def _model_bound(
    model: Any,
    names: Iterable[str],
    *,
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return an optional model bound broadcast to a reference tensor."""
    for name in names:
        if not hasattr(model, name):
            continue
        raw_value = getattr(model, name)
        if torch.is_tensor(raw_value):
            raw_value = raw_value.detach().cpu().numpy()
        value = _bus_time_array(
            raw_value,
            n_bus=int(model.n_bus),
            T=int(model.T),
            name=name,
        )
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return None


def build_microgrid_profile(
    constants: Dict[str, Any],
    p_mg: np.ndarray,
    *,
    duplicate_one_dimensional: bool = False,
) -> np.ndarray:
    """Map MW follower output to a per-bus, per-unit active-power profile."""
    T = int(constants["T"])
    n_bus = int(constants["n_bus"])
    mg_buses = [int(bus) for bus in constants.get("mg_buses", [])]
    s_base_mva = float(constants["s_base_mva"])
    values = np.asarray(p_mg, dtype=float)
    profile = np.zeros((n_bus, T), dtype=float)
    if values.shape == (n_bus, T):
        return values / s_base_mva
    if values.shape == (len(mg_buses), T):
        for index, bus in enumerate(mg_buses):
            profile[bus] = values[index] / s_base_mva
        return profile
    if values.shape == (T,):
        if not duplicate_one_dimensional and len(mg_buses) != 1:
            raise ValueError("1-D p_mg requires exactly one microgrid bus.")
        if duplicate_one_dimensional and not mg_buses:
            raise ValueError("1-D p_mg requires at least one microgrid bus.")
        for bus in mg_buses:
            profile[bus] = values / s_base_mva
        return profile
    raise ValueError(
        f"Unexpected p_mg shape {values.shape}; expected (T,), "
        f"({len(mg_buses)}, T), or (n_bus, T)."
    )


class CoptDistFlowEvaluator:
    """Persistent hard-constraint DistFlow model for repeated evaluations.

    Optional ``p_net_min``, ``p_net_max``, ``q_net_min``, and ``q_net_max``
    constants implement the nodal transformer bounds.  Each accepts a scalar,
    a bus vector, a time vector, or a full bus-by-time array.  Missing bounds
    default to +/-15 MW and +/-15 MVar and are converted to per unit using
    ``s_base_mva``.
    """

    def __init__(
        self,
        constants: Dict[str, Any],
        *,
        failure: Literal["raise", "penalty"],
        revenue: Literal["aggregate", "single_profile"],
        accept_suboptimal: bool,
        duplicate_one_dimensional: bool = False,
        verbose: Optional[bool] = None,
    ):
        self.constants = constants
        self.failure = failure
        self.revenue = revenue
        self.accept_suboptimal = accept_suboptimal
        self.duplicate_one_dimensional = duplicate_one_dimensional
        self.T = int(constants["T"])
        self.n_bus = int(constants["n_bus"])
        self.n_line = int(constants["n_line"])
        self.slack = int(constants["slack"])
        self.mg_buses = [int(bus) for bus in constants.get("mg_buses", [])]
        self.from_bus = [int(value) for value in constants["from_bus"]]
        self.to_bus = [int(value) for value in constants["to_bus"]]
        self.children, self.parent_edge = build_topology(
            self.n_bus, self.from_bus, self.to_bus
        )
        self.p_passive = np.asarray(constants["p_passive"], dtype=float)
        self.q_passive = np.asarray(constants["q_passive"], dtype=float)
        self.s_base = float(constants["s_base_mva"])
        if self.s_base <= 0.0:
            raise ValueError("s_base_mva must be positive.")
        self.lambda_g = np.asarray(
            constants.get("lambda_g", constants.get("lamda_g")), dtype=float
        )
        expected_profile_shape = (self.n_bus, self.T)
        if self.p_passive.shape != expected_profile_shape:
            raise ValueError(
                "p_passive must have shape "
                f"{expected_profile_shape}; got {self.p_passive.shape}."
            )
        if self.q_passive.shape != expected_profile_shape:
            raise ValueError(
                "q_passive must have shape "
                f"{expected_profile_shape}; got {self.q_passive.shape}."
            )
        if self.lambda_g.shape != (self.T,):
            raise ValueError(
                f"lambda_g must have shape ({self.T},); got {self.lambda_g.shape}."
            )

        # Transformer/net-injection limits are stored in per unit.  Their
        # physical defaults are +/-15 MW and +/-15 MVar, respectively.
        p_net_limit_pu = (
            float(constants.get("p_net_limit_mw", 15.0)) / self.s_base
        )
        q_net_limit_pu = (
            float(constants.get("q_net_limit_mvar", 15.0)) / self.s_base
        )
        if p_net_limit_pu < 0.0:
            raise ValueError("p_net_limit_mw must be nonnegative.")
        if q_net_limit_pu < 0.0:
            raise ValueError("q_net_limit_mvar must be nonnegative.")

        # The long names are canonical; p_min/p_max/q_min/q_max remain
        # accepted as convenient aliases and override the physical defaults.
        self.p_net_min = _constant_bound(
            constants,
            ("p_net_min", "p_min"),
            n_bus=self.n_bus,
            T=self.T,
            default=-p_net_limit_pu,
        )
        self.p_net_max = _constant_bound(
            constants,
            ("p_net_max", "p_max"),
            n_bus=self.n_bus,
            T=self.T,
            default=p_net_limit_pu,
        )
        self.q_net_min = _constant_bound(
            constants,
            ("q_net_min", "q_min"),
            n_bus=self.n_bus,
            T=self.T,
            default=-q_net_limit_pu,
        )
        self.q_net_max = _constant_bound(
            constants,
            ("q_net_max", "q_max"),
            n_bus=self.n_bus,
            T=self.T,
            default=q_net_limit_pu,
        )
        if np.any(self.p_net_min > self.p_net_max):
            raise ValueError("p_net_min must not exceed p_net_max.")
        if np.any(self.q_net_min > self.q_net_max):
            raise ValueError("q_net_min must not exceed q_net_max.")
        if any(
            np.any(np.isnan(bound))
            for bound in (
                self.p_net_min,
                self.p_net_max,
                self.q_net_min,
                self.q_net_max,
            )
        ):
            raise ValueError("Net-injection bounds must not contain NaN.")

        self._env = cp.Envr()
        model = self._env.createModel("distflow")
        model.setParam(
            COPT.Param.Logging,
            int(
                constants.get("verbose", False) if verbose is None else verbose
            ),
        )
        model.setParam(COPT.Param.Threads, 1)
        model.setParam(COPT.Param.NonConvex, 2)
        self.model = model

        self.p0 = model.addVars(
            self.T, lb=0.0, ub=constants["p0_max"], nameprefix="p0"
        )
        self.q0 = model.addVars(
            self.T, lb=0.0, ub=constants["q0_max"], nameprefix="q0"
        )
        self.voltage = model.addVars(
            self.n_bus,
            self.T,
            lb=constants["v_min"],
            ub=constants["v_max"],
            nameprefix="v",
        )
        self.current = model.addVars(
            self.n_line, self.T, lb=0.0, nameprefix="l"
        )
        self.p_flow = model.addVars(
            self.n_line,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="p",
        )
        self.q_flow = model.addVars(
            self.n_line,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="q",
        )
        self.p_net = model.addVars(
            self.n_bus,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="p_net",
        )
        self.q_control = model.addVars(
            self.n_bus,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="q_c",
        )
        self.q_net = model.addVars(
            self.n_bus,
            self.T,
            lb=-COPT.INFINITY,
            ub=COPT.INFINITY,
            nameprefix="q_net",
        )
        # Backwards-compatible name used by existing result-inspection code.
        self.q_injection = self.q_net
        model.setObjective(
            cp.quicksum(
                self.lambda_g[t] * self.p0[t] * self.s_base
                for t in range(self.T)
            ),
            COPT.MINIMIZE,
        )

        r = np.asarray(constants["r"], dtype=float)
        x = np.asarray(constants["x"], dtype=float)
        l_max = np.asarray(constants["l_max"], dtype=float)
        self.p_net_definition = {}
        for t in range(self.T):
            model.addConstr(
                self.voltage[self.slack, t] == float(constants["v_ref"]) ** 2,
                name=f"v_ref[{t}]",
            )
            for bus in range(self.n_bus):
                self.p_net_definition[bus, t] = model.addConstr(
                    self.p_net[bus, t] == self.p_passive[bus, t],
                    name=f"p_net_def[{bus},{t}]",
                )
                model.addConstr(
                    self.q_net[bus, t]
                    == self.q_control[bus, t] + self.q_passive[bus, t],
                    name=f"q_net_def[{bus},{t}]",
                )
                if np.isfinite(self.p_net_min[bus, t]):
                    model.addConstr(
                        self.p_net[bus, t] >= self.p_net_min[bus, t],
                        name=f"p_net_lb[{bus},{t}]",
                    )
                if np.isfinite(self.p_net_max[bus, t]):
                    model.addConstr(
                        self.p_net[bus, t] <= self.p_net_max[bus, t],
                        name=f"p_net_ub[{bus},{t}]",
                    )
                if np.isfinite(self.q_net_min[bus, t]):
                    model.addConstr(
                        self.q_net[bus, t] >= self.q_net_min[bus, t],
                        name=f"q_net_lb[{bus},{t}]",
                    )
                if np.isfinite(self.q_net_max[bus, t]):
                    model.addConstr(
                        self.q_net[bus, t] <= self.q_net_max[bus, t],
                        name=f"q_net_ub[{bus},{t}]",
                    )
            for edge in range(self.n_line):
                source, target = self.from_bus[edge], self.to_bus[edge]
                model.addConstr(
                    self.current[edge, t] <= l_max[edge],
                    name=f"l_bound[{edge},{t}]",
                )
                model.addConstr(
                    self.voltage[target, t]
                    == self.voltage[source, t]
                    - 2
                    * (
                        r[edge] * self.p_flow[edge, t]
                        + x[edge] * self.q_flow[edge, t]
                    )
                    + (r[edge] ** 2 + x[edge] ** 2) * self.current[edge, t],
                    name=f"voltage_drop[{edge},{t}]",
                )
                model.addQConstr(
                    self.p_flow[edge, t] ** 2 + self.q_flow[edge, t] ** 2
                    <= self.current[edge, t] * self.voltage[source, t],
                    name=f"soc[{edge},{t}]",
                )
            for bus in range(self.n_bus):
                parent = self.parent_edge[bus]
                if parent is None:
                    continue
                model.addConstr(
                    cp.quicksum(
                        self.p_flow[edge, t] for edge in self.children[bus]
                    )
                    == self.p_flow[parent, t]
                    - self.p_net[bus, t]
                    - r[parent] * self.current[parent, t],
                    name=f"p_balance[{bus},{t}]",
                )
                model.addConstr(
                    cp.quicksum(
                        self.q_flow[edge, t] for edge in self.children[bus]
                    )
                    == self.q_flow[parent, t]
                    - self.q_net[bus, t]
                    - x[parent] * self.current[parent, t],
                    name=f"q_balance[{bus},{t}]",
                )
            model.addConstr(
                self.p0[t]
                == cp.quicksum(
                    self.p_flow[edge, t] for edge in self.children[self.slack]
                )
                + self.p_net[self.slack, t],
                name=f"slack_p_balance[{t}]",
            )
            model.addConstr(
                self.q0[t]
                == cp.quicksum(
                    self.q_flow[edge, t] for edge in self.children[self.slack]
                )
                + self.q_net[self.slack, t],
                name=f"slack_q_balance[{t}]",
            )
        model.update()

    def evaluate(self, pi: np.ndarray, p_mg: np.ndarray) -> float:
        """Update parameter-dependent RHS/objective constants and optimize."""
        raw_price = np.asarray(pi, dtype=float)
        raw_p_mg = np.asarray(p_mg, dtype=float)
        if raw_price.shape == (self.T,):
            price_matrix = None
        elif raw_price.shape == (len(self.mg_buses), self.T):
            price_matrix = raw_price
            if raw_p_mg.shape != (len(self.mg_buses), self.T):
                raise ValueError(
                    "Independent microgrid prices require p_mg with shape "
                    f"({len(self.mg_buses)}, {self.T})."
                )
        else:
            raise ValueError(
                f"Unexpected price shape {raw_price.shape}; expected ({self.T},) or "
                f"({len(self.mg_buses)}, {self.T})."
            )
        if not np.all(np.isfinite(raw_price)):
            raise ValueError("pi must contain only finite values.")
        pi_min = float(self.constants.get("pi_min", -np.inf))
        pi_max = float(self.constants.get("pi_max", np.inf))
        if pi_min > pi_max:
            raise ValueError("pi_min must not exceed pi_max.")
        price_is_feasible = bool(
            np.all(raw_price >= pi_min) and np.all(raw_price <= pi_max)
        )
        if not price_is_feasible:
            if self.failure == "penalty":
                return 1e10
            raise ValueError(f"pi must lie in [{pi_min}, {pi_max}].")

        profile = build_microgrid_profile(
            self.constants,
            raw_p_mg,
            duplicate_one_dimensional=self.duplicate_one_dimensional,
        )
        p_net_values = self.p_passive + profile
        if not np.all(np.isfinite(p_net_values)):
            raise ValueError("p_mg must produce only finite p_net values.")
        for (bus, t), constraint in self.p_net_definition.items():
            rhs = float(p_net_values[bus, t])
            constraint.lb = rhs
            constraint.ub = rhs

        if price_matrix is not None:
            revenue_values = np.sum(price_matrix * raw_p_mg, axis=0)
        elif self.revenue == "single_profile":
            if raw_p_mg.shape != (self.T,):
                raise ValueError(
                    "single_profile revenue requires p_mg with shape (T,)."
                )
            revenue_values = raw_price * raw_p_mg
        else:
            revenue_values = raw_price * profile.sum(axis=0) * self.s_base
        self.model.setObjConst(-float(np.sum(revenue_values)))
        self.model.solve()

        accepted = self.model.status == COPT.OPTIMAL
        if self.accept_suboptimal:
            accepted = accepted or bool(self.model.hassol)
        if not accepted:
            if self.failure == "penalty":
                return 1e10
            raise RuntimeError(f"COPT failed with status {self.model.status}")
        return float(self.model.objval)


def solve_copt_distflow(
    constants: Dict[str, Any],
    pi: np.ndarray,
    p_mg: np.ndarray,
    *,
    failure: Literal["raise", "penalty"],
    revenue: Literal["aggregate", "single_profile"],
    accept_suboptimal: bool,
    duplicate_one_dimensional: bool = False,
    verbose: Optional[bool] = None,
) -> float:
    """One-shot compatibility wrapper around the persistent evaluator."""
    evaluator = CoptDistFlowEvaluator(
        constants,
        failure=failure,
        revenue=revenue,
        accept_suboptimal=accept_suboptimal,
        duplicate_one_dimensional=duplicate_one_dimensional,
        verbose=verbose,
    )
    return evaluator.evaluate(pi, p_mg)


# Backwards-compatible aliases for callers that still import the old names.
GurobiDistFlowEvaluator = CoptDistFlowEvaluator
solve_gurobi_distflow = solve_copt_distflow


def torch_penalty_distflow(
    model: Any,
    pi: torch.Tensor,
    p_mg_prediction: torch.Tensor,
    penalty_weights: Dict[str, float],
    *,
    square_voltage_limits: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Evaluate the differentiable counterpart of the DN DistFlow model."""
    if p_mg_prediction.ndim == 1:
        p_mg = {bus: p_mg_prediction for bus in model.mg_buses}
    elif p_mg_prediction.shape == (len(model.mg_buses), model.T):
        p_mg = {
            bus: p_mg_prediction[index]
            for index, bus in enumerate(model.mg_buses)
        }
    else:
        raise ValueError(
            "p_mg_prediction must have shape "
            f"({model.T},) or ({len(model.mg_buses)}, {model.T}); "
            f"got {tuple(p_mg_prediction.shape)}."
        )
    p_net = model.p_passive.clone()
    # Existing optimizer classes expose q_injection as q_net.  q_control is
    # therefore the implied controllable component q_net - q_passive.
    q_net = model.q_net if hasattr(model, "q_net") else model.q_injection
    q_control = q_net - model.q_passive
    for bus in model.mg_buses:
        p_net[bus] = p_net[bus] + p_mg[bus] / model.s_base_mva

    objective = torch.sum(model.lambda_g * model.p0 * model.s_base_mva)
    if pi.ndim == 1:
        for bus in model.mg_buses:
            objective = objective - torch.sum(pi * p_mg[bus])
    elif pi.shape == (len(model.mg_buses), model.T):
        for index, bus in enumerate(model.mg_buses):
            objective = objective - torch.sum(pi[index] * p_mg[bus])
    else:
        raise ValueError(
            f"pi must have shape ({model.T},) or "
            f"({len(model.mg_buses)}, {model.T}); got {tuple(pi.shape)}."
        )

    zero = lambda: p_net.new_zeros(())
    equality_penalty = zero()
    inequality_penalty = zero()
    soc_penalty = zero()
    for t in range(model.T):
        equality_penalty = (
            equality_penalty + (model.v[model.slack, t] - model.v_ref**2) ** 2
        )
        for edge in range(model.n_line):
            source = model.from_bus[edge]
            target = model.to_bus[edge]
            inequality_penalty = inequality_penalty + F.relu(
                model.l[edge, t] - model.l_max[edge]
            ) ** 2
            voltage_drop = model.v[target, t] - (
                model.v[source, t]
                - 2
                * (
                    model.r[edge] * model.p[edge, t]
                    + model.x[edge] * model.q[edge, t]
                )
                + (model.r[edge] ** 2 + model.x[edge] ** 2) * model.l[edge, t]
            )
            equality_penalty = equality_penalty + voltage_drop**2
            soc_violation = (
                model.p[edge, t] ** 2
                + model.q[edge, t] ** 2
                - model.l[edge, t] * model.v[source, t]
            )
            soc_penalty = soc_penalty + F.relu(soc_violation) ** 2

        for bus in range(model.n_bus):
            parent = model.parent_edge[bus]
            if parent is None:
                continue
            outgoing_p = sum(model.p[edge, t] for edge in model.child_edges[bus])
            outgoing_q = sum(model.q[edge, t] for edge in model.child_edges[bus])
            equality_penalty = equality_penalty + (
                outgoing_p
                - (
                    model.p[parent, t]
                    - p_net[bus, t]
                    - model.r[parent] * model.l[parent, t]
                )
            ) ** 2
            equality_penalty = equality_penalty + (
                outgoing_q
                - (
                    model.q[parent, t]
                    - q_net[bus, t]
                    - model.x[parent] * model.l[parent, t]
                )
            ) ** 2
        slack_out_p = sum(
            model.p[edge, t] for edge in model.child_edges[model.slack]
        )
        slack_out_q = sum(
            model.q[edge, t] for edge in model.child_edges[model.slack]
        )
        equality_penalty = equality_penalty + (
            model.p0[t] - (slack_out_p + p_net[model.slack, t])
        ) ** 2
        equality_penalty = equality_penalty + (
            model.q0[t] - (slack_out_q + q_net[model.slack, t])
        ) ** 2

    lower_voltage = model.v_min**2 if square_voltage_limits else model.v_min
    upper_voltage = model.v_max**2 if square_voltage_limits else model.v_max
    bound_penalty = (
        torch.sum(F.relu(-model.p0) ** 2 + F.relu(model.p0 - model.p0_max) ** 2)
        + torch.sum(F.relu(-model.q0) ** 2 + F.relu(model.q0 - model.q0_max) ** 2)
        + torch.sum(
            F.relu(lower_voltage - model.v) ** 2
            + F.relu(model.v - upper_voltage) ** 2
        )
        + torch.sum(F.relu(-model.l) ** 2)
    )
    if hasattr(model, "pi_min"):
        bound_penalty = bound_penalty + torch.sum(
            F.relu(float(model.pi_min) - pi) ** 2
        )
    if hasattr(model, "pi_max"):
        bound_penalty = bound_penalty + torch.sum(
            F.relu(pi - float(model.pi_max)) ** 2
        )

    p_net_min = _model_bound(
        model, ("p_net_min", "p_min"), reference=p_net
    )
    p_net_max = _model_bound(
        model, ("p_net_max", "p_max"), reference=p_net
    )
    q_net_min = _model_bound(
        model, ("q_net_min", "q_min"), reference=q_net
    )
    q_net_max = _model_bound(
        model, ("q_net_max", "q_max"), reference=q_net
    )
    s_base_mva = float(model.s_base_mva)
    if s_base_mva <= 0.0:
        raise ValueError("s_base_mva must be positive.")
    p_net_limit_pu = float(getattr(model, "p_net_limit_mw", 15.0)) / s_base_mva
    q_net_limit_pu = (
        float(getattr(model, "q_net_limit_mvar", 15.0)) / s_base_mva
    )
    if p_net_limit_pu < 0.0:
        raise ValueError("p_net_limit_mw must be nonnegative.")
    if q_net_limit_pu < 0.0:
        raise ValueError("q_net_limit_mvar must be nonnegative.")
    if p_net_min is None:
        p_net_min = torch.full_like(p_net, -p_net_limit_pu)
    if p_net_max is None:
        p_net_max = torch.full_like(p_net, p_net_limit_pu)
    if q_net_min is None:
        q_net_min = torch.full_like(q_net, -q_net_limit_pu)
    if q_net_max is None:
        q_net_max = torch.full_like(q_net, q_net_limit_pu)
    if bool(torch.any(p_net_min > p_net_max).item()):
        raise ValueError("p_net_min must not exceed p_net_max.")
    if bool(torch.any(q_net_min > q_net_max).item()):
        raise ValueError("q_net_min must not exceed q_net_max.")
    bound_penalty = (
        bound_penalty
        + torch.sum(F.relu(p_net_min - p_net) ** 2)
        + torch.sum(F.relu(p_net - p_net_max) ** 2)
        + torch.sum(F.relu(q_net_min - q_net) ** 2)
        + torch.sum(F.relu(q_net - q_net_max) ** 2)
    )
    loss = (
        objective
        + penalty_weights["eq"] * equality_penalty
        + penalty_weights["ineq"] * inequality_penalty
        + penalty_weights["soc"] * soc_penalty
        + penalty_weights["bound"] * bound_penalty
    )
    return loss, {
        "obj": objective.detach(),
        "eq_pen": equality_penalty.detach(),
        "ineq_pen": inequality_penalty.detach(),
        "soc_pen": soc_penalty.detach(),
        "bound_pen": bound_penalty.detach(),
        "pi": pi.detach(),
        "p0": model.p0.detach(),
        "p_net": p_net.detach(),
        "q_net": q_net.detach(),
        "q_control": q_control.detach(),
        "q_injection": q_net.detach(),
    }
