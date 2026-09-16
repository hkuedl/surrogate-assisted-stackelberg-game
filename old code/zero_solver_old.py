import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import gurobipy as gp
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import cma

np.random.seed(42)

def electricity_price(T: int):
    # randomly generate electricity price for each hour
    # Based on the real DA price data for 124 days
    # Randomly select one day and multiply by a random multiplier between 0.5 and 1.5.
    DA_price = pd.read_csv("data/DA_RT_data.csv")["DA"].values
    # unit price change USD per MWh to HKD per KWh
    DA_price = DA_price * 7.3 / 1000 # one (124 * 24,) array
    if T != 24:
        raise ValueError("electricity_price only supports 24-hour horizons.")
    segments = np.reshape(DA_price, (124, T))
    random_idx = np.random.randint(0, 124)
    selected_segment = segments[random_idx]
    multipliers = np.random.uniform(low=0.5, high=1.5, size=T)
    return selected_segment * multipliers


def _default_load_curve(T: int) -> np.ndarray:
    # Residential-like daily profile: night valley + morning/evening peaks.
    hour = np.linspace(0.0, 24.0, T, endpoint=False)
    base = 0.75 + 0.10 * np.cos(2.0 * np.pi * (hour - 14.0) / 24.0)
    morning_peak = 0.35 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening_peak = 0.55 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    curve = base + morning_peak + evening_peak
    return np.clip(curve, 0.5, 1.5)


def extract_lower_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    load_curve = _default_load_curve(T)

    base_p = float(net.load["p_mw"].sum()) if len(net.load) > 0 else 0.0
    base_q = float(net.load["q_mvar"].sum()) if len(net.load) > 0 else 0.0
    e_max = 0.2 * base_p  # battery capacity

    constants = {
        "T": T,
        "load_curve": load_curve,
        "p_load": base_p * load_curve,
        "q_load": base_q * load_curve,
        # TODO: Fill all constants below.
        "c_g": 1,  # linear cost coefficient
        "q_g": 0.1,  # quadratic cost coefficient
        "p_mg_max": 1.5 * base_p,  # maximum microgrid power from distribution network
        "s_max": base_p,  # maximum microgrid load shedding power
        "p_ch_max": 0.25 * e_max,  # maximum charging power
        "p_dis_max": 0.25 * e_max,  # maximum discharging power
        "eta_ch": 0.95,  # charging efficiency
        "eta_dis": 0.92,  # discharging efficiency
        "e_min": 0.0,  # minimum energy storage level
        "e_max": e_max,  # maximum energy storage level / battery capacity
        "e_init": 0.5 * e_max,  # initial energy storage level
    }
    return constants


def extract_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    """
    Build constants for leader optimization in per-unit system.
    case33bw native data (kV, MW/Mvar, ohm, kA) are converted to p.u. here.
    """
    n_bus = len(net.bus)
    line_df = net.line[net.line["in_service"]].reset_index(drop=True)
    n_line = len(line_df)
    slack = int(net.ext_grid.iloc[0]["bus"]) if len(net.ext_grid) > 0 else 0
    
    # Base values for p.u. conversion (TODO: adjust S_base if needed).
    v_base_kv = float(net.bus["vn_kv"].iloc[slack]) if n_bus > 0 else 12.66
    s_base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    z_base_ohm = (v_base_kv ** 2) / s_base_mva
    i_base_ka = s_base_mva / (np.sqrt(3.0) * v_base_kv)

    load_curve = _default_load_curve(T)

    from_bus = [int(line_df.loc[e, "from_bus"]) for e in range(n_line)]
    to_bus = [int(line_df.loc[e, "to_bus"]) for e in range(n_line)]
    
    r = np.array(
        [float(line_df.loc[e, "r_ohm_per_km"] * line_df.loc[e, "length_km"]) for e in range(n_line)],
        dtype=float,
    ) / z_base_ohm
    x = np.array(
        [float(line_df.loc[e, "x_ohm_per_km"] * line_df.loc[e, "length_km"]) for e in range(n_line)],
        dtype=float,
    ) / z_base_ohm
    
    if "i_max_ka" in line_df.columns:
        i_max_ka = np.asarray(line_df["i_max_ka"], dtype=float)
    else:  
        i_max_ka = np.full(n_line, 10.0, dtype=float)
    l_max = (i_max_ka / i_base_ka) ** 2
    lambda_g = electricity_price(T)

    constants = {
        "T": T,
        # Extracted and converted from case33bw
        "n_bus": n_bus,
        "n_line": n_line,
        "slack": slack,
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
        "lambda_g": lambda_g, # shape (T,)
        "pi_min": 0.0, # minimum price
        "pi_max": 1.5, # maximum price
        "p0_max": 10.0, # maximal active power from the transmission network
        "q0_max": 10.0, # maximal reactive power from the transmission network
        "v_ref": 1.0, # reference voltage
        "v_min": 0.80**2, # minimum squared voltage
        "v_max": 1.05**2, # maximum voltage
        "mg_buses": input["mg_buses"] if "mg_buses" in input else [],  # e.g. [5, 10]
        "p_mg": input["p_mg"] if "p_mg" in input else 0, # active power of the microgrid
        "q_mg": input["q_mg"] if "q_mg" in input else 0, # reactive power of the microgrid
        # "p_mg_min": 0.0, # minimum active power of the microgrid
        # "p_mg_max": 0.0, # maximum active power of the microgrid
        # "q_mg_min": 0.0, # minimum reactive power of the microgrid
        # "q_mg_max": 0.0, # maximum reactive power of the microgrid        
        "p_passive": _build_passive_profiles(net, T, load_curve, s_base_mva=s_base_mva)["p_passive"],
        "q_passive": _build_passive_profiles(net, T, load_curve, s_base_mva=s_base_mva)["q_passive"],
        "verbose": False,
    }
    return constants


def _build_passive_profiles(net: pp.pandapowerNet, T: int, load_curve: np.ndarray, s_base_mva: float) -> Dict[str, np.ndarray]:
    # Per-bus passive load profile in p.u.: each bus base load * load_curve / S_base.
    n_bus = len(net.bus)
    p_passive = np.zeros((n_bus, T), dtype=float)
    q_passive = np.zeros((n_bus, T), dtype=float)

    if len(net.load) == 0:
        return {"p_passive": p_passive, "q_passive": q_passive}
    
    for _, row in net.load.iterrows():
        bus = int(row["bus"])
        p_base_pu = float(row["p_mw"]) / s_base_mva
        q_base_pu = float(row["q_mvar"]) / s_base_mva
        p_passive[bus, :] += p_base_pu * load_curve
        q_passive[bus, :] += q_base_pu * load_curve
    return {"p_passive": p_passive, "q_passive": q_passive}


class UpperSolver:
    """
    Solver for upper-level optimization in bilevel Stackelberg game.
    """
    def __init__(self, constants: Dict[str, Any], lower_solver: Optional["LowerSolver"] = None):
        self.constants = constants
        self.lower_solver = lower_solver

        self.T = int(constants["T"])
        self.n_bus = int(constants["n_bus"])
        self.n_line = int(constants["n_line"])
        self.slack = int(constants["slack"])
        self.from_bus = [int(v) for v in constants["from_bus"]]
        self.to_bus = [int(v) for v in constants["to_bus"]]
        self.r = np.asarray(constants["r"], dtype=float)
        self.x = np.asarray(constants["x"], dtype=float)
        self.l_max = np.asarray(constants["l_max"], dtype=float)
        self.load_curve = np.asarray(constants["load_curve"], dtype=float)
        self.v_base_kv = float(constants["v_base_kv"])
        self.s_base_mva = float(constants["s_base_mva"])
        self.lambda_g = np.asarray(
            constants["lambda_g"] if "lambda_g" in constants else constants["lamda_g"],
            dtype=float,
        )
        self.pi_min = float(constants["pi_min"])
        self.pi_max = float(constants["pi_max"])
        self.p0_max = float(constants["p0_max"])
        self.q0_max = float(constants["q0_max"])
        self.v_ref = float(constants["v_ref"])
        self.v_min = float(constants["v_min"])
        self.v_max = float(constants["v_max"])
        self.mg_buses = [int(b) for b in constants.get("mg_buses", [])]
        self.p_passive = np.asarray(constants["p_passive"], dtype=float)
        self.q_passive = np.asarray(constants["q_passive"], dtype=float)
        self.verbose = bool(constants.get("verbose", False))
        self.interaction_count = 0
        self.interaction_history = []

        self.children = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = self.from_bus[e]
            j = self.to_bus[e]
            self.children[i].append(e)
            self.parent_edge[j] = e
    
    
    def distflow(self, pi: np.ndarray, p_mg: np.ndarray) -> Dict[str, Any]:
        """
        Calculate the distribution flow constraints.
        """
        pi = np.asarray(pi, dtype=float).reshape(self.T)
        p_mg_profile = self._build_mg_profile(p_mg)

        model = gp.Model("distflow")
        model.Params.OutputFlag = int(bool(self.verbose))
        model.Params.NonConvex = 2
        # optimization variables setup
        p0 = model.addVars(self.T, lb=0.0, ub=self.p0_max, name="p0")
        q0 = model.addVars(self.T, lb=0.0, ub=self.q0_max, name="q0")
        v = model.addVars(self.n_bus, self.T, lb=self.v_min, ub=self.v_max, name="v")
        p = model.addVars(self.n_line, self.T, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="p")
        q = model.addVars(self.n_line, self.T, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="q")
        l = model.addVars(self.n_line, self.T, lb=0.0, name="l")
        p_inj = model.addVars(self.n_bus, self.T, lb=-gp.GRB.INFINITY, name="p_inj")
        q_inj = model.addVars(self.n_bus, self.T, lb=-gp.GRB.INFINITY, name="q_inj")
        # objective function setup
        p_mg_total = p_mg_profile.sum(axis=0) * self.s_base_mva
        cost = gp.quicksum(self.lambda_g[t] * p0[t] * self.s_base_mva for t in range(self.T))
        revenue = gp.quicksum(p_mg_total[t] * pi[t] for t in range(self.T))
        model.setObjective(
            cost - revenue,
            sense=gp.GRB.MINIMIZE,
        )
        # constraints setup
        for t in range(self.T):
            model.addConstr(v[self.slack, t] == self.v_ref**2, name=f"v_ref[{t}]")
            for e in range(self.n_line):
                # lineflow constraints
                i = self.from_bus[e]
                j = self.to_bus[e]
                model.addConstr(l[e, t] <= self.l_max[e], name=f"l_bound[{e,t}]")
                model.addConstr(v[j, t] == v[i, t] - 2*(self.r[e]*p[e,t] + self.x[e]*q[e,t]) + (self.r[e]**2 + self.x[e]**2)*l[e,t])
                model.addQConstr(
                    p[e, t] * p[e, t] + q[e, t] * q[e, t] <= l[e, t] * v[i, t],
                    name=f"soc[{e,t}]",
                )

            for j in range(self.n_bus):
                model.addConstr(
                    p_inj[j, t] == self.p_passive[j, t] + p_mg_profile[j, t],
                    name=f"p_inj_def[{j,t}]",
                )
                # model.addConstr(q_inj[j, t] == self.q_passive[j, t], name=f"q_inj_def[{j,t}]")

                pe = self.parent_edge[j]
                if pe is None:
                    continue
                model.addConstr(
                    gp.quicksum(p[e, t] for e in self.children[j])
                    == p[pe, t] - p_inj[j, t] - self.r[pe] * l[pe, t],
                    name=f"p_balance[{j,t}]",
                )
                model.addConstr(
                    gp.quicksum(q[e, t] for e in self.children[j])
                    == q[pe, t] - q_inj[j, t] - self.x[pe] * l[pe, t],
                    name=f"q_balance[{j,t}]",
                )

            model.addConstr(
                p0[t] == gp.quicksum(p[e, t] for e in self.children[self.slack]) + p_inj[self.slack, t],
                name=f"slack_p_balance[{t}]",
            )
            model.addConstr(
                q0[t] == gp.quicksum(q[e, t] for e in self.children[self.slack]) + q_inj[self.slack, t],
                name=f"slack_q_balance[{t}]",
            )
        model.optimize()
        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            return 1e10 # large penalty for infeasible solution
        else:
            return model.ObjVal

    def _build_mg_profile(self, p_mg: np.ndarray) -> np.ndarray:
        """
        Convert lower-level microgrid demand to a per-bus active-power profile.
        """
        arr = np.asarray(p_mg, dtype=float)
        profile = np.zeros((self.n_bus, self.T), dtype=float)

        if arr.shape == (self.n_bus, self.T):
            return arr / self.s_base_mva
        if arr.shape == (len(self.mg_buses), self.T):
            for idx, bus in enumerate(self.mg_buses):
                profile[bus, :] = arr[idx, :] / self.s_base_mva
            return profile
        if arr.shape == (self.T,):
            if len(self.mg_buses) != 1:
                raise ValueError("1-D p_mg requires exactly one microgrid bus.")
            profile[self.mg_buses[0], :] = arr / self.s_base_mva
            return profile

        raise ValueError(
            f"Unexpected p_mg shape {arr.shape}; expected (T,), "
            f"({len(self.mg_buses)}, T), or (n_bus, T)."
        )
    

    def evaluate(self, pi: np.ndarray) -> float:
        """
        Evaluate the objective function.
        """
        eval_obj = 0.0
        lower_result = self.lower_solver.lower_solve(pi)

        if lower_result is None:
            eval_obj = 1e10 # large penalty for infeasible solution
        else:
            eval_obj = self.distflow(pi, lower_result["p_mg"])
        self.interaction_count += 1
        self.interaction_history.append({
            "interaction": self.interaction_count,
            "upper_objective": float(eval_obj),
        })
        return eval_obj


    def upper_solve(self, pi_0: np.ndarray, maxiter: int = 1000, cma_verbose: int = 1) -> Dict[str, Any]:
        """
        Solve the upper-level optimization problem.
        Using the CMA-ES algorithm.
        """
        es = cma.CMAEvolutionStrategy(pi_0, (self.pi_max - self.pi_min) / 4, {
            'bounds': [self.pi_min, self.pi_max],
            'seed': 42,
            'verbose': cma_verbose, # 0: no output, 1: output best solution, 2: output all solutions
            'maxiter': maxiter,
        })
        while not es.stop():
            ys = es.ask()
            fits = [self.evaluate(y) for y in ys]
            es.tell(ys, fits)
            es.disp()
        return {"optimal pi": es.result.xbest, "optimal value": es.result.fbest}

    def get_interaction_history(self) -> pd.DataFrame:
        """
        Return the upper objective value after each lower-solver call.
        """
        return pd.DataFrame(self.interaction_history)

    


class LowerSolver:
    """
    Solver for lower-level optimization in bilevel Stackelberg game.
    """
    def __init__(self, lower_constants: Dict[str, Any]):
        self.T = int(lower_constants.get("T", np.asarray(lower_constants["p_load"]).size))
        self.eta_ch = float(lower_constants["eta_ch"])
        self.eta_dis = float(lower_constants["eta_dis"])
        self.e_min = float(lower_constants["e_min"])
        self.e_max = float(lower_constants["e_max"])
        self.e_init = float(lower_constants["e_init"])
        self.p_ch_max = float(lower_constants["p_ch_max"])
        self.p_dis_max = float(lower_constants["p_dis_max"])
        self.p_mg_max = float(lower_constants["p_mg_max"])
        self.s_max = float(lower_constants["s_max"])
        self.c_g = float(lower_constants["c_g"])
        self.q_g = float(lower_constants["q_g"])
        self.p_load = np.asarray(lower_constants["p_load"], dtype=float)
        self.verbose = bool(lower_constants.get("verbose", False))
    
    def lower_solve(self, pi: np.ndarray) -> Dict[str, Any]:
        """
        Solve the lower-level optimization problem.
        """
        model = gp.Model("lower_level")
        model.Params.OutputFlag = int(bool(self.verbose))

        p_mg = model.addVars(self.T, lb=0.0, ub=self.p_mg_max, name="p_mg")
        s = model.addVars(self.T, lb=0.0, ub=self.s_max, name="s") # load shedding power
        p_ch = model.addVars(self.T, lb=0.0, ub=self.p_ch_max, name="p_ch")
        p_dis = model.addVars(self.T, lb=0.0, ub=self.p_dis_max, name="p_dis")
        e = model.addVars(self.T, lb=self.e_min, ub=self.e_max, name="e")
        u_ch = model.addVars(self.T, vtype=gp.GRB.BINARY, name="u_ch")
        u_dis = model.addVars(self.T, vtype=gp.GRB.BINARY, name="u_dis")

        model.setObjective(
            gp.quicksum(pi[t] * p_mg[t] + self.c_g * s[t] + self.q_g * s[t] * s[t] for t in range(self.T)),
            sense=gp.GRB.MINIMIZE,
        )

        for t in range(self.T):
            model.addConstr(
                p_mg[t] + s[t] + p_dis[t] == self.p_load[t] + p_ch[t],
                name=f"power_balance[{t}]",
            )
            model.addConstr(p_ch[t] <= self.p_ch_max * u_ch[t], name=f"charge_logic[{t}]")
            model.addConstr(p_dis[t] <= self.p_dis_max * u_dis[t], name=f"discharge_logic[{t}]")
            model.addConstr(u_ch[t] + u_dis[t] <= 1, name=f"charge_discharge_mutex[{t}]")

            if t == 0:
                model.addConstr(
                    e[t] == self.e_init + self.eta_ch * p_ch[t] - p_dis[t] / self.eta_dis,
                    name=f"soc_dynamics[{t}]",
                )
            else:
                model.addConstr(
                    e[t] == e[t-1] + self.eta_ch * p_ch[t] - p_dis[t] / self.eta_dis,
                    name=f"soc_dynamics[{t}]",
                )
            model.addConstr(e[t] >= self.e_min, name=f"soc_min[{t}]")
            model.addConstr(e[t] <= self.e_max, name=f"soc_max[{t}]")
        model.addConstr(e[self.T-1] == self.e_init, name=f"terminal_soc=initial_soc")
        model.optimize()
        if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            return None # infeasible solution
        else:
            return {
                "p_mg": np.array([p_mg[t].X for t in range(self.T)]),
            }


def solve_gt_optimal(
    input: Optional[Dict[str, Any]] = None,
    upper_net: Optional[pp.pandapowerNet] = None,
    mg_net: Optional[pp.pandapowerNet] = None,
) -> Dict[str, Any]:
    """
    Solve the ground-truth Stackelberg game.

    Default setup:
        upper distribution network: pn.case33bw()
        microgrid network: pn.create_cigre_network_lv()
        horizon: 24 hours
    """
    if input is None:
        input = {}

    T = int(input.get("T", 24))
    if T != 24:
        raise ValueError("solve_gt_optimal only supports the 24-hour setup.")

    if upper_net is None:
        upper_net = pn.case33bw()
    if mg_net is None:
        mg_net = pn.create_cigre_network_lv()

    mg_buses = input.get("mg_buses", [18])
    upper_constants = extract_constants({"mg_buses": mg_buses}, net=upper_net, T=T)
    lower_constants = extract_lower_constants({}, net=mg_net, T=T)

    upper_constants["v_min"] = float(input.get("v_min", 0.80**2))
    if "verbose" in input:
        upper_constants["verbose"] = bool(input["verbose"])
        lower_constants["verbose"] = bool(input["verbose"])

    lower_solver = LowerSolver(lower_constants)
    upper_solver = UpperSolver(upper_constants, lower_solver=lower_solver)

    if "pi_0" in input:
        pi_0 = np.asarray(input["pi_0"], dtype=float).reshape(T)
    else:
        pi_0 = np.full(T, 0.5 * (upper_solver.pi_min + upper_solver.pi_max), dtype=float)

    time_start = time.time()
    result = upper_solver.upper_solve(
        pi_0,
        maxiter=int(input.get("maxiter", 1000)),
        cma_verbose=int(input.get("cma_verbose", 1)),
    )
    interaction_history = upper_solver.get_interaction_history()
    result["interaction_history"] = interaction_history
    if "interaction_history_path" in input:
        interaction_history.to_csv(input["interaction_history_path"], index=False)

    lower_result = lower_solver.lower_solve(result["optimal pi"])
    result["p_mg"] = None if lower_result is None else lower_result["p_mg"]
    result["mg_buses"] = mg_buses
    result["runtime_seconds"] = time.time() - time_start
    result["n_interactions"] = upper_solver.interaction_count
    return result


def save_zero_solver_result(result: Dict[str, Any], output_dir: str = "results") -> Dict[str, str]:
    """
    Save zero-solver outputs beyond the interaction history.

    The JSON keeps scalar metadata and arrays; the solution CSV is convenient
    for plotting or comparing the optimal hourly profiles.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    history_path = output_path / f"zero_solver_history_{timestamp}.csv"
    solution_path = output_path / f"zero_solver_solution_{timestamp}.csv"
    json_path = output_path / f"zero_solver_result_{timestamp}.json"

    interaction_history = result.get("interaction_history")
    if isinstance(interaction_history, pd.DataFrame):
        interaction_history.to_csv(history_path, index=False)

    optimal_pi = np.asarray(result["optimal pi"], dtype=float)
    p_mg = result.get("p_mg")
    p_mg_array = None if p_mg is None else np.asarray(p_mg, dtype=float)

    solution_df = pd.DataFrame({
        "t": np.arange(optimal_pi.size),
        "optimal_pi": optimal_pi,
        "p_mg": np.nan if p_mg_array is None else p_mg_array,
    })
    solution_df.to_csv(solution_path, index=False)

    result_to_save = {
        "optimal_pi": optimal_pi.tolist(),
        "optimal_value": float(result["optimal value"]),
        "p_mg": None if p_mg_array is None else p_mg_array.tolist(),
        "mg_buses": result.get("mg_buses"),
        "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
        "n_interactions": int(result.get("n_interactions", 0)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2)

    saved_paths = {"json": str(json_path), "solution_csv": str(solution_path)}
    if isinstance(interaction_history, pd.DataFrame):
        saved_paths["history_csv"] = str(history_path)
    return saved_paths


def test_gt_solver_smoke():
    """
    Smoke test for the ground-truth bilevel solver components.

    Run manually with:
        python -c "import gt_solver; gt_solver.test_gt_solver_smoke()"
    """
    T = 24
    upper_net = pn.case33bw()
    mg_net = pn.create_cigre_network_lv()
    mg_bus = 5

    upper_constants = extract_constants({"mg_buses": [mg_bus]}, net=upper_net, T=T)
    upper_constants["v_min"] = 0.80**2
    lower_constants = extract_lower_constants({}, net=mg_net, T=T)
    lower_solver = LowerSolver(lower_constants)
    upper_solver = UpperSolver(upper_constants, lower_solver=lower_solver)

    pi = np.full(T, 0.5, dtype=float)
    lower_result = lower_solver.lower_solve(pi)
    assert lower_result is not None
    assert lower_result["p_mg"].shape == (T,)
    assert np.all(lower_result["p_mg"] >= -1e-8)

    obj = upper_solver.evaluate(pi)
    assert np.isfinite(obj)
    assert obj < 1e10

    print("test_gt_solver_smoke passed.")


def main():
    result = solve_gt_optimal({
        "maxiter": 1000,
        "cma_verbose": 1,
        "interaction_history_path": "interaction_history.csv",
    })
    saved_paths = save_zero_solver_result(result)
    print(f"optimal pi: {result['optimal pi']}")
    print(f"optimal value: {result['optimal value']}")
    print(f"p_mg: {result['p_mg']}")
    print(f"runtime seconds: {result['runtime_seconds']:.4f}")
    print("interaction history saved to interaction_history.csv")
    print(f"Saved full result to: {saved_paths['json']}")
    print(f"Saved optimal solution to: {saved_paths['solution_csv']}")
    if "history_csv" in saved_paths:
        print(f"Saved timestamped interaction history to: {saved_paths['history_csv']}")


if __name__ == "__main__":
    main()

# optimal pi: [1.         0.99999712 1.         1.         1.         0.98489783
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.3527656  1.        ]
# optimal value: 33.07042394670213
