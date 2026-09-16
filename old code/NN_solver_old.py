import numpy as np
import pandapower as pp
import pandapower.networks as pn
import gurobipy as gp
from typing import Any, Dict, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from L2Otraining import L2OSolver, MODEL_DIR as L2O_MODEL_DIR
from zero_solver import (
    LowerSolver as ZeroLowerSolver,
    UpperSolver as ZeroUpperSolver,
    extract_lower_constants as extract_zero_lower_constants,
)

np.random.seed(42)


def load_l2o_map_net(model_type: str = "SB", device: Optional[torch.device] = None) -> nn.Module:
    """
    Build L2OSolver and load weights from NNmodel/L2O_{model_type}.pth.
    Pass the return value as map_net=... to TorchDistFlowOptimizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = L2OSolver().to(device)
    ckpt = L2O_MODEL_DIR / f"L2O_{model_type}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing L2O checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def electricity_price(T: int):
    # randomly generate electricity price for each hour
    # Based on the real DA price data for 124 days
    # Randomly select one day and multiply by a random multiplier between 0.5 and 1.5.
    DA_price = pd.read_csv("data/DA_RT_data.csv")["DA"].values
    # unit price change USD per MWh to HKD per KWh
    DA_price = DA_price * 7.3 / 1000 # one (124 * 24,) array
    segments = np.reshape(DA_price, (124, T))
    random_idx = np.random.randint(0, 124)
    selected_segment = segments[random_idx]
    multipliers = np.random.uniform(low=0.5, high=1.5, size=T)
    return selected_segment * multipliers



def _default_load_curve(T: int) -> np.ndarray:
    # Same profile style as follower_solver.
    hour = np.linspace(0.0, 24.0, T, endpoint=False)
    base = 0.75 + 0.10 * np.cos(2.0 * np.pi * (hour - 14.0) / 24.0)
    morning_peak = 0.35 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening_peak = 0.55 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    return np.clip(base + morning_peak + evening_peak, 0.5, 1.5)


def extract_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    """
    Build constants for leader optimization in per-unit system.
    case33bw native data (kV, MW/Mvar, ohm, kA) are converted to p.u. here.
    """
    n_bus = len(net.bus)
    line_df = net.line[net.line["in_service"]].reset_index(drop=True)
    n_line = len(line_df)
    slack = int(net.ext_grid.iloc[0]["bus"]) if len(net.ext_grid) > 0 else 0

    # Base values for p.u. conversion.
    v_base_kv = float(net.bus["vn_kv"].iloc[slack]) if n_bus > 0 else 12.66
    s_base_mva = float(getattr(net, "sn_mva", 1.0) or 1.0)
    z_base_ohm = (v_base_kv ** 2) / s_base_mva
    i_base_ka = s_base_mva / (np.sqrt(3.0) * v_base_kv)

    load_curve = _default_load_curve(T)

    from_bus = [int(line_df.loc[e, "from_bus"]) for e in range(n_line)]
    to_bus = [int(line_df.loc[e, "to_bus"]) for e in range(n_line)]

    # Convert line impedance to p.u.
    r = np.array(
        [float(line_df.loc[e, "r_ohm_per_km"] * line_df.loc[e, "length_km"]) for e in range(n_line)],
        dtype=float,
    ) / z_base_ohm
    x = np.array(
        [float(line_df.loc[e, "x_ohm_per_km"] * line_df.loc[e, "length_km"]) for e in range(n_line)],
        dtype=float,
    ) / z_base_ohm

    # Current limit squared in p.u.
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
        # TODO: Fill all non-case33bw parameters below (ALL in p.u.).
        "lambda_g": lambda_g,  # shape (T,)
        "pi_min": 0.0, # minimum price
        "pi_max": 1.5, # maximum price
        "p0_max": 10.0, # maximal active power from the transmission network
        "q0_max": 10.0, # maximal reactive power from the transmission network
        "v_ref": 1.0, # reference voltage
        "v_min": 0.8, # minimum voltage
        "v_max": 1.05, # maximum voltage
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


def test_extract_constants():
    T = 24
    net = pn.case33bw()
    constants = extract_constants({}, net=net, T=T)

    assert constants["n_bus"] == len(net.bus)
    assert constants["n_line"] == int(net.line["in_service"].sum())
    assert constants["load_curve"].shape == (T,)
    assert constants["l_max"].shape == (constants["n_line"],)
    assert np.all(constants["load_curve"] >= 0.5)
    assert np.all(constants["load_curve"] <= 1.5)
    assert constants["v_ref"] == 1.0
    assert constants["z_base_ohm"] > 0.0
    assert constants["i_base_ka"] > 0.0

    print("test_extract_constants passed.")
    print(
        f"n_bus={constants['n_bus']}, n_line={constants['n_line']}, "
        f"v_base={constants['v_base_kv']}kV, s_base={constants['s_base_mva']}MVA"
    )


class TorchDistFlowOptimizer(nn.Module):
    def __init__(self, constants: Dict[str, Any], map_net: nn.Module, device="cpu"):
        super().__init__()
        self.device = device
        self.map_net = map_net.to(device)   # 你训练好的 L2OSolver，输出 p_mg
        # 常量
        self.T = constants["T"]
        self.n_bus = constants["n_bus"]
        self.n_line = constants["n_line"]
        self.slack = constants["slack"]
        self.from_bus = torch.tensor(constants["from_bus"], dtype=torch.long, device=device)
        self.to_bus   = torch.tensor(constants["to_bus"], dtype=torch.long, device=device)
        self.r = torch.tensor(constants["r"], dtype=torch.float32, device=device)
        self.x = torch.tensor(constants["x"], dtype=torch.float32, device=device)
        self.l_max = torch.tensor(constants["l_max"], dtype=torch.float32, device=device)
        self.lambda_g = torch.tensor(constants["lambda_g"], dtype=torch.float32, device=device)
        self.s_base_mva = float(constants["s_base_mva"])
        self.pi_min = float(constants["pi_min"])
        self.pi_max = float(constants["pi_max"])
        self.p0_max = float(constants["p0_max"])
        self.q0_max = float(constants["q0_max"])
        self.v_ref = float(constants["v_ref"])
        self.v_min = float(constants["v_min"])
        self.v_max = float(constants["v_max"])
        self.mg_buses = constants["mg_buses"]
        self.mg_index = {b: i for i, b in enumerate(self.mg_buses)}
        self.p_passive = torch.as_tensor(constants["p_passive"], dtype=torch.float32, device=device)  # [n_bus, T]
        # ========== 待优化变量 ==========
        # 只优化 pi 时，可把下面这些当“辅助变量”一起优化以满足约束
        self.q_passive = nn.Parameter(
            torch.as_tensor(constants["q_passive"], dtype=torch.float32, device=device).clone()
        )
        self.pi_raw = nn.Parameter(torch.zeros(self.T, device=device))
        self.p0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.q0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.v  = nn.Parameter(torch.ones(self.n_bus, self.T, device=device) * (self.v_ref**2))
        self.l  = nn.Parameter(torch.ones(self.n_line, self.T, device=device) * 1e-3)
        self.p  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        self.q  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        # 构建 children / parent_edge
        child_edges = {i: [] for i in range(self.n_bus)}
        parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            child_edges[i].append(e)
            parent_edge[j] = e
        self.child_edges = child_edges
        self.parent_edge = parent_edge
    def bounded_pi(self):
        # 映射到 [pi_min, pi_max]
        return self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(self.pi_raw)
    def forward(self, penalty_weights=None):
        if penalty_weights is None:
            penalty_weights = {
                "eq": 1000.0,      # 等式约束罚
                "ineq": 100.0,    # 不等式约束罚
                "soc": 100.0,     # SOC 罚
                "bound": 10.0     # 简单上下界罚
            }
        pi = self.bounded_pi()  # [T]
        # ---- NN 映射：p_mg = f(pi) ----
        # 假设 map_net 输入 [B, T]，输出 [B, T] 或 [B, n_mg*T]
        # 单微网示例：
        p_mg_pred = self.map_net(pi.unsqueeze(0)).squeeze(0)  # [T]
        # 如有多个微网，你可把输出改成 [n_mg, T]
        p_mg = {m: p_mg_pred for m in self.mg_buses}  # 临时：每个mg同一个输出（你需按真实模型改）
        # q_mg 若也有映射可同理；没有就先置0或另一个网络
        q_mg = {m: torch.zeros_like(p_mg_pred) for m in self.mg_buses}
        # 注入功率
        p_inj = self.p_passive.clone()
        # Avoid in-place writes on leaf Parameter views.
        q_inj = self.q_passive.clone()
        for m in self.mg_buses:
            p_inj[m, :] = p_inj[m, :] + p_mg[m] / self.s_base_mva
            q_inj[m, :] = q_inj[m, :] + q_mg[m]
        # ========== objective function ==========
        obj = torch.sum(self.lambda_g * self.p0 * self.s_base_mva)
        for m in self.mg_buses:
            obj = obj - torch.sum(pi * p_mg[m])
        # ========== penalty terms ==========
        eq_pen = torch.tensor(0.0, device=self.device)  # equality constraint penalty
        ineq_pen = torch.tensor(0.0, device=self.device)  # inequality constraint penalty
        soc_pen = torch.tensor(0.0, device=self.device)  # SOC constraint penalty
        bound_pen = torch.tensor(0.0, device=self.device)  # bound constraint penalty
        # voltage reference + DistFlow + SOC + power balance
        for t in range(self.T):
            eq_pen = eq_pen + (self.v[self.slack, t] - self.v_ref**2) ** 2
            for e in range(self.n_line):
                i = self.from_bus[e]
                j = self.to_bus[e]
                # l <= l_max
                ineq_pen = ineq_pen + F.relu(self.l[e, t] - self.l_max[e])**2
                # 电压降等式
                vd = self.v[j, t] - (self.v[i, t] - 2*(self.r[e]*self.p[e,t] + self.x[e]*self.q[e,t]) + (self.r[e]**2 + self.x[e]**2)*self.l[e,t])
                eq_pen = eq_pen + vd**2
                # SOC: p^2 + q^2 <= l * v_i
                soc_violation = self.p[e,t]**2 + self.q[e,t]**2 - self.l[e,t]*self.v[i,t]
                soc_pen = soc_pen + F.relu(soc_violation)**2
            # 节点平衡
            for j in range(self.n_bus):
                pe = self.parent_edge[j]
                if pe is None:
                    continue
                outgoing_p = sum(self.p[e, t] for e in self.child_edges[j]) if len(self.child_edges[j]) > 0 else 0.0
                outgoing_q = sum(self.q[e, t] for e in self.child_edges[j]) if len(self.child_edges[j]) > 0 else 0.0
                eq_pen = eq_pen + (outgoing_p - (self.p[pe, t] - p_inj[j, t] - self.r[pe]*self.l[pe, t]))**2
                eq_pen = eq_pen + (outgoing_q - (self.q[pe, t] - q_inj[j, t] - self.x[pe]*self.l[pe, t]))**2
            # slack balance
            slack_out_p = sum(self.p[e, t] for e in self.child_edges[self.slack]) if len(self.child_edges[self.slack]) > 0 else 0.0
            slack_out_q = sum(self.q[e, t] for e in self.child_edges[self.slack]) if len(self.child_edges[self.slack]) > 0 else 0.0
            eq_pen = eq_pen + (self.p0[t] - (slack_out_p + p_inj[self.slack, t]))**2
            eq_pen = eq_pen + (self.q0[t] - (slack_out_q + q_inj[self.slack, t]))**2
        # 变量上下界软罚
        bound_pen = bound_pen + torch.sum(F.relu(-self.p0)**2 + F.relu(self.p0 - self.p0_max)**2)
        bound_pen = bound_pen + torch.sum(F.relu(-self.q0)**2 + F.relu(self.q0 - self.q0_max)**2)
        bound_pen = bound_pen + torch.sum(F.relu(self.v_min**2 - self.v)**2 + F.relu(self.v - self.v_max**2)**2)
        bound_pen = bound_pen + torch.sum(F.relu(-self.l)**2)
        loss = obj \
               + penalty_weights["eq"] * eq_pen \
               + penalty_weights["ineq"] * ineq_pen \
               + penalty_weights["soc"] * soc_pen \
               + penalty_weights["bound"] * bound_pen
        aux = {
            "obj": obj.detach(),
            "eq_pen": eq_pen.detach(),
            "ineq_pen": ineq_pen.detach(),
            "soc_pen": soc_pen.detach(),
            "bound_pen": bound_pen.detach(),
            "pi": pi.detach(),
        }
        return loss, aux


def solve_with_torch(
    model,
    steps=5000,
    lr=1e-2,
    history_path="results/NN_solver_history.csv",
    save_every=100,
    true_eval_constants: Optional[Dict[str, Any]] = None,
    true_eval_mg_net: Optional[pp.pandapowerNet] = None,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for k in range(steps):
        opt.zero_grad()
        loss, aux = model()
        loss.backward()
        opt.step()
        if save_every > 0 and k % save_every == 0:
            with torch.no_grad():
                loss_log, aux_log = model()
            pi_np = aux_log["pi"].detach().cpu().numpy()
            record = {
                "iteration": k,
                "loss": loss_log.item(),
                "obj": aux_log["obj"].item(),
                "eq_pen": aux_log["eq_pen"].item(),
                "ineq_pen": aux_log["ineq_pen"].item(),
                "soc_pen": aux_log["soc_pen"].item(),
                "bound_pen": aux_log["bound_pen"].item(),
            }
            log_msg = (
                f"[{k}] loss={record['loss']:.4f}, obj={record['obj']:.4f}, "
                f"eq={record['eq_pen']:.4f}, soc={record['soc_pen']:.4f}, "
                f"bound={record['bound_pen']:.4f}"
            )
            if true_eval_constants is not None:
                true_eval = evaluate_true_objective(
                    pi_np,
                    true_eval_constants,
                    mg_net=true_eval_mg_net,
                )
                record["true_loss"] = true_eval["true_obj"]
                log_msg += f", true_loss={record['true_loss']:.4f}"
            print(log_msg)
            record.update({f"pi_{i}": value for i, value in enumerate(pi_np)})
            history.append(record)
    with torch.no_grad():
        _, aux = model()
    if history_path is not None and history:
        history_dir = os.path.dirname(history_path)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
    return aux["pi"].cpu().numpy()


def evaluate_true_objective(
    pi: np.ndarray,
    upper_constants: Dict[str, Any],
    mg_net: Optional[pp.pandapowerNet] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate NN-produced pi with the true lower-level response and hard DistFlow model.
    """
    if mg_net is None:
        mg_net = pn.create_cigre_network_lv()

    T = int(upper_constants["T"])
    lower_constants = extract_zero_lower_constants({"verbose": verbose}, net=mg_net, T=T)
    lower_constants["verbose"] = verbose

    eval_constants = dict(upper_constants)
    eval_constants["verbose"] = verbose
    # zero_solver.UpperSolver expects squared-voltage limits.
    eval_constants["v_min"] = upper_constants["v_min"] ** 2
    eval_constants["v_max"] = upper_constants["v_max"] ** 2

    lower_solver = ZeroLowerSolver(lower_constants)
    upper_solver = ZeroUpperSolver(eval_constants, lower_solver=lower_solver)
    lower_result = lower_solver.lower_solve(pi)
    if lower_result is None:
        return {"true_obj": 1e10, "p_mg_true": None}

    p_mg_true = lower_result["p_mg"]
    true_obj = upper_solver.distflow(pi, p_mg_true)
    return {"true_obj": true_obj, "p_mg_true": p_mg_true}


def test_torch_distflow_smoke(
    steps: int = 100,
    lr: float = 1e-2,
    model_type: str = "SB",
    mg_bus: int = 18,
) -> np.ndarray:
    """
    One-shot smoke test: build case33bw + TorchDistFlowOptimizer, run a few Adam steps.
    Uses NNmodel/L2O_{model_type}.pth if present; otherwise random-init L2OSolver (still T=24).
    Run from project root so data/DA_RT_data.csv resolves.
    """
    T = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = pn.case33bw()
    constants = extract_constants({"mg_buses": [mg_bus]}, net=net, T=T)
    ckpt = L2O_MODEL_DIR / f"L2O_{model_type}.pth"
    if ckpt.is_file():
        map_net = load_l2o_map_net(model_type=model_type, device=device)
        for p in map_net.parameters():
            p.requires_grad = False
    else:
        print(f"No checkpoint at {ckpt}; using random L2OSolver for smoke test.")
        map_net = L2OSolver().to(device)
        for p in map_net.parameters():
            p.requires_grad = False

    model = TorchDistFlowOptimizer(constants, map_net=map_net, device=device).to(device)
    loss0, _ = model()
    print(f"Initial loss={loss0.item():.4f}")
    pi = solve_with_torch(model, steps=steps, lr=lr, true_eval_constants=constants)
    print("pi:", pi)
    true_eval = evaluate_true_objective(pi, constants)
    print(f"true objective: {true_eval['true_obj']:.4f}")
    print("true p_mg:", true_eval["p_mg_true"])
    return pi


if __name__ == "__main__":
    # test_extract_constants()
    np.random.seed(42)
    start_time = time.time()
    test_torch_distflow_smoke(steps=5000, lr=1e-2)
    end_time = time.time()
    print(f"Time: {end_time - start_time:.1f}s")
    # # pi = np.array([0.979727, 0.8551583, 0.9011953, 1.154334, 0.53833634, 0.8070916,
    # #              0.83882296, 0.8913337, 0.92085683, 1.0081077, 1.015762, 1.016812,
    # #              0.98006505, 0.9240768, 0.99690413, 0.9587043, 0.9882177, 0.9242619,
    # #              1.0174826, 1.0582225, 0.9876231, 1.0304792, 1.011621, 0.84445477])
    # T = 24
    # mg_bus = 18
    # upper_net = pn.case33bw()
    # constants = extract_constants({"mg_buses": [mg_bus]}, net=upper_net, T=T)
    # true_eval = evaluate_true_objective(pi, constants)
    # print(f"true objective: {true_eval['true_obj']:.4f}")
    # print("true p_mg:", true_eval["p_mg_true"])


# pi: [0.979727   0.8551583  0.9011953  1.154334   0.53833634 0.8070916
#  0.83882296 0.8913337  0.92085683 1.0081077  1.015762   1.016812
#  0.98006505 0.9240768  0.99690413 0.9587043  0.9882177  0.9242619
#  1.0174826  1.0582225  0.9876231  1.0304792  1.011621   0.84445477]
# Time: 761.2s
# obj: 23.2288