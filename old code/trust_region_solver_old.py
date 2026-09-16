import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import gurobipy as gp
from typing import Any, Dict, Optional
import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from L2Otraining import L2OSolver, MODEL_DIR as L2O_MODEL_DIR
import logging
import time
import json
from scipy.linalg import svd
from scipy.stats.qmc import LatinHypercube, Sobol, scale
np.random.seed(42)
logger = logging.getLogger(__name__)

#region Part I: Helper Functions and Lower Solver class Setup
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
        "verbose": bool(input.get("verbose", False)),
    }
    return constants


class LowerSolver:
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
#endregion


#region Part II: Surrogate Model class Setup
class SurrogateModel(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    

class GlobalSurrogate(SurrogateModel):
    def __init__(self, map_net: nn.Module, device: str = 'cpu'):
        """
        input: map_net: nn.Module, device: str = 'cpu'
        """
        self.device = device
        self.map_net = map_net.to(device)
        self.map_net.eval()
        self._residuals: list["ResidualModel"] = []
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.map_net(x_t).cpu().numpy()
        for res in self._residuals:
            y += np.array([res.predict(xi) for xi in x])
        return y[0] if single else y

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable surrogate prediction used by leader.solve().
        Keeps the map_net path in the torch graph so pi receives the NN gradient.
        """
        single = (x.ndim == 1)
        x_batch = x.unsqueeze(0) if single else x
        y = self.map_net(x_batch.to(self.device))
        for res in self._residuals:
            if single:
                y = y + res.predict_torch(x).unsqueeze(0)
            else:
                y = y + torch.stack([res.predict_torch(xi) for xi in x], dim=0)
        return y[0] if single else y

    def merge(self, residual: "ResidualModel") -> None:
        self._residuals.append(residual)
    
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


def _unique_rows(X: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the first occurrence of rows separated by more than tol in Euclidean norm.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X.reshape(0, X.shape[-1] if X.ndim == 2 else 0), np.array([], dtype=int)
    keep = []
    kept_rows = []
    for idx, row in enumerate(X):
        if not kept_rows:
            keep.append(idx)
            kept_rows.append(row)
            continue
        distances = np.linalg.norm(np.asarray(kept_rows) - row, axis=1)
        if np.all(distances > tol):
            keep.append(idx)
            kept_rows.append(row)
    keep_idx = np.asarray(keep, dtype=int)
    return X[keep_idx], keep_idx


class ResidualModel:
    def __init__(self, model_type: str = "quadratic", lasso_alpha: float = 1e-4,
                 lasso_max_iter: int = 5000, lasso_tol: float = 1e-6,
                 rbf_kernel: str = "cubic", rbf_smoothing: float = 1e-8,
                 rbf_degree: int = 1, rbf_scale: str = "trust_radius",
                 rbf_epsilon: Optional[float] = None):
        """
        Input: model_type = "quadratic", "linear", or "rbf"
        Quadratic model: f(x) = x^T * A * x + b^T * x + c
        Linear model: f(x) = b^T * x + c
        RBF model: f(x) = P(xi) beta + sum_i lambda_i phi(||xi - xi_i||)
        lasso_alpha controls the L1 penalty used by quadratic fitting.
        """
        self.model_type = model_type
        self.lasso_alpha = lasso_alpha
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.rbf_kernel = rbf_kernel
        self.rbf_smoothing = rbf_smoothing
        self.rbf_degree = rbf_degree
        self.rbf_scale = rbf_scale
        self.rbf_epsilon = rbf_epsilon
        self.x_center = None
        self.A = None
        self.b = None
        self.c = None 
        self.rbf_centers = None
        self.rbf_lambda = None
        self.rbf_beta = None
        self.rbf_input_scale = None
        self.rbf_condition = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, x_center: np.ndarray = None,
            radius: float = None) -> None:
        if self.model_type == "quadratic":
            self._fit_quadratic(X, y, x_center=x_center)
        elif self.model_type == "linear":
            self._fit_linear(X, y)
        elif self.model_type == "rbf":
            self._fit_rbf(X, y, x_center=x_center, radius=radius)
        else:
            raise ValueError(f"Invalid residual model type: {self.model_type}")

    def _fit_quadratic(self, X: np.ndarray, y: np.ndarray,
                       x_center: np.ndarray = None) -> None:
        n, d_x = X.shape
        self.x_center = (
            np.asarray(x_center, dtype=float).copy()
            if x_center is not None
            else np.zeros(d_x, dtype=float)
        )
        X_local = X - self.x_center
        # 构造局部二次特征：(x_i-xc_i) * (x_j-xc_j)，i <= j
        quad_feats = []
        quad_index = []                                                        # 记录 (i,j) 对应位置
        for i in range(d_x):
            for j in range(i, d_x):
                quad_feats.append(X_local[:, i] * X_local[:, j])             # (n,)
                quad_index.append((i, j))
        quad_feats = np.stack(quad_feats, axis=1)                             # (n, n_quad)
        # 增广特征矩阵：[二次项, 线性项, 常数项]
        X_feat = np.hstack([quad_feats, X_local])                            # (n, n_quad+d_x)
        lasso_alpha = max(float(self.lasso_alpha), 0.0)
        if lasso_alpha == 0.0:
            X_aug = np.hstack([X_feat, np.ones((n, 1))])                     # (n, n_quad+d_x+1)
            W, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)              # (n_quad+d_x+1, d_y)
        else:
            # Lasso objective: 1/(2n)||XW-y||^2 + alpha*||W||_1.
            # The intercept is fitted after centering and is not L1-penalized.
            y_2d = y if y.ndim == 2 else y.reshape(-1, 1)
            x_mean = X_feat.mean(axis=0)
            x_scale = X_feat.std(axis=0)
            x_scale[x_scale < 1e-12] = 1.0
            y_mean = y_2d.mean(axis=0)
            X_std = (X_feat - x_mean) / x_scale
            y_center = y_2d - y_mean

            n_feat = X_std.shape[1]
            coef = np.zeros((n_feat, y_2d.shape[1]))
            z = coef.copy()
            t = 1.0
            lipschitz = (np.linalg.norm(X_std, ord=2) ** 2) / max(n, 1)
            step = 1.0 / max(lipschitz, 1e-12)

            for _ in range(int(self.lasso_max_iter)):
                grad = X_std.T @ (X_std @ z - y_center) / n
                coef_next = np.sign(z - step * grad) * np.maximum(
                    np.abs(z - step * grad) - step * lasso_alpha, 0.0
                )
                t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                z = coef_next + ((t - 1.0) / t_next) * (coef_next - coef)
                denom = max(np.linalg.norm(coef), 1.0)
                if np.linalg.norm(coef_next - coef) / denom < self.lasso_tol:
                    coef = coef_next
                    break
                coef = coef_next
                t = t_next

            coef_unscaled = coef / x_scale[:, None]
            intercept = y_mean - x_mean @ coef_unscaled
            W = np.vstack([coef_unscaled, intercept])                        # (n_quad+d_x+1, d_y)
        n_quad = len(quad_index)
        # 从 W 中还原 Q（对称），A，b
        # 这里取所有输出维度的平均 Q 作为共享二次项
        w_quad = W[:n_quad, :]                                                # (n_quad, d_y)
        w_quad_mean = w_quad.mean(axis=1)                                     # (n_quad,)，各输出平均
        self.A = np.zeros((d_x, d_x))
        for idx, (i, j) in enumerate(quad_index):
            if i == j:
                self.A[i, i] = w_quad_mean[idx]
            else:
                self.A[i, j] = w_quad_mean[idx] / 2
                self.A[j, i] = w_quad_mean[idx] / 2
        self.b = W[n_quad:n_quad + d_x, :].T                                 # (d_y, d_x)
        self.c = W[-1]      
    
    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d_x = X.shape
        X_aug = np.hstack([X, np.ones((n, 1))])                  # (n, d_x+1)
        W, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)              # (d_x+1, d_y)
        self.b = W[:d_x, :].T                                 # (d_y, d_x)
        self.c = W[-1]      

    def _fit_rbf(self, X: np.ndarray, y: np.ndarray,
                 x_center: np.ndarray = None, radius: float = None) -> None:
        X = np.asarray(X, dtype=float)
        y_2d = y if y.ndim == 2 else y.reshape(-1, 1)
        n, d_x = X.shape
        self.x_center = (
            np.asarray(x_center, dtype=float).copy()
            if x_center is not None
            else X.mean(axis=0)
        )
        self.rbf_input_scale = self._resolve_rbf_scale(X, radius)
        centers = (X - self.x_center) / self.rbf_input_scale
        centers, keep = _unique_rows(centers, tol=1e-10)
        y_2d = y_2d[keep]
        n = centers.shape[0]
        if n == 0:
            raise ValueError("RBF residual model needs at least one sample.")

        pairwise = centers[:, None, :] - centers[None, :, :]
        R = np.linalg.norm(pairwise, axis=2)
        K = self._rbf_phi_np(R)
        smoothing = max(float(self.rbf_smoothing), 0.0)
        if smoothing > 0.0:
            K = K + smoothing * np.eye(n)

        P = self._rbf_poly_np(centers)
        n_poly = P.shape[1]
        lhs = np.block([
            [K, P],
            [P.T, np.zeros((n_poly, n_poly))],
        ])
        rhs = np.vstack([y_2d, np.zeros((n_poly, y_2d.shape[1]))])
        self.rbf_condition = float(np.linalg.cond(lhs))
        try:
            coeff = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coeff, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        self.rbf_centers = centers
        self.rbf_lambda = coeff[:n]
        self.rbf_beta = coeff[n:]

    def _resolve_rbf_scale(self, X: np.ndarray, radius: float = None) -> float:
        if self.rbf_scale == "trust_radius" and radius is not None:
            return max(float(radius), 1e-8)
        spread = np.max(np.linalg.norm(X - self.x_center, axis=1))
        return max(float(spread), 1e-8)

    def _rbf_poly_np(self, x_local: np.ndarray) -> np.ndarray:
        x_local = np.atleast_2d(x_local)
        if self.rbf_degree < 0:
            return np.zeros((x_local.shape[0], 0))
        if self.rbf_degree == 0:
            return np.ones((x_local.shape[0], 1))
        if self.rbf_degree == 1:
            return np.hstack([np.ones((x_local.shape[0], 1)), x_local])
        raise ValueError("Only RBF polynomial degrees -1, 0, and 1 are supported.")

    def _rbf_poly_torch(self, x_local: torch.Tensor) -> torch.Tensor:
        x_local = x_local.unsqueeze(0) if x_local.ndim == 1 else x_local
        if self.rbf_degree < 0:
            return torch.empty((x_local.shape[0], 0), dtype=x_local.dtype, device=x_local.device)
        ones = torch.ones((x_local.shape[0], 1), dtype=x_local.dtype, device=x_local.device)
        if self.rbf_degree == 0:
            return ones
        if self.rbf_degree == 1:
            return torch.cat([ones, x_local], dim=1)
        raise ValueError("Only RBF polynomial degrees -1, 0, and 1 are supported.")

    def _rbf_phi_np(self, r: np.ndarray) -> np.ndarray:
        kernel = self.rbf_kernel
        eps = 1.0 if self.rbf_epsilon is None else float(self.rbf_epsilon)
        if kernel == "linear":
            return r
        if kernel == "cubic":
            return r ** 3
        if kernel == "thin_plate_spline":
            return np.where(r > 0.0, r ** 2 * np.log(r), 0.0)
        if kernel == "gaussian":
            return np.exp(-(eps * r) ** 2)
        if kernel == "multiquadric":
            return np.sqrt(1.0 + (eps * r) ** 2)
        if kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
        raise ValueError(f"Unknown RBF kernel: {kernel}")

    def _rbf_phi_torch(self, r: torch.Tensor) -> torch.Tensor:
        kernel = self.rbf_kernel
        eps = 1.0 if self.rbf_epsilon is None else float(self.rbf_epsilon)
        if kernel == "linear":
            return r
        if kernel == "cubic":
            return r ** 3
        if kernel == "thin_plate_spline":
            safe_r = torch.clamp(r, min=1e-12)
            return torch.where(r > 0.0, r ** 2 * torch.log(safe_r), torch.zeros_like(r))
        if kernel == "gaussian":
            return torch.exp(-(eps * r) ** 2)
        if kernel == "multiquadric":
            return torch.sqrt(1.0 + (eps * r) ** 2)
        if kernel == "inverse_multiquadric":
            return 1.0 / torch.sqrt(1.0 + (eps * r) ** 2)
        raise ValueError(f"Unknown RBF kernel: {kernel}")

    def _rbf_dphi_dr_np(self, r: np.ndarray) -> np.ndarray:
        kernel = self.rbf_kernel
        eps = 1.0 if self.rbf_epsilon is None else float(self.rbf_epsilon)
        if kernel == "linear":
            return np.ones_like(r)
        if kernel == "cubic":
            return 3.0 * r ** 2
        if kernel == "thin_plate_spline":
            return np.where(r > 0.0, 2.0 * r * np.log(r) + r, 0.0)
        if kernel == "gaussian":
            return -2.0 * eps ** 2 * r * np.exp(-(eps * r) ** 2)
        if kernel == "multiquadric":
            return eps ** 2 * r / np.sqrt(1.0 + (eps * r) ** 2)
        if kernel == "inverse_multiquadric":
            return -eps ** 2 * r / (1.0 + (eps * r) ** 2) ** 1.5
        raise ValueError(f"Unknown RBF kernel: {kernel}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_type == "quadratic":
            x_local = x - self.x_center
            return x_local.T @ self.A @ x_local + self.b @ x_local + self.c
        elif self.model_type == "linear":
            return self.b @ x + self.c
        elif self.model_type == "rbf":
            if self.rbf_centers is None or self.rbf_lambda is None or self.rbf_beta is None:
                raise RuntimeError("RBF ResidualModel must be fitted before prediction.")
            single = (x.ndim == 1)
            x_2d = x[np.newaxis, :] if single else x
            x_local = (x_2d - self.x_center) / self.rbf_input_scale
            R = np.linalg.norm(x_local[:, None, :] - self.rbf_centers[None, :, :], axis=2)
            y = self._rbf_phi_np(R) @ self.rbf_lambda + self._rbf_poly_np(x_local) @ self.rbf_beta
            return y[0] if single else y
        else:
            raise ValueError(f"Invalid residual model type: {self.model_type}")

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Torch version of predict() for differentiable leader subproblem solves.
        """
        if self.model_type != "rbf" and (self.b is None or self.c is None):
            raise RuntimeError("ResidualModel must be fitted before prediction.")
        if self.model_type == "quadratic":
            if self.A is None:
                raise RuntimeError("Quadratic ResidualModel must be fitted before prediction.")
            b = torch.as_tensor(self.b, dtype=x.dtype, device=x.device)
            c = torch.as_tensor(self.c, dtype=x.dtype, device=x.device)
            A = torch.as_tensor(self.A, dtype=x.dtype, device=x.device)
            x_center = torch.as_tensor(self.x_center, dtype=x.dtype, device=x.device)
            x_local = x - x_center
            return x_local.T @ A @ x_local + b @ x_local + c
        elif self.model_type == "linear":
            b = torch.as_tensor(self.b, dtype=x.dtype, device=x.device)
            c = torch.as_tensor(self.c, dtype=x.dtype, device=x.device)
            return b @ x + c
        elif self.model_type == "rbf":
            if self.rbf_centers is None or self.rbf_lambda is None or self.rbf_beta is None:
                raise RuntimeError("RBF ResidualModel must be fitted before prediction.")
            centers = torch.as_tensor(self.rbf_centers, dtype=x.dtype, device=x.device)
            rbf_lambda = torch.as_tensor(self.rbf_lambda, dtype=x.dtype, device=x.device)
            rbf_beta = torch.as_tensor(self.rbf_beta, dtype=x.dtype, device=x.device)
            x_center = torch.as_tensor(self.x_center, dtype=x.dtype, device=x.device)
            x_local = (x - x_center) / self.rbf_input_scale
            r = torch.linalg.norm(x_local.unsqueeze(0) - centers, dim=1)
            phi = self._rbf_phi_torch(r)
            poly = self._rbf_poly_torch(x_local)
            return phi @ rbf_lambda + poly.squeeze(0) @ rbf_beta
        else:
            raise ValueError(f"Invalid residual model type: {self.model_type}")
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        if self.model_type == "quadratic":
            x_local = x - self.x_center
            return (self.A + self.A.T) @ x_local + self.b
        elif self.model_type == "linear":
            return self.b
        elif self.model_type == "rbf":
            if self.rbf_centers is None or self.rbf_lambda is None or self.rbf_beta is None:
                raise RuntimeError("RBF ResidualModel must be fitted before computing gradients.")
            x_local = (np.asarray(x, dtype=float) - self.x_center) / self.rbf_input_scale
            diff = x_local[None, :] - self.rbf_centers
            r = np.linalg.norm(diff, axis=1)
            direction = np.zeros_like(diff)
            nonzero = r > 1e-12
            direction[nonzero] = diff[nonzero] / r[nonzero, None]
            dphi_dx = (
                self._rbf_dphi_dr_np(r)[:, None] * direction / self.rbf_input_scale
            )                                                       # (n_centers, d_x)
            grad = self.rbf_lambda.T @ dphi_dx                     # (d_y, d_x)
            if self.rbf_degree == 1:
                grad += self.rbf_beta[1:, :].T / self.rbf_input_scale
            return grad
        else:
            raise ValueError(f"Invalid residual model type: {self.model_type}")
#endregion


#region Part III: Trust Region class Setup 
class TrustRegion:
    def __init__(self, R_0: float, R_max: float, gamma_dec: float, gamma_inc: float):
        """
        Input  : R_0       — 初始半径
                 R_max     — 最大半径
                 gamma_dec — 收缩因子 ∈ (0,1)
                 gamma_inc — 扩张因子 > 1
        """
        self.R = R_0
        self.R_max = R_max
        self.gamma_dec = gamma_dec
        self.gamma_inc = gamma_inc
    def get_radius(self) -> float:
        return self.R
    def shrink(self) -> float:
        self.R = self.gamma_dec * self.R
        return self.R
    def keep(self) -> float:
        return self.R
    def expand(self) -> float:
        self.R = min(self.gamma_inc * self.R, self.R_max)
        return self.R
    def contains(self, x_center: np.ndarray, x_query: np.ndarray) -> bool:
        """
        Input  : x_center — ndarray (d_x,)
                 x_query  — ndarray (d_x,)
        Output : bool
        """
        return np.max(np.abs(x_query - x_center)) <= self.R

# Acitve Subspace Selection
class ActiveSubspace:
    """
    Based on the global surrogate model, select the active subspace.
    The active subspace of a 24-dimension electricity price should be 5-6 dimensions.
    Use this fact to reduce sampling number.
    """
    def __init__(self, n_active: int = 6):
        self.n_active = n_active
        self.directions = None
        self.singular_values = None
        self.explained_ratio = None

    def identify(self, x_center: np.ndarray, radius: float, surrogate_fn, n_probe: int = 80) -> np.ndarray:
        """
        Identify the active subspace based on the global surrogate model.
        Input  : x_center — ndarray (d_x,): center of the trust region
                 radius   — float: radius of the trust region
                 surrogate_fn — function:  surrogate model function
                 n_probe  — int:    number of directions to probe
        Output : directions — ndarray (d_x, n_active)
        """
        d_x = len(x_center)
        h = radius * 0.05 
        grad_samples = []
        for _ in range(n_probe):
            d = np.random.randn(d_x)
            d = d / np.linalg.norm(d)
            y_plus = np.atleast_1d(surrogate_fn.predict(x_center + h * d))
            y_minus = np.atleast_1d(surrogate_fn.predict(x_center - h * d))
            dir_derive = np.linalg.norm(y_plus - y_minus) / (2 * h)
            grad_samples.append(dir_derive * d)
        G = np.vstack(grad_samples)
        _, S, Vh = svd(G, full_matrices=False)
        self.directions = Vh[:self.n_active, :]  # (n_active, d_x)
        self.singular_values = S[:self.n_active]
        self.explained_ratio = (np.sum(S[:self.n_active] ** 2) / (np.sum(S ** 2) + 1e-6))
        return self.directions

    def project(self, x: np.ndarray, x_center: np.ndarray) -> np.ndarray:
        """高维偏移 → 低维坐标"""
        offset = np.atleast_2d(x) - x_center
        return offset @ self.directions.T  # (n, n_active)
    def lift(self, z: np.ndarray, x_center: np.ndarray) -> np.ndarray:
        """低维坐标 → 高维点"""
        z = np.atleast_2d(z)
        return x_center + z @ self.directions  # (n, d_x)
    def get_info(self) -> dict:
        return {
            'n_active': self.n_active,
            'explained_ratio': self.explained_ratio,
            'singular_values': self.singular_values
        }


def sample_in_trust_region(x_center: np.ndarray,
                            tr: TrustRegion,
                            n: int,
                            feasible_set: dict,
                            method: str = 'lhs',
                            active_subspace: ActiveSubspace = None,
                            n_supplement: int = 10,
                            seed: int = None,
                            include_star_points: bool = False,
                            star_alpha: float = 0.5,
                            min_dist: float = 1e-8) -> np.ndarray:
    """
    在信赖域 ∩ 可行域内采样 n 个点。
    Input  : x_center    — ndarray (d_x,)
             radius      — 信赖域半径
             n           — 采样数量
             feasible_set— {'lb': ndarray(d_x,), 'ub': ndarray(d_x,)}
             method      — 'random' | 'lhs' | 'sobol'
             active_subspace - 
    Output : samples     — ndarray (n, d_x)
    """
    radius = tr.get_radius()
    d_x = len(x_center)
    lb = np.maximum(feasible_set['lb'], x_center - radius)
    ub = np.minimum(feasible_set['ub'], x_center + radius)
    rng = np.random.default_rng(seed)
    def _sample_box(n_points: int) -> np.ndarray:
        if n_points <= 0:
            return np.empty((0, d_x))
        if method == 'random':
            return rng.uniform(lb, ub, size=(n_points, len(lb)))
        elif method == 'lhs':
            sampler = LatinHypercube(d=len(lb), seed=seed)
            samples = sampler.random(n_points)                     # (n, len(lb))，值域 [0,1]
            return scale(samples, lb, ub)   # 映射到 [lb, ub]
        elif method == 'sobol':
            sampler = Sobol(d=len(lb), scramble=True, seed=seed)
            samples = sampler.random(n_points)                     # (n, len(lb))，值域 [0,1]
            return scale(samples, lb, ub)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _star_points(directions: np.ndarray, max_points: int) -> np.ndarray:
        if (not include_star_points) or max_points <= 0:
            return np.empty((0, d_x))
        points = []
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            unit = direction / norm
            points.append(np.clip(x_center + star_alpha * radius * unit, lb, ub))
            points.append(np.clip(x_center - star_alpha * radius * unit, lb, ub))
            if len(points) >= max_points:
                break
        return np.asarray(points[:max_points]) if points else np.empty((0, d_x))

    def _finalize(points: np.ndarray, target_n: int) -> np.ndarray:
        if points.size == 0:
            points = np.empty((0, d_x))
        unique, _ = _unique_rows(points, tol=min_dist)
        refill_tries = 0
        while unique.shape[0] < target_n and refill_tries < 5:
            refill = _sample_box(target_n - unique.shape[0])
            unique, _ = _unique_rows(np.vstack([unique, refill]), tol=min_dist)
            refill_tries += 1
        return unique[:target_n]

    if active_subspace is None:   
        star = _star_points(np.eye(d_x), max_points=min(2 * d_x, n))
        fill = _sample_box(n - len(star))
        return _finalize(np.vstack([star, fill]), n)
    else:
        n_active = active_subspace.n_active
        star = _star_points(active_subspace.directions, max_points=min(2 * n_active, n))
        n_supp = min(n_supplement, max(n - len(star), 0))
        n_act_samples = max(n - len(star) - n_supp, 0)
        if n_act_samples > 0:
            if method == 'lhs':
                sampler = LatinHypercube(d=n_active, optimization='random-cd', seed=seed)
                raw = sampler.random(n_act_samples)
            elif method == 'sobol':
                sampler = Sobol(d=n_active, scramble=True, seed=seed)
                raw = sampler.random(n_act_samples)
            else:
                raw = rng.random((n_act_samples, n_active))
            z = 2 * raw - 1  # (n, n_active)
            # 2) 径向缩放：让点在球内均匀分布而非集中在角落
            norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-10
            z = z / norms  # 单位化方向
            # r^(1/d) 分布使得径向均匀
            radii = rng.uniform(0.2, 1.0, size=(n_act_samples, 1)) ** (1.0 / n_active)
            z = z * radii  # (n, n_active), 值域在单位球内
            # 3) 映射回高维
            x_main = active_subspace.lift(z * radius, x_center)  # (n, d_x)
        else:
            x_main = np.empty((0, d_x))
        # 4) 正交补方向补充
        x_supp = _sample_orthogonal_complement(
            x_center, radius, active_subspace.directions, n_supp, d_x, rng=rng
        )
        # 5) 合并: 主方向 + 几何骨架 + 补充；中心点由调用方统一添加。
        all_points = np.vstack([star, x_main, x_supp])
        # 6) Box 约束截断
        all_points = np.clip(all_points, lb, ub)
        all_points = _finalize(all_points, n)
        print(f"Number of samples: {all_points.shape[0]}")
        return all_points


def _sample_orthogonal_complement(x_center: np.ndarray, radius: float,
                                  active_dirs: np.ndarray,
                                  n_points: int, d_x: int,
                                  rng: np.random.Generator = None) -> np.ndarray:
    """
    在主动子空间的正交补方向采样探测点。
    目的：检测降维是否遗漏了对残差有显著影响的方向。
    """
    # 正交补投影矩阵: I - V^T V
    proj_null = np.eye(d_x) - active_dirs.T @ active_dirs
    if rng is None:
        rng = np.random.default_rng()
    if n_points <= 0:
        return np.empty((0, d_x))
    points = []
    for _ in range(n_points):
        d = rng.standard_normal(d_x)
        d = proj_null @ d
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            d = rng.standard_normal(d_x)
            d = proj_null @ d
            norm = np.linalg.norm(d)
        d = d / norm * radius * rng.uniform(0.3, 0.7)
        points.append(x_center + d)
    return np.array(points)


def compute_ratio(F_current: float, F_candidate: float,
                  F_tilde_current: float, F_tilde_candidate: float) -> float:
    """
    rho_k = Ared_k / Pred_k
    Input  : F_current         — 真实目标在 x^k 处的值
             F_candidate       — 真实目标在 x^{k,cand} 处的值
             F_tilde_current   — 代理目标在 x^k 处的值
             F_tilde_candidate — 代理目标在 x^{k,cand} 处的值
    Output : rho — float, Pred_k <= 0 时返回 -inf
    """
    Pred = F_tilde_current - F_tilde_candidate
    Ared = F_current - F_candidate
    if Pred <= 0:
        return -np.inf
    return Ared / Pred


def decide_step(rho: float, eta_1: float, eta_2: float) -> str:
    """
    Input  : rho   — compute_ratio 的返回值
             eta_1 — 下阈值
             eta_2 — 上阈值
    Output : 'reject' | 'accept_keep' | 'accept_expand'
    """
    if rho < eta_1:
        return 'reject'
    elif rho < eta_2:
        return 'accept_keep'
    else:
        return 'accept_expand'

#endregion


#region Part V: Trust Region Solver class Setup
class _LeaderSubproblem(nn.Module):
    """
    仅在 DistributionNetworkLeader 内部使用。
    fixed_pi=None, fixed_p_mg=None → 完整优化模式（用于 solve)
    fixed_pi=x,    fixed_p_mg=y    → 固定模式（用于 evaluate_objective,此时用 Gurobi)
    """
    def __init__(self, constants: dict, surrogate: SurrogateModel, device: str = 'cpu',
                 fixed_pi: np.ndarray = None, fixed_p_mg: np.ndarray = None):
        """
        Input  : constants  — extract_constants() 的返回值
                 surrogate  — 当前代理模型（固定模式时可为 None)
                 device     — 'cpu' 或 'cuda'
                 fixed_pi   — ndarray (T,)，固定模式时传入
                 fixed_p_mg — ndarray (T,)，固定模式时传入
        """
        super().__init__()
        self.device    = device
        self.surrogate = surrogate
        self.fixed_pi  = fixed_pi
        self.fixed_p_mg = fixed_p_mg
        # 常量
        self.T       = constants["T"]
        self.n_bus   = constants["n_bus"]
        self.n_line  = constants["n_line"]
        self.slack   = constants["slack"]
        self.from_bus  = torch.tensor(constants["from_bus"],  dtype=torch.long,    device=device)
        self.to_bus    = torch.tensor(constants["to_bus"],    dtype=torch.long,    device=device)
        self.r         = torch.tensor(constants["r"],         dtype=torch.float32, device=device)
        self.x         = torch.tensor(constants["x"],         dtype=torch.float32, device=device)
        self.l_max     = torch.tensor(constants["l_max"],     dtype=torch.float32, device=device)
        self.lambda_g  = torch.tensor(constants["lambda_g"],  dtype=torch.float32, device=device)
        self.s_base_mva = float(constants["s_base_mva"])
        self.pi_min    = float(constants["pi_min"])
        self.pi_max    = float(constants["pi_max"])
        self.p0_max    = float(constants["p0_max"])
        self.q0_max    = float(constants["q0_max"])
        self.v_ref     = float(constants["v_ref"])
        self.v_min     = float(constants["v_min"])
        self.v_max     = float(constants["v_max"])
        self.mg_buses  = constants["mg_buses"]
        self.p_passive = torch.as_tensor(
            constants["p_passive"], dtype=torch.float32, device=device
        )
        # 拓扑结构
        self.child_edges = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            self.child_edges[i].append(e)
            self.parent_edge[j] = e
        # 优化变量
        if fixed_pi is None:
            self.pi_raw = nn.Parameter(torch.zeros(self.T, device=device))
        else:
            self.pi_fixed = torch.tensor(fixed_pi, dtype=torch.float32, device=device)
        self.q_passive = nn.Parameter(
            torch.as_tensor(constants["q_passive"], dtype=torch.float32, device=device).clone()
        )
        self.p0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.q0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.v  = nn.Parameter(
            torch.ones(self.n_bus, self.T, device=device) * (self.v_ref ** 2)
        )
        self.l  = nn.Parameter(torch.ones(self.n_line, self.T, device=device) * 1e-3)
        self.p  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        self.q  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
    def bounded_pi(self) -> torch.Tensor:
        """
        Output : pi — Tensor (T,)，值域 [pi_min, pi_max]
        """
        if self.fixed_pi is None:
            return self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(self.pi_raw)
        else:
            return self.pi_fixed
    def forward(self, penalty_weights: dict = None) -> tuple:
        """
        Input  : penalty_weights — dict, keys: 'eq','ineq','soc','bound'
        Output : (loss: Tensor, aux: dict)
        """
        if penalty_weights is None:
            penalty_weights = {'eq': 1000.0, 'ineq': 100.0, 'soc': 100.0, 'bound': 10.0}
        pi = self.bounded_pi()                                  # (T,)
        # p_mg 来源
        if self.fixed_p_mg is not None:
            p_mg_t = torch.tensor(self.fixed_p_mg, dtype=torch.float32, device=self.device)
        else:
            if hasattr(self.surrogate, "predict_torch"):
                p_mg_t = self.surrogate.predict_torch(pi)
            else:
                pi_np = pi.detach().cpu().numpy()
                p_mg_t = torch.tensor(
                    self.surrogate.predict(pi_np), dtype=torch.float32, device=self.device
                )
        p_mg = {m: p_mg_t for m in self.mg_buses}
        q_mg = {m: torch.zeros_like(p_mg_t) for m in self.mg_buses}
        # 注入功率
        p_inj = self.p_passive.clone()
        q_inj = self.q_passive.clone()
        for m in self.mg_buses:
            p_inj[m, :] = p_inj[m, :] + p_mg[m] / self.s_base_mva
            q_inj[m, :] = q_inj[m, :] + q_mg[m]
        # 目标函数
        obj = torch.sum(self.lambda_g * self.p0 * self.s_base_mva)
        for m in self.mg_buses:
            obj = obj - torch.sum(pi * p_mg[m])
        # 罚项
        eq_pen    = torch.tensor(0.0, device=self.device)
        ineq_pen  = torch.tensor(0.0, device=self.device)
        soc_pen   = torch.tensor(0.0, device=self.device)
        bound_pen = torch.tensor(0.0, device=self.device)
        for t in range(self.T):
            eq_pen = eq_pen + (self.v[self.slack, t] - self.v_ref ** 2) ** 2
            for e in range(self.n_line):
                i = self.from_bus[e]
                j = self.to_bus[e]
                ineq_pen = ineq_pen + F.relu(self.l[e, t] - self.l_max[e]) ** 2
                vd = (self.v[j, t]
                      - self.v[i, t]
                      + 2 * (self.r[e] * self.p[e, t] + self.x[e] * self.q[e, t])
                      - (self.r[e] ** 2 + self.x[e] ** 2) * self.l[e, t])
                eq_pen  = eq_pen + vd ** 2
                soc_pen = soc_pen + F.relu(
                    self.p[e, t] ** 2 + self.q[e, t] ** 2 - self.l[e, t] * self.v[i, t]
                ) ** 2
            for j in range(self.n_bus):
                pe = self.parent_edge[j]
                if pe is None:
                    continue
                out_p = (sum(self.p[e, t] for e in self.child_edges[j])
                         if self.child_edges[j] else torch.tensor(0.0, device=self.device))
                out_q = (sum(self.q[e, t] for e in self.child_edges[j])
                         if self.child_edges[j] else torch.tensor(0.0, device=self.device))
                eq_pen = eq_pen + (
                    out_p - (self.p[pe, t] - p_inj[j, t] - self.r[pe] * self.l[pe, t])
                ) ** 2
                eq_pen = eq_pen + (
                    out_q - (self.q[pe, t] - q_inj[j, t] - self.x[pe] * self.l[pe, t])
                ) ** 2
            slack_out_p = (sum(self.p[e, t] for e in self.child_edges[self.slack])
                           if self.child_edges[self.slack] else torch.tensor(0.0, device=self.device))
            slack_out_q = (sum(self.q[e, t] for e in self.child_edges[self.slack])
                           if self.child_edges[self.slack] else torch.tensor(0.0, device=self.device))
            eq_pen = eq_pen + (self.p0[t] - (slack_out_p + p_inj[self.slack, t])) ** 2
            eq_pen = eq_pen + (self.q0[t] - (slack_out_q + q_inj[self.slack, t])) ** 2
        bound_pen = (bound_pen
            + torch.sum(F.relu(-self.p0) ** 2 + F.relu(self.p0 - self.p0_max) ** 2)
            + torch.sum(F.relu(-self.q0) ** 2 + F.relu(self.q0 - self.q0_max) ** 2)
            + torch.sum(F.relu(self.v_min - self.v) ** 2
                        + F.relu(self.v - self.v_max) ** 2)
            + torch.sum(F.relu(-self.l) ** 2))
        loss = (obj
                + penalty_weights['eq']    * eq_pen
                + penalty_weights['ineq']  * ineq_pen
                + penalty_weights['soc']   * soc_pen
                + penalty_weights['bound'] * bound_pen)
        aux = {
            'obj':       obj.detach(),
            'eq_pen':    eq_pen.detach(),
            'ineq_pen':  ineq_pen.detach(),
            'soc_pen':   soc_pen.detach(),
            'bound_pen': bound_pen.detach(),
            'pi':        pi.detach(),
            'p0':        self.p0.detach(),
        }
        return loss, aux


class DistributionNetworkLeader:
    def __init__(self, constants: dict, device: str = 'cpu'):
        """
        Input  : constants — extract_constants() 的返回值
                device - 'cpu' | 'cuda'
        """
        self.constants = constants
        self.device    = device
        self.T         = constants["T"]
        self.n_bus     = constants["n_bus"]
        self.n_line    = constants["n_line"]
        # 拓扑结构（供 _solve_power_flow 使用）
        self.child_edges = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            self.child_edges[i].append(e)
            self.parent_edge[j] = e


    def evaluate_objective(self, x: np.ndarray, y : np.ndarray) -> float:
        """
        fix pi = x, p_mg = y, use Gurobi to solve the power flow problem.
        Input  : x — ndarray (T,)
                 y — ndarray (T,)
        Output : 目标值 - float
        """
        return self._solve_power_flow(x, y)

    
    def evaluate_surrogate_objective(self, x: np.ndarray, surrogate: SurrogateModel) -> float:
        """
        Input  : x         — ndarray (T,), pi
                 surrogate — 当前代理模型
        Output : float
        """
        return self._solve_power_flow(x, surrogate.predict(x))

    def _solve_power_flow(self, pi: np.ndarray, p_mg: np.ndarray) -> float:
        """
        固定 pi 和 p_mg, 用 Gurobi SOCP 求解潮流变量，返回目标值。
        Input  : pi   — ndarray (T,)
                 p_mg — ndarray (T,)
        Output : float
        """
        import gurobipy as gp
        from gurobipy import GRB
        c        = self.constants
        T        = c["T"]
        n_bus    = c["n_bus"]
        n_line   = c["n_line"]
        slack    = c["slack"]
        r        = c["r"]
        x        = c["x"]
        l_max    = c["l_max"]
        lambda_g = c["lambda_g"]
        p_passive = c["p_passive"]          # (n_bus, T)
        s_base   = c["s_base_mva"]
        mg_buses = c["mg_buses"]
        # 注入功率（固定）
        p_inj = p_passive.copy()
        for m in mg_buses:
            p_inj[m, :] += p_mg / s_base
        mdl = gp.Model()
        mdl.setParam("OutputFlag", 0)
        # 变量
        p0     = mdl.addVars(T,        lb=0,            ub=c["p0_max"],  name="p0")
        q0     = mdl.addVars(T,        lb=0,            ub=c["q0_max"],  name="q0")
        v      = mdl.addVars(n_bus, T, lb=c["v_min"],   ub=c["v_max"],   name="v")
        l      = mdl.addVars(n_line, T,lb=0,            ub=GRB.INFINITY, name="l")
        p      = mdl.addVars(n_line, T,lb=-GRB.INFINITY,ub=GRB.INFINITY, name="p")
        q      = mdl.addVars(n_line, T,lb=-GRB.INFINITY,ub=GRB.INFINITY, name="q")
        q_pass = mdl.addVars(n_bus, T, lb=-GRB.INFINITY,ub=GRB.INFINITY, name="qpass")
        # 目标
        mdl.setObjective(
            gp.quicksum(lambda_g[t] * p0[t] * s_base for t in range(T))
            - gp.quicksum(pi[t] * p_mg[t] for t in range(T)),
            GRB.MINIMIZE
        )
        for t in range(T):
            # slack 电压
            mdl.addConstr(v[slack, t] == c["v_ref"] ** 2)
            for e in range(n_line):
                i = c["from_bus"][e]
                j = c["to_bus"][e]
                # 电流上限
                mdl.addConstr(l[e, t] <= l_max[e])
                # 电压降等式
                mdl.addConstr(
                    v[j, t] == v[i, t]
                    - 2 * (r[e] * p[e, t] + x[e] * q[e, t])
                    + (r[e] ** 2 + x[e] ** 2) * l[e, t]
                )
                # SOC：p^2 + q^2 <= l * v_i
                mdl.addQConstr(
                    p[e, t] * p[e, t] + q[e, t] * q[e, t] <= l[e, t] * v[i, t]
                )
            # 节点平衡
            for j in range(n_bus):
                pe = self.parent_edge[j]
                if pe is None:
                    continue
                out_p = (gp.quicksum(p[e, t] for e in self.child_edges[j])
                         if self.child_edges[j] else 0.0)
                out_q = (gp.quicksum(q[e, t] for e in self.child_edges[j])
                         if self.child_edges[j] else 0.0)
                mdl.addConstr(out_p == p[pe, t] - p_inj[j, t] - r[pe] * l[pe, t])
                mdl.addConstr(out_q == q[pe, t] - q_pass[j, t] - x[pe] * l[pe, t])
            # slack 平衡
            slack_out_p = (gp.quicksum(p[e, t] for e in self.child_edges[slack])
                           if self.child_edges[slack] else 0.0)
            slack_out_q = (gp.quicksum(q[e, t] for e in self.child_edges[slack])
                           if self.child_edges[slack] else 0.0)
            mdl.addConstr(p0[t] == slack_out_p + p_inj[slack, t])
            mdl.addConstr(q0[t] == slack_out_q + q_pass[slack, t])
        mdl.optimize()
        if mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed with status {mdl.Status}")
        p0_val = np.array([p0[t].X for t in range(T)])
        return float(np.sum(lambda_g * p0_val * s_base) - np.sum(pi * p_mg))

    
    def solve(self, surrogate: SurrogateModel,
            trust_region:TrustRegion = None,
              x_center: np.ndarray = None,
              solve_params: dict = None) -> tuple:
        """
        完整优化模式,pi 也参与优化,p_mg 来自 surrogate。
        Input  : surrogate    — 当前代理(GlobalSurrogate,已含历史残差）
                 trust_region — 非 None 时加 L_inf 软罚项
                 x_center     — 信赖域中心,shape (T,)，用于热启动和信赖域罚项
                 solve_params — {'steps', 'lr', 'penalty_weights', 'tr_weight'}
        Output : (x_opt: ndarray (T,), F_tilde: float)
                  x_opt   — 最优 pi
                  F_tilde — 对应的代理目标值（用于 compute_ratio)
        """
        if solve_params is None:
            solve_params = {'steps': 5000, 'lr': 1e-2}
        _model = _LeaderSubproblem(self.constants, surrogate, self.device)
        uses_torch_surrogate = hasattr(surrogate, "predict_torch")
        # 热启动
        if x_center is not None:
            with torch.no_grad():
                _model.pi_raw.copy_(
                    torch.tensor(
                        self._inverse_sigmoid(x_center), dtype=torch.float32
                    )
                )
        opt       = torch.optim.Adam(_model.parameters(), lr=solve_params['lr'])
        pw        = solve_params.get('penalty_weights',
                                     {'eq': 1000., 'ineq': 100., 'soc': 100., 'bound': 10.})
        tr_weight = solve_params.get('tr_weight', 500.0)
        tr_lb = tr_ub = None
        if trust_region is not None and x_center is not None:
            x_c = torch.tensor(x_center, dtype=torch.float32, device=self.device)
            radius = trust_region.get_radius()
            tr_lb = torch.clamp(x_c - radius, min=_model.pi_min, max=_model.pi_max)
            tr_ub = torch.clamp(x_c + radius, min=_model.pi_min, max=_model.pi_max)
        for _ in range(solve_params['steps']):
            opt.zero_grad()
            loss, _ = _model.forward(pw)
            # L_inf 信赖域软罚项
            if tr_lb is not None and tr_ub is not None:
                pi_cur = _model.bounded_pi()
                tr_viol = torch.maximum(
                    torch.relu(tr_lb - pi_cur).max(),
                    torch.relu(pi_cur - tr_ub).max(),
                )
                loss = loss + tr_weight * tr_viol ** 2
            loss.backward()
            # Fallback for legacy numpy-only surrogate models.
            if (not uses_torch_surrogate) and getattr(surrogate, "_residuals", []):
                self._add_residual_gradient(_model, surrogate)
            opt.step()
            if tr_lb is not None and tr_ub is not None:
                with torch.no_grad():
                    pi_projected = torch.clamp(_model.bounded_pi(), min=tr_lb, max=tr_ub)
                    z = (pi_projected - _model.pi_min) / (_model.pi_max - _model.pi_min)
                    z = torch.clamp(z, 1e-6, 1 - 1e-6)
                    _model.pi_raw.copy_(torch.log(z / (1 - z)))
        with torch.no_grad():
            _, aux = _model.forward(pw)
        x_opt   = aux['pi'].cpu().numpy()
        F_tilde = float(aux['obj'])
        return x_opt, F_tilde

    def get_feasible_set(self) -> dict:
        """
        Output : {'lb': ndarray(T,), 'ub': ndarray(T,)}
        """
        return {
            'lb': np.full(self.T, self.constants['pi_min']),
            'ub': np.full(self.T, self.constants['pi_max']),
        }
    # ------------------------------------------------------------------
    def _inverse_sigmoid(self, pi: np.ndarray) -> np.ndarray:
        """
        Input  : pi — ndarray (T,)，值域 [pi_min, pi_max]
        Output :    — ndarray (T,)，对应的 pi_raw
        """
        pi_min = self.constants['pi_min']
        pi_max = self.constants['pi_max']
        z = np.clip((pi - pi_min) / (pi_max - pi_min), 1e-6, 1 - 1e-6)
        return np.log(z / (1 - z))
    def _add_residual_gradient(self, _model: _LeaderSubproblem,
                                surrogate: GlobalSurrogate) -> None:
        """
        解析累加所有历史残差对 pi_raw 的梯度，直接修改 pi_raw.grad。
        Input  : _model    — 当前 _LeaderSubproblem 实例
                 surrogate — GlobalSurrogate,含 _residuals 列表
        Output : —
        """
        with torch.no_grad():
            pi    = _model.bounded_pi().cpu().numpy()   # (T,)
            dU_dy = -pi                                 # ∂U/∂p_mg = -pi，shape (T,)
            total = np.zeros_like(pi)
            for res in surrogate._residuals:
                dDelta_dpi = res.gradient(pi)           # (d_y, T)
                total += dU_dy @ dDelta_dpi             # (T,)
            pi_min   = self.constants['pi_min']
            pi_max   = self.constants['pi_max']
            sig      = (pi - pi_min) / (pi_max - pi_min)
            dpi_draw = (pi_max - pi_min) * sig * (1 - sig)
            correction = torch.tensor(
                total * dpi_draw, dtype=torch.float32, device=self.device
            )
            if _model.pi_raw.grad is not None:
                _model.pi_raw.grad += correction
            else:
                _model.pi_raw.grad = correction.clone()
# endregion


# region Part IV: Trust Region Solver class Setup
class TRResidualController:
    def __init__(self,
                 leader: DistributionNetworkLeader,
                 follower: LowerSolver,
                 global_surrogate: GlobalSurrogate,
                 residual_model: ResidualModel,
                 tr_params: dict,
                 algo_params: dict,
                 subspace: ActiveSubspace):
        """
        Input  : leader          — DistributionNetworkLeader 实例
                 follower        — FollowerBase 实例
                 global_surrogate— GlobalSurrogate 实例（已训练好的神经网络）
                 residual_model  — ResidualModel 实例
                 tr_params       — 信赖域参数
                                   {'R_0', 'R_max', 'gamma_dec', 'gamma_inc', 'eta_1', 'eta_2'}
                 algo_params     — 算法参数
                                   {'n_samples', 'K_max', 'x0',
                                    'eps_x', 'eps_f', 'eps_R',
                                    'sampling_method', 'solve_params'}
        """
        self.leader           = leader
        self.follower         = follower
        self.global_surrogate = global_surrogate
        self.residual_model   = residual_model
        self.residual_model_type = residual_model.model_type
        self.residual_model_kwargs = {
            'model_type': residual_model.model_type,
            'lasso_alpha': residual_model.lasso_alpha,
            'lasso_max_iter': residual_model.lasso_max_iter,
            'lasso_tol': residual_model.lasso_tol,
            'rbf_kernel': residual_model.rbf_kernel,
            'rbf_smoothing': residual_model.rbf_smoothing,
            'rbf_degree': residual_model.rbf_degree,
            'rbf_scale': residual_model.rbf_scale,
            'rbf_epsilon': residual_model.rbf_epsilon,
        }
        self.tr = TrustRegion(
            R_0       = tr_params['R_0'],
            R_max     = tr_params['R_max'],
            gamma_dec = tr_params['gamma_dec'],
            gamma_inc = tr_params['gamma_inc'],
        )
        self.eta_1 = tr_params['eta_1']
        self.eta_2 = tr_params['eta_2']
        self.n_samples       = algo_params['n_samples']
        self.K_max           = algo_params['K_max']
        self.x0              = algo_params['x0']
        self.eps_x           = algo_params['eps_x']
        self.eps_f           = algo_params['eps_f']
        self.eps_R           = algo_params['eps_R']
        self.sampling_method = algo_params.get('sampling_method', 'lhs')
        self.solve_params    = algo_params.get('solve_params', None)
        self.random_seed     = algo_params.get('random_seed', None)
        self.subspace        = subspace
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        执行完整的 TR 迭代。
        Output : dict
                 'x_opt'     : ndarray (T,)，最终 leader 决策
                 'surrogate' : GlobalSurrogate,最终代理模型
                 'history'   : list[dict]，每轮迭代记录
                 'converged' : bool
                 'n_iter'    : int
        """
        # 用全局代理求解初始点
        print("Using global surrogate to solve initial point...")
        time_start = time.time()
        # x_k, _ = self.leader.solve(
        #     surrogate=self.global_surrogate,
        #     solve_params= {**self.solve_params, 'steps': self.solve_params['steps_init']}
        # )
        x_k = np.asarray(self.x0.copy(), dtype=np.float32)
        time_end = time.time()
        print(f"Time taken to solve initial point {time_end - time_start:.4f} seconds.")
        history   = []
        converged = False
        for k in range(self.K_max):
            time_start = time.time()
            # if k % 10 == 0:
            #     self.subspace.identify(x_k, self.tr.get_radius(), self.global_surrogate, n_probe=80)
            #     print(f"Active subspace explained ratio: {self.subspace.explained_ratio:.4f}")
            if self.subspace is not None:
                self.subspace.identify(x_k, self.tr.get_radius(), self.global_surrogate, n_probe=80)
                print(f"Active subspace explained ratio: {self.subspace.explained_ratio:.4f}")
            print(f"Iteration {k} | R={self.tr.get_radius():.4f} | x_{k}={x_k}")
            logger.info(f"Iteration {k} | R={self.tr.get_radius():.4f} | x_{k}={x_k}")
            record, x_next, converged = self._one_iteration(k, x_k)
            time_end = time.time()
            print(f"Time taken to solve iteration {k} {time_end - time_start:.4f} seconds.")
            history.append(record)
            x_k = x_next
            if converged:
                logger.info(f"Converged at iteration {k}.")
                print(f"Converged at iteration {k}.")
                break
        return {
            'x_opt'    : x_k,
            'surrogate': self.global_surrogate,
            'history'  : history,
            'converged': converged,
            'n_iter'   : len(history),
        }

    def _one_iteration(self, k: int, x_k: np.ndarray, is_first: bool = False) -> tuple:
        """
        单次迭代。
        Input  : k   — 当前迭代索引
                 x_k — 当前 leader 决策,ndarray (T,)
        Output : (record: dict, x_next: ndarray (T,), converged: bool)
        """
        steps = (self.solve_params['steps_init'] if is_first
             else self.solve_params['steps_warm'])
        cur_solve_params = {**self.solve_params, 'steps': steps}
        feasible_set = self.leader.get_feasible_set()
        # Step 1: 在信赖域内采样
        print(f"Sampling in trust region with {self.n_samples} samples...")
        X_samples = sample_in_trust_region(
            x_center     = x_k,
            tr           = self.tr,
            n            = self.n_samples,
            feasible_set = feasible_set,
            method       = self.sampling_method,
            active_subspace = self.subspace,
            n_supplement = 10,
            seed         = None if self.random_seed is None else self.random_seed + k,
            include_star_points = self.residual_model_type == "rbf",
        )                                                       # (n, T)
        # Step 2: 查询 follower 真实响应，计算残差
        X_fit, _ = _unique_rows(np.vstack([x_k.reshape(1, -1), X_samples]), tol=1e-8)
        print(f"Querying follower with {len(X_fit)} residual-fitting samples...")
        lower_results = [self.follower.lower_solve(xi) for xi in X_fit]
        if any(res is None for res in lower_results):
            raise RuntimeError("Lower-level solve failed for at least one trust-region sample.")
        Y_true = np.array([res["p_mg"] for res in lower_results])   # (n, T)
        Y_hat  = self.global_surrogate.predict(X_fit)                  # (n, T)
        Delta  = Y_true - Y_hat                                             # (n, T)
        # Step 3: 拟合局部残差模型
        print(f"Fitting residual model with {len(X_fit)} samples...")
        residual_model = ResidualModel(**self.residual_model_kwargs)
        residual_model.fit(X_fit, Delta, x_center=x_k, radius=self.tr.get_radius())
        if residual_model.model_type == "rbf":
            print(f"RBF interpolation condition number: {residual_model.rbf_condition:.4e}")
        # Step 4: 求解信赖域子问题，得到候选点
        # 注意：此时 global_surrogate 还未 merge，surrogate 是旧的
        # 先临时构造一个 CorrectedSurrogate 用于本轮求解
        print(f"Constructing temporary surrogate and solving...")
        temp_surrogate = _TempCorrectedSurrogate(
            self.global_surrogate, residual_model
        )
        x_cand, F_tilde_candidate = self.leader.solve(
            surrogate    = temp_surrogate,
            trust_region = self.tr,
            x_center     = x_k,
            solve_params = cur_solve_params,
        )                                                       # x_cand: (T,)
        # Step 5: 计算代理目标值（不查询 follower）
        print(f"Evaluating surrogate and actual objective...")
        F_tilde_current = self.leader.evaluate_surrogate_objective(x_k, temp_surrogate)
        F_tilde_candidate = self.leader.evaluate_surrogate_objective(x_cand, temp_surrogate)
        # Step 6: 查询真实目标值（需要查询 follower）
        lower_k    = self.follower.lower_solve(x_k)
        lower_cand = self.follower.lower_solve(x_cand)
        if lower_k is None or lower_cand is None:
            raise RuntimeError("Lower-level solve failed while evaluating the true objective.")
        y_k    = lower_k["p_mg"]                      # (T,)
        y_cand = lower_cand["p_mg"]                   # (T,)
        F_current   = self.leader.evaluate_objective(x_k,    y_k)
        F_candidate = self.leader.evaluate_objective(x_cand, y_cand)
        # Step 7: 计算 ρ，决定接受/拒绝
        print(f"Computing ratio and deciding step...")
        rho      = compute_ratio(F_current, F_candidate, F_tilde_current, F_tilde_candidate)
        decision = decide_step(rho, self.eta_1, self.eta_2)
        n_residuals_after = (
            len(self.global_surrogate._residuals)
            if decision == 'reject'
            else len(self.global_surrogate._residuals) + 1
        )
        print(f"[Iter {k:3d}] "
              f"F_current={F_current:.4f} | "
              f"F_candidate={F_candidate:.4f} | "
              f"F_tilde_current={F_tilde_current:.4f} | "
              f"F_tilde_candidate={F_tilde_candidate:.4f} | "
              f"Ared={F_current - F_candidate:.4f} | "
              f"Pred={F_tilde_current - F_tilde_candidate:.4f} | "
              f"rho={rho:.4f} | "
              f"decision={decision} | "
              f"R={self.tr.get_radius():.4f} | "
              f"n_residuals={n_residuals_after}")
        logger.info(f"[Iter {k:3d}] "
              f"F_current={F_current:.4f} | "
              f"F_candidate={F_candidate:.4f} | "
              f"F_tilde_current={F_tilde_current:.4f} | "
              f"F_tilde_candidate={F_tilde_candidate:.4f} | "
              f"Ared={F_current - F_candidate:.4f} | "
              f"Pred={F_tilde_current - F_tilde_candidate:.4f} | "
              f"rho={rho:.4f} | "
              f"decision={decision} | "
              f"R={self.tr.get_radius():.4f} | "
              f"n_residuals={n_residuals_after}")
        # Step 8: 更新状态
        if decision == 'reject':
            print(f"Rejecting step {k}...")
            x_next = x_k
            self.tr.shrink()
            # 残差模型丢弃，不合并进全局代理
        elif decision == 'accept_keep':
            print(f"Accepting step {k} and keeping...")
            x_next = x_cand
            self.global_surrogate.merge(residual_model)   # 合并本轮独立残差
            self.tr.keep()
        else:  # accept_expand
            print(f"Accepting step {k} and expanding...")
            x_next = x_cand
            self.global_surrogate.merge(residual_model)   # 合并本轮独立残差
            self.tr.expand()
        # Step 9: 检查终止条件
        # Reject 时 x_next == x_k 是算法动作，不代表 leader 决策已经收敛。
        if decision == 'reject':
            converged = self.tr.get_radius() <= self.eps_R
        else:
            converged = (
                self.tr.get_radius()             <= self.eps_R or
                np.linalg.norm(x_next - x_k)    <= self.eps_x or
                abs(F_current - F_candidate)     <= self.eps_f
            )
        record = {
            'k'              : k,
            'x_k'           : x_k,
            'x_cand'        : x_cand,
            'x_next'        : x_next,
            'y_k'           : y_k,
            'y_cand'        : y_cand,
            'F_current'     : F_current,
            'F_candidate'   : F_candidate,
            'F_tilde_current'  : F_tilde_current,
            'F_tilde_candidate': F_tilde_candidate,
            'rho'           : rho,
            'decision'      : decision,
            'R'             : self.tr.get_radius(),
            'n_residuals'   : len(self.global_surrogate._residuals),
        }
        return record, x_next, converged


class _TempCorrectedSurrogate(SurrogateModel):
    """
    在当前轮 solve 时临时组合 global_surrogate + residual_model。
    不持久化，不影响 global_surrogate 的状态。
    """
    def __init__(self, base: GlobalSurrogate, residual: ResidualModel):
        """
        Input  : base     — 当前 GlobalSurrogate(未 merge 当前轮残差）
                 residual — 当前轮已拟合的 ResidualModel
        """
        self.base     = base
        self.residual = residual
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        ỹ(x) = base.predict(x) + residual.predict(x)
        Input  : x     — ndarray (T,) 或 (n, T)
        Output : y_hat — ndarray (T,) 或 (n, T)
        """
        single = (x.ndim == 1)
        base_pred = self.base.predict(x)
        if single:
            res_pred = self.residual.predict(x)
        else:
            res_pred = np.array([self.residual.predict(xi) for xi in x])
        return base_pred + res_pred

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable corrected prediction for the leader subproblem.
        """
        return self.base.predict_torch(x) + self.residual.predict_torch(x)

    @property
    def _residuals(self):
        """
        透传 base 的历史残差列表，再加上当前轮残差。
        供 _add_residual_gradient 遍历使用。
        """
        return self.base._residuals + [self.residual]
    def update(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass
# endregion


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def save_trust_region_result(result: dict, output_dir: str = "results") -> dict:
    """
    Save the final result and per-iteration history.
    JSON keeps full arrays; CSV keeps scalar iteration metrics for quick plotting.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(output_dir, f"trust_region_result_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"trust_region_history_{timestamp}.csv")

    result_to_save = {
        "x_opt": result["x_opt"],
        "params": result.get("params", {}),
        "history": result["history"],
        "converged": result["converged"],
        "n_iter": result["n_iter"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result_to_save), f, indent=2)

    summary_keys = [
        "k", "F_current", "F_candidate", "F_tilde_current",
        "F_tilde_candidate", "rho", "decision", "R", "n_residuals",
    ]
    summary_rows = [
        {key: record.get(key) for key in summary_keys}
        for record in result["history"]
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


# Region Part V: Main Function
if __name__ == "__main__":
    T = 24
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    hot_start_point = np.array([0.979727, 0.8551583, 0.9011953, 1.154334, 0.53833634, 0.8070916,
                 0.83882296, 0.8913337, 0.92085683, 1.0081077, 1.015762, 1.016812,
                 0.98006505, 0.9240768, 0.99690413, 0.9587043, 0.9882177, 0.9242619,
                 1.0174826, 1.0582225, 0.9876231, 1.0304792, 1.011621, 0.84445477])
    cold_start_point = np.full(T, 0.75)
    # 配电网参数
    net       = pn.case33bw()
    lower_net = pn.create_cigre_network_lv()
    mg_buses = [18]
    constants = extract_constants(input={"mg_buses": mg_buses}, net=net, T=T)
    lower_constants = extract_lower_constants(input={}, net=lower_net, T=T)
    # 初始化各模块
    map_net          = load_l2o_map_net(model_type='SB', device='cpu')                                      # 你已训练好的神经网络
    global_surrogate = GlobalSurrogate(map_net=map_net, device='cpu')
    residual_model_params = {
        'model_type': 'rbf',
        'lasso_alpha': 1e-4,
        'lasso_max_iter': 5000,
        'lasso_tol': 1e-8,
        'rbf_kernel': 'cubic',  # 'cubic', 'linear', 'gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline'
        'rbf_smoothing': 1e-6,
        'rbf_degree': 1,
        'rbf_scale': 'trust_radius',
        'rbf_epsilon': None,
    }
    residual_model   = ResidualModel(**residual_model_params)
    follower         = LowerSolver(lower_constants=lower_constants)
    leader           = DistributionNetworkLeader(constants=constants, device='cpu')
    tr_params = {
        'R_0'      : 0.2,
        'R_max'    : 1.0,
        'gamma_dec': 0.7,
        'gamma_inc': 1.5,
        'eta_1'    : 0.1,
        'eta_2'    : 0.9,
    }
    algo_params = {
        'n_samples'      : 60,
        'K_max'          : 500,
        'x0'             : cold_start_point,
        'eps_x'          : 1e-4,
        'eps_f'          : 1e-4,
        'eps_R'          : 1e-4,
        'sampling_method': 'lhs',
        'random_seed'    : random_seed,
        'solve_params'   : {'steps_init': 300,
                            'steps_warm': 300,
                            'lr': 1e-2,
                            'penalty_weights': {'eq': 1000, 'ineq': 100., 'soc': 100., 'bound': 10.},
                            'tr_weight'  : 1e6, 
                            'verbose': True,
                            },
    }
    controller = TRResidualController(
        leader           = leader,
        follower         = follower,
        global_surrogate = global_surrogate,
        residual_model   = residual_model,
        tr_params        = tr_params,
        algo_params      = algo_params,
        subspace         = None,
    )
    time_start = time.time()
    result = controller.run()
    time_end = time.time()
    result["params"] = {
        "tr_params": tr_params,
        "algo_params": algo_params,
        "residual_model": {
            "model_type": residual_model.model_type,
            "lasso_alpha": residual_model.lasso_alpha,
            "lasso_max_iter": residual_model.lasso_max_iter,
            "lasso_tol": residual_model.lasso_tol,
            "rbf_kernel": residual_model.rbf_kernel,
            "rbf_smoothing": residual_model.rbf_smoothing,
            "rbf_degree": residual_model.rbf_degree,
            "rbf_scale": residual_model.rbf_scale,
            "rbf_epsilon": residual_model.rbf_epsilon,
        },
    }
    saved_paths = save_trust_region_result(result)
    print(f"Time taken to run {time_end - time_start:.4f} seconds.")
    print(f"Converged : {result['converged']}")
    print(f"Iterations: {result['n_iter']}")
    print(f"Optimal pi: {result['x_opt']}")
    final_objective = result['history'][-1]['F_current'] if result['history'] else None
    best_objective = final_objective
    for record in result['history']:
        if record['decision'].startswith('accept'):
            best_objective = min(best_objective, record['F_candidate'])
    print(f"Final objective: {final_objective}")
    print(f"Best accepted objective: {best_objective}")
    print(f"Saved full result to: {saved_paths['json']}")
    print(f"Saved iteration summary to: {saved_paths['csv']}")