"""Trust-region residual-correction solver implementation."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats.qmc import LatinHypercube, Sobol, scale

from follower_solver import build_solver as build_scenario_follower

from .distflow import CoptDistFlowEvaluator, torch_penalty_distflow
from .follower import CoptFollowerSolver, extract_follower_constants
from .followers import build_composite_follower
from .l2o import load_l2o_model, load_l2o_models
from .network import (
    build_passive_profiles,
    create_case33bw_20kv,
    extract_upper_constants,
)
from .instances import resolve_follower_configuration
from .pricing import sample_electricity_price
from .profiles import daily_load_curve
from .randomness import DEFAULT_SEED, set_global_seed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUE_RESPONSE_DIR = PROJECT_ROOT / "data"
TRUE_RESPONSE_TAG = "tr"

#region Part 0: Constants and Lower Solver Setup
def electricity_price(T: int) -> np.ndarray:
    return sample_electricity_price(T, require_24_hours=True)


def _default_load_curve(T: int) -> np.ndarray:
    return daily_load_curve(T)


def extract_lower_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    scenario = str(input.get("scenario", "SB")).upper()
    if scenario != "SB":
        raise NotImplementedError(
            f"The {scenario} follower optimization problem is not implemented."
        )
    constants = extract_follower_constants(
        net, T, scenario=scenario, verbose=bool(input.get("verbose", False))
    )
    constants.pop("random_seed")
    return constants


class LowerSolver(CoptFollowerSolver):
    def __init__(self, lower_constants: Dict[str, Any]):
        super().__init__(lower_constants, failure_mode="none", scenario="lower_level")


def _build_passive_profiles(net: pp.pandapowerNet, T: int, load_curve: np.ndarray, s_base_mva: float) -> Dict[str, np.ndarray]:
    return build_passive_profiles(net, T, load_curve, s_base_mva)


def extract_constants(input: Dict[str, Any], net: pp.pandapowerNet, T: int) -> Dict[str, Any]:
    return extract_upper_constants(
        input, net, T, squared_voltage_limits=True, price_sampler=electricity_price
    )


def load_l2o_map_net(model_type: str = "SB", device: Optional[torch.device] = None) -> nn.Module:
    return load_l2o_model(model_type, device)
#endregion Part 0: Constants and Lower Solver Setup

#region Part I: Global Surrogate Model Loading
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

    def predict_base(self, x: np.ndarray) -> np.ndarray:
        """Predict with the original neural surrogate, without corrections."""
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.map_net(x_t).cpu().numpy()
        return y[0] if single else y

    def predict_base_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable prediction from the original neural surrogate."""
        single = (x.ndim == 1)
        x_batch = x.unsqueeze(0) if single else x
        y = self.map_net(x_batch.to(self.device))
        return y[0] if single else y
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        single = (x.ndim == 1)
        x_batch = x[np.newaxis, :] if single else x
        y = self.predict_base(x_batch)
        for res in self._residuals:
            y += np.array([res.predict(xi) for xi in x_batch])
        return y[0] if single else y

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable surrogate prediction used by leader.solve().
        Keeps the map_net path in the torch graph so pi receives the NN gradient.
        """
        single = (x.ndim == 1)
        x_batch = x.unsqueeze(0) if single else x
        y = self.predict_base_torch(x_batch)
        for res in self._residuals:
            if single:
                y = y + res.predict_torch(x).unsqueeze(0)
            else:
                y = y + torch.stack([res.predict_torch(xi) for xi in x], dim=0)
        return y[0] if single else y

    def merge(self, residual: "ResidualModel") -> None:
        self._residuals.append(residual)

    def replace(self, residual: "ResidualModel") -> None:
        """Keep only the newest local residual correction."""
        self._residuals = [residual]
    
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


class CompositeGlobalSurrogate(SurrogateModel):
    """One independent differentiable surrogate per follower."""

    def __init__(self, surrogates: Sequence[GlobalSurrogate], T: int):
        self.surrogates = list(surrogates)
        self.T = int(T)
        if not self.surrogates:
            raise ValueError("At least one surrogate is required.")
        self.n_followers = len(self.surrogates)
        self.decision_size = self.n_followers * self.T

    @property
    def n_corrections(self) -> int:
        return min(len(surrogate._residuals) for surrogate in self.surrogates)

    @property
    def _residuals(self) -> list[tuple["ResidualModel", ...]]:
        return list(zip(*(surrogate._residuals for surrogate in self.surrogates)))

    def _reshape_numpy(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(x, dtype=float)
        if values.shape == (self.decision_size,):
            return values.reshape(1, self.n_followers, self.T), True
        if values.shape == (self.n_followers, self.T):
            return values.reshape(1, self.n_followers, self.T), True
        if values.ndim == 2 and values.shape[1] == self.decision_size:
            return values.reshape(-1, self.n_followers, self.T), False
        raise ValueError(
            f"Expected price shape ({self.decision_size},), "
            f"({self.n_followers}, {self.T}), or (n, {self.decision_size}); "
            f"got {values.shape}."
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        prices, single = self._reshape_numpy(x)
        outputs = [
            self.surrogates[index].predict(prices[:, index, :])
            for index in range(self.n_followers)
        ]
        response = np.stack(outputs, axis=1)
        flattened = response.reshape(response.shape[0], self.decision_size)
        return flattened[0] if single else flattened

    def predict_base(self, x: np.ndarray) -> np.ndarray:
        """Predict all followers without any persisted local corrections."""
        prices, single = self._reshape_numpy(x)
        outputs = [
            self.surrogates[index].predict_base(prices[:, index, :])
            for index in range(self.n_followers)
        ]
        response = np.stack(outputs, axis=1)
        flattened = response.reshape(response.shape[0], self.decision_size)
        return flattened[0] if single else flattened

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(x), dtype=float).reshape(self.n_followers, self.T)

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        single = x.ndim in {1, 2} and (
            x.ndim == 1 or tuple(x.shape) == (self.n_followers, self.T)
        )
        if x.ndim == 1:
            prices = x.reshape(1, self.n_followers, self.T)
        elif tuple(x.shape) == (self.n_followers, self.T):
            prices = x.unsqueeze(0)
        elif x.ndim == 2 and x.shape[1] == self.decision_size:
            prices = x.reshape(-1, self.n_followers, self.T)
            single = False
        else:
            raise ValueError(f"Unexpected composite price tensor shape {tuple(x.shape)}.")
        outputs = [
            self.surrogates[index].predict_torch(prices[:, index, :])
            for index in range(self.n_followers)
        ]
        response = torch.stack(outputs, dim=1)
        flattened = response.reshape(response.shape[0], self.decision_size)
        return flattened[0] if single else flattened

    def predict_base_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable multi-follower prediction without corrections."""
        single = x.ndim in {1, 2} and (
            x.ndim == 1 or tuple(x.shape) == (self.n_followers, self.T)
        )
        if x.ndim == 1:
            prices = x.reshape(1, self.n_followers, self.T)
        elif tuple(x.shape) == (self.n_followers, self.T):
            prices = x.unsqueeze(0)
        elif x.ndim == 2 and x.shape[1] == self.decision_size:
            prices = x.reshape(-1, self.n_followers, self.T)
            single = False
        else:
            raise ValueError(f"Unexpected composite price tensor shape {tuple(x.shape)}.")
        outputs = [
            self.surrogates[index].predict_base_torch(prices[:, index, :])
            for index in range(self.n_followers)
        ]
        response = torch.stack(outputs, dim=1)
        flattened = response.reshape(response.shape[0], self.decision_size)
        return flattened[0] if single else flattened

    def predict_torch_matrix(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_torch(x).reshape(self.n_followers, self.T)

    def merge(self, residuals: Sequence["ResidualModel"]) -> None:
        residual_list = list(residuals)
        if len(residual_list) != self.n_followers:
            raise ValueError(
                f"Expected {self.n_followers} residual models, got {len(residual_list)}."
            )
        for surrogate, residual in zip(self.surrogates, residual_list):
            surrogate.merge(residual)

    def replace(self, residuals: Sequence["ResidualModel"]) -> None:
        residual_list = list(residuals)
        if len(residual_list) != self.n_followers:
            raise ValueError(
                f"Expected {self.n_followers} residual models, got {len(residual_list)}."
            )
        for surrogate, residual in zip(self.surrogates, residual_list):
            surrogate.replace(residual)

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


def _normalize_scenario_name(scenario: str) -> str:
    name = str(scenario).upper()
    return "IP" if name == "IE" else name
#endregion Part I: Global Surrogate Model Loading

#region Part II: Residual Model Setup and Fitting
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
        # Construct local quadratic features: (x_i-xc_i) * (x_j-xc_j), i <= j
        quad_feats = []
        quad_index = []                                                        # Record the position for each (i,j)
        for i in range(d_x):
            for j in range(i, d_x):
                quad_feats.append(X_local[:, i] * X_local[:, j])             # (n,)
                quad_index.append((i, j))
        quad_feats = np.stack(quad_feats, axis=1)                             # (n, n_quad)
        # Augmented feature matrix: [quadratic terms, linear terms, constant term]
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
        # Recover symmetric Q, A, and b from W
        # Average Q across all output dimensions as the shared quadratic term
        w_quad = W[:n_quad, :]                                                # (n_quad, d_y)
        w_quad_mean = w_quad.mean(axis=1)                                     # (n_quad,), averaged across outputs
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
        (
            self.x_center,
            self.rbf_input_scale,
            centers,
            keep,
            P,
            lhs,
        ) = self._build_rbf_system(X, x_center=x_center, radius=radius)
        y_2d = y_2d[keep]
        n = centers.shape[0]
        if n == 0:
            raise ValueError("RBF residual model needs at least one sample.")

        n_poly = P.shape[1]
        rhs = np.vstack([y_2d, np.zeros((n_poly, y_2d.shape[1]))])
        self.rbf_condition = float(np.linalg.cond(lhs))
        try:
            coeff = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coeff, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        self.rbf_centers = centers
        self.rbf_lambda = coeff[:n]
        self.rbf_beta = coeff[n:]

    def _build_rbf_system(
        self,
        X: np.ndarray,
        x_center: np.ndarray = None,
        radius: float = None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the exact normalized augmented system used by RBF fitting."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("RBF residual model needs a non-empty 2D design.")
        center = (
            np.asarray(x_center, dtype=float).copy()
            if x_center is not None
            else X.mean(axis=0)
        )
        input_scale = self._resolve_rbf_scale(
            X, radius=radius, x_center=center
        )
        centers = (X - center) / input_scale
        centers, keep = _unique_rows(centers, tol=1e-10)
        pairwise = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(pairwise, axis=2)
        K = self._rbf_phi_np(distances)
        smoothing = max(float(self.rbf_smoothing), 0.0)
        if smoothing > 0.0:
            K = K + smoothing * np.eye(len(centers))
        P = self._rbf_poly_np(centers)
        lhs = np.block([
            [K, P],
            [P.T, np.zeros((P.shape[1], P.shape[1]))],
        ])
        return center, input_scale, centers, keep, P, lhs

    def rbf_geometry_metrics(
        self,
        X: np.ndarray,
        x_center: np.ndarray = None,
        radius: float = None,
    ) -> dict:
        """Return rank and conditioning for the exact prospective RBF system."""
        _, _, centers, _, P, lhs = self._build_rbf_system(
            X, x_center=x_center, radius=radius
        )
        try:
            condition = float(np.linalg.cond(lhs))
        except np.linalg.LinAlgError:
            condition = float("inf")
        return {
            "sample_count": int(len(centers)),
            "polynomial_rank": (
                int(np.linalg.matrix_rank(P)) if P.shape[1] else 0
            ),
            "polynomial_columns": int(P.shape[1]),
            "condition_number": condition,
        }

    def _resolve_rbf_scale(
        self,
        X: np.ndarray,
        radius: float = None,
        x_center: np.ndarray = None,
    ) -> float:
        if self.rbf_scale == "trust_radius" and radius is not None:
            return max(float(radius), 1e-8)
        center = self.x_center if x_center is None else x_center
        spread = np.max(np.linalg.norm(X - center, axis=1))
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
            values = np.zeros_like(r)
            positive = r > 0.0
            values[positive] = r[positive] ** 2 * np.log(r[positive])
            return values
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
        Input  : R_0       — initial radius
                 R_max     — maximum radius
                 gamma_dec — contraction factor in (0,1)
                 gamma_inc — expansion factor > 1
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

# Sampling in Trust Region
def sample_in_trust_region(x_center: np.ndarray,
                            tr: TrustRegion,
                            n: int,
                            feasible_set: dict,
                            method: str = 'lhs',
                            seed: int = None,
                            include_star_points: bool = False,
                            max_star_points: int = None,
                            star_alpha: float = 1.0,
                            min_dist: float = 1e-8) -> np.ndarray:
    """
    Sample n points within the intersection of the trust region and feasible set.
    Input  : x_center    — ndarray (d_x,)
             radius      — trust-region radius
             n           — number of samples
             feasible_set— {'lb': ndarray(d_x,), 'ub': ndarray(d_x,)}
             method      — 'random' | 'lhs' | 'sobol'
    Output : samples     — ndarray (n, d_x)
    """
    radius = tr.get_radius()
    d_x = len(x_center)
    lb = np.maximum(feasible_set['lb'], x_center - radius)
    ub = np.minimum(feasible_set['ub'], x_center + radius)
    rng = np.random.default_rng(seed)
    def _sample_box(n_points: int) -> np.ndarray:
        """
        Sample n points from the box defined by the lower and upper bounds.
        Input  : n_points — number of points to sample
        Output : samples  — ndarray (n_points, d_x)
        """
        if n_points <= 0:
            return np.empty((0, d_x))
        if method == 'random':
            return rng.uniform(lb, ub, size=(n_points, len(lb)))
        elif method == 'lhs':
            sampler = LatinHypercube(d=len(lb), seed=seed)
            samples = sampler.random(n_points)                     # (n, len(lb)), range [0,1]
            return scale(samples, lb, ub)   # Map to [lb, ub]
        elif method == 'sobol':
            sampler = Sobol(d=len(lb), scramble=True, seed=seed)
            samples = sampler.random(n_points)                     # (n, len(lb)), range [0,1]
            return scale(samples, lb, ub)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _star_points(directions: np.ndarray, max_points: int) -> np.ndarray:
        """
        Sample star points from the directions.
        Used for the RBF ResidualModel.
        Input  : directions — ndarray (d_x, d_x)
                 max_points — maximum number of points to sample
        Output : star_points — ndarray (max_points, d_x)
        """
        if (not include_star_points) or max_points <= 0:
            return np.empty((0, d_x))
        points = []
        n_directions = min(len(directions), (max_points + 1) // 2)
        direction_indices = np.linspace(
            0, len(directions) - 1, n_directions, dtype=int
        )
        for direction in directions[direction_indices]:
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
        """
        Finalize the points by sampling from the box and ensuring uniqueness.
        Input  : points    — ndarray (n_points, d_x)
                 target_n  — number of points to sample
        Output : unique    — ndarray (target_n, d_x)
        """
        if points.size == 0:
            points = np.empty((0, d_x))
        unique, _ = _unique_rows(points, tol=min_dist)
        refill_tries = 0
        while unique.shape[0] < target_n and refill_tries < 5:
            refill = _sample_box(target_n - unique.shape[0])
            unique, _ = _unique_rows(np.vstack([unique, refill]), tol=min_dist)
            refill_tries += 1
        return unique[:target_n]

    star_limit = min(2 * d_x, n)
    if max_star_points is not None:
        star_limit = min(star_limit, max(int(max_star_points), 0))
    star = _star_points(np.eye(d_x), max_points=star_limit)
    fill = _sample_box(n - len(star))
    return _finalize(np.vstack([star, fill]), n)


def compute_ratio(F_current: float, F_candidate: float,
                  F_tilde_current: float, F_tilde_candidate: float) -> float:
    """
    Compute the ratio of the actual improvement to the predicted improvement.
    rho_k = Ared_k / Pred_k
    Input  : F_current         — true objective value at x^k
             F_candidate       — true objective value at x^{k,cand}
             F_tilde_current   — surrogate objective value at x^k
             F_tilde_candidate — surrogate objective value at x^{k,cand}
    Output : rho — float, returns -inf when Pred_k <= 0
    """
    Pred = F_tilde_current - F_tilde_candidate
    Ared = F_current - F_candidate
    if Pred <= 0:
        # This means that the candidate is not better than the current point. Then, the candidate should be rejected directly.
        return -np.inf
    return Ared / Pred


def decide_step(rho: float, eta_1: float, eta_2: float) -> str:
    """
    Input  : rho   — return value from compute_ratio
             eta_1 — lower threshold
             eta_2 — upper threshold
    Output : 'reject' | 'accept_keep' | 'accept_expand'
    """
    if rho < eta_1:
        # This means that the candidate is not better than the current point. 
        return 'reject'
    elif rho < eta_2:
        # This means that the candidate is better than the current point.
        return 'accept_keep'
    else:
        # This means that the candidate is significantly better than the current point.
        return 'accept_expand'

#endregion


#region Part IV: Leader problem class Setup
class _LeaderSubproblem(nn.Module):
    """
    Used only within DistributionNetworkLeader.
    fixed_pi=None, fixed_p_mg=None → full optimization mode (used by solve)
    fixed_pi=x,    fixed_p_mg=y    → fixed mode (used by evaluate_objective with COPT)
    """
    def __init__(self, constants: dict, surrogate: SurrogateModel, device: str = 'cpu',
                 fixed_pi: np.ndarray = None, fixed_p_mg: np.ndarray = None):
        """
        Input  : constants  — return value from extract_constants()
                 surrogate  — current surrogate model (may be None in fixed mode)
                 device     — 'cpu' or 'cuda'
                 fixed_pi   — ndarray (T,), provided in fixed mode
                 fixed_p_mg — ndarray (T,), provided in fixed mode
        """
        super().__init__()
        self.device    = device
        self.surrogate = surrogate
        self.fixed_pi  = fixed_pi
        self.fixed_p_mg = fixed_p_mg
        # Constants
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
        self.p_net_min = constants.get("p_net_min", -15.0 / self.s_base_mva)
        self.p_net_max = constants.get("p_net_max", 15.0 / self.s_base_mva)
        self.q_net_min = constants.get("q_net_min", -15.0 / self.s_base_mva)
        self.q_net_max = constants.get("q_net_max", 15.0 / self.s_base_mva)
        self.v_ref     = float(constants["v_ref"])
        self.v_min     = float(constants["v_min"])
        self.v_max     = float(constants["v_max"])
        self.mg_buses  = constants["mg_buses"]
        self.independent_prices = bool(constants.get("independent_prices", False))
        self.n_followers = len(self.mg_buses) if self.independent_prices else 1
        self.p_passive = torch.as_tensor(
            constants["p_passive"], dtype=torch.float32, device=device
        )
        # Topology
        self.child_edges = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            self.child_edges[i].append(e)
            self.parent_edge[j] = e
        # Optimization variables
        if fixed_pi is None:
            pi_shape = (
                (self.n_followers, self.T)
                if self.independent_prices
                else (self.T,)
            )
            self.pi_raw = nn.Parameter(torch.zeros(pi_shape, device=device))
        else:
            self.pi_fixed = torch.tensor(fixed_pi, dtype=torch.float32, device=device)
        # Keep the passive reactive-power profile as immutable reference data.
        self.register_buffer(
            "q_passive",
            torch.as_tensor(
                constants["q_passive"], dtype=torch.float32, device=device
            ).clone(),
        )
        self.p0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.q0 = nn.Parameter(torch.zeros(self.T, device=device))
        self.v  = nn.Parameter(
            torch.ones(self.n_bus, self.T, device=device) * (self.v_ref ** 2)
        )
        self.l  = nn.Parameter(torch.ones(self.n_line, self.T, device=device) * 1e-3)
        self.p  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        self.q  = nn.Parameter(torch.zeros(self.n_line, self.T, device=device))
        # Explicit auxiliary variable from the DN formulation.  The shared
        # penalty model enforces q_injection == q_passive.
        self.q_injection = nn.Parameter(self.q_passive.detach().clone())

    def bounded_pi(self) -> torch.Tensor:
        """
        Output : pi — Tensor (T,), range [pi_min, pi_max]
        """
        if self.fixed_pi is None:
            return self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(self.pi_raw)
        else:
            return self.pi_fixed

    def forward(self, penalty_weights: dict = None) -> tuple:
        if penalty_weights is None:
            penalty_weights = {
                "eq": 1000.0, "ineq": 100.0, "soc": 100.0, "bound": 10.0
            }
        pi = self.bounded_pi()
        if self.fixed_p_mg is not None:
            p_mg_prediction = torch.as_tensor(
                self.fixed_p_mg, dtype=torch.float32, device=self.device
            )
        elif hasattr(self.surrogate, "predict_torch"):
            p_mg_prediction = self.surrogate.predict_torch(pi)
            if self.independent_prices:
                p_mg_prediction = p_mg_prediction.reshape(self.n_followers, self.T)
        else:
            p_mg_prediction = torch.as_tensor(
                self.surrogate.predict(pi.detach().cpu().numpy()),
                dtype=torch.float32,
                device=self.device,
            )
        return torch_penalty_distflow(
            self,
            pi,
            p_mg_prediction,
            penalty_weights,
            square_voltage_limits=False,
        )


class DistributionNetworkLeader:
    def __init__(self, constants: dict, device: str = 'cpu'):
        """
        Input  : constants — return value from extract_constants()
                device - 'cpu' | 'cuda'
        """
        self.constants = constants
        self.device    = device
        self.T         = constants["T"]
        self.n_bus     = constants["n_bus"]
        self.n_line    = constants["n_line"]
        self.independent_prices = bool(constants.get("independent_prices", False))
        self.n_followers = (
            len(constants["mg_buses"]) if self.independent_prices else 1
        )
        self.decision_size = self.n_followers * self.T
        self.distflow_evaluator = CoptDistFlowEvaluator(
            constants,
            failure="raise",
            revenue="single_profile",
            accept_suboptimal=False,
            duplicate_one_dimensional=True,
            verbose=False,
        )
        # Populated only when solve_params explicitly enables diagnostic
        # tracing.  Keeping the trace on the leader avoids changing the
        # long-standing two-value return contract of solve().
        self.last_solve_diagnostics: dict | None = None
        # Topology used by _solve_power_flow
        self.child_edges = {i: [] for i in range(self.n_bus)}
        self.parent_edge = {i: None for i in range(self.n_bus)}
        for e in range(self.n_line):
            i = int(constants["from_bus"][e])
            j = int(constants["to_bus"][e])
            self.child_edges[i].append(e)
            self.parent_edge[j] = e


    def evaluate_objective(self, x: np.ndarray, y : np.ndarray) -> float:
        """
        fix pi = x, p_mg = y, use COPT to solve the power flow problem.
        Input  : x — ndarray (T,)
                 y — ndarray (T,)
        Output : objective value - float
        """
        return self._solve_power_flow(x, y)

    
    def evaluate_surrogate_objective(self, x: np.ndarray, surrogate: SurrogateModel) -> float:
        """
        Input  : x         — ndarray (T,), pi
                 surrogate — current surrogate model
        Output : float
        """
        response = surrogate.predict(x)
        if self.independent_prices:
            response = np.asarray(response).reshape(self.n_followers, self.T)
        return self._solve_power_flow(self._reshape_price(x), response)

    def _solve_power_flow(self, pi: np.ndarray, p_mg: np.ndarray) -> float:
        """Evaluate fixed decisions with the shared hard DistFlow model."""
        return self.distflow_evaluator.evaluate(
            self._reshape_price(pi),
            p_mg,
        )

    def _copt_network_state(
        self,
        x_center: np.ndarray,
        surrogate: SurrogateModel,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Solve hard DistFlow at a TR center and return its network state."""
        price = np.asarray(x_center, dtype=float).reshape(-1)
        response = np.asarray(surrogate.predict(price), dtype=float)
        if self.independent_prices:
            response = response.reshape(self.n_followers, self.T)
        objective = self.distflow_evaluator.evaluate(
            self._reshape_price(price), response
        )
        evaluator = self.distflow_evaluator

        def one_dimensional(variables: Any, size: int) -> np.ndarray:
            return np.asarray(
                [variables[index].x for index in range(size)], dtype=float
            )

        def two_dimensional(
            variables: Any,
            first: int,
            second: int,
        ) -> np.ndarray:
            return np.asarray(
                [
                    [variables[i, t].x for t in range(second)]
                    for i in range(first)
                ],
                dtype=float,
            )

        return {
            "p0": one_dimensional(evaluator.p0, self.T),
            "q0": one_dimensional(evaluator.q0, self.T),
            "v": two_dimensional(evaluator.voltage, self.n_bus, self.T),
            "l": two_dimensional(evaluator.current, self.n_line, self.T),
            "p": two_dimensional(evaluator.p_flow, self.n_line, self.T),
            "q": two_dimensional(evaluator.q_flow, self.n_line, self.T),
            "q_injection": two_dimensional(
                evaluator.q_injection, self.n_bus, self.T
            ),
        }, float(objective)

    def _initialize_torch_state(
        self,
        model: _LeaderSubproblem,
        state: Mapping[str, np.ndarray],
    ) -> None:
        """Validate and copy a hard-DistFlow state into Torch parameters."""
        with torch.no_grad():
            for name in ("p0", "q0", "v", "l", "p", "q", "q_injection"):
                parameter = getattr(model, name)
                if name not in state:
                    raise RuntimeError(
                        f"COPT warm-start state is missing parameter {name}."
                    )
                value = torch.as_tensor(
                    state[name], dtype=parameter.dtype, device=parameter.device
                )
                if value.shape != parameter.shape:
                    raise RuntimeError(
                        f"COPT warm-start shape mismatch for {name}: "
                        f"expected {tuple(parameter.shape)}, got {tuple(value.shape)}."
                    )
                if value.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"COPT warm-start dtype mismatch for {name}: "
                        f"expected {parameter.dtype}, got {value.dtype}."
                    )
                if value.device != parameter.device:
                    raise RuntimeError(
                        f"COPT warm-start device mismatch for {name}: "
                        f"expected {parameter.device}, got {value.device}."
                    )
                parameter.copy_(value)

    def solve(self, surrogate: SurrogateModel,
            trust_region:TrustRegion = None,
              x_center: np.ndarray = None,
              solve_params: dict = None) -> tuple:
        """
        Full optimization mode: pi is optimized and p_mg comes from the surrogate.
        Input  : surrogate    — current surrogate (GlobalSurrogate with historical residuals)
                 trust_region — add an L_inf soft penalty when not None
                 x_center     — trust-region center, shape (T,), used for warm start and penalty
                 solve_params — {'steps', 'lr', 'penalty_weights', 'tr_weight',
                                 'copt_initialize_network_state'}
        Output : (x_opt: ndarray (T,), F_tilde: float)
                  x_opt   — optimal pi
                  F_tilde — corresponding surrogate objective value (used by compute_ratio)
        """
        if solve_params is None:
            solve_params = {'steps': 5000, 'lr': 1e-2}
        self.last_solve_diagnostics = None
        _model = _LeaderSubproblem(self.constants, surrogate, self.device)
        uses_torch_surrogate = hasattr(surrogate, "predict_torch")
        # Warm start
        if x_center is not None:
            with torch.no_grad():
                _model.pi_raw.copy_(
                    torch.tensor(
                        self._inverse_sigmoid(self._reshape_price(x_center)),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        copt_warm_start_seconds = 0.0
        copt_warm_start_objective = None
        if (
            trust_region is not None
            and x_center is not None
            and bool(solve_params.get("copt_initialize_network_state", False))
        ):
            start = time.perf_counter()
            with torch.no_grad():
                state, copt_warm_start_objective = self._copt_network_state(
                    x_center, surrogate
                )
                self._initialize_torch_state(_model, state)
            copt_warm_start_seconds = time.perf_counter() - start

        opt       = torch.optim.Adam(_model.parameters(), lr=solve_params['lr'])
        pw        = solve_params.get('penalty_weights',
                                     {'eq': 1000., 'ineq': 100., 'soc': 100., 'bound': 10.})
        tr_weight = solve_params.get('tr_weight', 500.0)
        diagnostic_trace_every = int(
            solve_params.get('diagnostic_trace_every', 0) or 0
        )
        if diagnostic_trace_every < 0:
            raise ValueError("diagnostic_trace_every must be nonnegative.")
        collect_diagnostics = diagnostic_trace_every > 0
        diagnostic_trace = []
        best_diagnostics = None
        tr_lb = tr_ub = None
        if trust_region is not None and x_center is not None:
            x_c = torch.tensor(
                self._reshape_price(x_center),
                dtype=torch.float32,
                device=self.device,
            )
            radius = trust_region.get_radius()
            tr_lb = torch.clamp(x_c - radius, min=_model.pi_min, max=_model.pi_max)
            tr_ub = torch.clamp(x_c + radius, min=_model.pi_min, max=_model.pi_max)
        best_merit = np.inf
        best_x = None
        best_objective = None

        def diagnostic_snapshot(
            iteration: int,
            loss: torch.Tensor,
            aux: dict,
            tr_penalty: torch.Tensor,
            gradient_available: bool,
        ) -> dict:
            pi_value = aux['pi'].detach().cpu().numpy().reshape(-1)
            eq_pen = float(aux['eq_pen'].detach().cpu())
            ineq_pen = float(aux['ineq_pen'].detach().cpu())
            soc_pen = float(aux['soc_pen'].detach().cpu())
            bound_pen = float(aux['bound_pen'].detach().cpu())
            gradients = [
                parameter.grad.detach().reshape(-1)
                for parameter in _model.parameters()
                if parameter.grad is not None
            ] if gradient_available else []
            if gradients:
                gradient = torch.cat(gradients)
                gradient_l2 = float(torch.linalg.vector_norm(gradient).cpu())
                gradient_inf = float(torch.max(torch.abs(gradient)).cpu())
            else:
                gradient_l2 = float('nan')
                gradient_inf = float('nan')
            distance_inf = (
                float(np.max(np.abs(pi_value - np.asarray(x_center).reshape(-1))))
                if x_center is not None
                else float('nan')
            )
            return {
                'iteration': int(iteration),
                'merit': float(loss.detach().cpu()),
                'raw_surrogate_objective': float(aux['obj'].detach().cpu()),
                'eq_pen': eq_pen,
                'ineq_pen': ineq_pen,
                'soc_pen': soc_pen,
                'bound_pen': bound_pen,
                'weighted_penalty': float(
                    pw['eq'] * eq_pen
                    + pw['ineq'] * ineq_pen
                    + pw['soc'] * soc_pen
                    + pw['bound'] * bound_pen
                ),
                'tr_penalty': float(tr_penalty.detach().cpu()),
                'gradient_l2': gradient_l2,
                'gradient_inf': gradient_inf,
                'distance_inf': distance_inf,
                'distance_over_R': (
                    distance_inf / trust_region.get_radius()
                    if trust_region is not None and np.isfinite(distance_inf)
                    else float('nan')
                ),
            }

        def record_best(
            loss: torch.Tensor,
            aux: dict,
            iteration: int,
            tr_penalty: torch.Tensor,
        ) -> None:
            nonlocal best_merit, best_x, best_objective, best_diagnostics
            merit = float(loss.detach().cpu())
            if np.isfinite(merit) and merit < best_merit:
                best_merit = merit
                best_x = aux['pi'].detach().cpu().numpy().copy()
                best_objective = float(aux['obj'].detach().cpu())
                if collect_diagnostics:
                    best_diagnostics = diagnostic_snapshot(
                        iteration, loss, aux, tr_penalty, gradient_available=False
                    )

        for iteration in range(solve_params['steps']):
            opt.zero_grad()
            loss, aux = _model.forward(pw)
            tr_penalty = loss.new_zeros(())
            # L_inf trust-region soft penalty
            if tr_lb is not None and tr_ub is not None:
                pi_cur = _model.bounded_pi()
                tr_viol = torch.maximum(
                    torch.relu(tr_lb - pi_cur).max(),
                    torch.relu(pi_cur - tr_ub).max(),
                )
                tr_penalty = tr_weight * tr_viol ** 2
                loss = loss + tr_penalty
            # Keep the best surrogate-merit iterate. This is an entirely
            # surrogate-side comparison and does not query the follower.
            record_best(loss, aux, iteration, tr_penalty)
            loss.backward()
            # Fallback for legacy numpy-only surrogate models.
            if (not uses_torch_surrogate) and getattr(surrogate, "_residuals", []):
                self._add_residual_gradient(_model, surrogate)
            if collect_diagnostics and (
                iteration == 0
                or (iteration + 1) % diagnostic_trace_every == 0
            ):
                diagnostic_trace.append(
                    diagnostic_snapshot(
                        iteration, loss, aux, tr_penalty,
                        gradient_available=True,
                    )
                )
            opt.step()
            if tr_lb is not None and tr_ub is not None:
                with torch.no_grad():
                    pi_projected = torch.clamp(_model.bounded_pi(), min=tr_lb, max=tr_ub)
                    z = (pi_projected - _model.pi_min) / (_model.pi_max - _model.pi_min)
                    z = torch.clamp(z, 1e-6, 1 - 1e-6)
                    _model.pi_raw.copy_(torch.log(z / (1 - z)))
        with torch.no_grad():
            final_loss, final_aux = _model.forward(pw)
            final_tr_penalty = final_loss.new_zeros(())
            if tr_lb is not None and tr_ub is not None:
                pi_cur = _model.bounded_pi()
                tr_viol = torch.maximum(
                    torch.relu(tr_lb - pi_cur).max(),
                    torch.relu(pi_cur - tr_ub).max(),
                )
                final_tr_penalty = tr_weight * tr_viol ** 2
                final_loss = final_loss + final_tr_penalty
            record_best(
                final_loss, final_aux, int(solve_params['steps']),
                final_tr_penalty,
            )
            if collect_diagnostics:
                diagnostic_trace.append(
                    diagnostic_snapshot(
                        int(solve_params['steps']), final_loss, final_aux,
                        final_tr_penalty, gradient_available=False,
                    )
                )
        if best_x is None or best_objective is None:
            raise RuntimeError("Leader surrogate optimization produced no finite iterate.")
        x_opt = best_x
        if self.independent_prices:
            x_opt = x_opt.reshape(-1)
        F_tilde = best_objective
        if collect_diagnostics:
            self.last_solve_diagnostics = {
                'steps': int(solve_params['steps']),
                'lr': float(solve_params['lr']),
                'trace_every': diagnostic_trace_every,
                'copt_warm_start_seconds': copt_warm_start_seconds,
                'copt_warm_start_objective': copt_warm_start_objective,
                'best': best_diagnostics,
                'initial': diagnostic_trace[0] if diagnostic_trace else None,
                'final': diagnostic_trace[-1] if diagnostic_trace else None,
                'trace': diagnostic_trace,
            }
        return x_opt, F_tilde

    def get_feasible_set(self) -> dict:
        """
        Output : {'lb': ndarray(T,), 'ub': ndarray(T,)}
        """
        return {
            'lb': np.full(self.decision_size, self.constants['pi_min']),
            'ub': np.full(self.decision_size, self.constants['pi_max']),
        }
    # ------------------------------------------------------------------
    def _inverse_sigmoid(self, pi: np.ndarray) -> np.ndarray:
        """
        Input  : pi — ndarray (T,), range [pi_min, pi_max]
        Output :    — ndarray (T,), corresponding pi_raw
        """
        pi_min = self.constants['pi_min']
        pi_max = self.constants['pi_max']
        z = np.clip((pi - pi_min) / (pi_max - pi_min), 1e-6, 1 - 1e-6)
        return np.log(z / (1 - z))

    def _reshape_price(self, pi: np.ndarray) -> np.ndarray:
        price = np.asarray(pi, dtype=float)
        if not self.independent_prices:
            return price.reshape(self.T)
        return price.reshape(self.n_followers, self.T)
    def _add_residual_gradient(self, _model: _LeaderSubproblem,
                                surrogate: GlobalSurrogate) -> None:
        """
        Analytically accumulate gradients of all historical residuals with respect to pi_raw,
        directly modifying pi_raw.grad.
        Input  : _model    — current _LeaderSubproblem instance
                 surrogate — GlobalSurrogate containing the _residuals list
        Output : —
        """
        with torch.no_grad():
            pi    = _model.bounded_pi().cpu().numpy()   # (T,)
            dU_dy = -pi                                 # ∂U/∂p_mg = -pi, shape (T,)
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


#region Part V: Trust Region Controller Setup
class TRResidualController:
    def __init__(self,
                 leader: DistributionNetworkLeader,
                 follower: LowerSolver,
                 global_surrogate: GlobalSurrogate,
                 residual_model: ResidualModel,
                 tr_params: dict,
                 algo_params: dict):
        """
        Input  : leader          — DistributionNetworkLeader instance
                 follower        — FollowerBase instance
                 global_surrogate— GlobalSurrogate instance (trained neural network)
                 residual_model  — ResidualModel instance
                 tr_params       — trust-region parameters
                                   {'R_0', 'R_max', 'gamma_dec', 'gamma_inc', 'eta_1', 'eta_2'}
                 algo_params     — algorithm parameters
                                   {'n_samples', 'K_max', 'x0',
                                    'eps_x', 'eps_f', 'eps_R',
                                    'sampling_method',
                                    'residual_update_mode',
                                    'rbf_geometry_condition_max',
                                    'save_true_responses',
                                    'radius_policy', 'acceptance_policy',
                                    'fixed_radius',
                                    'max_consecutive_rejections',
                                    'solve_params'}
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
        self.radius_policy = str(algo_params.get("radius_policy", "adaptive"))
        self.acceptance_policy = str(algo_params.get("acceptance_policy", "filter"))
        self.fixed_radius = float(algo_params.get("fixed_radius", 0.05))
        self.max_consecutive_rejections = int(
            algo_params.get("max_consecutive_rejections", 10)
        )
        if self.radius_policy not in {"adaptive", "fixed"}:
            raise ValueError("radius_policy must be 'adaptive' or 'fixed'.")
        if self.acceptance_policy not in {"filter", "always_accept"}:
            raise ValueError("acceptance_policy must be 'filter' or 'always_accept'.")
        if not np.isfinite(self.fixed_radius) or self.fixed_radius <= 0:
            raise ValueError("fixed_radius must be finite and positive.")
        if self.max_consecutive_rejections <= 0:
            raise ValueError("max_consecutive_rejections must be positive.")
        if self.radius_policy == "fixed":
            self.tr.R = self.fixed_radius
        self.consecutive_rejections = 0
        # Total points in each follower-local design, including its center.
        self.n_samples       = int(algo_params['n_samples'])
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.K_max           = algo_params['K_max']
        self.x0              = algo_params['x0']
        self.eps_x           = algo_params['eps_x']
        self.eps_f           = algo_params['eps_f']
        self.eps_R           = algo_params['eps_R']
        self.sampling_method = algo_params.get('sampling_method', 'lhs')
        self.solve_params    = algo_params.get('solve_params', None)
        self.random_seed     = algo_params.get('random_seed', None)
        self.residual_update_mode = str(
            algo_params.get("residual_update_mode", "replace")
        ).strip().lower()
        if self.residual_update_mode not in {"replace", "accumulate"}:
            raise ValueError(
                "residual_update_mode must be 'replace' or 'accumulate', got "
                f"{self.residual_update_mode!r}."
            )
        # The samples are still used to fit the current residual model.  This
        # flag only controls whether copies are retained across iterations for
        # later export.
        self.save_true_responses = bool(
            algo_params.get("save_true_responses", False)
        )
        self.rbf_geometry_condition_max = float(
            algo_params.get("rbf_geometry_condition_max", 1e12)
        )
        if self.rbf_geometry_condition_max <= 0.0:
            raise ValueError("rbf_geometry_condition_max must be positive.")
        self.rbf_geometry_max_attempts = int(
            algo_params.get("rbf_geometry_max_attempts", 6)
        )
        if self.rbf_geometry_max_attempts <= 0:
            raise ValueError("rbf_geometry_max_attempts must be positive.")
        self.sample_tolerance = 1e-8
        self._last_geometry_resample_attempts = 0
        self._last_rbf_geometry_metrics: list[dict] = []
        self.scenarios = [
            _normalize_scenario_name(value)
            for value in algo_params.get("scenarios", [])
        ]
        self._reset_true_response_buffers()

    def _n_response_models(self) -> int:
        if isinstance(self.global_surrogate, CompositeGlobalSurrogate):
            return int(self.global_surrogate.n_followers)
        return 1

    def _reset_true_response_buffers(self) -> None:
        n_models = self._n_response_models()
        self._sampled_X: list[list[np.ndarray]] = [[] for _ in range(n_models)]
        self._sampled_Y: list[list[np.ndarray]] = [[] for _ in range(n_models)]

    def _record_true_responses(
        self, fit_data: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        if not self.save_true_responses:
            return
        for index, (X_fit, Y_true) in enumerate(fit_data):
            self._sampled_X[index].append(np.asarray(X_fit, dtype=float).copy())
            self._sampled_Y[index].append(np.asarray(Y_true, dtype=float).copy())

    def _follower_scenarios(self) -> list[str]:
        if self.scenarios:
            return list(self.scenarios)
        follower_ids = getattr(self.follower, "follower_ids", None)
        if follower_ids:
            return [_normalize_scenario_name(value) for value in follower_ids]
        children = getattr(self.follower, "followers", None)
        if children:
            return [
                _normalize_scenario_name(getattr(child, "scenario", f"F{index}"))
                for index, child in enumerate(children)
            ]
        return [_normalize_scenario_name(getattr(self.follower, "scenario", "SB"))]

    def collected_true_responses(self) -> dict[str, dict[str, np.ndarray]]:
        """Collect unique high-fidelity samples separately by follower ID."""
        if not self.save_true_responses:
            return {}
        scenarios = self._follower_scenarios()
        n_models = self._n_response_models()
        if len(scenarios) != n_models:
            raise RuntimeError(
                f"Expected {n_models} follower scenarios, got {len(scenarios)}."
            )
        grouped_x: dict[str, list[np.ndarray]] = {}
        grouped_y: dict[str, list[np.ndarray]] = {}
        for index, scenario in enumerate(scenarios):
            if not self._sampled_X[index]:
                continue
            grouped_x.setdefault(scenario, []).append(np.vstack(self._sampled_X[index]))
            grouped_y.setdefault(scenario, []).append(np.vstack(self._sampled_Y[index]))
        collected: dict[str, dict[str, np.ndarray]] = {}
        for scenario, blocks in grouped_x.items():
            X = np.vstack(blocks)
            Y = np.vstack(grouped_y[scenario])
            X_unique, keep = _unique_rows(X, tol=self.sample_tolerance)
            collected[scenario] = {"X": X_unique, "Y": Y[keep]}
        return collected

    def _rbf_design_quality(
        self,
        X: np.ndarray,
        x_center: np.ndarray,
        trust_radius: float,
    ) -> dict:
        """Check a proposed or final design before it is used for RBF fitting."""
        X = np.asarray(X, dtype=float)
        center = np.asarray(x_center, dtype=float).reshape(-1)
        center_present = bool(
            len(X)
            and np.any(
                np.max(np.abs(X - center), axis=1)
                <= self.sample_tolerance
            )
        )
        model = ResidualModel(**self.residual_model_kwargs)
        metrics = model.rbf_geometry_metrics(
            X, x_center=center, radius=trust_radius
        )
        expected_rank = (
            center.size + 1
            if model.rbf_degree == 1
            else metrics["polynomial_columns"]
        )
        metrics.update({
            "input_sample_count": int(len(X)),
            "center_present": center_present,
            "expected_polynomial_rank": int(expected_rank),
            "within_sample_cap": bool(len(X) <= self.n_samples),
        })
        metrics["is_valid"] = bool(
            center_present
            and metrics["within_sample_cap"]
            and metrics["polynomial_rank"] == expected_rank
            and np.isfinite(metrics["condition_number"])
            and metrics["condition_number"]
            <= self.rbf_geometry_condition_max
        )
        return metrics

    def _sample_unique_local_points(
        self,
        x_center: np.ndarray,
        feasible_set: dict,
        n: int,
        seed: int | None,
        include_star_points: bool,
        max_star_points: int | None = None,
        selection_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return coordinate-star points first, then fill with unique LHS points."""
        x_center = np.asarray(x_center, dtype=float).reshape(-1)
        if n <= 0:
            return np.empty((0, x_center.size))
        references = (
            np.asarray(selection_points, dtype=float).reshape(-1, x_center.size)
            if selection_points is not None
            else x_center.reshape(1, -1)
        )
        selected: list[np.ndarray] = []
        for refill in range(8):
            deficit = n - len(selected)
            if deficit <= 0:
                break
            refill_seed = (
                None if seed is None else seed + refill * 10_007
            )
            use_star_points = include_star_points and refill == 0
            sampled = sample_in_trust_region(
                x_center=x_center,
                tr=self.tr,
                n=deficit,
                feasible_set=feasible_set,
                method=self.sampling_method,
                seed=refill_seed,
                include_star_points=use_star_points,
                max_star_points=(
                    min(2 * x_center.size, deficit)
                    if use_star_points and max_star_points is None
                    else max_star_points
                ),
            )
            for candidate in sampled:
                uniqueness_references = (
                    references
                    if not selected
                    else np.vstack([references, np.asarray(selected)])
                )
                if len(uniqueness_references) and np.any(
                    np.max(np.abs(uniqueness_references - candidate), axis=1)
                    <= self.sample_tolerance
                ):
                    continue
                selected.append(np.asarray(candidate, dtype=float).copy())
                if len(selected) == n:
                    break
        if not selected:
            return np.empty((0, x_center.size))
        return np.asarray(selected[:n])

    def _sample_composite_rows(
        self,
        k: int,
        x_center: np.ndarray,
        feasible_set: dict,
        missing_counts: Sequence[int],
        attempt: int,
        fit_inputs: Sequence[np.ndarray] | None = None,
        trust_radius: float | None = None,
    ) -> np.ndarray:
        n_new = max((int(value) for value in missing_counts), default=0)
        decision_size = np.asarray(x_center).size
        if n_new <= 0:
            return np.empty((0, decision_size))
        seed_offset = attempt * 100_003
        use_star_points = self.residual_model_type == "rbf"
        is_composite = isinstance(self.global_surrogate, CompositeGlobalSurrogate)
        if not is_composite:
            seed = (
                None
                if self.random_seed is None
                else self.random_seed + k + seed_offset
            )
            return self._sample_unique_local_points(
                x_center=x_center,
                n=n_new,
                feasible_set=feasible_set,
                seed=seed,
                include_star_points=use_star_points,
                max_star_points=min(2 * x_center.size, n_new),
                selection_points=(
                    fit_inputs[0] if fit_inputs is not None else None
                ),
            )

        n_followers = self.global_surrogate.n_followers
        T = self.global_surrogate.T
        x_blocks = np.asarray(x_center, dtype=float).reshape(n_followers, T)
        lb_blocks = np.asarray(feasible_set["lb"]).reshape(n_followers, T)
        ub_blocks = np.asarray(feasible_set["ub"]).reshape(n_followers, T)
        sample_blocks = []
        for index in range(n_followers):
            block_seed = (
                None
                if self.random_seed is None
                else self.random_seed
                + k * n_followers
                + index
                + seed_offset
            )
            sample_blocks.append(
                self._sample_unique_local_points(
                    x_center=x_blocks[index],
                    n=n_new,
                    feasible_set={
                        "lb": lb_blocks[index],
                        "ub": ub_blocks[index],
                    },
                    seed=block_seed,
                    include_star_points=use_star_points,
                    max_star_points=min(2 * T, n_new),
                    selection_points=(
                        fit_inputs[index] if fit_inputs is not None else None
                    ),
                )
            )
        n_rows = min(len(block) for block in sample_blocks)
        if n_rows == 0:
            return np.empty((0, decision_size))
        return np.concatenate([block[:n_rows] for block in sample_blocks], axis=1)

    def _collect_residual_fit_data(
        self,
        k: int,
        x_k: np.ndarray,
        feasible_set: dict,
        trust_radius: float,
    ) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]]]:
        is_composite = isinstance(self.global_surrogate, CompositeGlobalSurrogate)
        n_models = self.global_surrogate.n_followers if is_composite else 1
        target = self.n_samples
        centers = (
            np.asarray(x_k, dtype=float).reshape(
                self.global_surrogate.n_followers,
                self.global_surrogate.T,
            )
            if is_composite
            else np.asarray(x_k, dtype=float).reshape(1, -1)
        )
        fit_inputs = [centers[index:index + 1] for index in range(n_models)]
        missing_counts = [self.n_samples - 1] * n_models
        self._last_geometry_resample_attempts = 0
        self._last_rbf_geometry_metrics = []

        last_failed_index = 0
        last_metrics = {
            "polynomial_rank": 0,
            "condition_number": float("inf"),
        }
        for attempt in range(self.rbf_geometry_max_attempts):
            new_rows = self._sample_composite_rows(
                k,
                x_k,
                feasible_set,
                missing_counts,
                attempt,
                fit_inputs=fit_inputs,
                trust_radius=trust_radius,
            )
            if len(new_rows) == 0:
                continue

            row_blocks = (
                new_rows.reshape(
                    len(new_rows),
                    self.global_surrogate.n_followers,
                    self.global_surrogate.T,
                )
                if is_composite
                else new_rows.reshape(len(new_rows), 1, -1)
            )
            proposed_designs = []
            all_valid = True
            for index in range(n_models):
                combined = np.vstack([
                    fit_inputs[index],
                    row_blocks[:, index, :],
                ])
                proposed = combined
                proposed_designs.append(proposed)
                if self.residual_model_type == "rbf":
                    quality = self._rbf_design_quality(
                        proposed, centers[index], trust_radius
                    )
                    quality["is_valid"] = bool(
                        quality["is_valid"]
                        and quality["sample_count"] == target
                    )
                else:
                    quality = {
                        "is_valid": len(proposed) == target,
                        "polynomial_rank": 0,
                        "condition_number": float("nan"),
                    }
                if not quality["is_valid"] and all_valid:
                    all_valid = False
                    last_failed_index = index
                    last_metrics = quality
            if not all_valid:
                self._last_geometry_resample_attempts = attempt + 1
                continue

            # Every follower design has passed before the first expensive query.
            lower_k = self.follower.lower_solve(
                np.asarray(x_k, dtype=float).copy()
            )
            if lower_k is None:
                raise RuntimeError("Lower-level solve failed at the current point.")
            center_outputs = np.asarray(
                lower_k["p_mg"], dtype=float
            ).reshape(n_models, -1)
            row_results = []
            for row in new_rows:
                result = self.follower.lower_solve(
                    np.asarray(row, dtype=float).copy()
                )
                if result is None:
                    raise RuntimeError(
                        "Lower-level solve failed for a trust-region sample."
                    )
                row_results.append(result)

            row_output_blocks = np.stack([
                np.asarray(result["p_mg"], dtype=float)
                for result in row_results
            ]).reshape(len(row_results), n_models, -1)
            final_fit_data = [
                (
                    proposed_designs[index],
                    np.vstack([
                        center_outputs[index:index + 1],
                        row_output_blocks[:, index, :],
                    ]),
                )
                for index in range(n_models)
            ]
            final_missing = [
                target - len(values) for values, _ in final_fit_data
            ]
            final_metrics = []
            for index in range(n_models):
                final_X = final_fit_data[index][0]
                if self.residual_model_type == "rbf":
                    quality = self._rbf_design_quality(
                        final_X, centers[index], trust_radius
                    )
                    quality["is_valid"] = bool(
                        quality["is_valid"]
                        and quality["sample_count"] == target
                    )
                    if not quality["is_valid"]:
                        raise RuntimeError(
                            "Final RBF geometry check failed after residual "
                            f"sampling: follower_index={index}, "
                            f"missing_count={final_missing[index]}, "
                            f"polynomial_rank={quality['polynomial_rank']}, "
                            f"condition_number="
                            f"{quality['condition_number']:.6e}, "
                            f"attempt_count={attempt + 1}."
                        )
                    final_metrics.append(quality)
            if max(final_missing, default=0) <= 0:
                self._last_geometry_resample_attempts = attempt
                self._last_rbf_geometry_metrics = final_metrics
                self._record_true_responses(final_fit_data)
                return lower_k, final_fit_data
            raise RuntimeError(
                "Geometry-approved residual sampling did not fill every local "
                f"design; missing_counts={final_missing}."
            )

        raise RuntimeError(
            "RBF geometry check failed before high-fidelity residual queries: "
            f"follower_index={last_failed_index}, "
            f"missing_count={missing_counts[last_failed_index]}, "
            f"polynomial_rank={last_metrics['polynomial_rank']}, "
            f"condition_number={last_metrics['condition_number']:.6e}, "
            f"attempt_count={self.rbf_geometry_max_attempts}."
        )
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the complete TR iteration process.
        Output : dict
                 'x_opt'     : ndarray (T,), final leader decision
                 'x_init'    : ndarray, leader decision after the steps_init solve
                 'surrogate' : GlobalSurrogate, final surrogate model
                 'history'   : list[dict], per-iteration records
                 'true_responses': dict, unique high-fidelity samples by scenario;
                                   present only when save_true_responses is enabled
                 'converged' : bool
                 'n_iter'    : int
        """
        if self.random_seed is not None:
            set_global_seed(self.random_seed)
        self._reset_true_response_buffers()
        self.consecutive_rejections = 0
        if self.solve_params is None:
            raise RuntimeError("solve_params must include steps_init and steps_warm.")
        # From x0, run an unconstrained global-surrogate solve for steps_init
        # steps. Local sampling and residual-corrected re-solves then use
        # steps_warm.
        print("Using global surrogate to solve initial point...")
        time_start = time.time()
        init_steps = int(self.solve_params["steps_init"])
        x_k, _ = self.leader.solve(
            surrogate=self.global_surrogate,
            x_center=np.asarray(self.x0, dtype=float),
            solve_params={**self.solve_params, "steps": init_steps},
        )
        x_k = np.asarray(x_k, dtype=np.float32)
        x_init = np.asarray(x_k, dtype=float).copy()
        print(
            f"Initial surrogate solve used steps_init={init_steps}. "
            f"Time taken to solve initial point "
            f"{time.time() - time_start:.4f} seconds."
        )
        history   = []
        converged = False
        stop_reason = "max_iterations"
        for k in range(self.K_max):
            time_start = time.time()
            print(f"Iteration {k} | R={self.tr.get_radius():.4f} ")
            logger.info(f"Iteration {k} | R={self.tr.get_radius():.4f} | x_{k}={x_k}")
            record, x_next, converged = self._one_iteration(k, x_k)
            time_end = time.time()
            print(f"Time taken to solve iteration {k} {time_end - time_start:.4f} seconds.")
            history.append(record)
            x_k = x_next
            if record["stop_reason"] is not None:
                stop_reason = record["stop_reason"]
                logger.info(f"Stopped at iteration {k}: {stop_reason}.")
                print(f"Stopped at iteration {k}: {stop_reason}.")
                break
        result = {
            'x_opt'    : x_k,
            'x_init'   : x_init,
            'surrogate': self.global_surrogate,
            'history'  : history,
            'converged': converged,
            'stop_reason': stop_reason,
            'n_iter'   : len(history),
        }
        if self.save_true_responses:
            result['true_responses'] = self.collected_true_responses()
        return result

    def _one_iteration(self, k: int, x_k: np.ndarray) -> tuple:
        """
        Run one iteration.
        Input  : k   — current iteration index
                 x_k — current leader decision, ndarray (T,)
        Output : (record: dict, x_next: ndarray (T,), converged: bool)
        """
        steps = int(self.solve_params["steps_warm"])
        cur_solve_params = {**self.solve_params, "steps": steps}
        feasible_set = self.leader.get_feasible_set()
        trust_radius = self.tr.get_radius()
        is_composite = isinstance(
            self.global_surrogate, CompositeGlobalSurrogate
        )
        # Steps 1-2: Build a fresh local residual design for this iteration.
        if is_composite:
            n_followers = self.global_surrogate.n_followers
            print(
                f"Building {self.n_samples}-point local designs for each of "
                f"{n_followers} followers (center + coordinate star + LHS) "
                f"within R_tr={trust_radius:.4f}..."
            )
        else:
            print(
                f"Building a {self.n_samples}-point local design "
                f"(center + coordinate star + LHS) within "
                f"R_tr={trust_radius:.4f}..."
            )
        lower_k, fit_data = self._collect_residual_fit_data(
            k, x_k, feasible_set, trust_radius
        )
        # Step 3: Fit the local residual model
        rbf_condition_numbers = []
        if is_composite:
            print(
                "Fitting follower-local residual models with sample counts "
                f"{[len(values) for values, _ in fit_data]}..."
            )
            residual_models = []
            for index in range(n_followers):
                X_fit, Y_true = fit_data[index]
                follower_surrogate = self.global_surrogate.surrogates[index]
                if self.residual_update_mode == "accumulate":
                    Y_hat = follower_surrogate.predict(X_fit)
                else:
                    Y_hat = follower_surrogate.predict_base(X_fit)
                residual = ResidualModel(**self.residual_model_kwargs)
                residual.fit(
                    X_fit,
                    Y_true - Y_hat,
                    x_center=x_k.reshape(
                        n_followers, self.global_surrogate.T
                    )[index],
                    radius=trust_radius,
                )
                residual_models.append(residual)
                if residual.model_type == "rbf":
                    rbf_condition_numbers.append(
                        float(residual.rbf_condition)
                    )
                    print(
                        f"Follower {index} RBF interpolation condition number: "
                        f"{residual.rbf_condition:.4e}"
                    )
            residual_update = residual_models
        else:
            X_fit, Y_true = fit_data[0]
            if self.residual_update_mode == "accumulate":
                Y_hat = self.global_surrogate.predict(X_fit)
            else:
                Y_hat = self.global_surrogate.predict_base(X_fit)
            print(f"Fitting residual model with {len(X_fit)} samples...")
            residual_model = ResidualModel(**self.residual_model_kwargs)
            residual_model.fit(
                X_fit,
                Y_true - Y_hat,
                x_center=x_k,
                radius=trust_radius,
            )
            residual_update = residual_model
            if residual_model.model_type == "rbf":
                rbf_condition_numbers.append(
                    float(residual_model.rbf_condition)
                )
                print(
                    f"RBF interpolation condition number: "
                    f"{residual_model.rbf_condition:.4e}"
                )
        # Step 4: Solve the trust-region subproblem to obtain a candidate
        # Note: global_surrogate has not been merged yet, so the surrogate is still the old one
        # Temporarily construct a CorrectedSurrogate for this iteration
        print(f"Constructing temporary surrogate and solving with steps_warm={steps}...")
        if isinstance(self.global_surrogate, CompositeGlobalSurrogate):
            temp_surrogate = _TempCompositeCorrectedSurrogate(
                self.global_surrogate,
                residual_update,
                update_mode=self.residual_update_mode,
            )
        else:
            temp_surrogate = _TempCorrectedSurrogate(
                self.global_surrogate,
                residual_update,
                update_mode=self.residual_update_mode,
            )
        x_cand, F_tilde_candidate = self.leader.solve(
            surrogate    = temp_surrogate,
            trust_region = self.tr,
            x_center     = x_k,
            solve_params = cur_solve_params,
        )                                                       # x_cand: (T,)
        # Step 5: Compute surrogate objective values without querying the follower
        print(f"Evaluating surrogate and actual objective...")
        F_tilde_current = self.leader.evaluate_surrogate_objective(x_k, temp_surrogate)
        F_tilde_candidate = self.leader.evaluate_surrogate_objective(x_cand, temp_surrogate)
        # Step 6: Query true objective values, which requires querying the follower
        lower_cand = self.follower.lower_solve(
            np.asarray(x_cand, dtype=float).copy()
        )
        if lower_k is None or lower_cand is None:
            raise RuntimeError("Lower-level solve failed while evaluating the true objective.")
        y_k    = lower_k["p_mg"]                      # (T,)
        y_cand = lower_cand["p_mg"]                   # (T,)
        F_current   = self.leader.evaluate_objective(x_k,    y_k)
        F_candidate = self.leader.evaluate_objective(x_cand, y_cand)
        # Step 7: Compute ρ and decide whether to accept or reject
        print(f"Computing ratio and deciding step...")
        rho = compute_ratio(
            F_current, F_candidate, F_tilde_current, F_tilde_candidate
        )
        actual_reduction = F_current - F_candidate
        if actual_reduction > self.eps_f and rho < self.eta_1:
            # Preserve a meaningful true improvement, but shrink because the
            # local model predicted its magnitude or direction poorly.
            would_be_decision = 'accept_shrink'
        else:
            would_be_decision = decide_step(rho, self.eta_1, self.eta_2)
        decision = (
            'accept_forced_shrink'
            if self.acceptance_policy == 'always_accept'
            and would_be_decision == 'reject'
            else would_be_decision
        )

        current_n_residuals = len(self.global_surrogate._residuals)
        if decision == 'reject':
            n_residuals_after = current_n_residuals
        elif self.residual_update_mode == "accumulate":
            n_residuals_after = current_n_residuals + 1
        else:
            n_residuals_after = 1
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
              f"residual_update={self.residual_update_mode} | "
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
              f"residual_update={self.residual_update_mode} | "
              f"n_residuals={n_residuals_after}")
        # Step 8: Update the state
        if decision == 'reject':
            print(f"Rejecting step {k}...")
            x_next = x_k
            # Discard the residual model without merging it into the global surrogate
        elif decision in {'accept_shrink', 'accept_forced_shrink'}:
            print(f"Accepting step {k} and shrinking...")
            x_next = x_cand
            self._commit_residual_update(residual_update)
        elif decision == 'accept_keep':
            print(f"Accepting step {k} and keeping...")
            x_next = x_cand
            self._commit_residual_update(residual_update)
        else:  # accept_expand
            print(f"Accepting step {k} and expanding...")
            x_next = x_cand
            self._commit_residual_update(residual_update)
        if self.radius_policy == 'adaptive':
            if would_be_decision in {'reject', 'accept_shrink'}:
                self.tr.shrink()
            elif would_be_decision == 'accept_expand':
                self.tr.expand()
            else:
                self.tr.keep()
        # Step 9: Check termination conditions
        # On rejection, x_next == x_k is an algorithmic action, not leader-decision convergence
        if decision == 'reject':
            converged = self.tr.get_radius() <= self.eps_R
        else:
            converged = (
                self.tr.get_radius()             <= self.eps_R or
                np.linalg.norm(x_next - x_k)    <= self.eps_x or
                abs(F_current - F_candidate)     <= self.eps_f
            )
        self.consecutive_rejections = (
            self.consecutive_rejections + 1 if decision == 'reject' else 0
        )
        rejection_limit_reached = (
            self.radius_policy == 'fixed'
            and self.acceptance_policy == 'filter'
            and self.consecutive_rejections >= self.max_consecutive_rejections
        )
        stop_reason = (
            'consecutive_rejection_limit' if rejection_limit_reached
            else 'converged' if converged else None
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
            'would_be_decision': would_be_decision,
            'actual_update_action': decision,
            'residual_committed': decision != 'reject',
            'consecutive_rejections': self.consecutive_rejections,
            'stop_reason': stop_reason,
            'radius_policy': self.radius_policy,
            'acceptance_policy': self.acceptance_policy,
            'R_before': trust_radius,
            'R_after': self.tr.get_radius(),
            'R'             : self.tr.get_radius(),
            'F_next': F_current if decision == 'reject' else F_candidate,
            'surrogate_error_current': abs(F_current - F_tilde_current),
            'surrogate_error_candidate': abs(F_candidate - F_tilde_candidate),
            'composite_queries': self.n_samples + 1,
            'cumulative_composite_queries': (k + 1) * (self.n_samples + 1),
            'residual_update_mode': self.residual_update_mode,
            'n_residuals'   : len(self.global_surrogate._residuals),
            'rbf_condition_numbers': rbf_condition_numbers,
            'max_rbf_condition': (
                max(rbf_condition_numbers)
                if rbf_condition_numbers
                else None
            ),
            'geometry_resample_attempts':
                self._last_geometry_resample_attempts,
        }
        return record, x_next, converged

    def _commit_residual_update(self, residual_update) -> None:
        if self.residual_update_mode == "accumulate":
            self.global_surrogate.merge(residual_update)
        else:
            self.global_surrogate.replace(residual_update)


class _TempCorrectedSurrogate(SurrogateModel):
    """
    Temporarily combine global_surrogate and residual_model during the current solve.
    This is not persisted and does not affect the state of global_surrogate.
    """
    def __init__(
        self,
        base: GlobalSurrogate,
        residual: ResidualModel,
        update_mode: str = "accumulate",
    ):
        """
        Input  : base     — current GlobalSurrogate before merging this iteration's residual
                 residual — ResidualModel fitted in the current iteration
        """
        self.base     = base
        self.residual = residual
        self.update_mode = update_mode

    def _base_predict(self, x: np.ndarray) -> np.ndarray:
        if self.update_mode == "accumulate":
            return self.base.predict(x)
        return self.base.predict_base(x)

    def _base_predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.update_mode == "accumulate":
            return self.base.predict_torch(x)
        return self.base.predict_base_torch(x)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        ỹ(x) = base.predict(x) + residual.predict(x)
        Input  : x     — ndarray (T,) or (n, T)
        Output : y_hat — ndarray (T,) or (n, T)
        """
        single = (x.ndim == 1)
        base_pred = self._base_predict(x)
        if single:
            res_pred = self.residual.predict(x)
        else:
            res_pred = np.array([self.residual.predict(xi) for xi in x])
        return base_pred + res_pred

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable corrected prediction for the leader subproblem.
        """
        return self._base_predict_torch(x) + self.residual.predict_torch(x)

    @property
    def _residuals(self):
        """
        Forward the historical residual list from base and append the current residual.
        Used by _add_residual_gradient for iteration.
        """
        if self.update_mode == "accumulate":
            return self.base._residuals + [self.residual]
        return [self.residual]
    def update(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass


class _TempCompositeCorrectedSurrogate(SurrogateModel):
    """Temporary per-follower corrections for a composite surrogate."""

    def __init__(
        self,
        base: CompositeGlobalSurrogate,
        residuals: Sequence[ResidualModel],
        update_mode: str = "accumulate",
    ) -> None:
        self.base = base
        self.residuals = list(residuals)
        self.n_followers = base.n_followers
        self.T = base.T
        self.decision_size = base.decision_size
        self.update_mode = update_mode
        if len(self.residuals) != self.n_followers:
            raise ValueError("One residual model is required per follower.")

    @property
    def _residuals(self):
        if self.update_mode == "accumulate":
            return self.base._residuals + [tuple(self.residuals)]
        return [tuple(self.residuals)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        prices, single = self.base._reshape_numpy(x)
        if self.update_mode == "accumulate":
            base_prediction = self.base.predict(x)
        else:
            base_prediction = self.base.predict_base(x)
        base_batch = np.asarray(base_prediction).reshape(
            prices.shape[0], self.n_followers, self.T
        )
        correction = np.stack(
            [
                np.asarray(
                    [
                        self.residuals[index].predict(sample)
                        for sample in prices[:, index, :]
                    ]
                )
                for index in range(self.n_followers)
            ],
            axis=1,
        )
        output = (base_batch + correction).reshape(prices.shape[0], -1)
        return output[0] if single else output

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        matrix_input = tuple(x.shape) == (self.n_followers, self.T)
        flat_input = x.ndim == 1
        if matrix_input:
            prices = x
        elif flat_input:
            prices = x.reshape(self.n_followers, self.T)
        else:
            raise ValueError(
                "Composite leader prediction expects one flattened or matrix price."
            )
        if self.update_mode == "accumulate":
            base = self.base.predict_torch(prices)
        else:
            base = self.base.predict_base_torch(prices)
        base = base.reshape(self.n_followers, self.T)
        correction = torch.stack(
            [
                residual.predict_torch(prices[index])
                for index, residual in enumerate(self.residuals)
            ]
        )
        output = base + correction
        return output if matrix_input else output.reshape(-1)

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
        "x_init": result.get("x_init"),
        "params": result.get("params", {}),
        "history": result["history"],
        "converged": result["converged"],
        "stop_reason": result.get("stop_reason"),
        "n_iter": result["n_iter"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result_to_save), f, indent=2)

    summary_keys = [
        "k", "F_current", "F_candidate", "F_tilde_current",
        "F_tilde_candidate", "F_next", "rho", "decision",
        "would_be_decision", "actual_update_action", "residual_committed",
        "consecutive_rejections", "stop_reason",
        "radius_policy", "acceptance_policy", "R_before", "R_after",
        "R", "composite_queries", "cumulative_composite_queries",
        "surrogate_error_current", "surrogate_error_candidate", "n_residuals",
        "rbf_condition_numbers", "max_rbf_condition",
        "geometry_resample_attempts",
    ]
    summary_rows = [
        {key: record.get(key) for key in summary_keys}
        for record in result["history"]
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def save_trust_region_true_responses(
    true_responses: dict,
    output_dir: str | os.PathLike | None = None,
    tag: str = TRUE_RESPONSE_TAG,
) -> dict:
    """Save high-fidelity pairs as one tagged dataset per follower ID.

    These files sit next to the original per-instance training
    data and use a distinct tag so they can retrain an NN solver without
    overwriting the original sample files.
    """
    dest = Path(DEFAULT_TRUE_RESPONSE_DIR if output_dir is None else output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, str]] = {}
    for scenario, payload in true_responses.items():
        scenario_name = _normalize_scenario_name(scenario)
        x_values = np.asarray(payload["X"], dtype=float)
        y_values = np.asarray(payload["Y"], dtype=float)
        x_path = dest / f"X_{scenario_name}_{tag}.npy"
        y_path = dest / f"Y_{scenario_name}_{tag}.npy"
        np.save(x_path, x_values)
        np.save(y_path, y_values)
        saved[scenario_name] = {"X": str(x_path), "Y": str(y_path)}
        print(
            f"Saved {len(x_values)} {scenario_name} trust-region samples to "
            f"{x_path.name} and {y_path.name}."
        )
    return saved


@dataclass
class TrustRegionExperimentConfig:
    """Configuration shared by scripts and notebooks."""

    upper_net: pp.pandapowerNet
    lower_nets: Sequence[pp.pandapowerNet]
    mg_buses: Sequence[int]
    scenarios: Sequence[str]
    follower_kwargs: Sequence[Mapping[str, Any]] | None = None
    T: int = 24
    seed: int = DEFAULT_SEED
    p_net_min_mw: float = -15.0
    p_net_max_mw: float = 15.0
    q_net_min_mvar: float = -15.0
    q_net_max_mvar: float = 15.0
    device: str = "cpu"
    x0: np.ndarray | None = None
    tr_params: Dict[str, Any] = field(default_factory=dict)
    algo_params: Dict[str, Any] = field(default_factory=dict)
    radius_policy: str = "adaptive"
    acceptance_policy: str = "filter"
    fixed_radius: float = 0.05
    max_consecutive_rejections: int = 10
    residual_model_params: Dict[str, Any] = field(default_factory=dict)
    map_nets: Sequence[nn.Module] | None = None
    lower_time_limit: float | None = None
    lower_mip_gap: float | None = None
    save_results: bool = True
    output_dir: str = "results"
    # Disabled by default: do not accumulate extra sample copies for export.
    save_true_responses: bool = False
    true_response_dir: str = str(DEFAULT_TRUE_RESPONSE_DIR)
    true_response_tag: str = TRUE_RESPONSE_TAG
    copt_initialize_network_state: bool = False


def build_trust_region_experiment(
    config: TrustRegionExperimentConfig,
) -> tuple[TRResidualController, dict]:
    """Validate configuration and construct a reusable experiment controller."""
    scenarios, scenario_types, follower_kwargs = resolve_follower_configuration(
        config.scenarios, config.follower_kwargs
    )
    lower_nets = list(config.lower_nets)
    mg_buses = [int(value) for value in config.mg_buses]
    n_followers = len(scenarios)
    if n_followers == 0:
        raise ValueError("At least one follower scenario is required.")
    if len(lower_nets) != n_followers or len(mg_buses) != n_followers:
        raise ValueError(
            "scenarios, lower_nets, and mg_buses must have the same nonzero length."
        )
    if config.T <= 0:
        raise ValueError("T must be positive.")

    random_seed = set_global_seed(config.seed)
    independent_prices = n_followers > 1
    upper_constants = extract_constants(
        input={
            "mg_buses": mg_buses,
            "independent_prices": independent_prices,
            "seed": random_seed,
            "p_net_min_mw": config.p_net_min_mw,
            "p_net_max_mw": config.p_net_max_mw,
            "q_net_min_mvar": config.q_net_min_mvar,
            "q_net_max_mvar": config.q_net_max_mvar,
        },
        net=config.upper_net,
        T=config.T,
    )

    if config.map_nets is None:
        map_nets = load_l2o_models(
            scenarios, device=torch.device(config.device)
        )
    else:
        map_nets = list(config.map_nets)
        if len(map_nets) != n_followers:
            raise ValueError(
                f"Expected {n_followers} map_nets, got {len(map_nets)}."
            )

    if n_followers == 1:
        single_kwargs = {
            "verbose": False,
            "time_limit": config.lower_time_limit,
            "mip_gap": config.lower_mip_gap,
            **follower_kwargs[0],
        }
        follower = build_scenario_follower(
            scenario_types[0],
            T=config.T,
            net=lower_nets[0],
            **single_kwargs,
        )
        follower.follower_id = scenarios[0]
        global_surrogate: SurrogateModel = GlobalSurrogate(
            map_net=map_nets[0], device=config.device
        )
    else:
        follower = build_composite_follower(
            scenarios,
            lower_nets,
            T=config.T,
            follower_kwargs=follower_kwargs,
            verbose=False,
            time_limit=config.lower_time_limit,
            mip_gap=config.lower_mip_gap,
        )
        global_surrogate = CompositeGlobalSurrogate(
            [
                GlobalSurrogate(map_net=map_net, device=config.device)
                for map_net in map_nets
            ],
            T=config.T,
        )

    tr_params = {
        "R_0": 0.05,
        "R_max": 1.0,
        "gamma_dec": 0.7,
        "gamma_inc": 1.5,
        "eta_1": 0.1,
        "eta_2": 0.9,
        **config.tr_params,
    }
    if config.radius_policy == "fixed":
        tr_params["R_0"] = config.fixed_radius
    decision_size = n_followers * config.T
    x0 = (
        np.full(decision_size, 0.75, dtype=float)
        if config.x0 is None
        else np.asarray(config.x0, dtype=float).reshape(-1)
    )
    if x0.size != decision_size:
        raise ValueError(
            f"x0 must contain {decision_size} prices, got {x0.size}."
        )
    algo_params = {
        "n_samples": 60,
        "K_max": 500,
        "x0": x0,
        "eps_x": 1e-4,
        "eps_f": 1e-4,
        "eps_R": 1e-4,
        "sampling_method": "lhs",
        "residual_update_mode": "replace",
        "rbf_geometry_condition_max": 1e12,
        "rbf_geometry_max_attempts": 6,
        "random_seed": random_seed,
        "solve_params": {
            "steps_init": 300,
            "steps_warm": 300,
            "lr": 1e-2,
            "penalty_weights": {
                "eq": 1000.0,
                "ineq": 100.0,
                "soc": 100.0,
                "bound": 10.0,
            },
            "tr_weight": 1e6,
            "verbose": True,
        },
        **config.algo_params,
    }
    algo_params["radius_policy"] = config.radius_policy
    algo_params["acceptance_policy"] = config.acceptance_policy
    algo_params["fixed_radius"] = config.fixed_radius
    algo_params["max_consecutive_rejections"] = config.max_consecutive_rejections
    solve_params = dict(algo_params.get("solve_params", {}))
    solve_params["copt_initialize_network_state"] = bool(
        config.copt_initialize_network_state
    )
    algo_params["solve_params"] = solve_params
    # Keep this experiment-level switch authoritative even if algo_params was
    # assembled elsewhere. It controls both in-memory retention and .npy export.
    algo_params["save_true_responses"] = bool(config.save_true_responses)
    algo_params["scenarios"] = scenarios
    algo_params["n_samples_includes_center"] = True
    residual_params = {
        "model_type": "rbf",
        "lasso_alpha": 1e-4,
        "lasso_max_iter": 5000,
        "lasso_tol": 1e-8,
        "rbf_kernel": "cubic",
        "rbf_smoothing": 1e-6,
        "rbf_degree": 1,
        "rbf_scale": "trust_radius",
        "rbf_epsilon": None,
        **config.residual_model_params,
    }
    residual_model = ResidualModel(**residual_params)
    controller = TRResidualController(
        leader=DistributionNetworkLeader(upper_constants, device=config.device),
        follower=follower,
        global_surrogate=global_surrogate,
        residual_model=residual_model,
        tr_params=tr_params,
        algo_params=algo_params,
    )
    metadata = {
        "T": config.T,
        "scenarios": scenarios,
        "scenario_types": scenario_types,
        "follower_kwargs": follower_kwargs,
        "mg_buses": mg_buses,
        "independent_prices": independent_prices,
        "lower_time_limit": config.lower_time_limit,
        "lower_mip_gap": config.lower_mip_gap,
        "tr_params": tr_params,
        "algo_params": algo_params,
        "residual_model": residual_params,
    }
    return controller, metadata


def run_trust_region_experiment(config: TrustRegionExperimentConfig) -> dict:
    """Build, run, optionally save, and return a configured experiment."""
    controller, metadata = build_trust_region_experiment(config)
    start = time.time()
    result = controller.run()
    result["elapsed_seconds"] = time.time() - start
    result["params"] = metadata
    saved_paths: dict[str, Any] = {}
    if config.save_results:
        saved_paths.update(
            save_trust_region_result(result, output_dir=config.output_dir)
        )
    if config.save_true_responses:
        saved_paths["true_responses"] = save_trust_region_true_responses(
            result.get("true_responses", {}),
            output_dir=config.true_response_dir,
            tag=config.true_response_tag,
        )
    if saved_paths:
        result["saved_paths"] = saved_paths
    return result


def main(seed: int = DEFAULT_SEED, scenario: str = "SB") -> dict:
    """Run the original single-SB demonstration through the experiment API."""
    config = TrustRegionExperimentConfig(
        upper_net=create_case33bw_20kv(),
        lower_nets=[pn.create_cigre_network_lv()],
        mg_buses=[18],
        scenarios=[scenario],
        seed=seed,
    )
    result = run_trust_region_experiment(config)
    print(f"Time taken to run {result['elapsed_seconds']:.4f} seconds.")
    print(f"Converged : {result['converged']}")
    print(f"Iterations: {result['n_iter']}")
    print(f"Optimal pi: {result['x_opt']}")
    if result.get("saved_paths"):
        saved_paths = result["saved_paths"]
        if saved_paths.get("json"):
            print(f"Saved full result to: {saved_paths['json']}")
        if saved_paths.get("csv"):
            print(f"Saved iteration summary to: {saved_paths['csv']}")
        true_response_paths = saved_paths.get("true_responses", {})
        for scenario, paths in true_response_paths.items():
            print(
                f"Saved {scenario} trust-region samples to: "
                f"{paths['X']} and {paths['Y']}"
            )
    return result
