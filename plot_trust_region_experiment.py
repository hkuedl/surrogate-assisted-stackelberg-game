"""Plot the newest NN, CMA-ES, and trust-region experiment results.

Reads timestamped files in ``results/`` and writes figures next to them.
This script does not re-run any solver.

Example::

    python -u plot_trust_region_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def print_frame(frame: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    print(frame.to_string())
    print()


def latest_file(results_dir: Path, pattern: str) -> Path:
    candidates = list(results_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_nn_result(results_dir: Path) -> dict[str, Any]:
    history_path = latest_file(results_dir, "NN_solver_history_*.csv")
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"NN solver history is empty: {history_path}")

    last_row = history.sort_values("iteration").iloc[-1]
    pi_columns = sorted(
        (column for column in history.columns if column.startswith("pi_")),
        key=lambda column: int(column.removeprefix("pi_")),
    )
    prices = last_row[pi_columns].to_numpy(dtype=float)
    print(f"Loaded NN history: {history_path}")
    print(f"NN saved iteration: {int(last_row['iteration'])}")
    return {
        "path": history_path,
        "prices": prices,
        "true_obj": float(last_row["true_loss"]) if "true_loss" in last_row.index else np.nan,
        "runtime_seconds": np.nan,
    }


def load_cma_result(results_dir: Path) -> dict[str, Any]:
    json_path = latest_file(results_dir, "zero_solver_result_*.json")
    payload = load_json(json_path)
    timestamp = json_path.stem.removeprefix("zero_solver_result_")
    history_path = results_dir / f"zero_solver_history_{timestamp}.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Matching CMA-ES history not found: {history_path}")

    print(f"Loaded CMA-ES result: {json_path}")
    print(f"Loaded CMA-ES history: {history_path}")
    return {
        "path": json_path,
        "history_path": history_path,
        "scenarios": [str(value) for value in payload["scenarios"]],
        "prices": np.asarray(payload["optimal_pi"], dtype=float),
        "optimal_value": float(payload["optimal_value"]),
        "runtime_seconds": float(payload.get("runtime_seconds", np.nan)),
        "n_interactions": int(payload.get("n_interactions", 0)),
        "interaction_history": pd.read_csv(history_path),
    }


def load_trust_region_result(results_dir: Path) -> dict[str, Any]:
    json_path = latest_file(results_dir, "trust_region_result_*.json")
    payload = load_json(json_path)
    params = payload.get("params", {})
    algo_params = params.get("algo_params", {})
    timestamp = json_path.stem.removeprefix("trust_region_result_")
    history_path = results_dir / f"trust_region_history_{timestamp}.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame(payload["history"])

    print(f"Loaded trust-region result: {json_path}")
    if history_path.is_file():
        print(f"Loaded trust-region history: {history_path}")
    return {
        "path": json_path,
        "history_path": history_path if history_path.is_file() else json_path,
        "scenarios": [str(value) for value in params.get("scenarios", [])],
        "T": int(params.get("T", 24)),
        "n_samples": int(algo_params.get("n_samples", 60)),
        "n_samples_includes_center": bool(
            algo_params.get("n_samples_includes_center", False)
        ),
        "prices": np.asarray(payload["x_opt"], dtype=float),
        "history": history,
        "converged": bool(payload.get("converged", False)),
        "n_iter": int(payload.get("n_iter", len(history))),
        "elapsed_seconds": payload.get("elapsed_seconds", np.nan),
    }


def reshape_prices(prices: np.ndarray, n_followers: int, horizon: int) -> np.ndarray:
    array = np.asarray(prices, dtype=float)
    if array.size != n_followers * horizon:
        raise ValueError(
            f"Expected {n_followers * horizon} prices, got shape {array.shape}."
        )
    return array.reshape(n_followers, horizon)


def trust_region_objective(history: pd.DataFrame) -> float:
    last_iteration = history.iloc[-1]
    if str(last_iteration["decision"]) == "reject":
        return float(last_iteration["F_current"])
    return float(last_iteration["F_candidate"])


def save_history_plot(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    history.plot(x="k", y=["F_current", "F_candidate"], ax=axes[0], title="Objective")
    history.plot(x="k", y="rho", ax=axes[1], title="Trust ratio")
    history.plot(x="k", y="R", ax=axes[2], title="Trust-region radius")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trust-region history plot: {output_path}")


def save_query_plot(
    nn_true_loss: float,
    zero_history: pd.DataFrame,
    tr_history: pd.DataFrame,
    queries_per_iteration: int,
    output_path: Path,
) -> None:
    zero_plot_df = zero_history.copy().sort_values("interaction").reset_index(drop=True)
    tr_plot_df = tr_history.copy()
    tr_plot_df["accepted"] = tr_plot_df["decision"].astype(str).str.startswith("accept")
    tr_plot_df["cumulative_composite_queries"] = (
        tr_plot_df["k"] + 1
    ) * queries_per_iteration
    # Both high-fidelity methods are warm-started from the NN solution. Treat
    # the NN true loss as the query-zero incumbent so that worse evaluations
    # do not appear above the NN baseline in the best-so-far comparison.
    common_initial_objective = float(nn_true_loss)

    zero_plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "interaction": [0],
                    "upper_objective": [common_initial_objective],
                }
            ),
            zero_plot_df[["interaction", "upper_objective"]],
        ],
        ignore_index=True,
    )
    zero_plot_df["best_so_far_loss"] = np.minimum(
        zero_plot_df["upper_objective"].cummin(),
        nn_true_loss,
    )

    best_accepted = common_initial_objective
    tr_best_values = []
    for _, row in tr_plot_df.iterrows():
        if row["accepted"]:
            best_accepted = min(best_accepted, float(row["F_candidate"]))
        tr_best_values.append(best_accepted)
    tr_plot_df["best_true_objective"] = np.minimum(
        tr_best_values,
        nn_true_loss,
    )
    tr_plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "cumulative_composite_queries": [0],
                    "best_true_objective": [common_initial_objective],
                }
            ),
            tr_plot_df[["cumulative_composite_queries", "best_true_objective"]],
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        zero_plot_df["interaction"],
        zero_plot_df["best_so_far_loss"],
        linewidth=2.5,
        label="CMA-ES best-so-far",
    )
    ax.plot(
        tr_plot_df["cumulative_composite_queries"],
        tr_plot_df["best_true_objective"],
        linewidth=2.5,
        label="Trust region best accepted",
    )
    ax.axhline(
        nn_true_loss,
        color="tab:purple",
        linestyle="--",
        linewidth=2.5,
        label=f"NN solver true loss = {nn_true_loss:.3f}",
    )
    ax.scatter(
        [0],
        [common_initial_objective],
        color="black",
        s=70,
        zorder=5,
        label="Shared initial point",
    )
    x_max = max(
        int(zero_plot_df["interaction"].max()),
        int(tr_plot_df["cumulative_composite_queries"].max()),
    )
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Cumulative composite high-fidelity queries")
    ax.set_ylabel("Best upper objective / Loss")
    ax.set_title("Composite-query accounting")
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved composite-query plot: {output_path}")


def main() -> None:
    _configure_stdio()
    results_dir = RESULTS_DIR
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    nn_result = load_nn_result(results_dir)
    zero_result = load_cma_result(results_dir)
    tr_result = load_trust_region_result(results_dir)

    scenarios = tr_result["scenarios"] or zero_result["scenarios"]
    horizon = int(tr_result["T"])
    n_followers = len(scenarios)
    nn_prices = reshape_prices(nn_result["prices"], n_followers, horizon)
    zero_prices = reshape_prices(zero_result["prices"], n_followers, horizon)
    tr_prices = reshape_prices(tr_result["prices"], n_followers, horizon)
    tr_objective = trust_region_objective(tr_result["history"])
    queries_per_iteration = tr_result["n_samples"] + (
        1 if tr_result["n_samples_includes_center"] else 2
    )
    tr_sampling_queries = tr_result["n_iter"] * queries_per_iteration

    comparison = pd.DataFrame(
        [
            {
                "method": "NN solver",
                "true_objective": nn_result["true_obj"],
                "runtime_seconds": nn_result["runtime_seconds"],
                "composite_queries": np.nan,
                "individual_follower_solves": np.nan,
            },
            {
                "method": "CMA-ES",
                "true_objective": zero_result["optimal_value"],
                "runtime_seconds": zero_result["runtime_seconds"],
                "composite_queries": zero_result["n_interactions"],
                "individual_follower_solves": zero_result["n_interactions"] * n_followers,
            },
            {
                "method": "Trust region",
                "true_objective": tr_objective,
                "runtime_seconds": tr_result["elapsed_seconds"],
                "composite_queries": tr_sampling_queries,
                "individual_follower_solves": tr_sampling_queries * n_followers,
            },
        ]
    ).set_index("method")

    print("Trust region converged:", tr_result["converged"])
    print("Trust region iterations:", tr_result["n_iter"])
    print_frame(comparison, "Method comparison")
    print_frame(
        pd.concat(
            {
                "NN solver": pd.DataFrame(nn_prices, index=scenarios),
                "CMA-ES": pd.DataFrame(zero_prices, index=scenarios),
                "Trust region": pd.DataFrame(tr_prices, index=scenarios),
            },
            names=["method", "scenario"],
        ),
        "Optimal prices",
    )
    print_frame(
        tr_result["history"][
            [
                "k",
                "F_current",
                "F_candidate",
                "rho",
                "decision",
                "R",
                "max_rbf_condition",
                "geometry_resample_attempts",
            ]
        ].tail(),
        "Trust-region history tail",
    )

    save_history_plot(tr_result["history"], results_dir / "trust_region_history.png")
    save_query_plot(
        nn_true_loss=float(nn_result["true_obj"]),
        zero_history=zero_result["interaction_history"],
        tr_history=tr_result["history"],
        queries_per_iteration=queries_per_iteration,
        output_path=results_dir / "composite_query_comparison.png",
    )


if __name__ == "__main__":
    main()
