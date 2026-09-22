"""Reusable runner and analysis for the 2x2 trust-region mechanism ablation.

The companion notebook runs one editable seed at a time. Completed groups are
saved separately and skipped on a resumed invocation. The price scenario stays
at seed 42 so the supplied original-NN objective remains comparable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib-cache")
)

import numpy as np
import pandas as pd
import pandapower.networks as pn
from scipy.stats import t as student_t

from stackelberg.followers.bg import create_cigre_residential_network
from stackelberg.followers.ip import create_industrial_park_network
from stackelberg.network import create_case33bw_20kv
from stackelberg.trust_region import (
    TrustRegionExperimentConfig,
    run_trust_region_experiment,
)


GROUPS = {
    "adaptive_filter": ("adaptive", "filter"),
    "adaptive_always_accept": ("adaptive", "always_accept"),
    "fixed_filter": ("fixed", "filter"),
    "fixed_always_accept": ("fixed", "always_accept"),
}
SCENARIOS = ["SB1", "SB2", "BG1", "BG2", "IP1"]
MG_BUSES = [17, 32, 15, 21, 1]
T = 24
PROBLEM_SEED = 42


def load_warm_start(path: Path) -> tuple[np.ndarray, float]:
    history = pd.read_csv(path)
    if history.empty:
        raise ValueError(f"NN warm-start history is empty: {path}")
    price_columns = sorted(
        (col for col in history if col.startswith("pi_")),
        key=lambda col: int(col[3:]),
    )
    if len(price_columns) != len(SCENARIOS) * T:
        raise ValueError(f"Expected {len(SCENARIOS) * T} pi columns in {path}.")
    last = history.sort_values("iteration").iloc[-1]
    return last[price_columns].to_numpy(dtype=float), float(last["true_loss"])


def build_config(
    warm_start: np.ndarray,
    *,
    problem_seed: int,
    trial_seed: int,
    radius_policy: str,
    acceptance_policy: str,
    max_iterations: int,
    steps_warm: int,
    output_dir: Path,
) -> TrustRegionExperimentConfig:
    return TrustRegionExperimentConfig(
        upper_net=create_case33bw_20kv(),
        lower_nets=[
            pn.create_cigre_network_lv(),
            pn.create_cigre_network_lv(),
            create_cigre_residential_network(),
            create_cigre_residential_network(),
            create_industrial_park_network(),
        ],
        mg_buses=MG_BUSES,
        scenarios=SCENARIOS,
        T=T,
        seed=problem_seed,
        device="cpu",
        x0=warm_start,
        lower_time_limit=30.0,
        lower_mip_gap=1e-3,
        radius_policy=radius_policy,
        acceptance_policy=acceptance_policy,
        fixed_radius=0.05,
        max_consecutive_rejections=10,
        tr_params={
            "R_0": 0.05,
            "R_max": 1.0,
            "gamma_dec": 0.7,
            "gamma_inc": 1.5,
            "eta_1": 0.1,
            "eta_2": 0.75,
        },
        algo_params={
            "n_samples": 60,
            "K_max": max_iterations,
            "sampling_method": "lhs",
            "residual_update_mode": "accumulate",
            "rbf_geometry_condition_max": 1e12,
            "random_seed": trial_seed,
            "solve_params": {
                "steps_init": 0,
                "steps_warm": steps_warm,
                "lr": 1e-2,
                "penalty_weights": {
                    "eq": 1000.0,
                    "ineq": 100.0,
                    "soc": 100.0,
                    "bound": 10.0,
                },
                "tr_weight": 1e6,
                "verbose": False,
            },
        },
        residual_model_params={
            "model_type": "rbf",
            "rbf_kernel": "cubic",
            "rbf_smoothing": 1e-3,
            "rbf_degree": 1,
            "rbf_scale": "trust_radius",
        },
        save_results=True,
        output_dir=str(output_dir),
        save_true_responses=False,
    )


def run_single_seed(
    *,
    seed: int,
    warm_start_csv: Path,
    output_dir: Path,
    max_iterations: int = 60,
    steps_warm: int = 2000,
) -> None:
    """Run all four groups for one seed, resuming completed groups if present."""
    if max_iterations < 1 or steps_warm < 1:
        raise ValueError("max_iterations and steps_warm must be positive.")
    seed = int(seed)
    warm_start_csv = Path(warm_start_csv)
    output_dir = Path(output_dir)
    warm_start, nn_baseline = load_warm_start(warm_start_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, (radius_policy, acceptance_policy) in GROUPS.items():
        run_dir = output_dir / f"seed_{seed}" / group
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "problem_seed": PROBLEM_SEED,
                "max_iterations": max_iterations,
                "steps_warm": steps_warm,
                "max_consecutive_rejections": 10,
                "warm_start_csv": str(warm_start_csv.resolve()),
            }
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing run has different settings: {manifest_path}")
            print(f"Skipping completed run: seed={seed}, group={group}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(
            warm_start,
            problem_seed=PROBLEM_SEED,
            trial_seed=seed,
            radius_policy=radius_policy,
            acceptance_policy=acceptance_policy,
            max_iterations=max_iterations,
            steps_warm=steps_warm,
            output_dir=run_dir,
        )
        print(f"Running seed={seed}, group={group}", flush=True)
        result = run_trust_region_experiment(config)
        manifest = {
            "seed": seed,
            "problem_seed": PROBLEM_SEED,
            "group": group,
            "radius_policy": radius_policy,
            "acceptance_policy": acceptance_policy,
            "warm_start_csv": str(warm_start_csv.resolve()),
            "nn_baseline": nn_baseline,
            "max_iterations": max_iterations,
            "steps_warm": steps_warm,
            "max_consecutive_rejections": 10,
            "elapsed_seconds": result["elapsed_seconds"],
            "result_json": str(Path(result["saved_paths"]["json"]).resolve()),
            "history_csv": str(Path(result["saved_paths"]["csv"]).resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _curve(history: list[dict], max_iterations: int) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        raise ValueError("An ablation result has no iterations.")
    n_queries = int(history[0].get("composite_queries", 61))
    baseline = float(history[0]["F_current"])
    values = np.empty(max_iterations + 1, dtype=float)
    values[0] = baseline
    best = baseline
    for k in range(1, max_iterations + 1):
        if k <= len(history):
            record = history[k - 1]
            best = min(best, float(record["F_current"]), float(record["F_candidate"]))
        values[k] = best
    return np.arange(max_iterations + 1) * n_queries, values


def _confidence_interval(differences: np.ndarray) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"n": n, "mean": mean, "ci_low": float("nan"), "ci_high": float("nan")}
    half_width = float(
        student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    )
    return {"n": n, "mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width}


def analyze(
    output_dir: Path,
    max_iterations: int,
    *,
    allow_partial: bool = False,
    target: float | None = None,
) -> None:
    runs: dict[tuple[int, str], tuple[dict, dict]] = {}
    for manifest_path in output_dir.glob("seed_*/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(Path(manifest["result_json"]).read_text(encoding="utf-8"))
        runs[(int(manifest["seed"]), manifest["group"])] = (manifest, result)
    seeds = sorted({seed for seed, _ in runs})
    paired_seeds = [
        seed for seed in seeds if all((seed, group) in runs for group in GROUPS)
    ]
    if not paired_seeds:
        if allow_partial:
            print("No complete four-group seed yet; analysis deferred.")
            return
        raise ValueError("No seed has all four completed groups.")
    if not allow_partial and paired_seeds != seeds:
        raise ValueError("Some seeds do not have all four groups.")
    problem_seeds = {runs[key][0]["problem_seed"] for key in runs if key[0] in paired_seeds}
    warm_starts = {runs[key][0]["warm_start_csv"] for key in runs if key[0] in paired_seeds}
    if len(problem_seeds) != 1 or len(warm_starts) != 1:
        raise ValueError("Paired runs must share one problem seed and warm-start file.")
    max_iterations = max(int(runs[(seed, group)][0]["max_iterations"])
                         for seed in paired_seeds for group in GROUPS)
    target_is_derived = target is None
    if target_is_derived:
        initial = np.mean([
            runs[(seed, "adaptive_filter")][1]["history"][0]["F_current"]
            for seed in paired_seeds
        ])
        pooled_best = min(
            min(float(row["F_candidate"]) for row in runs[(seed, group)][1]["history"])
            for seed in paired_seeds for group in GROUPS
        )
        target = float(initial - 0.9 * max(initial - pooled_best, 0.0))
    for seed in paired_seeds:
        initial_values = [
            float(runs[(seed, group)][1]["history"][0]["F_current"])
            for group in GROUPS
        ]
        if max(initial_values) - min(initial_values) > 1e-3:
            raise ValueError(f"Seed {seed} does not share one initial true objective.")
    rows = []
    curves = {}
    for seed in paired_seeds:
        for group in GROUPS:
            manifest, result = runs[(seed, group)]
            history = result["history"]
            budgets, best = _curve(history, max_iterations)
            curves[(seed, group)] = best
            hit = np.flatnonzero(best <= target)
            rows.append({
                "seed": seed,
                "group": group,
                "best_at_budget": best[-1],
                "final_returned": float(history[-1].get(
                    "F_next",
                    history[-1]["F_current"] if history[-1]["decision"] == "reject"
                    else history[-1]["F_candidate"],
                )),
                "auc_normalized": float(np.trapezoid(best, budgets) / budgets[-1]),
                "queries_to_target": float(budgets[hit[0]]) if len(hit) else np.nan,
                "elapsed_seconds": float(manifest["elapsed_seconds"]),
                "iterations": len(history),
                "stop_reason": result.get("stop_reason", history[-1].get("stop_reason")),
                "queries_used": int(history[-1].get("cumulative_composite_queries", len(history) * 61)),
                "would_reject_rate": float(np.mean([
                    row.get("would_be_decision", row["decision"]) == "reject"
                    for row in history
                ])),
                "actual_reject_rate": float(np.mean([
                    row["decision"] == "reject" for row in history
                ])),
                "mean_candidate_surrogate_error": float(np.mean([
                    abs(row["F_candidate"] - row["F_tilde_candidate"])
                    for row in history
                ])),
                "final_radius": float(history[-1].get("R_after", history[-1]["R"])),
                "nn_external_baseline": float(manifest["nn_baseline"]),
            })
    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "paired_seeds": paired_seeds,
        "problem_seed": next(iter(problem_seeds)),
        "warm_start_csv": next(iter(warm_starts)),
        "max_iterations": max_iterations,
        "queries_per_iteration": int(budgets[1]),
        "target_objective": target,
        "target_definition": (
            "90% of pooled best improvement from mean initial objective"
            if target_is_derived else "user-specified objective"
        ),
        "best_so_far_includes_rejected_evaluated_candidates": True,
        "effect_sign": "Positive adaptive_benefit/filter_benefit means lower objective with that mechanism.",
        "interaction_definition": "(fixed_always_accept - fixed_filter) - (adaptive_always_accept - adaptive_filter)",
    }, indent=2), encoding="utf-8")
    effects = []
    paired_differences = []
    for metric in ("best_at_budget", "auc_normalized", "final_returned"):
        wide = metrics.pivot(index="seed", columns="group", values=metric)
        af = wide["adaptive_filter"]
        aa = wide["adaptive_always_accept"]
        ff = wide["fixed_filter"]
        fa = wide["fixed_always_accept"]
        contrasts = {
            "adaptive_benefit": ((ff + fa) - (af + aa)) / 2,
            "filter_benefit": ((aa + fa) - (af + ff)) / 2,
            "interaction": (fa - ff) - (aa - af),
        }
        for effect, values in contrasts.items():
            effects.append({"metric": metric, "effect": effect,
                            **_confidence_interval(values.to_numpy())})
            paired_differences.extend({
                "seed": int(seed), "metric": metric, "effect": effect,
                "difference": float(value),
            } for seed, value in values.items())
    pd.DataFrame(effects).to_csv(output_dir / "paired_effects.csv", index=False)
    pd.DataFrame(paired_differences).to_csv(
        output_dir / "paired_differences.csv", index=False
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for group in GROUPS:
        mean_curve = np.mean([curves[(seed, group)] for seed in paired_seeds], axis=0)
        ax.plot(budgets, mean_curve, label=group.replace("_", " "))
    ax.axhline(metrics["nn_external_baseline"].iloc[0], color="gray",
               linestyle="--", label="Original NN (external baseline)")
    ax.set(xlabel="Composite high-fidelity queries", ylabel="Best observed true objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "budget_curves.png", dpi=180)
    plt.close(fig)
    print(f"Analyzed {len(paired_seeds)} paired seeds in {output_dir}")
