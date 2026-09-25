"""Fast analysis checks using saved-result-shaped synthetic runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stackelberg.tr_ablation as tr_ablation
from stackelberg.tr_ablation import GROUPS, _curve, analyze


def test_curve_fills_early_stop_to_common_budget() -> None:
    budgets, best = _curve([{
        "F_current": 10.0,
        "F_candidate": 8.0,
        "composite_queries": 61,
    }], 3)

    assert budgets.tolist() == [0, 61, 122, 183]
    assert best.tolist() == [10.0, 8.0, 8.0, 8.0]


def test_single_seed_runner_runs_four_groups_and_resumes(tmp_path, monkeypatch) -> None:
    configs = []
    monkeypatch.setattr(
        tr_ablation, "load_warm_start",
        lambda path: (np.full(120, 0.75), -98.0),
    )

    def fake_build_config(warm_start, **kwargs):
        configs.append(kwargs)
        return kwargs

    monkeypatch.setattr(tr_ablation, "build_config", fake_build_config)
    monkeypatch.setattr(
        tr_ablation,
        "run_trust_region_experiment",
        lambda config: {
            "elapsed_seconds": 1.0,
            "saved_paths": {
                "json": str(config["output_dir"] / "result.json"),
                "csv": str(config["output_dir"] / "history.csv"),
            },
        },
    )
    kwargs = {
        "seed": 47,
        "warm_start_csv": Path("example.csv"),
        "output_dir": tmp_path,
        "max_iterations": 2,
        "steps_warm": 3,
    }

    tr_ablation.run_single_seed(**kwargs)
    tr_ablation.run_single_seed(**kwargs)

    assert len(configs) == 4
    assert {entry["trial_seed"] for entry in configs} == {47}
    assert {entry["problem_seed"] for entry in configs} == {42}
    assert {entry["radius_policy"] for entry in configs} == {"adaptive", "fixed"}
    assert {entry["acceptance_policy"] for entry in configs} == {"filter", "always_accept"}
    assert len(list(tmp_path.glob("seed_47/*/manifest.json"))) == 4


def test_analysis_writes_paired_effects_and_budget_curve(tmp_path) -> None:
    candidates = {
        "adaptive_filter": 8.0,
        "adaptive_always_accept": 8.5,
        "fixed_filter": 9.0,
        "fixed_always_accept": 9.5,
    }
    for seed in (42, 43):
        for group in GROUPS:
            run_dir = tmp_path / f"seed_{seed}" / group
            run_dir.mkdir(parents=True)
            result_path = run_dir / "result.json"
            candidate = candidates[group] + (seed - 42) * 0.1
            result_path.write_text(json.dumps({"history": [{
                "F_current": 10.0,
                "F_candidate": candidate,
                "F_tilde_candidate": candidate - 0.2,
                "F_next": candidate,
                "decision": "accept_keep",
                "would_be_decision": "accept_keep",
                "R": 0.05,
                "composite_queries": 61,
                "cumulative_composite_queries": 61,
            }]}), encoding="utf-8")
            (run_dir / "manifest.json").write_text(json.dumps({
                "seed": seed,
                "problem_seed": 42,
                "group": group,
                "warm_start_csv": "/same/warm_start.csv",
                "nn_baseline": 10.0,
                "max_iterations": 3,
                "elapsed_seconds": 1.0,
                "result_json": str(result_path),
            }), encoding="utf-8")

    analyze(tmp_path, 3)

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    effects = pd.read_csv(tmp_path / "paired_effects.csv")
    assert len(metrics) == 8
    assert metrics.loc[metrics["group"] == "adaptive_filter", "best_at_budget"].tolist() == pytest.approx([8.0, 8.1])
    assert all(metrics["queries_used"] == 61)
    adaptive = effects.query("metric == 'best_at_budget' and effect == 'adaptive_benefit'").iloc[0]
    assert adaptive["mean"] == pytest.approx(1.0)
    assert (tmp_path / "budget_curves.pdf").exists()
