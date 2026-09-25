"""Command-line entry point for the isolated MV Oberrhein scaling study."""

from __future__ import annotations

import argparse

from stackelberg.mv_oberrhein_scaling import (
    MICROGRID_COUNTS,
    MVScalingExperimentConfig,
    run_mv_oberrhein_scaling_experiment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run NN-warm-started trust-region experiments on the modified "
            "108-bus, 20 kV MV Oberrhein network."
        )
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(MICROGRID_COUNTS),
        choices=MICROGRID_COUNTS,
        help="Microgrid counts to run (default: 5 10 15 20).",
    )
    parser.add_argument("--output-dir", default="results/mv_oberrhein_scaling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nn-steps", type=int, default=10_000)
    parser.add_argument("--tr-iterations", type=int, default=10)
    parser.add_argument("--tr-samples", type=int, default=60)
    parser.add_argument("--tr-steps-warm", type=int, default=1_000)
    parser.add_argument("--tr-lr", type=float, default=1e-2)
    parser.add_argument(
        "--warm-start",
        default=None,
        help=(
            "Load an existing .npy NN warm start instead of rerunning the "
            "NN initialization."
        ),
    )
    parser.add_argument(
        "--collect-diagnostics",
        action="store_true",
        help=(
            "Evaluate all local true samples with hard DistFlow and record "
            "the differentiable optimizer trace."
        ),
    )
    parser.add_argument("--diagnostic-trace-every", type=int, default=25)
    parser.add_argument(
        "--copt-initialize-network-state",
        action="store_true",
        help=(
            "For this 108-bus experiment only, initialize the Torch DistFlow "
            "network variables from a hard COPT solution at the TR center."
        ),
    )
    parser.add_argument(
        "--no-feasibility-check",
        action="store_true",
        help="Skip the untimed hard-DistFlow check at the NN warm start.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse completed manifests in the output directory.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = MVScalingExperimentConfig(
        microgrid_counts=tuple(args.counts),
        seed=args.seed,
        device=args.device,
        nn_steps=args.nn_steps,
        tr_max_iterations=args.tr_iterations,
        tr_n_samples=args.tr_samples,
        tr_steps_warm=args.tr_steps_warm,
        tr_lr=args.tr_lr,
        output_dir=args.output_dir,
        warm_start_path=args.warm_start,
        collect_diagnostics=args.collect_diagnostics,
        diagnostic_trace_every=args.diagnostic_trace_every,
        copt_initialize_network_state=args.copt_initialize_network_state,
        resume=not args.no_resume,
        run_feasibility_check=not args.no_feasibility_check,
    )
    summary = run_mv_oberrhein_scaling_experiment(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
