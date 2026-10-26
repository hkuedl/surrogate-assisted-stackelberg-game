# Surrogate-Based Stackelberg Game for Coordinated Operation of Distribution Networks and Microgrids

Code and experimental data accompanying the manuscript:

**Surrogate-Based Stackelberg Game for Coordinated Operation of Distribution Networks and Microgrids**

**Authors:** Qi Li, Xueyuan Cui, and Yi Wang

## Overview

This work studies the coordinated operation of a distribution network (DN) and heterogeneous microgrids (MGs) through a multi-follower Stackelberg game.

The DN operator acts as the leader and determines electricity prices. Each MG acts as a follower and optimizes its internal operation in response to its assigned price trajectory. The leader must anticipate these responses while accounting for distribution-network operating constraints.

The proposed approach learns the MGs' price-response mappings using neural-network surrogates and improves them during optimization through local residual correction. This allows the leader to use observed responses without embedding each MG's internal optimization model into its own problem. The follower models included in this repository provide the response evaluations used in the numerical experiments.

## Main Ideas

The manuscript develops three connected components:

1. **Surrogate-based coordination.** Pretrained models approximate the responses of heterogeneous MGs and support optimization of the leader's pricing decisions.
2. **Objective-gap analysis.** The theoretical analysis relates response approximation errors to the leader's objective gap and motivates improving surrogate accuracy near candidate decisions.
3. **Adaptive local residual correction.** Additional responses are sampled locally, radial basis function (RBF) models are fitted to prediction residuals, and corrected surrogates generate new candidate solutions.

Candidates are evaluated using actual follower responses. Acceptance filtering preserves improvements, while agreement between predicted and actual objective reductions guides trust-region radius adjustment.

The numerical studies compare accuracy and communication requirements, examine the roles of residual correction and trust-region mechanisms, and assess computational scaling.

## Methods and Test Systems

| Method | Description |
|---|---|
| NN solver | Optimizes the leader problem using pretrained neural surrogates |
| CMA-ES | Searches over prices using repeated follower-response evaluations |
| Proposed method | Combines neural surrogates, local RBF residual correction, and adaptive trust-region optimization |

The main case study uses a modified IEEE 33-bus distribution network over 24 hourly periods, with five MG instances:

| Instance | Type |
|---|---|
| `SB1`, `SB2` | Single-bus microgrids |
| `BG1`, `BG2` | Building-group microgrids |
| `IP1` | Industrial-park microgrid |

Four NN initializations are considered: constant prices of `0.0`, `0.75`, and `1.5`, and a random initialization with seed `42`. Their result directories are `000`, `075`, `150`, and `rng42`.

The scalability study uses a modified 108-bus, 20 kV MV Oberrhein network with 5, 10, 15, and 20 MGs.

## Repository Contents

| Path | Purpose |
|---|---|
| [`stackelberg/`](stackelberg/) | Solver implementations, follower models, network utilities, and experiment helpers |
| [`stackelberg/README.md`](stackelberg/README.md) | Module documentation and API example |
| `BG_data_prepare.py`, `BG_data_prepare.ipynb` | Building-profile preparation |
| `buildings.py` | Building data and model utilities |
| `data_generation.py` | Generation of follower price-response samples |
| `L2Otraining.py` | Neural-surrogate architecture and training functions |
| `train_l2o_model.py`, `train_l2o_model.ipynb` | Training entry points |
| `follower_solver.py`, `NN_solver.py`, `zero_solver.py`, `trust_region_solver.py` | Root-level solver interfaces |
| `run_*.ipynb`, `run_mv_oberrhein_scaling.py` | Experiment entry points |
| `plot_*.ipynb`, `plot_trust_region_experiment.py` | Result analysis and plotting |
| `retraining_NN_solver.ipynb` | Additional-sample and surrogate-retraining study |
| `data/` | Processed inputs and training arrays |
| `building data/` | Source building-profile data |
| `NNmodel/` | Model metadata and locally generated checkpoints |
| `results/` | Saved comparison, ablation, and scalability results |

Run commands and notebooks from the repository root.

## Reproduce Figures from Saved Results

The saved-result plotting notebooks read the supplied data without running the optimization experiments. They do not require trained model checkpoints or COPT.

Clone the repository:

```bash
git clone https://github.com/hkuedl/surrogate-assisted-stackelberg-game.git
cd surrogate-assisted-stackelberg-game
```

Install the plotting dependencies and start JupyterLab:

```bash
python -m pip install numpy pandas matplotlib jupyterlab
jupyter lab
```

Open the relevant notebook and run its cells in order:

| Notebook | Content | Saved inputs |
|---|---|---|
| `plot_trust_region_four_runs.ipynb` | Four-initialization comparison, convergence curves, and summary tables | `results/000/`, `results/075/`, `results/150/`, `results/rng42/` |
| `plot_trust_region_ablation.ipynb` | Trust-region mechanism ablation | `results/tr_ablation_seed_42/` |
| `plot_mv_oberrhein_scaling.ipynb` | Scalability summary and iteration-time decomposition | `results/mv_oberrhein_coptinit_scaling_k10_s1000/` |

The four-run notebook also reads:

```text
results/trust_region_five_runs_20260924-211815.csv
```

Keep this filename unchanged: despite its historical name, the current four-run plotting notebook uses it.

Plot styles, labels, and input paths can be adjusted in each notebook's configuration cells.

## Run Optimization Experiments

### Dependencies

The experiment code uses NumPy, pandas, SciPy, Matplotlib, NetworkX, pandapower, PyTorch, CMA-ES, and the COPT Python interface.

```bash
python -m pip install numpy pandas scipy matplotlib networkx pandapower torch cma coptpy jupyterlab
```

Follower optimization and hard-constraint network evaluation require a working COPT installation and a suitable license. See the [COPT installation guide](https://guide.coap.online/copt/en-doc/install.html).

### Train the Surrogates

The repository includes training arrays for the five standard MG instances. Train their models with:

```bash
python train_l2o_model.py --scenarios SB1 SB2 BG1 BG2 IP1
```

The script writes checkpoint and metadata pairs to `NNmodel/`:

```text
L2O_SB1.pth    L2O_SB1.json
L2O_SB2.pth    L2O_SB2.json
L2O_BG1.pth    L2O_BG1.json
L2O_BG2.pth    L2O_BG2.json
L2O_IP1.pth    L2O_IP1.json
```

The current repository contains model metadata; `.pth` weights must be generated locally or supplied separately. Metadata JSON files alone cannot be used for inference.

To regenerate the training samples:

```bash
python data_generation.py --scenarios SB1 SB2 BG1 BG2 IP1 --samples 10000 --seed 42
```

### Experiment Entry Points

| Entry point | Purpose |
|---|---|
| `run_trust_region_experiment.ipynb` | NN initialization, CMA-ES comparison, and trust-region optimization |
| `run_four_trust_region_experiments.ipynb` | Trust-region runs from the four saved NN warm starts |
| `run_trust_region_ablation.ipynb` | Adaptive/fixed radius crossed with filtered/unconditional acceptance |
| `retraining_NN_solver.ipynb` | Comparison of surrogate retraining and local correction |
| `run_mv_oberrhein_scaling.py` | NN-initialized trust-region experiments at different system sizes |

In `run_trust_region_experiment.ipynb`, `RUN_NN_SOLVER` and `RUN_CMA_ES` control whether those methods are rerun or their saved results are loaded.

Before rerunning the mechanism ablation, configure a valid `WARM_START_CSV`. The historical warm-start file currently referenced by that notebook is not included in the repository. For a new experiment, an available input is:

```python
WARM_START_CSV = Path(
    "results/rng42/NN_solver_history_20260914-120155.csv"
)
OUTPUT_DIR = Path("results/tr_ablation_rerun_seed_42")
```

This starts a new ablation from the selected warm start. Reproducing the archived ablation numerically requires its original warm start and settings, recorded in the saved metadata and manifests.

An example scalability command is:

```bash
python run_mv_oberrhein_scaling.py \
    --counts 5 10 15 20 \
    --seed 42 \
    --tr-iterations 10 \
    --tr-samples 60 \
    --tr-steps-warm 1000 \
    --copt-initialize-network-state \
    --output-dir results/mv_oberrhein_rerun
```

To plot the new scalability results, update `RESULTS_ROOT` in `plot_mv_oberrhein_scaling.ipynb` accordingly.

The current scalability entry point runs the NN + trust-region workflow. Saved CMA-ES scalability results are also provided in the result archive.

## Results and Interpretation

- The optimization code minimizes the leader objective `F`. The manuscript reports DN profit as `-F`, so larger profit corresponds to a smaller objective.
- Empirical gaps use a best-known feasible objective as their reference. This reference is not a certified global optimum.
- One communication round consists of sending a price trajectory to every MG and receiving all corresponding responses. It can involve multiple individual follower solves.
- The reported online communication budget excludes offline training-data collection and reuse of cached responses.
- The four-initialization plots use specific saved filenames. Update their input mapping when analyzing newly generated runs.
- Saved manifests and parameter records describe the archived experiments. Editable notebook defaults may differ from those archived settings.

Use separate output directories for new experiments so the supplied results remain available for figure reproduction. Retraining models or changing solver versions, tolerances, seeds, or hardware may change numerical results and runtimes.

## Related Manuscript

Qi Li, Xueyuan Cui, and Yi Wang,  
*Surrogate-Based Stackelberg Game for Coordinated Operation of Distribution Networks and Microgrids.*

Please cite the companion manuscript when using this code or its experimental results.
