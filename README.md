# Availability-Aware ANPR Forecasting

This repository contains the non-visualization code used for the experiments
in "Availability-Aware Probabilistic Travel Time Forecasting and Service-Level
Reliability for an ANPR-Instrumented Urban Arterial."

The implementation covers completion-aware ANPR preprocessing, complete-period
chronological splitting, snapshot and recurrent mixture density networks,
quantile gradient-boosted trees, validation-only model and policy selection,
positive-support diagnostics, moving-period-block inference, frozen test
evaluation, service-level reliability metrics, and CPU inference benchmarking.

## Privacy and release boundary

The original ANPR event records, vehicle identifiers, camera identifiers,
camera metadata, trained checkpoints, per-case outputs, and per-period outputs
are not included. They are governed by the data provider and cannot be made
public. The configuration files therefore contain generic camera identifiers
and relative input paths. No plotting or figure-generation code is included.

The paper's exact split counts and experiment grids are retained in
`experiment_config.yaml`. Reproducing the reported numerical results
requires authorized access to the original records. The included unit tests and
synthetic event generator exercise the public code without using those records.

## Repository contents

- `data_processing.py`: event matching, availability-aware features,
  trajectory alignment, and complete-period splitting.
- `forecast_models.py`: snapshot and recurrent neural models and training code.
- `probabilistic_metrics.py`: probabilistic scores, support corrections, block
  resampling, and multiplicity-aware utilities.
- `experiment_runner.py`: training/validation selection stages and
  selection freezing.
- `frozen_evaluation.py`: one-shot frozen test evaluation and service KPIs.
- `scripts/benchmark_inference.py`: CPU inference benchmark used in
  the computational-efficiency experiment.
- `docs/experiment_protocol.md`: execution order and data-use boundaries.
- `tests/`: deterministic synthetic unit tests.

## Environment

The reported experiments used Python 3.9, scikit-learn 1.6.1, and PyTorch
2.7.1. Install the platform-appropriate PyTorch build when GPU training is
required.

```powershell
conda activate tf_env
python -m pip install -r requirements.txt
```

## Input schema

Set `plate1_path`, `plate2_path`, `camera_up`, and `camera_down` in
`study_config.yaml`. Each event table must contain:

| Column | Meaning |
| --- | --- |
| `leave_time` | Event timestamp parseable by pandas |
| `vehicle_id` | Identifier used only for upstream/downstream matching |
| `camera_id` | Camera identifier used by the configured corridor filter |
| `turn_id` | Movement identifier; the study filter uses value `2` |

Vehicle and camera identifiers must be handled according to the applicable
data-governance agreement. Do not commit event tables or generated outputs.

The synthetic generator illustrates this schema with fictional identifiers:

```powershell
conda activate tf_env
python examples/generate_synthetic_events.py
```

The synthetic files are for interface checks only and do not reproduce the
paper's sample size or results.

## Tests

```powershell
conda activate tf_env
python -m unittest discover -s tests -p "test_*.py"
```

## Reproducing the experiment workflow

After placing authorized input tables at the configured paths, execute the
development stages in order. All model, threshold, and block-length choices use
training and validation data only.

```powershell
conda activate tf_env
python experiment_runner.py prepare
python experiment_runner.py stage-a --family snapshot
python experiment_runner.py stage-a --family recurrent
python experiment_runner.py stage-b
python experiment_runner.py k-scan
python experiment_runner.py fixed-references
python experiment_runner.py trees --kind snapshot
python experiment_runner.py trees --kind lagged
python experiment_runner.py ap03
python experiment_runner.py ap01
python experiment_runner.py ap06
python experiment_runner.py final-models
python experiment_runner.py freeze
```

Open the frozen test set once, then run the CPU benchmark:

```powershell
conda activate tf_env
python frozen_evaluation.py
python scripts/benchmark_inference.py
```

The workflow writes generated artifacts under `outputs/paper_results/`, which
is excluded from version control.

## License

The code is released under the MIT License. See `LICENSE`.
