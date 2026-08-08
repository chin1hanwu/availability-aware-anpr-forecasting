# Experiment protocol

## Data-use boundary

The workflow uses a chronological train/validation/test split at complete
five-minute update-period boundaries. Training and validation data determine
features, scalers, model architecture, mixture count, quantile-tree settings,
block length, interval construction, and publication thresholds. The test set
is scored only after these choices are written to `selection_lock.json`.

## Development stages

1. `prepare` constructs completion-aware features and materializes the frozen
   complete-period split.
2. `stage-a` and `stage-b` compare snapshot and recurrent density-model
   configurations on training and validation data.
3. `k-scan` applies the validation one-standard-error rule to mixture count.
4. `fixed-references` fits the naive and ridge reference distributions.
5. `trees` selects snapshot and explicit-lag quantile GBDTs on validation data.
6. `ap03` evaluates native quantiles, induced Gaussian distributions, and
   positive-support corrections on development data.
7. `ap01` runs completion-delay and feature-block diagnostics.
8. `ap06` selects publication policies on complete validation periods.
9. `final-models` materializes the selected four-seed neural components.
10. `freeze` hashes all selection artifacts and records the test-access guard.

## Confirmatory evaluation

`frozen_evaluation.py` performs one frozen test batch for predictive scores,
paired representation effects, support diagnostics, moving-period-block
intervals, Holm-adjusted families, service policies, and service KPIs. The
inference benchmark uses one prepared update state per call and excludes model
loading and feature construction, matching the paper's computational protocol.
