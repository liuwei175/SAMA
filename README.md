# SSMSA Numerical Experiment Package 6.16

This folder contains the June 16 unified reproducibility package for the
SSMSA numerical experiments.

## Contents

- `exp1_toy_example`: toy minimax experiment and final figure scripts.
- `exp2_robust_logistic_regression/parkinsons`: Parkinsons robust logistic
  regression experiment.
- `exp2_robust_logistic_regression/breast_cancer`: Breast Cancer robust
  logistic regression experiment.

## Experiment 2 Design

The robust logistic regression experiments use the final grid

```text
N = {1, 2, 4, 8, 16, 64, 256}.
```

Direct and Majorant are trained on the same nested uncertainty bank, with the
same split, standardization, initialization, bank prefix length, stage update
count, and exact vertex evaluation at each `(epsilon, N)`.

ERM is trained on the clean logistic objective with a matched seven-stage
update budget. The publication-facing table uses the final `N=256` ERM
reference for each epsilon and repeats that reference across all N values,
because ERM does not use the uncertainty sample size N.

Each dataset folder contains fixed final parameters, scripts, generated
tables, and a manifest with file hashes.
