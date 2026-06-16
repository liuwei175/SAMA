# Breast Cancer Robust Logistic Experiment

This folder contains the unified Breast Cancer experiment package used to
reproduce the final Direct, Majorant, and ERM tables.

## Contents

- `data/raw/breast_cancer.csv`: Wisconsin Breast Cancer data generated from
  `sklearn.datasets.load_breast_cancer`.
- `config/final_selected_configs.csv`: fixed final configurations for all
  robust stages and the clean ERM baseline.
- `scripts/training_core.py`: data loading, objective functions, schedules,
  and exact robust/CVaR evaluation routines.
- `scripts/train_final_models.py`: unified training entrypoint for
  `N={1,2,4,8,16,64,256}`.
- `scripts/build_final_tables.py`: aggregation, ERM reference table,
  advantage tables, fairness audit, and manifest generation.
- `results/`: final generated tables.

## Protocol

- Dataset: Wisconsin Breast Cancer data.
- Splits: stratified train/validation/test splits with seeds
  `202600, 202601, 202602, 202603, 202604`.
- Feature groups: size, texture, surface, irregularity, shape.
- Epsilon grid: `{0.15, 0.25, 0.35}`.
- N grid: `{1, 2, 4, 8, 16, 64, 256}`.
- Stage updates: `600` per N stage.
- Projection: `w in [-2, 2]`.

Direct and Majorant use the same split, standardization, nested uncertainty
bank, bank prefix, initialization, stage count, and exact vertex evaluation at
each `(epsilon, N)`. The final robust parameters are fixed in
`config/final_selected_configs.csv`.

ERM is trained on the clean logistic objective with the same seven-stage update
budget. Since ERM does not use N, `full_loss_table_erm_n256_reference.csv`
uses the final `N=256` ERM evaluation for each epsilon and repeats it across
all N rows.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate split-level results and tables:

```bash
python scripts/train_final_models.py
python scripts/build_final_tables.py
```

Primary outputs:

- `data/processed/final_split_results.csv`
- `results/final_loss_table.csv`
- `results/full_loss_table_erm_n256_reference.csv`
- `results/advantage_majorant_vs_direct.csv`
- `results/advantage_majorant_vs_erm_reference.csv`
- `results/fairness_audit_direct_majorant.csv`
- `results/tables.md`
- `results/manifest.csv`
