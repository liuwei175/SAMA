from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
SPLIT_RESULTS = DATA_DIR / "final_split_results.csv"
N_GRID = [1, 2, 4, 8, 16, 64, 256]
EPS_GRID = [0.15, 0.25, 0.35]
LOSS_COLS = ["erm_robust", "direct_robust", "majorant_robust", "erm_cvar", "direct_cvar", "majorant_cvar"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_results() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_RESULTS)
    dataset = str(df["dataset"].iloc[0])
    split_count = 4 if dataset == "parkinsons" else 5
    expected = split_count * len(EPS_GRID) * len(N_GRID) * 3
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} split rows for {dataset}, found {len(df)}")
    if set(df["N"].astype(int)) != set(N_GRID):
        raise RuntimeError(f"Unexpected N grid: {sorted(set(df['N'].astype(int)))}")
    if bool(df["failed"].astype(bool).any()):
        raise RuntimeError("At least one training row failed")
    return df


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "te_robust_loss",
        "te_cvar_loss",
        "te_clean_loss",
        "val_robust_loss",
        "val_cvar_loss",
        "val_clean_loss",
        "runtime_sec",
        "stationarity_proxy",
        "actual_update_count",
    ]
    group_cols = ["dataset", "sampling_policy", "comparison", "model", "epsilon", "N", "alpha0", "lambda_l1", "lmaj_factor"]
    out = df.groupby(group_cols, as_index=False, dropna=False)[metrics].agg(["mean", "std"])
    out.columns = ["_".join(col).rstrip("_") for col in out.columns.to_flat_index()]
    order = {"erm": 0, "direct": 1, "majorant": 2}
    out["_model_order"] = out["model"].map(order)
    return out.sort_values(["epsilon", "N", "_model_order"]).drop(columns="_model_order").reset_index(drop=True)


def make_loss_table(summary: pd.DataFrame) -> pd.DataFrame:
    robust = summary.pivot_table(index=["epsilon", "N"], columns="model", values="te_robust_loss_mean", aggfunc="first")
    cvar = summary.pivot_table(index=["epsilon", "N"], columns="model", values="te_cvar_loss_mean", aggfunc="first")
    out = pd.DataFrame(index=robust.index)
    out["erm_robust"] = robust["erm"]
    out["direct_robust"] = robust["direct"]
    out["majorant_robust"] = robust["majorant"]
    out["erm_cvar"] = cvar["erm"]
    out["direct_cvar"] = cvar["direct"]
    out["majorant_cvar"] = cvar["majorant"]
    out["majorant_vs_direct_robust_adv_pct"] = 100.0 * (out["direct_robust"] - out["majorant_robust"]) / out["direct_robust"]
    out["majorant_vs_direct_cvar_adv_pct"] = 100.0 * (out["direct_cvar"] - out["majorant_cvar"]) / out["direct_cvar"]
    return out.reset_index()


def make_erm_n256_reference_table(loss: pd.DataFrame) -> pd.DataFrame:
    out = loss.copy()
    ref = out[out["N"].astype(int) == 256].set_index("epsilon")[["erm_robust", "erm_cvar"]]
    for eps, vals in ref.iterrows():
        mask = out["epsilon"].eq(eps)
        out.loc[mask, "erm_robust"] = vals["erm_robust"]
        out.loc[mask, "erm_cvar"] = vals["erm_cvar"]
    return out


def advantage_table(summary: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    majorant = summary[summary["model"] == "majorant"][
        ["epsilon", "N", "te_robust_loss_mean", "te_cvar_loss_mean", "val_robust_loss_mean", "val_cvar_loss_mean"]
    ].rename(
        columns={
            "te_robust_loss_mean": "majorant_te_robust",
            "te_cvar_loss_mean": "majorant_te_cvar",
            "val_robust_loss_mean": "majorant_val_robust",
            "val_cvar_loss_mean": "majorant_val_cvar",
        }
    )
    baseline = summary[summary["model"] == baseline_model][
        ["epsilon", "N", "alpha0", "lambda_l1", "te_robust_loss_mean", "te_cvar_loss_mean", "val_robust_loss_mean", "val_cvar_loss_mean"]
    ].rename(
        columns={
            "alpha0": f"{baseline_model}_alpha0",
            "lambda_l1": f"{baseline_model}_lambda_l1",
            "te_robust_loss_mean": f"{baseline_model}_te_robust",
            "te_cvar_loss_mean": f"{baseline_model}_te_cvar",
            "val_robust_loss_mean": f"{baseline_model}_val_robust",
            "val_cvar_loss_mean": f"{baseline_model}_val_cvar",
        }
    )
    out = majorant.merge(baseline, on=["epsilon", "N"], how="inner", validate="one_to_one")
    out["baseline_model"] = baseline_model
    out["majorant_test_robust_adv_pct"] = 100.0 * (out[f"{baseline_model}_te_robust"] - out["majorant_te_robust"]) / out[f"{baseline_model}_te_robust"]
    out["majorant_test_cvar_adv_pct"] = 100.0 * (out[f"{baseline_model}_te_cvar"] - out["majorant_te_cvar"]) / out[f"{baseline_model}_te_cvar"]
    out["majorant_val_robust_adv_pct"] = 100.0 * (out[f"{baseline_model}_val_robust"] - out["majorant_val_robust"]) / out[f"{baseline_model}_val_robust"]
    out["majorant_val_cvar_adv_pct"] = 100.0 * (out[f"{baseline_model}_val_cvar"] - out["majorant_val_cvar"]) / out[f"{baseline_model}_val_cvar"]
    return out.sort_values(["epsilon", "N"]).reset_index(drop=True)


def fairness_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair = df[(df["model"].isin(["direct", "majorant"])) & (df["sampling_policy"] == "uniform_iid_nested")].copy()
    checks = [
        "bank_hash",
        "bank_seed",
        "inner_steps_per_stage",
        "actual_update_count",
        "init_w_norm",
        "init_b",
        "x_box",
        "m_groups",
        "eps_tol_k",
        "mu_k",
        "warm_start_from_N",
    ]
    for (split_seed, epsilon, n), group in pair.groupby(["split_seed", "epsilon", "N"]):
        direct = group[group["model"] == "direct"]
        majorant = group[group["model"] == "majorant"]
        if len(direct) != 1 or len(majorant) != 1:
            rows.append({"split_seed": split_seed, "epsilon": epsilon, "N": n, "check": "pair_presence", "pass": False})
            continue
        d = direct.iloc[0]
        m = majorant.iloc[0]
        for check in checks:
            dval = d[check]
            mval = m[check]
            if pd.isna(dval) and pd.isna(mval):
                passed = True
            elif isinstance(dval, (float, np.floating)) or isinstance(mval, (float, np.floating)):
                passed = bool(np.isclose(float(dval), float(mval), atol=1e-12))
            else:
                passed = bool(dval == mval)
            rows.append({"split_seed": split_seed, "epsilon": epsilon, "N": n, "check": check, "pass": passed, "direct_value": dval, "majorant_value": mval})
    return pd.DataFrame(rows)


def write_markdown(stage_loss: pd.DataFrame, full_loss: pd.DataFrame, adv_direct: pd.DataFrame) -> None:
    dataset = str(stage_loss.attrs.get("dataset", "Experiment"))
    text = "\n".join(
        [
            f"# {dataset.title()} Final Tables",
            "",
            "## Stage-Wise Test Loss",
            "",
            stage_loss.round(4).to_markdown(index=False),
            "",
            "## Test Loss With ERM N=256 Reference",
            "",
            full_loss.round(4).to_markdown(index=False),
            "",
            "## Majorant Advantage vs Direct",
            "",
            adv_direct.round(4).to_markdown(index=False),
            "",
        ]
    )
    (RESULTS_DIR / "tables.md").write_text(text)


def write_manifest() -> None:
    rows = []
    for path in sorted([*DATA_DIR.glob("*.csv"), *RESULTS_DIR.glob("*.csv"), RESULTS_DIR / "tables.md"]):
        if path.exists():
            rows.append({"relative_path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "manifest.csv", index=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_results = load_split_results()
    summary = aggregate_summary(split_results)
    stage_loss = make_loss_table(summary)
    stage_loss.attrs["dataset"] = str(split_results["dataset"].iloc[0])
    full_loss = make_erm_n256_reference_table(stage_loss)
    adv_direct = advantage_table(summary, "direct")
    adv_erm = advantage_table(summary, "erm")
    audit = fairness_audit(split_results)
    if not bool(audit["pass"].all()):
        raise RuntimeError("Direct/Majorant fairness audit failed")
    for eps, group in full_loss.groupby("epsilon"):
        if group["erm_robust"].nunique() != 1 or group["erm_cvar"].nunique() != 1:
            raise RuntimeError(f"ERM N=256 reference is not constant for epsilon={eps}")
    summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)
    stage_loss.to_csv(RESULTS_DIR / "final_loss_table.csv", index=False)
    full_loss.to_csv(RESULTS_DIR / "full_loss_table_erm_n256_reference.csv", index=False)
    adv_direct.to_csv(RESULTS_DIR / "advantage_majorant_vs_direct.csv", index=False)
    adv_erm.to_csv(RESULTS_DIR / "advantage_majorant_vs_erm_reference.csv", index=False)
    audit.to_csv(RESULTS_DIR / "fairness_audit_direct_majorant.csv", index=False)
    write_markdown(stage_loss, full_loss, adv_direct)
    write_manifest()
    print(f"Wrote final tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
