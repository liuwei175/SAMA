from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split_results() -> pd.DataFrame:
    split_results = pd.read_csv(DATA_DIR / "final_split_results.csv")
    selected_configs = pd.read_csv(CONFIG_DIR / "final_selected_configs.csv")
    keys = ["dataset", "model", "epsilon", "N", "alpha0", "lambda_l1", "lmaj_factor", "sampling_policy"]
    observed = split_results[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    expected = selected_configs[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    return split_results


def aggregate_summary(split_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "te_robust_loss",
        "te_cvar_loss",
        "te_clean_loss",
        "val_robust_loss",
        "val_cvar_loss",
        "val_clean_loss",
        "runtime_sec",
        "stationarity_proxy",
    ]
    group_cols = [
        "dataset",
        "sampling_policy",
        "comparison",
        "model",
        "epsilon",
        "N",
        "alpha0",
        "lambda_l1",
        "lmaj_factor",
    ]
    out = split_results.groupby(group_cols, as_index=False, dropna=False)[metrics].agg(["mean", "std"])
    out.columns = ["_".join(col).rstrip("_") for col in out.columns.to_flat_index()]
    order = {"erm": 0, "majorant": 1, "direct": 2}
    out["_model_order"] = out["model"].map(order)
    return out.sort_values(["epsilon", "N", "_model_order"]).drop(columns="_model_order").reset_index(drop=True)


def make_loss_table(summary: pd.DataFrame) -> pd.DataFrame:
    robust = summary.pivot_table(index=["epsilon", "N"], columns="model", values="te_robust_loss_mean", aggfunc="first")
    cvar = summary.pivot_table(index=["epsilon", "N"], columns="model", values="te_cvar_loss_mean", aggfunc="first")
    out = pd.DataFrame(index=robust.index)
    out["erm_robust"] = robust["erm"]
    out["erm_cvar"] = cvar["erm"]
    out["direct_robust"] = robust["direct"]
    out["direct_cvar"] = cvar["direct"]
    out["majorant_robust"] = robust["majorant"]
    out["majorant_cvar"] = cvar["majorant"]
    out["robust_adv_pct"] = 100.0 * (out["direct_robust"] - out["majorant_robust"]) / out["direct_robust"]
    out["cvar_adv_pct"] = 100.0 * (out["direct_cvar"] - out["majorant_cvar"]) / out["direct_cvar"]
    out["erm_to_direct_robust_improve_pct"] = 100.0 * (out["erm_robust"] - out["direct_robust"]) / out["erm_robust"]
    out["erm_to_majorant_robust_improve_pct"] = 100.0 * (out["erm_robust"] - out["majorant_robust"]) / out["erm_robust"]
    out["erm_to_direct_cvar_improve_pct"] = 100.0 * (out["erm_cvar"] - out["direct_cvar"]) / out["erm_cvar"]
    out["erm_to_majorant_cvar_improve_pct"] = 100.0 * (out["erm_cvar"] - out["majorant_cvar"]) / out["erm_cvar"]
    return out.reset_index()


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
        [
            "epsilon",
            "N",
            "alpha0",
            "lambda_l1",
            "te_robust_loss_mean",
            "te_cvar_loss_mean",
            "val_robust_loss_mean",
            "val_cvar_loss_mean",
        ]
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
    out["majorant_test_robust_adv_pct"] = 100.0 * (
        out[f"{baseline_model}_te_robust"] - out["majorant_te_robust"]
    ) / out[f"{baseline_model}_te_robust"]
    out["majorant_test_cvar_adv_pct"] = 100.0 * (
        out[f"{baseline_model}_te_cvar"] - out["majorant_te_cvar"]
    ) / out[f"{baseline_model}_te_cvar"]
    out["majorant_val_robust_adv_pct"] = 100.0 * (
        out[f"{baseline_model}_val_robust"] - out["majorant_val_robust"]
    ) / out[f"{baseline_model}_val_robust"]
    out["majorant_val_cvar_adv_pct"] = 100.0 * (
        out[f"{baseline_model}_val_cvar"] - out["majorant_val_cvar"]
    ) / out[f"{baseline_model}_val_cvar"]
    return out.sort_values(["epsilon", "N"]).reset_index(drop=True)


def fairness_audit(split_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair = split_results[
        (split_results["model"].isin(["direct", "majorant"]))
        & (split_results["sampling_policy"] == "uniform_iid_nested")
    ].copy()
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
            rows.append(
                {
                    "split_seed": split_seed,
                    "epsilon": epsilon,
                    "N": n,
                    "check": check,
                    "pass": passed,
                    "direct_value": dval,
                    "majorant_value": mval,
                }
            )
    return pd.DataFrame(rows)


def write_markdown(loss_table: pd.DataFrame, adv_erm: pd.DataFrame, adv_direct: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Breast Final Tables",
            "",
            "## Test Loss",
            "",
            loss_table.round(4).to_markdown(index=False),
            "",
            "## Majorant vs ERM Reference",
            "",
            adv_erm[["epsilon", "N", "majorant_test_robust_adv_pct", "majorant_test_cvar_adv_pct"]]
            .round(2)
            .to_markdown(index=False),
            "",
            "## Majorant vs Direct",
            "",
            adv_direct[["epsilon", "N", "majorant_test_robust_adv_pct", "majorant_test_cvar_adv_pct"]]
            .round(2)
            .to_markdown(index=False),
            "",
        ]
    )
    (RESULTS_DIR / "tables.md").write_text(text)


def write_manifest() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            rows.append(
                {
                    "relative_path": str(rel),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "manifest.csv", index=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_results = load_split_results()
    summary = aggregate_summary(split_results)
    loss_table = make_loss_table(summary)
    adv_erm = advantage_table(summary, "erm")
    adv_direct = advantage_table(summary, "direct")
    audit = fairness_audit(split_results)

    summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)
    loss_table.to_csv(RESULTS_DIR / "final_loss_table.csv", index=False)
    adv_erm.to_csv(RESULTS_DIR / "advantage_majorant_vs_erm_reference.csv", index=False)
    adv_direct.to_csv(RESULTS_DIR / "advantage_majorant_vs_direct.csv", index=False)
    audit.to_csv(RESULTS_DIR / "fairness_audit_direct_majorant.csv", index=False)
    write_markdown(loss_table, adv_erm, adv_direct)
    write_manifest()
    if not bool(audit["pass"].all()):
        raise RuntimeError("Direct/Majorant fairness audit failed.")
    print(f"Wrote final tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
