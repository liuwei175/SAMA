from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import training_core as base


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FINAL_SPLIT_RESULTS = PROCESSED_DIR / "final_split_results.csv"
CONFIG_FILE = ROOT / "config" / "final_selected_configs.csv"

REPRODUCE_N_GRID = [1, 2, 4, 8, 16]
LARGE_N_GRID = [64, 256]
FINAL_N_GRID = REPRODUCE_N_GRID + LARGE_N_GRID
FULL_BANK_N = 256


def build_uncertainty_bank(m: int, eps: float, split_seed: int) -> np.ndarray:
    rng = np.random.default_rng(split_seed + base.BANK_SEED_OFFSET)
    return rng.uniform(-eps, eps, size=(FULL_BANK_N, m)).astype(np.float64)


def make_row(
    *,
    comparison: str,
    model: str,
    split: dict[str, np.ndarray],
    spec: base.DatasetSpec,
    eps: float,
    N: int,
    stage_index: int,
    alpha0: float,
    lambda_l1: float,
    lmaj_factor: float | None,
    Lmaj: float,
    eps_tol: float,
    mu_smooth: float,
    step_size: float,
    updates: int,
    train_objective: float,
    stationarity_proxy: float,
    runtime_sec: float,
    bank: np.ndarray,
    w_np: np.ndarray,
    b_np: float,
    failed: bool,
    source_config_N: int,
    warm_start_from_N: float,
    checkpoint_path: str,
) -> dict[str, object]:
    return {
        "comparison": comparison,
        "dataset": spec.name,
        "sampling_policy": "uniform_iid_nested",
        "model": model,
        "split_seed": int(split["split_seed"]),
        "epsilon": float(eps),
        "N": int(N),
        "stage_index": int(stage_index),
        "alpha0": float(alpha0),
        "lambda_l1": float(lambda_l1),
        "lmaj_factor": float(lmaj_factor) if model == "majorant" else np.nan,
        "Lmaj": float(Lmaj) if model == "majorant" else np.nan,
        "eps_tol_k": float(eps_tol),
        "mu_k": float(mu_smooth),
        "stepsize": float(step_size),
        "inner_steps_per_stage": int(base.INNER_STEPS),
        "actual_update_count": int(updates),
        "train_objective": float(train_objective),
        "stationarity_proxy": float(stationarity_proxy),
        "runtime_sec": float(runtime_sec),
        "x_box": float(spec.x_box),
        "m_groups": int(spec.G.shape[1]),
        "group_names": ",".join(spec.group_names),
        "bank_seed": int(split["split_seed"] + base.BANK_SEED_OFFSET),
        "bank_hash": base.bank_hash(bank),
        "init_w_norm": 0.0,
        "init_b": 0.0,
        "failed": bool(failed),
        "source_config_N": int(source_config_N),
        "warm_start_from_N": warm_start_from_N,
        "checkpoint_path": checkpoint_path,
        **base.evaluate_sampled_train(split, spec, w_np, b_np, bank),
        **base.evaluate_exact_subset(split, spec, eps, w_np, b_np, "tr"),
        **base.evaluate_exact_subset(split, spec, eps, w_np, b_np, "val"),
        **base.evaluate_exact_subset(split, spec, eps, w_np, b_np, "te"),
    }


def run_stage(
    *,
    model: str,
    tensors: dict[str, torch.Tensor],
    split: dict[str, np.ndarray],
    spec: base.DatasetSpec,
    eps: float,
    N: int,
    stage_index: int,
    w: torch.Tensor,
    b: torch.Tensor,
    bank_full: np.ndarray,
    alpha0: float,
    lambda_l1: float,
    lmaj_factor: float | None,
    Lmaj: float,
    mu0: float,
    updates: int,
) -> tuple[dict[str, object], int]:
    eps_tol, mu_smooth = base.schedule_for_stage(stage_index, N)
    step_size = alpha0 * mu_smooth / mu0
    Y_np = bank_full[:N].copy()
    Y_t = torch.tensor(Y_np, dtype=torch.float64)
    failed = False
    last_obj = np.nan
    last_proxy = np.nan
    start = base.time.time()
    for _ in range(base.INNER_STEPS):
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        if model == "direct":
            obj = base.direct_objective(w, b, tensors, Y_t, mu_smooth, lambda_l1)
        elif model == "majorant":
            obj = base.majorant_objective(w, b, tensors, Y_t, mu_smooth, lambda_l1, eps, float(Lmaj))
        else:
            raise ValueError(model)
        if not torch.isfinite(obj):
            failed = True
            break
        obj.backward()
        if w.grad is None or b.grad is None or not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
            failed = True
            break
        last_proxy = base.projected_gradient_proxy(w, b, w.grad, b.grad, step_size, spec.x_box)
        with torch.no_grad():
            w -= step_size * w.grad
            b -= step_size * b.grad
            w.clamp_(-spec.x_box, spec.x_box)
        updates += 1
        last_obj = float(obj.detach().cpu().item())
    runtime = base.time.time() - start
    with torch.no_grad():
        w_np = w.detach().cpu().numpy().copy()
        b_np = float(b.detach().cpu().item())
    row = make_row(
        comparison="final_path",
        model=model,
        split=split,
        spec=spec,
        eps=eps,
        N=N,
        stage_index=stage_index,
        alpha0=alpha0,
        lambda_l1=lambda_l1,
        lmaj_factor=lmaj_factor,
        Lmaj=Lmaj,
        eps_tol=eps_tol,
        mu_smooth=mu_smooth,
        step_size=step_size,
        updates=updates,
        train_objective=last_obj,
        stationarity_proxy=last_proxy,
        runtime_sec=runtime,
        bank=Y_np,
        w_np=w_np,
        b_np=b_np,
        failed=failed,
        source_config_N=N,
        warm_start_from_N=np.nan,
        checkpoint_path="",
    )
    return row, updates


def make_state_record(row: dict[str, object]) -> dict[str, object]:
    out = {key: row[key] for key in [
        "dataset",
        "model",
        "split_seed",
        "epsilon",
        "N",
        "stage_index",
        "alpha0",
        "lambda_l1",
        "lmaj_factor",
        "Lmaj",
        "actual_update_count",
        "bank_seed",
        "bank_hash",
    ]}
    out["checkpoint_path"] = "in_memory_N16_state"
    return out


def train_robust_path(
    *,
    model: str,
    split: dict[str, np.ndarray],
    spec: base.DatasetSpec,
    eps: float,
    reproduce_cfg: pd.Series,
    large_cfgs: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    tensors = base.make_tensors(split, spec.G)
    w = torch.zeros(split["Xtr_raw"].shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    _, mu0 = base.schedule_for_stage(0, REPRODUCE_N_GRID[0])
    bank_full = build_uncertainty_bank(spec.G.shape[1], eps, int(split["split_seed"]))
    lmaj_factor = None if pd.isna(reproduce_cfg["lmaj_factor"]) else float(reproduce_cfg["lmaj_factor"])
    Lmaj = base.estimate_lmaj_from_box(split, spec.G, spec.x_box, float(lmaj_factor)) if model == "majorant" else np.nan
    reproduce_rows: list[dict[str, object]] = []
    large_rows: list[dict[str, object]] = []
    updates = 0
    checkpoint_row: dict[str, object] | None = None

    for stage_index, N in enumerate(REPRODUCE_N_GRID):
        row, updates = run_stage(
            model=model,
            tensors=tensors,
            split=split,
            spec=spec,
            eps=eps,
            N=N,
            stage_index=stage_index,
            w=w,
            b=b,
            bank_full=bank_full,
            alpha0=float(reproduce_cfg["alpha0"]),
            lambda_l1=float(reproduce_cfg["lambda_l1"]),
            lmaj_factor=lmaj_factor,
            Lmaj=Lmaj,
            mu0=mu0,
            updates=updates,
        )
        row["comparison"] = "final_path"
        row["source_config_N"] = int(N)
        reproduce_rows.append(row)
        if N == 16:
            checkpoint_row = make_state_record(row)

    if checkpoint_row is None:
        raise RuntimeError(f"Missing N=16 checkpoint for {model}, eps={eps}, split={split['split_seed']}")

    previous_N = 16
    for N in LARGE_N_GRID:
        cfg = large_cfgs[(large_cfgs["model"] == model) & np.isclose(large_cfgs["epsilon"], eps) & (large_cfgs["_N_int"] == N)].iloc[0]
        large_lmaj = None if pd.isna(cfg["lmaj_factor"]) else float(cfg["lmaj_factor"])
        large_Lmaj = base.estimate_lmaj_from_box(split, spec.G, spec.x_box, float(large_lmaj)) if model == "majorant" else np.nan
        row, updates = run_stage(
            model=model,
            tensors=tensors,
            split=split,
            spec=spec,
            eps=eps,
            N=int(N),
            stage_index=REPRODUCE_N_GRID.index(16) + LARGE_N_GRID.index(N) + 1,
            w=w,
            b=b,
            bank_full=bank_full,
            alpha0=float(cfg["alpha0"]),
            lambda_l1=float(cfg["lambda_l1"]),
            lmaj_factor=large_lmaj,
            Lmaj=large_Lmaj,
            mu0=mu0,
            updates=updates,
        )
        row["comparison"] = "final_path"
        row["source_config_N"] = int(N)
        row["warm_start_from_N"] = float(previous_N)
        row["checkpoint_path"] = checkpoint_row["checkpoint_path"] if N == 64 else ""
        large_rows.append(row)
        previous_N = int(N)

    return reproduce_rows, large_rows, checkpoint_row


def train_erm_path(
    *,
    split: dict[str, np.ndarray],
    spec: base.DatasetSpec,
    alpha0: float,
    lambda_l1: float,
) -> list[dict[str, object]]:
    tensors = base.make_tensors(split, spec.G)
    w = torch.zeros(split["Xtr_raw"].shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    failed = False
    last_obj = np.nan
    last_proxy = np.nan
    start = base.time.time()
    updates = 0
    rows: list[dict[str, object]] = []
    previous_N = np.nan
    for stage_index, N in enumerate(FINAL_N_GRID):
        for _ in range(base.INNER_STEPS):
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            obj = base.clean_objective(w, b, tensors, lambda_l1)
            if not torch.isfinite(obj):
                failed = True
                break
            obj.backward()
            if w.grad is None or b.grad is None or not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
                failed = True
                break
            last_proxy = base.projected_gradient_proxy(w, b, w.grad, b.grad, alpha0, spec.x_box)
            with torch.no_grad():
                w -= alpha0 * w.grad
                b -= alpha0 * b.grad
                w.clamp_(-spec.x_box, spec.x_box)
            updates += 1
            last_obj = float(obj.detach().cpu().item())
        runtime = base.time.time() - start
        with torch.no_grad():
            w_np = w.detach().cpu().numpy().copy()
            b_np = float(b.detach().cpu().item())
        eps_tol, mu_smooth = base.schedule_for_stage(stage_index, N)
        for eps in base.EPS_GRID:
            rows.append(
                {
                    "comparison": "erm_clean_path",
                    "dataset": spec.name,
                    "sampling_policy": "clean",
                    "model": "erm",
                    "split_seed": int(split["split_seed"]),
                    "epsilon": float(eps),
                    "N": int(N),
                    "stage_index": int(stage_index),
                    "alpha0": float(alpha0),
                    "lambda_l1": float(lambda_l1),
                    "lmaj_factor": np.nan,
                    "Lmaj": np.nan,
                    "eps_tol_k": float(eps_tol),
                    "mu_k": float(mu_smooth),
                    "stepsize": float(alpha0),
                    "inner_steps_per_stage": int(base.INNER_STEPS),
                    "actual_update_count": int(updates),
                    "train_objective": float(last_obj),
                    "stationarity_proxy": float(last_proxy),
                    "runtime_sec": float(runtime),
                    "x_box": float(spec.x_box),
                    "m_groups": int(spec.G.shape[1]),
                    "group_names": ",".join(spec.group_names),
                    "bank_seed": np.nan,
                    "bank_hash": "clean",
                    "init_w_norm": 0.0,
                    "init_b": 0.0,
                    "failed": bool(failed),
                    "source_config_N": int(N),
                    "warm_start_from_N": previous_N,
                    "checkpoint_path": "",
                    "tr_sampled_robust_loss": np.nan,
                    "tr_sampled_mean_loss": np.nan,
                    **base.evaluate_exact_subset(split, spec, float(eps), w_np, b_np, "tr"),
                    **base.evaluate_exact_subset(split, spec, float(eps), w_np, b_np, "val"),
                    **base.evaluate_exact_subset(split, spec, float(eps), w_np, b_np, "te"),
                }
            )
        previous_N = int(N)
    return rows


def run_training() -> pd.DataFrame:
    spec = base.load_parkinsons()
    final_cfgs = pd.read_csv(CONFIG_FILE)
    final_cfgs["_N_int"] = pd.to_numeric(final_cfgs["N"], errors="coerce")
    rows: list[dict[str, object]] = []
    for split_seed in base.SPLIT_SEEDS:
        split = base.make_split(spec, split_seed)
        erm_cfg = final_cfgs[final_cfgs["model"] == "erm"].iloc[0]
        rows.extend(
            train_erm_path(
                split=split,
                spec=spec,
                alpha0=float(erm_cfg["alpha0"]),
                lambda_l1=float(erm_cfg["lambda_l1"]),
            )
        )
        for eps in base.EPS_GRID:
            for model in ["direct", "majorant"]:
                cfg = final_cfgs[
                    (final_cfgs["model"] == model)
                    & np.isclose(final_cfgs["epsilon"], eps)
                    & (final_cfgs["_N_int"].isin(REPRODUCE_N_GRID))
                ].iloc[0]
                repro, large, checkpoint = train_robust_path(
                    model=model,
                    split=split,
                    spec=spec,
                    eps=float(eps),
                    reproduce_cfg=cfg,
                    large_cfgs=final_cfgs,
                )
                rows.extend(repro)
                rows.extend(large)
        print(f"finished split {split_seed}", flush=True)
    return pd.DataFrame(rows)


def write_outputs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_results = run_training()
    base_cols = [
        "comparison",
        "dataset",
        "sampling_policy",
        "model",
        "split_seed",
        "epsilon",
        "N",
        "stage_index",
        "alpha0",
        "lambda_l1",
        "lmaj_factor",
        "Lmaj",
        "eps_tol_k",
        "mu_k",
        "stepsize",
        "inner_steps_per_stage",
        "actual_update_count",
        "train_objective",
        "stationarity_proxy",
        "runtime_sec",
        "x_box",
        "m_groups",
        "group_names",
        "bank_seed",
        "bank_hash",
        "init_w_norm",
        "init_b",
        "failed",
        "source_config_N",
        "warm_start_from_N",
        "checkpoint_path",
        "tr_sampled_robust_loss",
        "tr_sampled_mean_loss",
        "tr_robust_loss",
        "tr_cvar_loss",
        "tr_clean_loss",
        "val_robust_loss",
        "val_cvar_loss",
        "val_clean_loss",
        "te_robust_loss",
        "te_cvar_loss",
        "te_clean_loss",
    ]
    split_results[base_cols].sort_values(["model", "epsilon", "N", "split_seed"]).to_csv(
        FINAL_SPLIT_RESULTS,
        index=False,
    )
    print(f"Wrote {FINAL_SPLIT_RESULTS}")


if __name__ == "__main__":
    write_outputs()
