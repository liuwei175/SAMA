from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit


torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "parkinsons.data"
CONFIG_FILE = ROOT / "config" / "final_selected_configs.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

N_GRID = [1, 2, 4, 8, 16]
EPS_GRID = [0.15, 0.25, 0.35]
SPLIT_SEEDS = [20000517, 20000518, 20000617, 20000618]
INNER_STEPS = 500
TOTAL_UPDATES = INNER_STEPS * len(N_GRID)
BANK_SEED_OFFSET = 777
ALPHA_CVAR = 0.90

PARKINSON_GROUPS = {
    "frequency": ["mdvp:fo(hz)", "mdvp:fhi(hz)", "mdvp:flo(hz)"],
    "jitter": ["mdvp:jitter(%)", "mdvp:jitter(abs)", "mdvp:rap", "mdvp:ppq", "jitter:ddp"],
    "shimmer": ["mdvp:shimmer", "shimmer(db)", "mdvp:shimmer(db)", "shimmer:apq3", "shimmer:apq5", "mdvp:apq", "shimmer:dda"],
    "noise_tone": ["nhr", "hnr"],
    "nonlinear": ["rpde", "d2", "dfa", "spread1", "spread2", "ppe"],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    G: np.ndarray
    group_names: list[str]
    subject_ids: np.ndarray
    x_box: float = 1.0


def extract_subject_id(name: str) -> str:
    text = str(name)
    return text.rsplit("_", 1)[0] if "_" in text else text


def build_group_matrix(feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [str(name).strip().lower() for name in feature_names]
    group_names = list(PARKINSON_GROUPS.keys())
    G = np.zeros((len(names), len(group_names)), dtype=np.float64)
    for j, name in enumerate(names):
        assigned = None
        for group_name, keys in PARKINSON_GROUPS.items():
            if any(key in name for key in keys):
                assigned = group_name
                break
        if assigned is None:
            raise ValueError(f"Unassigned feature: {feature_names[j]}")
        G[j, group_names.index(assigned)] = 1.0
    return G, group_names


def load_parkinsons() -> DatasetSpec:
    df = pd.read_csv(RAW_DATA)
    df.columns = [str(c).strip() for c in df.columns]
    feature_names = [c for c in df.columns if c not in {"name", "status"}]
    y = np.where(df["status"].to_numpy(dtype=int) == 1, 1.0, -1.0)
    X = df[feature_names].to_numpy(dtype=np.float64)
    G, group_names = build_group_matrix(feature_names)
    subject_ids = df["name"].map(extract_subject_id).to_numpy()
    return DatasetSpec("parkinsons", X, y.astype(np.float64), feature_names, G, group_names, subject_ids)


def split_indices_groupwise(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups, first_idx = np.unique(groups, return_index=True)
    unique_groups = unique_groups[np.argsort(first_idx)]
    group_labels = np.array([y[np.flatnonzero(groups == group)[0]] for group in unique_groups])
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
    group_train_idx, group_hold_idx = next(sss1.split(unique_groups, group_labels))
    hold_groups = unique_groups[group_hold_idx]
    hold_labels = group_labels[group_hold_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    group_val_rel, group_test_rel = next(sss2.split(hold_groups, hold_labels))
    train_groups = unique_groups[group_train_idx]
    val_groups = hold_groups[group_val_rel]
    test_groups = hold_groups[group_test_rel]
    return (
        np.flatnonzero(np.isin(groups, train_groups)),
        np.flatnonzero(np.isin(groups, val_groups)),
        np.flatnonzero(np.isin(groups, test_groups)),
    )


def standardize(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu[None, :]) / sig[None, :]


def make_split(spec: DatasetSpec, seed: int) -> dict[str, np.ndarray]:
    idx_train, idx_val, idx_test = split_indices_groupwise(spec.y, spec.subject_ids, seed)
    split = {
        "Xtr_raw": spec.X[idx_train].copy(),
        "ytr": spec.y[idx_train].copy(),
        "Xval_raw": spec.X[idx_val].copy(),
        "yval": spec.y[idx_val].copy(),
        "Xte_raw": spec.X[idx_test].copy(),
        "yte": spec.y[idx_test].copy(),
        "split_seed": seed,
    }
    mu = split["Xtr_raw"].mean(axis=0)
    sig = split["Xtr_raw"].std(axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    split["mu"] = mu
    split["sig"] = sig
    return split


def enumerate_box_vertices(m: int, eps: float) -> np.ndarray:
    return np.array(list(itertools.product([-eps, eps], repeat=m)), dtype=np.float64)


def build_uniform_bank(m: int, eps: float, split_seed: int) -> np.ndarray:
    rng = np.random.default_rng(split_seed + BANK_SEED_OFFSET)
    return rng.uniform(-eps, eps, size=(max(N_GRID), m)).astype(np.float64)


def bank_hash(bank: np.ndarray) -> str:
    arr = np.ascontiguousarray(bank, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def apply_group_drift_np(X_raw: np.ndarray, y_unc: np.ndarray, G: np.ndarray) -> np.ndarray:
    return X_raw * (1.0 + G @ y_unc)[None, :]


def apply_group_drift_torch(X_raw: torch.Tensor, y_unc: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    return X_raw * (1.0 + G @ y_unc).unsqueeze(0)


def logistic_loss_np(
    w: np.ndarray,
    b: float,
    X_raw: np.ndarray,
    y: np.ndarray,
    y_unc: np.ndarray,
    G: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> float:
    Z = standardize(apply_group_drift_np(X_raw, y_unc, G), mu, sig)
    margins = y * (Z @ w + b)
    return float(np.logaddexp(0.0, -margins).mean())


def dataset_loss_torch(
    w: torch.Tensor,
    b: torch.Tensor,
    X_raw: torch.Tensor,
    y: torch.Tensor,
    y_unc: torch.Tensor,
    G: torch.Tensor,
    mu: torch.Tensor,
    sig: torch.Tensor,
) -> torch.Tensor:
    Z = (apply_group_drift_torch(X_raw, y_unc, G) - mu.unsqueeze(0)) / sig.unsqueeze(0)
    margins = y * (Z @ w + b)
    return F.softplus(-margins).mean()


def smooth_max(vals: torch.Tensor, mu: float) -> torch.Tensor:
    vmax = vals.max()
    return vmax + mu * torch.log(torch.sum(torch.exp((vals - vmax) / mu)))


def direct_objective(
    w: torch.Tensor,
    b: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    Ybank: torch.Tensor,
    mu_smooth: float,
    lambda_l1: float,
) -> torch.Tensor:
    vals = [
        dataset_loss_torch(w, b, tensors["Xtr_raw"], tensors["ytr"], yk, tensors["G"], tensors["mu"], tensors["sig"])
        for yk in Ybank
    ]
    return smooth_max(torch.stack(vals), mu_smooth) + lambda_l1 * torch.abs(w).sum()


def grad_y_loss(
    w: torch.Tensor,
    b: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    yk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    yvar = yk.detach().clone().requires_grad_(True)
    loss = dataset_loss_torch(w, b, tensors["Xtr_raw"], tensors["ytr"], yvar, tensors["G"], tensors["mu"], tensors["sig"])
    (grad,) = torch.autograd.grad(loss, yvar, retain_graph=True, create_graph=True)
    return loss, grad


def majorant_single_value(
    w: torch.Tensor,
    b: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    yk: torch.Tensor,
    eps: float,
    Lmaj: float,
) -> torch.Tensor:
    loss, grad = grad_y_loss(w, b, tensors, yk)
    z_star = torch.clamp(yk.detach() + grad.detach() / Lmaj, -eps, eps)
    delta = z_star - yk
    return loss + torch.dot(grad, delta) - 0.5 * Lmaj * torch.sum(delta**2)


def majorant_objective(
    w: torch.Tensor,
    b: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    Ybank: torch.Tensor,
    mu_smooth: float,
    lambda_l1: float,
    eps: float,
    Lmaj: float,
) -> torch.Tensor:
    vals = [majorant_single_value(w, b, tensors, yk, eps, Lmaj) for yk in Ybank]
    return smooth_max(torch.stack(vals), mu_smooth) + lambda_l1 * torch.abs(w).sum()


def clean_objective(
    w: torch.Tensor,
    b: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    lambda_l1: float,
) -> torch.Tensor:
    Z = (tensors["Xtr_raw"] - tensors["mu"].unsqueeze(0)) / tensors["sig"].unsqueeze(0)
    margins = tensors["ytr"] * (Z @ w + b)
    return F.softplus(-margins).mean() + lambda_l1 * torch.abs(w).sum()


def make_tensors(split: dict[str, np.ndarray], G: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "Xtr_raw": torch.tensor(split["Xtr_raw"], dtype=torch.float64),
        "ytr": torch.tensor(split["ytr"], dtype=torch.float64),
        "G": torch.tensor(G, dtype=torch.float64),
        "mu": torch.tensor(split["mu"], dtype=torch.float64),
        "sig": torch.tensor(split["sig"], dtype=torch.float64),
    }


def schedule_for_stage(k: int, N: int) -> tuple[float, float]:
    eps_tol = max(1e-8, 1e-1 * (0.9**k))
    mu_smooth = eps_tol / (2.0 * math.log(max(N, 2)))
    return eps_tol, mu_smooth


def estimate_lmaj_from_box(split: dict[str, np.ndarray], G: np.ndarray, x_box: float, lmaj_factor: float) -> float:
    w_ref = np.full(split["Xtr_raw"].shape[1], x_box, dtype=np.float64)
    U = split["Xtr_raw"] / split["sig"][None, :]
    A = (U * w_ref[None, :]) @ G
    gram = (A.T @ A) / A.shape[0]
    lam_max = float(np.linalg.eigvalsh(gram).max())
    return float(max(1e-6, 0.25 * lmaj_factor * lam_max))


def projected_gradient_proxy(
    w: torch.Tensor,
    b: torch.Tensor,
    grad_w: torch.Tensor,
    grad_b: torch.Tensor,
    alpha: float,
    x_box: float,
) -> float:
    with torch.no_grad():
        w_next = torch.clamp(w - alpha * grad_w, -x_box, x_box)
        b_next = b - alpha * grad_b
        mapping_w = (w - w_next) / alpha
        mapping_b = (b - b_next) / alpha
        return float(torch.sqrt(torch.sum(mapping_w**2) + mapping_b**2).detach().cpu().item())


def evaluate_exact_subset(
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    eps: float,
    w: np.ndarray,
    b: float,
    subset: str,
) -> dict[str, float]:
    X_raw = split[f"X{subset}_raw"]
    y = split[f"y{subset}"]
    vertices = enumerate_box_vertices(spec.G.shape[1], eps)
    dataset_losses = [
        logistic_loss_np(w, b, X_raw, y, vertex, spec.G, split["mu"], split["sig"])
        for vertex in vertices
    ]
    robust = float(np.max(dataset_losses))
    worst_losses = np.full(X_raw.shape[0], -np.inf, dtype=np.float64)
    for vertex in vertices:
        Z = standardize(apply_group_drift_np(X_raw, vertex, spec.G), split["mu"], split["sig"])
        margins = y * (Z @ w + b)
        worst_losses = np.maximum(worst_losses, np.logaddexp(0.0, -margins))
    tail_size = max(1, int(math.ceil((1.0 - ALPHA_CVAR) * worst_losses.size)))
    cvar = float(np.partition(worst_losses, worst_losses.size - tail_size)[worst_losses.size - tail_size :].mean())
    clean = logistic_loss_np(w, b, X_raw, y, np.zeros(spec.G.shape[1], dtype=np.float64), spec.G, split["mu"], split["sig"])
    return {
        f"{subset}_robust_loss": robust,
        f"{subset}_cvar_loss": cvar,
        f"{subset}_clean_loss": clean,
    }


def evaluate_sampled_train(
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    w: np.ndarray,
    b: float,
    Y_np: np.ndarray,
) -> dict[str, float]:
    losses = [
        logistic_loss_np(w, b, split["Xtr_raw"], split["ytr"], y_unc, spec.G, split["mu"], split["sig"])
        for y_unc in Y_np
    ]
    return {
        "tr_sampled_robust_loss": float(np.max(losses)),
        "tr_sampled_mean_loss": float(np.mean(losses)),
    }


def train_robust_path(
    model: str,
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    eps: float,
    alpha0: float,
    lambda_l1: float,
    lmaj_factor: float | None,
) -> list[dict[str, object]]:
    tensors = make_tensors(split, spec.G)
    w = torch.zeros(split["Xtr_raw"].shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    _, mu0 = schedule_for_stage(0, N_GRID[0])
    bank_full = build_uniform_bank(spec.G.shape[1], eps, int(split["split_seed"]))
    Lmaj = estimate_lmaj_from_box(split, spec.G, spec.x_box, float(lmaj_factor)) if model == "majorant" else np.nan
    rows: list[dict[str, object]] = []
    updates = 0
    for stage_index, N in enumerate(N_GRID):
        eps_tol, mu_smooth = schedule_for_stage(stage_index, N)
        step_size = alpha0 * mu_smooth / mu0
        Y_np = bank_full[:N].copy()
        Y_t = torch.tensor(Y_np, dtype=torch.float64)
        failed = False
        last_obj = np.nan
        last_proxy = np.nan
        start = time.time()
        for _ in range(INNER_STEPS):
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            if model == "direct":
                obj = direct_objective(w, b, tensors, Y_t, mu_smooth, lambda_l1)
            elif model == "majorant":
                obj = majorant_objective(w, b, tensors, Y_t, mu_smooth, lambda_l1, eps, float(Lmaj))
            else:
                raise ValueError(f"Unknown robust model: {model}")
            if not torch.isfinite(obj):
                failed = True
                break
            obj.backward()
            if w.grad is None or b.grad is None or not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
                failed = True
                break
            last_proxy = projected_gradient_proxy(w, b, w.grad, b.grad, step_size, spec.x_box)
            with torch.no_grad():
                w -= step_size * w.grad
                b -= step_size * b.grad
                w.clamp_(-spec.x_box, spec.x_box)
            updates += 1
            last_obj = float(obj.detach().cpu().item())
        runtime = time.time() - start
        with torch.no_grad():
            w_np = w.detach().cpu().numpy().copy()
            b_np = float(b.detach().cpu().item())
        rows.append(
            {
                "comparison": "path_validation",
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
                "inner_steps_per_stage": int(INNER_STEPS),
                "actual_update_count": int(updates),
                "train_objective": float(last_obj),
                "stationarity_proxy": float(last_proxy),
                "runtime_sec": float(runtime),
                "x_box": float(spec.x_box),
                "m_groups": int(spec.G.shape[1]),
                "group_names": ",".join(spec.group_names),
                "bank_seed": int(split["split_seed"] + BANK_SEED_OFFSET),
                "bank_hash": bank_hash(Y_np),
                "init_w_norm": 0.0,
                "init_b": 0.0,
                "failed": bool(failed),
                **evaluate_sampled_train(split, spec, w_np, b_np, Y_np),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "tr"),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "val"),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "te"),
            }
        )
    return rows


def train_erm_reference(
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    alpha0: float,
    lambda_l1: float,
) -> list[dict[str, object]]:
    tensors = make_tensors(split, spec.G)
    w = torch.zeros(split["Xtr_raw"].shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    failed = False
    last_obj = np.nan
    last_proxy = np.nan
    start = time.time()
    for _ in range(TOTAL_UPDATES):
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        obj = clean_objective(w, b, tensors, lambda_l1)
        if not torch.isfinite(obj):
            failed = True
            break
        obj.backward()
        if w.grad is None or b.grad is None or not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
            failed = True
            break
        last_proxy = projected_gradient_proxy(w, b, w.grad, b.grad, alpha0, spec.x_box)
        with torch.no_grad():
            w -= alpha0 * w.grad
            b -= alpha0 * b.grad
            w.clamp_(-spec.x_box, spec.x_box)
        last_obj = float(obj.detach().cpu().item())
    runtime = time.time() - start
    with torch.no_grad():
        w_np = w.detach().cpu().numpy().copy()
        b_np = float(b.detach().cpu().item())
    rows: list[dict[str, object]] = []
    for eps in EPS_GRID:
        for stage_index, N in enumerate(N_GRID):
            eps_tol, mu_smooth = schedule_for_stage(stage_index, N)
            rows.append(
                {
                    "comparison": "erm_reference",
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
                    "inner_steps_per_stage": int(INNER_STEPS),
                    "actual_update_count": int(TOTAL_UPDATES),
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
                    "tr_sampled_robust_loss": np.nan,
                    "tr_sampled_mean_loss": np.nan,
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "tr"),
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "val"),
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "te"),
                }
            )
    return rows


def run_training() -> pd.DataFrame:
    spec = load_parkinsons()
    configs = pd.read_csv(CONFIG_FILE)
    rows: list[dict[str, object]] = []
    for split_seed in SPLIT_SEEDS:
        split = make_split(spec, split_seed)
        erm_cfg = configs[configs["model"] == "erm"].iloc[0]
        rows.extend(train_erm_reference(split, spec, float(erm_cfg["alpha0"]), float(erm_cfg["lambda_l1"])))
        for eps in EPS_GRID:
            for model in ["direct", "majorant"]:
                cfg = configs[(configs["model"] == model) & np.isclose(configs["epsilon"], eps)].iloc[0]
                lmaj_factor = None if pd.isna(cfg["lmaj_factor"]) else float(cfg["lmaj_factor"])
                rows.extend(
                    train_robust_path(
                        model=model,
                        split=split,
                        spec=spec,
                        eps=eps,
                        alpha0=float(cfg["alpha0"]),
                        lambda_l1=float(cfg["lambda_l1"]),
                        lmaj_factor=lmaj_factor,
                    )
                )
        print(f"finished split {split_seed}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_results = run_training()
    cols = [
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
    split_results[cols].sort_values(["model", "epsilon", "split_seed", "N"]).to_csv(
        PROCESSED_DIR / "final_split_results.csv",
        index=False,
    )
    print(f"Wrote {PROCESSED_DIR / 'final_split_results.csv'}")


if __name__ == "__main__":
    main()
