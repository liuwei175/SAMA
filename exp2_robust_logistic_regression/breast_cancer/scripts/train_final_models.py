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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit


torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "breast_cancer.csv"
CONFIG_FILE = ROOT / "config" / "final_selected_configs.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

N_GRID = [1, 2, 4, 8, 16]
EPS_GRID = [0.15, 0.25, 0.35]
SPLIT_SEEDS = [202600, 202601, 202602, 202603, 202604]
INNER_STEPS = 600
TOTAL_UPDATES = INNER_STEPS * len(N_GRID)
BANK_SEED_OFFSET = 777
ALPHA_CVAR = 0.90

BREAST_GROUPS = {
    "size": ["radius", "perimeter", "area"],
    "texture": ["texture"],
    "surface": ["smoothness"],
    "irregularity": ["compactness", "concavity", "concave points"],
    "shape": ["symmetry", "fractal dimension"],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    G: np.ndarray
    group_names: list[str]
    x_box: float = 2.0


def write_raw_breast_data() -> None:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df.insert(0, "target", data.target.astype(int))
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA, index=False)


def build_group_matrix(feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [str(v).strip().lower() for v in feature_names]
    group_names = list(BREAST_GROUPS.keys())
    G = np.zeros((len(names), len(group_names)), dtype=np.float64)
    for j, name in enumerate(names):
        assigned = None
        for group_name, keys in BREAST_GROUPS.items():
            if any(key in name for key in keys):
                assigned = group_name
                break
        if assigned is None:
            raise ValueError(f"Unassigned feature: {feature_names[j]}")
        G[j, group_names.index(assigned)] = 1.0
    return G, group_names


def load_breast() -> DatasetSpec:
    if not RAW_DATA.exists():
        write_raw_breast_data()
    df = pd.read_csv(RAW_DATA)
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = np.where(df["target"].to_numpy(dtype=int) == 1, 1.0, -1.0).astype(np.float64)
    G, group_names = build_group_matrix(feature_names)
    return DatasetSpec("breast", X, y, feature_names, G, group_names)


def split_indices_samplewise(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(y.shape[0])
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
    idx_tr, idx_hold = next(sss1.split(idx_all, y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    idx_val_rel, idx_te_rel = next(sss2.split(idx_hold, y[idx_hold]))
    return idx_tr, idx_hold[idx_val_rel], idx_hold[idx_te_rel]


def standardize(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu[None, :]) / sig[None, :]


def make_split(spec: DatasetSpec, seed: int) -> dict[str, np.ndarray]:
    idx_train, idx_val, idx_test = split_indices_samplewise(spec.y, seed)
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
    split["mu"] = mu
    split["sig"] = np.where(sig < 1e-12, 1.0, sig)
    return split


def enumerate_box_corners(m: int, eps: float) -> np.ndarray:
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
    Xp = apply_group_drift_torch(X_raw, y_unc, G)
    Z = (Xp - mu.unsqueeze(0)) / sig.unsqueeze(0)
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
    margins = tensors["ytr"] * (((tensors["Xtr_raw"] - tensors["mu"].unsqueeze(0)) / tensors["sig"].unsqueeze(0)) @ w + b)
    return F.softplus(-margins).mean() + lambda_l1 * torch.abs(w).sum()


def make_tensors(split: dict[str, np.ndarray], G: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "Xtr_raw": torch.tensor(split["Xtr_raw"], dtype=torch.float64),
        "ytr": torch.tensor(split["ytr"], dtype=torch.float64),
        "G": torch.tensor(G, dtype=torch.float64),
        "mu": torch.tensor(split["mu"], dtype=torch.float64),
        "sig": torch.tensor(split["sig"], dtype=torch.float64),
    }


def estimate_lmaj_from_box(split: dict[str, np.ndarray], G: np.ndarray, x_box: float, lmaj_factor: float) -> float:
    d = split["Xtr_raw"].shape[1]
    w_ref = np.full(d, float(x_box), dtype=np.float64)
    U = split["Xtr_raw"] / split["sig"][None, :]
    A = (U * w_ref[None, :]) @ G
    gram = (A.T @ A) / A.shape[0]
    lam_max = float(np.linalg.eigvalsh(gram).max())
    return float(max(1e-6, 0.25 * float(lmaj_factor) * lam_max))


def schedule_for_stage(k: int, N: int) -> tuple[float, float]:
    eps_tol = 0.1 * (0.9**k)
    mu = eps_tol / (2.0 * math.log(max(N, 2)))
    return eps_tol, mu


def projected_gradient_proxy(
    w: torch.Tensor,
    b: torch.Tensor,
    gw: torch.Tensor,
    gb: torch.Tensor,
    alpha: float,
    x_box: float,
) -> float:
    with torch.no_grad():
        w_next = torch.clamp(w - alpha * gw, -x_box, x_box)
        b_next = b - alpha * gb
        return float(torch.sqrt(torch.sum((w - w_next) ** 2) + (b - b_next) ** 2).cpu().item() / alpha)


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
    corners = enumerate_box_corners(spec.G.shape[1], eps)
    dataset_losses = [
        logistic_loss_np(w, b, X_raw, y, corner, spec.G, split["mu"], split["sig"]) for corner in corners
    ]
    robust = float(np.max(dataset_losses))

    worst_losses = np.full(X_raw.shape[0], -np.inf, dtype=np.float64)
    for corner in corners:
        Xp = apply_group_drift_np(X_raw, corner, spec.G)
        Z = standardize(Xp, split["mu"], split["sig"])
        margins = y * (Z @ w + b)
        worst_losses = np.maximum(worst_losses, np.logaddexp(0.0, -margins))
    k_tail = max(1, int(math.ceil((1.0 - ALPHA_CVAR) * worst_losses.size)))
    cvar = float(np.partition(worst_losses, worst_losses.size - k_tail)[worst_losses.size - k_tail :].mean())
    clean = logistic_loss_np(
        w,
        b,
        X_raw,
        y,
        np.zeros(spec.G.shape[1], dtype=np.float64),
        spec.G,
        split["mu"],
        split["sig"],
    )
    return {
        f"{subset}_robust_loss": robust,
        f"{subset}_cvar_loss": cvar,
        f"{subset}_clean_loss": clean,
    }


def run_robust_path(
    model: str,
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    eps: float,
    alpha0: float,
    lambda_l1: float,
    lmaj_factor: float | None,
) -> list[dict[str, object]]:
    tensors = make_tensors(split, spec.G)
    d = split["Xtr_raw"].shape[1]
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    _, mu0 = schedule_for_stage(0, N_GRID[0])
    bank_full = build_uniform_bank(spec.G.shape[1], eps, int(split["split_seed"]))
    Lmaj = estimate_lmaj_from_box(split, spec.G, spec.x_box, float(lmaj_factor)) if model == "majorant" else np.nan
    rows: list[dict[str, object]] = []
    cumulative_updates = 0

    for k, N in enumerate(N_GRID):
        eps_tol, mu_smooth = schedule_for_stage(k, N)
        alpha = alpha0 * mu_smooth / mu0
        Y_np = bank_full[:N].copy()
        Y_t = torch.tensor(Y_np, dtype=torch.float64)
        t0 = time.time()
        last_obj = np.nan
        last_proxy = np.nan
        failed = False

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
                raise ValueError(model)
            if not torch.isfinite(obj):
                failed = True
                break
            obj.backward()
            if w.grad is None or b.grad is None or not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
                failed = True
                break
            last_proxy = projected_gradient_proxy(w, b, w.grad, b.grad, alpha, spec.x_box)
            with torch.no_grad():
                w -= alpha * w.grad
                b -= alpha * b.grad
                w.clamp_(-spec.x_box, spec.x_box)
            cumulative_updates += 1
            last_obj = float(obj.detach().cpu().item())

        runtime = time.time() - t0
        with torch.no_grad():
            w_np = w.detach().cpu().numpy().copy()
            b_np = float(b.detach().cpu().item())
        rows.append(
            {
                "comparison": "strict_fair" if model == "direct" else "majorant_refined",
                "dataset": spec.name,
                "sampling_policy": "uniform_iid_nested",
                "model": model,
                "split_seed": int(split["split_seed"]),
                "epsilon": eps,
                "N": N,
                "stage_index": k,
                "alpha0": alpha0,
                "lambda_l1": lambda_l1,
                "lmaj_factor": np.nan if model == "direct" else float(lmaj_factor),
                "Lmaj": Lmaj,
                "eps_tol_k": eps_tol,
                "mu_k": mu_smooth,
                "stepsize": alpha,
                "inner_steps_per_stage": INNER_STEPS,
                "actual_update_count": cumulative_updates,
                "train_objective": last_obj,
                "stationarity_proxy": last_proxy,
                "runtime_sec": runtime,
                "x_box": spec.x_box,
                "m_groups": spec.G.shape[1],
                "group_names": ",".join(spec.group_names),
                "bank_seed": int(split["split_seed"] + BANK_SEED_OFFSET),
                "bank_hash": bank_hash(Y_np),
                "init_w_norm": 0.0,
                "init_b": 0.0,
                "failed": failed,
                **evaluate_sampled_train(split, spec, w_np, b_np, Y_np),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "tr"),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "val"),
                **evaluate_exact_subset(split, spec, eps, w_np, b_np, "te"),
            }
        )
    return rows


def train_direct_path(*args, **kwargs) -> list[dict[str, object]]:
    return run_robust_path("direct", *args, **kwargs)


def train_majorant_path(*args, **kwargs) -> list[dict[str, object]]:
    return run_robust_path("majorant", *args, **kwargs)


def train_erm(
    split: dict[str, np.ndarray],
    spec: DatasetSpec,
    alpha0: float,
    lambda_l1: float,
) -> list[dict[str, object]]:
    tensors = make_tensors(split, spec.G)
    d = split["Xtr_raw"].shape[1]
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    failed = False
    last_obj = np.nan
    last_proxy = np.nan
    t0 = time.time()

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

    runtime = time.time() - t0
    with torch.no_grad():
        w_np = w.detach().cpu().numpy().copy()
        b_np = float(b.detach().cpu().item())
    rows: list[dict[str, object]] = []
    for eps in EPS_GRID:
        for k, N in enumerate(N_GRID):
            eps_tol, mu_smooth = schedule_for_stage(k, N)
            rows.append(
                {
                    "comparison": "erm_clean_stage600",
                    "dataset": spec.name,
                    "sampling_policy": "clean",
                    "model": "erm",
                    "split_seed": int(split["split_seed"]),
                    "epsilon": eps,
                    "N": N,
                    "stage_index": k,
                    "alpha0": alpha0,
                    "lambda_l1": lambda_l1,
                    "lmaj_factor": np.nan,
                    "Lmaj": np.nan,
                    "eps_tol_k": eps_tol,
                    "mu_k": mu_smooth,
                    "stepsize": alpha0,
                    "inner_steps_per_stage": INNER_STEPS,
                    "actual_update_count": TOTAL_UPDATES,
                    "train_objective": last_obj,
                    "stationarity_proxy": last_proxy,
                    "runtime_sec": runtime,
                    "x_box": spec.x_box,
                    "m_groups": spec.G.shape[1],
                    "group_names": ",".join(spec.group_names),
                    "bank_seed": np.nan,
                    "bank_hash": "clean",
                    "init_w_norm": 0.0,
                    "init_b": 0.0,
                    "failed": failed,
                    "tr_sampled_robust_loss": np.nan,
                    "tr_sampled_mean_loss": np.nan,
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "tr"),
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "val"),
                    **evaluate_exact_subset(split, spec, eps, w_np, b_np, "te"),
                }
            )
    return rows


def run_training() -> pd.DataFrame:
    write_raw_breast_data()
    spec = load_breast()
    configs = pd.read_csv(CONFIG_FILE)
    rows: list[dict[str, object]] = []

    for split_seed in SPLIT_SEEDS:
        split = make_split(spec, split_seed)
        erm_cfg = configs[configs["model"] == "erm"].iloc[0]
        rows.extend(train_erm(split, spec, float(erm_cfg["alpha0"]), float(erm_cfg["lambda_l1"])))

        for eps in EPS_GRID:
            bank_cfgs = configs[(configs["epsilon"] == eps) & (configs["model"].isin(["direct", "majorant"]))]
            for model in ["direct", "majorant"]:
                model_cfgs = bank_cfgs[bank_cfgs["model"] == model]
                for (alpha0, lambda_l1, lmaj_factor), cfg_group in model_cfgs.groupby(
                    ["alpha0", "lambda_l1", "lmaj_factor"], dropna=False
                ):
                    lmaj = None if pd.isna(lmaj_factor) else float(lmaj_factor)
                    path_rows = run_robust_path(
                        model,
                        split,
                        spec,
                        eps,
                        float(alpha0),
                        float(lambda_l1),
                        lmaj,
                    )
                    keep_ns = set(int(v) for v in cfg_group["N"])
                    rows.extend([row for row in path_rows if int(row["N"]) in keep_ns])
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
    split_results[cols].sort_values(["model", "epsilon", "N", "split_seed"]).to_csv(
        PROCESSED_DIR / "final_split_results.csv",
        index=False,
    )
    print(f"Wrote {PROCESSED_DIR / 'final_split_results.csv'}")


if __name__ == "__main__":
    main()
