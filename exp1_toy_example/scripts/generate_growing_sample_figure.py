#!/usr/bin/env python3
"""Generate the growing-sample stationarity and value-gap figure."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(slots=True)
class ExperimentConfig:
    l_smooth: float = 1.0
    k_stages: int = 200
    n0: int = 5
    dn: int = 5
    n_pool: int = 1000
    max_inner_iters: int = 10_000_000
    eps0: float = 1e-1
    eps_min: float = 1e-8
    n_repeats: int = 50
    ms: tuple[int, ...] = (2, 3, 5)
    seed: int = 42
    y_grid_size: int = 201
    x_grid_size: int = 81
    output_eps: Path = Path("results/figures/growing_samples_multi_m.eps")
    output_pdf: Path = Path("results/figures/growing_samples_multi_m.pdf")
    output_csv: Path = Path("results/data/growing_samples_multi_m_data.csv")

    def ns(self) -> np.ndarray:
        values = self.n0 + np.arange(self.k_stages) * self.dn
        if int(values[-1]) > self.n_pool:
            raise ValueError("The largest sample size exceeds the presampled pool.")
        return values.astype(int)

    def eps_sequence(self) -> np.ndarray:
        k = np.arange(self.k_stages, dtype=float)
        return np.maximum(self.eps_min, self.eps0 * (0.9 ** k))


@dataclass(slots=True)
class StageData:
    y: np.ndarray
    m: int
    odd: np.ndarray
    even: np.ndarray
    sum_odd: np.ndarray
    sin_odd_sum: np.ndarray
    cos_odd: np.ndarray
    sum_even: np.ndarray
    cos_even_sum: np.ndarray
    sin_even: np.ndarray
    gv_max: float


def psi(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] ** 2 - x[1] ** 2, 2.0 * x[0] * x[1]], dtype=float)


def psi_jacobian_transpose(x: np.ndarray) -> np.ndarray:
    return np.array([[2.0 * x[0], 2.0 * x[1]], [-2.0 * x[1], 2.0 * x[0]]], dtype=float)


def make_y_grid(size: int) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, size, dtype=float)


def one_dim_max_odd(x1: float, y_grid: np.ndarray) -> float:
    return float(np.max(0.2 * x1 * y_grid - np.sin(y_grid)))


def one_dim_max_even(x2: float, y_grid: np.ndarray) -> float:
    return float(np.max(0.2 * x2 * y_grid - np.cos(y_grid)))


def true_value_single_m(x: np.ndarray, m: int, y_grid: np.ndarray) -> float:
    n_odd = (m + 1) // 2
    n_even = m // 2
    psi_x = psi(x)
    value = 0.0
    if n_odd > 0:
        value += n_odd * one_dim_max_odd(float(psi_x[0]), y_grid)
    if n_even > 0:
        value += n_even * one_dim_max_even(float(psi_x[1]), y_grid)
    return value


def true_optimum_m(m: int, x_grid_size: int, y_grid: np.ndarray) -> tuple[float, np.ndarray]:
    x_grid = np.linspace(-1.0, 1.0, x_grid_size, dtype=float)
    best_value = np.inf
    x_star = np.zeros(2, dtype=float)
    for x1 in x_grid:
        for x2 in x_grid:
            x = np.array([x1, x2], dtype=float)
            value = true_value_single_m(x, m, y_grid)
            if value < best_value:
                best_value = value
                x_star = x
    return float(best_value), x_star


def build_stage_data(y: np.ndarray, m: int) -> StageData:
    odd = np.arange(0, m, 2, dtype=int)
    even = np.arange(1, m, 2, dtype=int)
    y_odd = y[:, odd]
    sum_odd = np.sum(y_odd, axis=1)
    sin_odd_sum = np.sum(np.sin(y_odd), axis=1)
    cos_odd = np.cos(y_odd)
    if even.size:
        y_even = y[:, even]
        sum_even = np.sum(y_even, axis=1)
        cos_even_sum = np.sum(np.cos(y_even), axis=1)
        sin_even = np.sin(y_even)
    else:
        n = y.shape[0]
        sum_even = np.zeros(n, dtype=float)
        cos_even_sum = np.zeros(n, dtype=float)
        sin_even = np.empty((n, 0), dtype=float)
    n_odd = odd.size
    n_even = even.size
    ay_max = 0.2 * np.pi * np.sqrt(n_odd**2 + n_even**2)
    gv_max = float(2.0 * np.sqrt(2.0) * ay_max)
    return StageData(y, m, odd, even, sum_odd, sin_odd_sum, cos_odd, sum_even, cos_even_sum, sin_even, gv_max)


def smooth_value_grad(x: np.ndarray, data: StageData, l_smooth: float, mu: float) -> tuple[float, np.ndarray]:
    n = data.y.shape[0]
    psi_x = psi(x)
    fxy = 0.2 * (psi_x[0] * data.sum_odd + psi_x[1] * data.sum_even) - data.sin_odd_sum - data.cos_even_sum
    gy = np.zeros((n, data.m), dtype=float)
    gy[:, data.odd] = 0.2 * psi_x[0] - data.cos_odd
    if data.even.size:
        gy[:, data.even] = 0.2 * psi_x[1] + data.sin_even
    y_star = np.clip(data.y + gy / l_smooth, -np.pi, np.pi)
    dy = y_star - data.y
    v = fxy + np.sum(gy * dy, axis=1) - 0.5 * l_smooth * np.sum(dy**2, axis=1)
    v_max = float(np.max(v))
    weights_raw = np.exp((v - v_max) / mu)
    weights = weights_raw / float(np.sum(weights_raw))
    v_tilde = v_max + mu * np.log(float(np.sum(weights_raw)))
    g1 = 0.2 * np.sum(y_star[:, data.odd], axis=1)
    g2 = 0.2 * np.sum(y_star[:, data.even], axis=1) if data.even.size else np.zeros(n, dtype=float)
    grad_psi = np.array([np.dot(g1, weights), np.dot(g2, weights)], dtype=float)
    grad_x = psi_jacobian_transpose(x) @ grad_psi
    return v_tilde, grad_x


def stationarity_residual_from_grad(x: np.ndarray, grad: np.ndarray) -> float:
    tol = 1e-6
    residual = np.zeros(2, dtype=float)
    for i in range(2):
        if -1.0 + tol < x[i] < 1.0 - tol:
            residual[i] = abs(grad[i])
        elif x[i] <= -1.0 + tol:
            residual[i] = max(0.0, -float(grad[i]))
        else:
            residual[i] = max(0.0, float(grad[i]))
    return float(np.linalg.norm(residual))


def stationarity_residual(x: np.ndarray, data: StageData, l_smooth: float, mu: float) -> float:
    _, grad = smooth_value_grad(x, data, l_smooth, mu)
    return stationarity_residual_from_grad(x, grad)


def inner_solve_stage(x0: np.ndarray, data: StageData, l_smooth: float, mu: float, eps_k: float, res_prev: float, max_iter: int) -> tuple[np.ndarray, int, float]:
    l_mu = (data.gv_max**2) / mu + l_smooth
    alpha = 1.0 / (2.0 * l_mu)
    x = np.array(x0, dtype=float, copy=True)
    _, grad0 = smooth_value_grad(x, data, l_smooth, mu)
    best_res = stationarity_residual_from_grad(x, grad0)
    best_x = x.copy()
    target = max(eps_k, 1e-4 * res_prev)
    it_used = 0
    for it in range(1, max_iter + 1):
        _, grad_x = smooth_value_grad(x, data, l_smooth, mu)
        x = np.clip(x - alpha * grad_x, -1.0, 1.0)
        _, grad_next = smooth_value_grad(x, data, l_smooth, mu)
        res = stationarity_residual_from_grad(x, grad_next)
        if res < best_res:
            best_res = res
            best_x = x.copy()
        it_used = it
        if res <= target:
            break
    return best_x, it_used, best_res


def save_results_figure(ns: np.ndarray, ms: tuple[int, ...], mean_residual: np.ndarray, mean_gap: np.ndarray, output_eps: Path, output_pdf: Path) -> None:
    output_eps.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.75))
    fig.subplots_adjust(wspace=0.34, left=0.10, right=0.98, top=0.96, bottom=0.23)
    markers = ("o", "s", "^", "D", "v", "P")
    markevery = max(1, len(ns) // 12)
    for idx, m in enumerate(ms):
        ax1.semilogy(ns, np.maximum(mean_residual[:, idx], 1e-16), "-", linewidth=1.0, marker=markers[idx], markersize=3.0, markevery=markevery, label=rf"$m={m}$")
        ax2.semilogy(ns, np.maximum(mean_gap[:, idx], 1e-16), "-", linewidth=1.0, marker=markers[idx], markersize=3.0, markevery=markevery, label=rf"$m={m}$")
    ax1.set_xlabel(r"sample size $N$")
    ax1.set_ylabel(r"mean $r_k^m(\hat{x}_k^m)$")
    ax1.set_xlim(int(ns[0]), int(ns[-1]))
    ax1.set_ylim(1e-4, 1e-1)
    ax1.legend(loc="best")
    ax2.set_xlabel(r"sample size $N$")
    ax2.set_ylabel(r"mean $V^m(\hat{x}_k^m)-\nu^{m,\star}$")
    ax2.set_xlim(int(ns[0]), int(ns[-1]))
    ax2.set_ylim(1e-4, 1e0)
    ax2.legend(loc="best")
    fig.savefig(output_pdf, format="pdf")
    fig.savefig(output_eps, format="eps")
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg.output_eps = root / cfg.output_eps
    cfg.output_pdf = root / cfg.output_pdf
    cfg.output_csv = root / cfg.output_csv
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    ns = cfg.ns()
    eps_seq = cfg.eps_sequence()
    y_grid = make_y_grid(cfg.y_grid_size)
    mean_residual = np.zeros((cfg.k_stages, len(cfg.ms)), dtype=float)
    mean_gap = np.zeros_like(mean_residual)
    fields = ["m", "repeat", "stage", "N", "eps", "mu", "x1", "x2", "inner_iterations", "smoothed_stationarity_residual", "reference_value", "reference_value_gap", "reference_v_star", "reference_x_star_1", "reference_x_star_2"]
    with cfg.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for im, m in enumerate(cfg.ms):
            v_star, x_star = true_optimum_m(m, cfg.x_grid_size, y_grid)
            gaps = np.zeros((cfg.k_stages, cfg.n_repeats), dtype=float)
            residuals = np.zeros_like(gaps)
            for r in range(cfg.n_repeats):
                y_pool = rng.uniform(-np.pi, np.pi, size=(cfg.n_pool, m))
                y_pool = y_pool[rng.permutation(cfg.n_pool), :]
                x = np.array([-0.5, 0.5], dtype=float)
                first_data = build_stage_data(y_pool[: ns[0], :], m)
                res_prev = stationarity_residual(x, first_data, cfg.l_smooth, eps_seq[0])
                for k in range(cfg.k_stages):
                    n_k = int(ns[k])
                    eps_k = float(eps_seq[k])
                    mu_k = eps_k / (2.0 * np.log(max(n_k, 2)))
                    data = build_stage_data(y_pool[:n_k, :], m)
                    x, it_used, res_out = inner_solve_stage(x, data, cfg.l_smooth, mu_k, eps_k, res_prev, cfg.max_inner_iters)
                    res_prev = res_out
                    ref_value = true_value_single_m(x, m, y_grid)
                    gap = ref_value - v_star
                    gaps[k, r] = gap
                    residuals[k, r] = res_out
                    writer.writerow({
                        "m": m,
                        "repeat": r + 1,
                        "stage": k + 1,
                        "N": n_k,
                        "eps": eps_k,
                        "mu": mu_k,
                        "x1": x[0],
                        "x2": x[1],
                        "inner_iterations": it_used,
                        "smoothed_stationarity_residual": res_out,
                        "reference_value": ref_value,
                        "reference_value_gap": gap,
                        "reference_v_star": v_star,
                        "reference_x_star_1": x_star[0],
                        "reference_x_star_2": x_star[1],
                    })
            mean_gap[:, im] = gaps.mean(axis=1)
            mean_residual[:, im] = residuals.mean(axis=1)
    save_results_figure(ns, cfg.ms, mean_residual, mean_gap, cfg.output_eps, cfg.output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the toy growing-sample figure.")
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--k-stages", type=int, default=200)
    parser.add_argument("--n-pool", type=int, default=1000)
    parser.add_argument("--max-inner-iters", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        n_repeats=args.n_repeats,
        k_stages=args.k_stages,
        n_pool=args.n_pool,
        max_inner_iters=args.max_inner_iters,
        seed=args.seed,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
