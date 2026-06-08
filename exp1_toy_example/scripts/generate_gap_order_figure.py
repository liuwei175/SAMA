#!/usr/bin/env python3
"""Generate the value-gap and beta-order figure for the toy experiment."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

A_COEFF = 0.2
Y_LO, Y_HI = -np.pi, np.pi
L_Y = 1.5


def split_odd_even_m(m: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(m)
    odd = (idx % 2) == 0
    even = ~odd
    return odd, even


def psi_values(x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x1**2 - x2**2, 2.0 * x1 * x2


def one_dim_max_linear_minus_sin(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    out = np.maximum(c * Y_LO - np.sin(Y_LO), c * Y_HI - np.sin(Y_HI))
    mask = np.abs(c) <= 1.0
    if np.any(mask):
        y0 = np.arccos(c[mask])
        vals = np.maximum(c[mask] * y0 - np.sin(y0), c[mask] * (-y0) - np.sin(-y0))
        out = out.copy()
        out[mask] = np.maximum(out[mask], vals)
    return out


def one_dim_max_linear_minus_cos(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    out = np.maximum(c * Y_LO - np.cos(Y_LO), c * Y_HI - np.cos(Y_HI))
    mask = np.abs(c) <= 1.0
    if np.any(mask):
        a = np.arcsin(-c[mask])
        trial_points = [a, np.pi - a, -np.pi - a]
        vals = []
        for y in trial_points:
            valid = (Y_LO <= y) & (y <= Y_HI)
            v = np.full_like(y, -np.inf, dtype=float)
            v[valid] = c[mask][valid] * y[valid] - np.cos(y[valid])
            vals.append(v)
        out = out.copy()
        out[mask] = np.maximum(out[mask], np.maximum.reduce(vals))
    return out


def true_value_grid(xgrid: np.ndarray, m: int) -> np.ndarray:
    x1, x2 = np.meshgrid(xgrid, xgrid, indexing="ij")
    psi1, psi2 = psi_values(x1, x2)
    n_odd = (m + 1) // 2
    n_even = m // 2
    value = np.zeros_like(x1, dtype=float)
    if n_odd:
        value += n_odd * one_dim_max_linear_minus_sin(A_COEFF * psi1)
    if n_even:
        value += n_even * one_dim_max_linear_minus_cos(A_COEFF * psi2)
    return value


def sample_pool(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    y = rng.uniform(Y_LO, Y_HI, size=(n, m))
    return y[rng.permutation(n)]


def sample_stats(y_pool: np.ndarray) -> dict[str, np.ndarray]:
    n, m = y_pool.shape
    odd, even = split_odd_even_m(m)
    return {
        "odd": odd,
        "even": even,
        "sum_odd": y_pool[:, odd].sum(axis=1) if odd.any() else np.zeros(n),
        "sum_even": y_pool[:, even].sum(axis=1) if even.any() else np.zeros(n),
        "sin_odd_sum": np.sin(y_pool[:, odd]).sum(axis=1) if odd.any() else np.zeros(n),
        "cos_even_sum": np.cos(y_pool[:, even]).sum(axis=1) if even.any() else np.zeros(n),
        "cos_odd": np.cos(y_pool[:, odd]) if odd.any() else np.zeros((n, 0)),
        "sin_even": np.sin(y_pool[:, even]) if even.any() else np.zeros((n, 0)),
        "y_odd": y_pool[:, odd] if odd.any() else np.zeros((n, 0)),
        "y_even": y_pool[:, even] if even.any() else np.zeros((n, 0)),
    }


def direct_value_grid(xgrid: np.ndarray, stats: dict[str, np.ndarray], n: int) -> np.ndarray:
    x1, x2 = np.meshgrid(xgrid, xgrid, indexing="ij")
    psi1, psi2 = psi_values(x1[..., None], x2[..., None])
    vals = (
        A_COEFF * (psi1 * stats["sum_odd"][:n] + psi2 * stats["sum_even"][:n])
        - stats["sin_odd_sum"][:n]
        - stats["cos_even_sum"][:n]
    )
    return vals.max(axis=2)


def majorant_value_grid(xgrid: np.ndarray, y_pool: np.ndarray, stats: dict[str, np.ndarray], n: int) -> np.ndarray:
    x1, x2 = np.meshgrid(xgrid, xgrid, indexing="ij")
    output = np.full_like(x1, -np.inf, dtype=float)
    y_odd = stats["y_odd"][:n]
    y_even = stats["y_even"][:n]
    cos_odd = stats["cos_odd"][:n]
    sin_even = stats["sin_even"][:n]

    for i in range(n):
        psi1, psi2 = psi_values(x1, x2)
        base = (
            A_COEFF * (psi1 * stats["sum_odd"][i] + psi2 * stats["sum_even"][i])
            - stats["sin_odd_sum"][i]
            - stats["cos_even_sum"][i]
        )
        correction = np.zeros_like(x1)
        if y_odd.shape[1]:
            grad_odd = A_COEFF * psi1[..., None] - cos_odd[i]
            y_star_odd = np.clip(y_odd[i] + grad_odd / L_Y, Y_LO, Y_HI)
            d_odd = y_star_odd - y_odd[i]
            correction += (grad_odd * d_odd).sum(axis=2) - 0.5 * L_Y * (d_odd**2).sum(axis=2)
        if y_even.shape[1]:
            grad_even = A_COEFF * psi2[..., None] + sin_even[i]
            y_star_even = np.clip(y_even[i] + grad_even / L_Y, Y_LO, Y_HI)
            d_even = y_star_even - y_even[i]
            correction += (grad_even * d_even).sum(axis=2) - 0.5 * L_Y * (d_even**2).sum(axis=2)
        output = np.maximum(output, base + correction)
    return output


def minimizer_gap(model_grid: np.ndarray, true_grid: np.ndarray, v_star: float) -> float:
    idx = int(np.argmin(model_grid))
    value_at_model_min = float(true_grid.ravel()[idx])
    return max(value_at_model_min - v_star, 1e-16)


def midpoint_tensor_cover(k: int, m: int) -> tuple[np.ndarray, float]:
    one_dim = Y_LO + (np.arange(k, dtype=float) + 0.5) * ((Y_HI - Y_LO) / k)
    mesh = np.meshgrid(*([one_dim] * m), indexing="ij")
    y_pool = np.stack([arr.ravel() for arr in mesh], axis=1)
    beta_n = float(np.sqrt(m) * ((Y_HI - Y_LO) / k) / 2.0)
    return y_pool, beta_n


def run_beta_order_experiment(m: int = 2, k_list: tuple[int, ...] = (6, 8, 10, 12, 16, 20, 24, 32, 40), nx: int = 61) -> dict[str, np.ndarray]:
    xgrid = np.linspace(-1.0, 1.0, nx)
    true_grid = true_value_grid(xgrid, m)
    v_star = float(true_grid.min())
    beta_values, direct_errors, majorant_errors, sample_sizes = [], [], [], []
    for k in k_list:
        y_pool, beta_n = midpoint_tensor_cover(k, m)
        stats = sample_stats(y_pool)
        n = y_pool.shape[0]
        direct_grid = direct_value_grid(xgrid, stats, n)
        majorant_grid = majorant_value_grid(xgrid, y_pool, stats, n)
        beta_values.append(beta_n)
        sample_sizes.append(n)
        direct_errors.append(float(np.max(np.abs(true_grid - direct_grid))))
        majorant_errors.append(float(np.max(np.abs(true_grid - majorant_grid))))
    return {
        "Ns": np.asarray(sample_sizes, dtype=int),
        "beta_N": np.asarray(beta_values, dtype=float),
        "uniform_gap_barVN": np.asarray(direct_errors, dtype=float),
        "uniform_gap_VN": np.asarray(majorant_errors, dtype=float),
        "V_star": np.asarray(v_star),
    }


def run_relative_gap_experiment(m_list: tuple[int, ...] = (2, 3, 5), n_total: int = 200, step_n: int = 10, n_repeats: int = 20, nx: int = 61, seeds: tuple[int, ...] = (1, 42, 2024)) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, float]]:
    xgrid = np.linspace(-1.0, 1.0, nx)
    ns = np.r_[1, np.arange(step_n, n_total + 1, step_n)]
    results: dict[int, dict[str, np.ndarray]] = {}
    v_star_by_m: dict[int, float] = {}
    for m in m_list:
        true_grid = true_value_grid(xgrid, m)
        v_star = float(true_grid.min())
        v_star_by_m[m] = v_star
        direct_gap = np.zeros((len(ns), n_repeats * len(seeds)))
        majorant_gap = np.zeros_like(direct_gap)
        for s_idx, seed_base in enumerate(seeds):
            for r in range(n_repeats):
                col = s_idx * n_repeats + r
                rng = np.random.default_rng(seed_base + 1000 * m + r)
                y_pool = sample_pool(n_total, m, rng)
                stats = sample_stats(y_pool)
                for k, n in enumerate(ns):
                    n_int = int(n)
                    direct_grid = direct_value_grid(xgrid, stats, n_int)
                    majorant_grid = majorant_value_grid(xgrid, y_pool, stats, n_int)
                    direct_gap[k, col] = minimizer_gap(direct_grid, true_grid, v_star)
                    majorant_gap[k, col] = minimizer_gap(majorant_grid, true_grid, v_star)
        results[m] = {
            "Ns": ns,
            "err_barVN": direct_gap.mean(axis=1),
            "err_VN": majorant_gap.mean(axis=1),
        }
    return results, v_star_by_m


def loglog_slope(x: np.ndarray, y: np.ndarray, tail: int = 5) -> float:
    mask = (x > 0.0) & (y > 0.0)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if len(x) > tail:
        x = x[-tail:]
        y = y[-tail:]
    return float(np.polyfit(np.log(x), np.log(y), deg=1)[0])


def plot_beta_order(beta_results: dict[str, np.ndarray], relative_results: dict[int, dict[str, np.ndarray]], v_star_by_m: dict[int, float], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    beta_n = beta_results["beta_N"]
    gap_direct = beta_results["uniform_gap_barVN"]
    gap_majorant = beta_results["uniform_gap_VN"]
    direct_slope = loglog_slope(beta_n, gap_direct)
    majorant_slope = loglog_slope(beta_n, gap_majorant)
    beta_ref = np.geomspace(beta_n.min(), beta_n.max(), 200)
    direct_ref = gap_direct[-1] * (beta_ref / beta_n[-1])
    majorant_ref = gap_majorant[-1] * (beta_ref / beta_n[-1]) ** 2

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7.6, 2.75))
    fig.subplots_adjust(wspace=0.34, left=0.10, right=0.98, top=0.96, bottom=0.23)

    ax1.loglog(beta_n, gap_direct, marker="o", linestyle="--", color="#1f77b4", markersize=3.0, linewidth=1.35, label=rf"$\bar{{V}}_N$ ($order={direct_slope:.2f}$)")
    ax1.loglog(beta_n, gap_majorant, marker="^", linestyle="-", color="#d62728", markersize=3.4, linewidth=1.35, label=rf"$V_N$ ($order={majorant_slope:.2f}$)")
    ax1.loglog(beta_ref, direct_ref, linestyle="-.", color="#1f77b4", linewidth=1.0, label=r"$O(\beta_N)$")
    ax1.loglog(beta_ref, majorant_ref, linestyle=":", color="#d62728", linewidth=1.2, label=r"$O(\beta_N^2)$")
    ax1.set_xlabel(r"coverage radius $\beta_N$")
    ax1.set_ylabel(r"uniform error $\|\cdot - V^2\|_\infty$")
    ax1.set_yticks([1e-1, 1e-2])
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.legend(loc="lower right", frameon=True, framealpha=0.86, fancybox=False, borderpad=0.3)

    style_map = {
        2: dict(color="#1f77b4", vn_marker="o", bar_marker="x"),
        3: dict(color="#ff7f0e", vn_marker="s", bar_marker="D"),
        5: dict(color="#2ca02c", vn_marker="^", bar_marker="v"),
    }
    handles, labels = [], []
    for m in sorted(relative_results):
        dat = relative_results[m]
        style = style_map[m]
        line_vn, = ax2.semilogy(dat["Ns"], dat["err_VN"] / abs(v_star_by_m[m]), linestyle="-", marker=style["vn_marker"], color=style["color"], markersize=3.0, markevery=2, linewidth=1.15)
        line_bar, = ax2.semilogy(dat["Ns"], dat["err_barVN"] / abs(v_star_by_m[m]), linestyle=(0, (4, 2)), marker=style["bar_marker"], markerfacecolor="white", markeredgecolor=style["color"], color=style["color"], markersize=3.0, markevery=2, linewidth=1.15)
        handles.extend([line_vn, line_bar])
        labels.extend([r"$V_N$", rf"$\bar{{V}}_N$, $m={m}$"])
    ax2.set_xlim(0, 200)
    ax2.set_xticks(np.arange(0, 201, 40))
    ax2.set_xlabel(r"sample size $N$")
    ax2.set_ylabel(r"relative value gap in $V^m$")
    ax2.set_yticks([1e0, 1e-1, 1e-2])
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.legend(handles, labels, loc="lower left", ncol=2, frameon=True, framealpha=0.86, fancybox=False, borderpad=0.3, handlelength=3.2, columnspacing=0.8, handletextpad=0.45)

    ax1.set_xlim(beta_n.min() * 0.9, beta_n.max() * 1.1)
    ax1.invert_xaxis()
    ax1.set_xticks([0.8, 0.4, 0.2, 0.1])
    ax1.set_xticklabels(["0.8", "0.4", "0.2", "0.1"])
    ax1.xaxis.set_minor_formatter(NullFormatter())
    for ax in (ax1, ax2):
        ax.grid(True, which="major", linewidth=0.32, alpha=0.22)

    fig.savefig(outdir / "toy_beta_order.pdf", bbox_inches="tight")
    fig.savefig(outdir / "toy_beta_order.eps", format="eps", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start = time.time()
    outdir = Path(__file__).resolve().parents[1] / "results" / "figures"
    beta_results = run_beta_order_experiment(m=2, nx=61)
    relative_results, v_star_by_m = run_relative_gap_experiment(m_list=(2, 3, 5), n_total=200, step_n=10, n_repeats=20, nx=61, seeds=(1, 42, 2024))
    plot_beta_order(beta_results, relative_results, v_star_by_m, outdir)
    print(f"Saved toy_beta_order figure in {outdir}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
