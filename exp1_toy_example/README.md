# Toy Example Reproducible Package

This package contains the code and final figure files for Experiment 1,
the toy minimax example in Section 4.1 of the paper.

## Experiment Setting

The toy family is

\[
\min_{x\in[-1,1]^2} V^m(x),\qquad
V^m(x)=\max_{y\in[-\pi,\pi]^m}
\psi(x)^\top A_m y-\mathbf{1}^\top(S_o\sin y+S_e\cos y),
\]

where

\[
\psi(x)=(x_1^2-x_2^2,\;2x_1x_2)^\top.
\]

The direct sampled model \(\bar V_N^m\) and the majorant model \(V_N^m\)
are always built from the same sample set.

## Figures

The final paper figures are stored in `results/figures`:

- `toy_fixedm_sweepN_N20.eps`, `toy_fixedm_sweepN_N50.eps`,
  `toy_fixedm_sweepN_N100.eps`, and `toy_fixedm_sweepN_N200.eps`:
  fixed \(m=5\), with one file per value of \(N\).
- `toy_fixedN_sweepm_m2.eps`, `toy_fixedN_sweepm_m3.eps`,
  `toy_fixedN_sweepm_m4.eps`, and `toy_fixedN_sweepm_m5.eps`:
  fixed \(N=20\), with one file per value of \(m\).
- `toy_beta_order_left.eps` and `toy_beta_order_right.eps`:
  value-gap and empirical beta-order comparison panels.
- `growing_samples_multi_m_left.eps` and `growing_samples_multi_m_right.eps`:
  growing-sample stationarity and value-gap panels.

PDF copies are included for quick preview.  The growing-sample data used for
the fourth figure are stored in `results/data/growing_samples_multi_m_data.csv`.

## Scripts

- `scripts/generate_surface_figures.py` regenerates the surface figure panels.
- `scripts/generate_gap_order_figure.py` regenerates the value-gap and beta-order panels.
- `scripts/generate_growing_sample_figure.py` regenerates the growing-sample panels and CSV. Use `--plot-only` to redraw the panels from the existing CSV without rerunning the experiment.

The growing-sample experiment uses \(K=200\) stages with
\(N_k=5(k+1)\), \(50\) independent repetitions, and nested sample streams.
The smoothing schedule is
\[
\varepsilon_k=\max\{10^{-8},10^{-1}(0.9)^k\},\qquad
\mu_k=\varepsilon_k/(2\log N_k).
\]

Run scripts from the package root, for example:

```bash
python scripts/generate_surface_figures.py
```
