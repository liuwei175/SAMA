import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch
from pathlib import Path
from PIL import Image

"""Generate the two surface-comparison figures for the toy experiment."""

A_M_SCALE = 1.0 / (5 * np.pi)  # 1/(5*pi) - normalized by pi since y in [-pi, pi]
Y_LO, Y_HI = -np.pi, np.pi
L_Y = 1.5  # Strictly > 1.0 (since actual L=1 for sin/cos)

def _max_1d_linear_minus_sin(c, ylo=Y_LO, yhi=Y_HI):
    cand = [ylo, yhi]
    if abs(c) <= 1.0:
        y0 = float(np.arccos(c))
        cand.extend([y0, -y0])
    vals = [c*y - np.sin(y) for y in cand]
    return max(vals)

def _max_1d_linear_minus_cos(c, ylo=Y_LO, yhi=Y_HI):
    cand = [ylo, yhi]
    s = -c
    if abs(s) <= 1.0:
        a = float(np.arcsin(s))
        cand.extend([a, np.pi - a, -np.pi - a])
    cand = [y for y in cand if (ylo - 1e-12) <= y <= (yhi + 1e-12)]
    vals = [c*y - np.cos(y) for y in cand]
    return max(vals)

def calc_V_true(X1, X2, m):
    V = np.zeros_like(X1, dtype=float)
    for j in range(m):
        # Apply psi mapping: psi1 = X1^2 - X2^2, psi2 = 2*X1*X2
        psi1 = X1**2 - X2**2
        psi2 = 2.0 * X1 * X2
        if (j % 2) == 0:
            c = A_M_SCALE * psi1
            V += np.vectorize(_max_1d_linear_minus_sin)(c)
        else:
            c = A_M_SCALE * psi2
            V += np.vectorize(_max_1d_linear_minus_cos)(c)
    return V

def sample_y_deterministic(N, m, seed_base=0):
    seed = int(seed_base + 100000*m + N)
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(N, m))

def precompute_sample_stats(Y):
    # For psi mapping, we need Ay = A_m @ y where:
    #   psi1 = x1^2 - x2^2 contributes to even indices (j=1,3,5...)
    #   psi2 = 2*x1*x2 contributes to odd indices (j=2,4,6...)
    even = np.arange(Y.shape[1]) % 2 == 0  # indices 0,2,4... correspond to j=1,3,5...
    odd  = ~even                           # indices 1,3,5... correspond to j=2,4,6...
    Ay_even = Y[:, even].sum(axis=1) if even.any() else np.zeros(Y.shape[0])
    Ay_odd  = Y[:, odd].sum(axis=1)  if odd.any()  else np.zeros(Y.shape[0])
    C = np.sin(Y[:, even]).sum(axis=1) if even.any() else np.zeros(Y.shape[0])
    D = np.cos(Y[:, odd]).sum(axis=1)  if odd.any()  else np.zeros(Y.shape[0])
    cos_even = np.cos(Y[:, even]) if even.any() else np.zeros((Y.shape[0], 0))
    sin_odd  = np.sin(Y[:, odd])  if odd.any()  else np.zeros((Y.shape[0], 0))
    return Ay_even, Ay_odd, C, D, even, odd, cos_even, sin_odd

def calc_V_direct(X1, X2, Ay_even, Ay_odd, C, D):
    x1 = X1.reshape(-1, 1)
    x2 = X2.reshape(-1, 1)
    # Apply psi mapping
    psi1 = x1**2 - x2**2
    psi2 = 2.0 * x1 * x2
    vals = A_M_SCALE*(psi1*Ay_even.reshape(1,-1) + psi2*Ay_odd.reshape(1,-1)) - (C + D).reshape(1,-1)
    Vn = vals.max(axis=1)
    return Vn.reshape(X1.shape)

def calc_V_majorant(X1, X2, Y, Ay_even, Ay_odd, C, D, even_mask, odd_mask, cos_even, sin_odd):
    N, m = Y.shape
    K = X1.size
    V_maj = np.empty(K, dtype=float)
    X1f, X2f = X1.ravel(), X2.ravel()
    Y_even = Y[:, even_mask] if even_mask.any() else np.zeros((N,0))
    Y_odd  = Y[:, odd_mask]  if odd_mask.any()  else np.zeros((N,0))
    base_const = -(C + D)

    for k in range(K):
        x1, x2 = X1f[k], X2f[k]
        # Apply psi mapping
        psi1 = x1**2 - x2**2
        psi2 = 2.0 * x1 * x2
        fxy = A_M_SCALE*(psi1*Ay_even + psi2*Ay_odd) + base_const

        if even_mask.any():
            grad_even = (A_M_SCALE*psi1) - cos_even
            ystar_even = np.clip(Y_even + grad_even / L_Y, -np.pi, np.pi)
            d_even = ystar_even - Y_even
            term_even = (grad_even * d_even).sum(axis=1) - 0.5*L_Y*(d_even**2).sum(axis=1)
        else:
            term_even = 0.0

        if odd_mask.any():
            grad_odd = (A_M_SCALE*psi2) + sin_odd
            ystar_odd = np.clip(Y_odd + grad_odd / L_Y, -np.pi, np.pi)
            d_odd = ystar_odd - Y_odd
            term_odd = (grad_odd * d_odd).sum(axis=1) - 0.5*L_Y*(d_odd**2).sum(axis=1)
        else:
            term_odd = 0.0

        qi = fxy + term_even + term_odd
        V_maj[k] = qi.max()

    return V_maj.reshape(X1.shape)


def make_grid(res=41):
    xs = np.linspace(-1, 1, res)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    return X1, X2
def calc_err_metric(V_true, V_majorant, V_direct, eps=1e-12):
    e_prop = float(np.max(np.abs(V_true - V_majorant)))
    e_base = float(np.max(np.abs(V_true - V_direct)))
    err = 1.0 - e_prop / max(e_base, eps)
    return err, e_prop, e_base


def style_axes(ax, zlim, xticks, yticks, label_axes=True):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.tick_params(axis='both', which='major', labelsize=10, pad=-1)
    ax.tick_params(axis='z', labelsize=10, pad=-1)

    if label_axes:
        ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=-1)
        ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=-1)
        ax.set_zlabel("Function value", fontsize=11, labelpad=0)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    ax.view_init(elev=20, azim=-55)


def overlay_three_surfaces(ax, X1, X2, V_true, V_majorant, V_direct):
    # 1. True V (Ceiling): Dark wireframe for structure, highly transparent body
    ax.plot_surface(
        X1, X2, V_true,
        color="dimgray", alpha=0.15,
        edgecolor="black", linewidth=0.3,
        antialiased=True, shade=False
    )

    # 2. Proposed V_N (Middle): Blue
    ax.plot_surface(
        X1, X2, V_majorant,
        color="dodgerblue", alpha=0.5,
        linewidth=0, antialiased=True, shade=True
    )

    # 3. Baseline \bar{V}_N (Bottom): Red
    ax.plot_surface(
        X1, X2, V_direct,
        color="tomato", alpha=0.5,
        linewidth=0, antialiased=True, shade=True
    )


def add_column_labels_bottom(fig, axes, labels, pad=0.050, fontsize=13):
    # For two-line labels, centering w.r.t. the full axes box is more stable
    for ax, lab in zip(axes, labels):
        ax_bb = ax.get_position()
        xcenter = 0.5 * (ax_bb.x0 + ax_bb.x1)
        y = ax_bb.y0 - pad
        fig.text(
            xcenter, y, lab,
            ha="center", va="top",
            fontsize=fontsize,
            linespacing=1.15
        )


def add_global_legend(fig):
    handles = [
        Patch(facecolor="dimgray", edgecolor="black", alpha=0.3, label=r"Reference surface $V^m$"),
        Patch(facecolor="dodgerblue", edgecolor="dodgerblue", alpha=0.6, label=r"Majorant approximation $V_N^m$"),
        Patch(facecolor="tomato", edgecolor="tomato", alpha=0.6, label=r"Standard approximation $\bar{V}_N^m$"),
    ]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.94)
    )


def build_pack(X1, X2, N_list, m_list, seed_base=0):
    pack = []
    for m, N in zip(m_list, N_list):
        Y = sample_y_deterministic(N, m, seed_base=seed_base)
        A, B, C, D, even, odd, cos_even, sin_odd = precompute_sample_stats(Y)

        V_t = calc_V_true(X1, X2, m)
        V_d = calc_V_direct(X1, X2, A, B, C, D)
        V_m = calc_V_majorant(X1, X2, Y, A, B, C, D, even, odd, cos_even, sin_odd)

        err, e_prop, e_base = calc_err_metric(V_t, V_m, V_d)

        pack.append((m, N, V_t, V_m, V_d, err, e_prop, e_base))
    return pack


def global_zlim_from_pack(pack):
    zmin, zmax = np.inf, -np.inf
    for item in pack:
        _, _, Vt, Vm, Vd, *_ = item
        zmin = min(zmin, Vt.min(), Vm.min(), Vd.min())
        zmax = max(zmax, Vt.max(), Vm.max(), Vd.max())
    pad = 0.02 * (zmax - zmin + 1e-12)
    return (zmin - pad, zmax + pad)


def render_single_row(outdir, pack, zlim, is_sweep_N=True, grid_res=41, out_name="toy_1row.pdf"):
    X1, X2 = make_grid(grid_res)
    xticks, yticks = [-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1]

    # Keep the same compact layout; only slightly enlarge bottom margin
    fig = plt.figure(figsize=(13.6, 4.15))
    axes = []

    for c, item in enumerate(pack):
        m, N, V_t, V_m, V_d, err, e_prop, e_base = item
        ax = fig.add_subplot(1, 4, c + 1, projection='3d')
        overlay_three_surfaces(ax, X1, X2, V_t, V_m, V_d)
        style_axes(ax, zlim, xticks, yticks, label_axes=True)
        if c == 0:
            ax.set_zlabel("Function value", fontsize=11, labelpad=6)
        else:
            ax.set_zlabel("")
        axes.append(ax)

    labels = [
        rf"$m={m},\, N={N}$"
        for (m, N, *_rest, err, e_prop, e_base) in pack
    ]

    add_column_labels_bottom(fig, axes, labels, pad=0.060, fontsize=13)
    add_global_legend(fig)

    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.03)

    out = Path(outdir) / out_name
    fig.savefig(out, format="pdf", bbox_inches='tight')
    png_out = Path(outdir) / out_name.replace(".pdf", ".png")
    fig.savefig(png_out, format="png", dpi=180, bbox_inches='tight')
    eps_out = Path(outdir) / out_name.replace(".pdf", ".eps")
    Image.open(png_out).convert("RGB").save(eps_out, "EPS")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")
    print(f"Saved: {eps_out.resolve()}")
    print(f"Saved: {png_out.resolve()}")


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parents[1] / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    grid_res = 41

    X1, X2 = make_grid(grid_res)

    # Figure 1: fixed m=5, vary N
    pack_sweep_N = build_pack(
        X1, X2,
        N_list=[20, 50, 100, 200],
        m_list=[5, 5, 5, 5]
    )

    # Figure 2: fixed N=20, vary m
    pack_sweep_m = build_pack(
        X1, X2,
        N_list=[20, 20, 20, 20],
        m_list=[2, 3, 4, 5]
    )

    # Shared z-limits across the two figures
    zlim_global = global_zlim_from_pack(pack_sweep_N + pack_sweep_m)

    print("Rendering Figure 1: Effect of Sample Size ...")
    render_single_row(
        outdir, pack_sweep_N, zlim_global,
        is_sweep_N=True,
        out_name="toy_fixedm_sweepN_1row.pdf"
    )

    print("Rendering Figure 2: Effect of Inner Dimension ...")
    render_single_row(
        outdir, pack_sweep_m, zlim_global,
        is_sweep_N=False,
        out_name="toy_fixedN_sweepm_1row.pdf"
    )
