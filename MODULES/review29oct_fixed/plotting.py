#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/plotting.py

Corner plots and 1-D marginals shared by reference_posterior.py and postprocessing.py.

Both the reference and the VAE posterior are drawn from plain, equally weighted samples with the
same density estimator: a Gaussian kernel (Scott bandwidth) reflected at the bounds [lbound, 1],
so mass piled up at z = 1 (healthy parameters) is not smeared outside the box. Overlay contours
enclose 50 % (solid) and 90 % (dashed) of the probability mass.
"""

import os
import logging
import functools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

LABELS_7D = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
TITLES_7D = ["Zone $z_1$", "Zone $z_2$", "Zone $z_3$", "Zone $z_4$", "Zone $z_5$",
             "Left bearing $z_{b,L}$", "Right bearing $z_{b,R}$"]
RAW_NAMES = ["z_1 (D1)", "z_2 (D2)", "z_3 (D3)", "z_4 (D4)", "z_5 (D5)", "z_6 (BL)", "z_7 (BR)"]
RED, BLUE = "#D62728", "#1F77B4"


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Journal style: Times-like serif text and STIX math, embedded TrueType fonts in PDF (ASCE requires embedded fonts),
# 300 dpi raster output. Applied on top of matplotlib's defaults, whatever style the console has set.
PAPER_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.9, "axes.edgecolor": "black", "axes.titleweight": "normal",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.framealpha": 0.95, "legend.edgecolor": "0.6",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def fixed_style(fn):
    """Draws with the journal style PAPER_RC, whatever style the console has set (e.g. seaborn in Spyder)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.style.context(["default", PAPER_RC]):
            return fn(*args, **kwargs)
    return wrapper


def save_figure(fig, path, dpi=300):
    """Saves `path` as PNG (dpi) and the same figure as vector PDF next to it (for the manuscript)."""
    stem = os.path.splitext(path)[0]
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")


# =============================================================================
# DENSITY ESTIMATES
# =============================================================================
def kde_1d(samples, grid, lb, ub=1.0):
    """Reflected Gaussian KDE (Scott bandwidth) on [lb, ub]."""
    s = np.asarray(samples, dtype=np.float64)
    if np.std(s) < 1e-9:
        s = s + np.random.default_rng(0).normal(0.0, 1e-4, size=s.shape)
    k = gaussian_kde(s)
    return k(grid) + k(2.0 * ub - grid) + k(2.0 * lb - grid)


def density_2d(x, y, lb, ub=1.0, n_bins=160):
    """Histogram smoothed with a reflected Gaussian filter (Scott bandwidth per axis). Returns X, Y, Z."""
    edges = np.linspace(lb, ub, n_bins + 1)
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    width = edges[1] - edges[0]
    n = len(x)
    sig = [max(np.std(v) * n ** (-1.0 / 6.0) / width, 0.6) for v in (x, y)]
    Z = gaussian_filter(H, sigma=sig, mode="reflect").T / (n * width * width)
    c = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(c, c)
    return X, Y, Z


def hpd_levels(Z, masses=(0.9, 0.5)):
    """Density thresholds of the highest-density regions holding `masses` (returned in increasing order)."""
    z = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(z) / (np.sum(z) + 1e-300)
    levels = [z[min(np.searchsorted(cum, m), len(z) - 1)] for m in masses]
    return sorted(set(levels))


def summary_stats(samples):
    return np.mean(samples, axis=0), np.std(samples, axis=0)


# =============================================================================
# SINGLE-POSTERIOR CORNER (style of review29oct "P{pos}_GT.png" / "P{pos}_VAE.png")
# =============================================================================
@fixed_style
def corner_single(samples, z_true, lb, title, cbar_label, save_path, cmap="viridis", dims=None):
    dims = list(range(samples.shape[1])) if dims is None else dims
    n = len(dims)
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n + 2, 2.3 * n), facecolor="white")
    grid = np.linspace(lb, 1.0, 300)
    levels = np.linspace(0.0, 1.0, 21)
    cf = None
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            dr, dc = dims[r], dims[c]
            if r == c:
                y = kde_1d(samples[:, dr], grid, lb)
                ax.set_facecolor("#EAEAF2")
                ax.fill_between(grid, y, color="steelblue", alpha=0.4)
                ax.plot(grid, y, color="steelblue", lw=2.0)
                ax.axvline(z_true[dr], color=RED, ls="--", lw=1.8)
                ax.set_xlim(lb, 1.0)
                ax.set_ylim(0, None)
                ax.set_yticks([])
            elif r > c:
                X, Y, Z = density_2d(samples[:, dc], samples[:, dr], lb)
                cf = ax.contourf(X, Y, Z / (Z.max() + 1e-300), levels=levels, cmap=cmap)
                ax.plot(z_true[dc], z_true[dr], "*", color=RED, ms=12, mec="white", zorder=10)
                ax.set_xlim(lb, 1.0)
                ax.set_ylim(lb, 1.0)
            else:
                ax.axis("off")
            if r >= c:
                ax.tick_params(labelsize=10)
                if c == 0 and r > 0:
                    ax.set_ylabel(LABELS_7D[dr], fontsize=15)
                else:
                    ax.tick_params(labelleft=False)
                if r == n - 1:
                    ax.set_xlabel(LABELS_7D[dc], fontsize=15)
                else:
                    ax.tick_params(labelbottom=False)
    if cf is not None:
        cax = fig.add_axes([0.87, 0.55, 0.02, 0.35])
        cb = fig.colorbar(cf, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cb.set_label(cbar_label, fontsize=13)
    gt_marker = mlines.Line2D([], [], color=RED, marker="*", ls="--", ms=13, mec="white", label=r"True state $\mathbf{z}^*$")
    fig.legend(handles=[gt_marker], loc="upper right", bbox_to_anchor=(0.86, 0.97), fontsize=13)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.85, bottom=0.06, top=0.94, wspace=0.12, hspace=0.12)
    save_figure(fig, save_path)
    plt.close(fig)


# =============================================================================
# OVERLAY CORNER: reference (red) vs VAE (blue)
# =============================================================================
def _draw_hpd(ax, X, Y, Z, fill_colors, line_color, alpha):
    """Filled 90 % / 50 % highest-density regions, plus the 99 % contour (dotted) that bounds the plausible region."""
    lv = hpd_levels(Z)
    if len(lv) == 2 and Z.max() > 0:
        ax.contourf(X, Y, Z, levels=[lv[0], lv[1], Z.max() * 1.001], colors=fill_colors, alpha=alpha)
        ax.contour(X, Y, Z, levels=lv, colors=[line_color, line_color], linestyles=["--", "-"], linewidths=[1.2, 1.8])
        lv99 = hpd_levels(Z, masses=(0.99,))
        if lv99[0] < lv[0]:
            ax.contour(X, Y, Z, levels=lv99, colors=[line_color], linestyles=[":"], linewidths=[1.0])


@fixed_style
def corner_overlay(x_gt, x_vae, z_true, dims, lb, title, save_path, legend_title, label_vae="Estimated KS-Copula VAE posterior"):
    n = len(dims)
    fig_sz = (19.5, 19.0) if n >= 7 else (3.1 * n + 3, 3.0 * n + 2.5)
    fig, axes = plt.subplots(n, n, figsize=fig_sz, facecolor="white")
    axes = np.atleast_2d(axes)
    grid = np.linspace(lb, 1.0, 300)
    m_gt, s_gt = summary_stats(x_gt)
    m_q, s_q = summary_stats(x_vae)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            dr, dc = dims[r], dims[c]
            if r == c:
                k_gt = kde_1d(x_gt[:, dr], grid, lb)
                k_q = kde_1d(x_vae[:, dr], grid, lb)
                ax.set_facecolor("#F7F8FA")
                ax.fill_between(grid, k_gt, color=RED, alpha=0.22)
                ax.plot(grid, k_gt, color=RED, lw=2.2)
                ax.fill_between(grid, k_q, color=BLUE, alpha=0.25)
                ax.plot(grid, k_q, color=BLUE, lw=2.2)
                ax.axvline(z_true[dr], color="#111111", ls="--", lw=1.9)
                ax.set_title(f"{LABELS_7D[dr]} (T={z_true[dr]:.2f}|GT={m_gt[dr]:.2f}|Est={m_q[dr]:.2f})",
                             fontsize=10.5 if n >= 7 else 12, fontweight="bold", pad=4)
                ax.set_xlim(lb, 1.0)
                ax.set_ylim(0, None)
                ax.set_yticks([])
            elif r > c:
                X, Y, Z = density_2d(x_gt[:, dc], x_gt[:, dr], lb)
                _draw_hpd(ax, X, Y, Z, ["#FADBD8", "#F1948A"], "#922B21", 0.6)
                X, Y, Z = density_2d(x_vae[:, dc], x_vae[:, dr], lb)
                _draw_hpd(ax, X, Y, Z, ["#D6EAF8", "#85C1E9"], "#1B4F72", 0.5)
                ax.plot(z_true[dc], z_true[dr], marker="*", color="#FFD700", ms=13, mec="#111111", mew=1.1, zorder=15)
                ax.set_xlim(lb, 1.0)
                ax.set_ylim(lb, 1.0)
                ax.grid(True, ls=":", alpha=0.35)
            else:
                ax.axis("off")
            if r >= c:
                ax.tick_params(labelsize=11)
                if c == 0 and r > 0:
                    ax.set_ylabel(LABELS_7D[dr], fontsize=16, fontweight="bold")
                else:
                    ax.tick_params(labelleft=False)
                if r == n - 1:
                    ax.set_xlabel(LABELS_7D[dc], fontsize=16, fontweight="bold")
                else:
                    ax.tick_params(labelbottom=False)

    handles = [
        mlines.Line2D([], [], color=RED, lw=3, label=r"Reference posterior $p(\mathbf{z}|\mathbf{m})$ (SMC)"),
        mlines.Line2D([], [], color=BLUE, lw=3, label=label_vae + r" $q_\phi(\mathbf{z}|\mathbf{m})$"),
        mlines.Line2D([], [], color="#555555", lw=1.8, ls="-", label="50 % (solid), 90 % (dashed), 99 % (dotted) HPD"),
        mlines.Line2D([], [], color="#111111", marker="*", markerfacecolor="#FFD700", ls="--", lw=1.8, ms=13,
                      label=r"True state $\mathbf{z}^*$"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.955), fontsize=12, frameon=True,
               facecolor="#F8F9F9", edgecolor="#888888", title=legend_title, title_fontsize=13)

    lines = [f"{'Param':<8} | {'True z*':<7} | {'Reference (mean±std)':<20} | {'VAE (mean±std)':<15} | std ratio",
             "-" * 78]
    for d in dims:
        lines.append(f"{RAW_NAMES[d]:<8} | {z_true[d]:<7.3f} | {m_gt[d]:.3f} ± {s_gt[d]:.3f}        "
                     f"| {m_q[d]:.3f} ± {s_q[d]:.3f}  | {s_q[d] / (s_gt[d] + 1e-12):.2f}")
    lines += ["-" * 78, f"Mean |mu_VAE - mu_ref| = {np.mean(np.abs(m_q[dims] - m_gt[dims])):.4f}   "
              f"mean std ratio = {np.mean(s_q[dims] / (s_gt[dims] + 1e-12)):.2f}"]
    # statistics box in the empty upper triangle, right of the diagonal panels it spans
    col_w, row_h = 0.925 / n, 0.89 / n
    x0, y_top = (0.47, 0.775) if n >= 7 else (0.06 + 2 * col_w + 0.015, 0.94 - 1.25 * row_h)
    fig.text(x0, y_top, "\n".join(lines), fontsize=10.5 if n >= 7 else 9.5, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#FDFEFE", edgecolor="#BFC9CA", lw=1.5))
    z_str = "[" + ", ".join(f"{float(v):.2f}" for v in z_true) + "]"
    fig.suptitle(f"{title} | True $\\mathbf{{z}}^* = {z_str}$", fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.05, top=0.94, wspace=0.12, hspace=0.25)
    save_figure(fig, save_path)
    plt.close(fig)


# =============================================================================
# 1-D MARGINALS STRIP
# =============================================================================
@fixed_style
def strip_1d(x_gt, x_vae, z_true, lb, save_path, title, analytic_pdf=None,
             analytic_label="VAE KS-mixture marginal (exact)"):
    """analytic_pdf: optional callable (grid, d) -> exact marginal density of q (KS mixture or a baseline family)."""
    n = x_gt.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.3), facecolor="white")
    grid = np.linspace(lb, 1.0, 300)
    for d in range(n):
        ax = axes[d]
        k_gt = kde_1d(x_gt[:, d], grid, lb)
        k_q = kde_1d(x_vae[:, d], grid, lb)
        ax.set_facecolor("#F8F9FA")
        ax.fill_between(grid, k_gt, color=RED, alpha=0.22)
        ax.plot(grid, k_gt, color=RED, lw=2.2, label="Reference posterior (SMC)")
        ax.fill_between(grid, k_q, color=BLUE, alpha=0.25)
        ax.plot(grid, k_q, color=BLUE, lw=2.2, label="VAE posterior (samples)")
        if analytic_pdf is not None:
            ax.plot(grid, analytic_pdf(grid, d), color="#2CA02C", ls="-.", lw=1.8, label=analytic_label)
        ax.axvline(z_true[d], color="#111111", ls="--", lw=2.0, label="True value $z_i^*$")
        ax.set_title(f"{TITLES_7D[d]}\nT={z_true[d]:.2f} | GT={np.mean(x_gt[:, d]):.2f} | Est={np.mean(x_vae[:, d]):.2f}",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlim(lb, 1.0)
        ax.set_ylim(0, None)
        ax.set_yticks([])
        ax.grid(True, ls=":", alpha=0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(h), fontsize=11.5, frameon=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    save_figure(fig, save_path)
    plt.close(fig)


# =============================================================================
# PHYSICAL PROFILE ALONG THE BRIDGE (violins)
# =============================================================================
@fixed_style
def physical_profile(x_gt, x_vae, z_true, lb, save_path, title):
    n = len(z_true)
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [4, 1]}, sharex=True,
                                  facecolor="white")
    zg = np.linspace(lb, 1.0, 400)
    for i in range(n):
        xc = i + 1
        q = kde_1d(x_vae[:, i], zg, lb)
        g = kde_1d(x_gt[:, i], zg, lb)
        q = 0.38 * q / (q.max() + 1e-12)
        g = 0.38 * g / (g.max() + 1e-12)
        col = "steelblue" if i < 5 else "#2CA02C"
        ax.fill_betweenx(zg, xc - q, xc + q, color=col, alpha=0.38,
                         label=("VAE posterior (deck)" if i == 0 else ("VAE posterior (bearings)" if i == 5 else None)))
        ax.plot(xc - g, zg, color=RED, ls="--", lw=1.6, label="Reference posterior" if i == 0 else None)
        ax.plot(xc + g, zg, color=RED, ls="--", lw=1.6)
        ax.plot(xc, z_true[i], "*", color="#FFD700", ms=17, mec="black", mew=1.2, zorder=10,
                label=r"True $\mathbf{z}^*$" if i == 0 else None)
    ax.axvline(5.5, color="#555555", ls="-.", lw=1.6)
    ax.set_ylabel(r"Stiffness factor $\mathbf{z}$", fontsize=16)
    ax.set_ylim(lb - 0.02, 1.02)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=12)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    im = axb.imshow(z_true.reshape(1, -1), cmap="YlOrRd_r", norm=mcolors.Normalize(lb, 1.0), aspect="auto",
                    extent=[0.5, n + 0.5, 0, 1])
    for j in range(n):
        axb.text(j + 1, 0.5, f"{z_true[j]:.2f}", ha="center", va="center", fontsize=15, fontweight="bold",
                 color="white" if z_true[j] < 0.62 else "black")
    axb.set_yticks([])
    axb.set_xticks(np.arange(1, n + 1), LABELS_7D[:n], fontsize=15)
    fig.colorbar(im, ax=axb, fraction=0.02, pad=0.01, label=r"$\mathbf{z}^*$")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, save_path)
    plt.close(fig)
