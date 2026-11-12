#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/plotting.py

Corner plots and 1-D marginals shared by reference_posterior.py and postprocessing.py.

Both the reference and the VAE posterior are drawn from plain, equally weighted samples with the
same density estimator: a Gaussian kernel (Scott bandwidth) reflected at the bounds [lbound, 1],
so mass piled up at z = 1 (healthy parameters) is not smeared outside the box. Overlay contours
enclose 50 % and 90 % of the probability mass.

corner_overlay, strip_1d and physical_profile are journal figures: they are created at their printed width
(FIG_WIDTH = 7 in, ASCE), so the font sizes of PUB_RC are the printed ones, and they carry no titles or statistics
(those belong in the caption and in the text). The same identity is used in all of them: reference posterior = dashed
outline, with a grey fill for the marginals and a gold scale for the joint densities (only in the overlays, where it
validates the VAE), VAE posterior = navy, true state = red.
"""

import os
import logging
import functools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

LABELS_7D = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
TITLES_7D = ["Zone $z_1$", "Zone $z_2$", "Zone $z_3$", "Zone $z_4$", "Zone $z_5$",
             "Left bearing $z_{b,L}$", "Right bearing $z_{b,R}$"]
RED, BLUE = "#D62728", "#1F77B4"

# Journal figures: one identity for each quantity, readable in colour, in greyscale and with colour-blind vision (fill vs
# line, dashed vs solid, star). The joint density is a single-hue gold scale, lightness falling monotonically from white,
# stopped at a mid tone so that the navy lines and the white-edged star stay readable on its darkest colour.
REF_FILL, REF_LINE, REF_DASH = "#CFCFCF", "#3F3F3F", (0, (3.2, 1.6))
VAE_FILL, VAE_LINE = "#9DC3E6", "#08306B"
TRUE_COLOR = "#C1121F"
TRUE_EDGE = "white"
REF_CMAP = mcolors.LinearSegmentedColormap.from_list("reference_density", ["#FFFFFF", "#FFF2B8", "#FDCF5F", "#F2A32C"])
REF_LABEL = r"Reference posterior $p(\mathbf{z}\,|\,\mathbf{m})$"
VAE_LABEL = r"VAE posterior $q_\phi(\mathbf{z}\,|\,\mathbf{m})$"
TRUE_LABEL = r"True state $\mathbf{z}^*$"
GROUP_COLORS = {"deck": "#0072B2", "bearings": "#E69F00"}   # deck zones / bearings (Okabe-Ito, colour-blind safe)


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

# Figures drawn at their printed size: ASCE asks for lettering legible when the figure is 3.5 in or 7 in wide. These sizes
# are therefore the printed ones (figures drawn at 19-23 in and reduced to 7 in print their 11-13 pt text at 4-5 pt).
FIG_WIDTH = 7.0
PUB_RC = {
    **PAPER_RC,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 9.5, "legend.frameon": False, "legend.handlelength": 1.7, "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.0, "legend.borderpad": 0.0,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "patch.linewidth": 0.7, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
    "ytick.major.pad": 2, "axes.labelpad": 2.5, "axes.titlepad": 3.5, "grid.linewidth": 0.4, "grid.color": "0.86",
}


def fixed_style(fn):
    """Draws with the journal style PAPER_RC, whatever style the console has set (e.g. seaborn in Spyder)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.style.context(["default", PAPER_RC]):
            return fn(*args, **kwargs)
    return wrapper


def pub_style(fn):
    """Draws with PUB_RC (figures created at their printed size), whatever style the console has set."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.style.context(["default", PUB_RC]):
            return fn(*args, **kwargs)
    return wrapper


def save_figure(fig, path, dpi=300, tight=True):
    """
    Saves `path` as PNG (dpi) and the same figure as vector PDF next to it (for the manuscript).
    tight=False keeps the exact figure size (figures created at their printed width).
    """
    stem = os.path.splitext(path)[0]
    bbox = "tight" if tight else None
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches=bbox)
    fig.savefig(stem + ".pdf", dpi=dpi, bbox_inches=bbox)


def despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


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
# OVERLAY CORNER: reference (shaded density + dashed contours) vs VAE (solid contours)
# =============================================================================
def _overlay_panel(ax, x_ref, x_vae, dc, dr, z_true, lb):
    """Off-diagonal panel: relative density of the reference, 50 % and 90 % contours of both posteriors, true state."""
    X, Y, Z = density_2d(x_ref[:, dc], x_ref[:, dr], lb)
    ax.imshow(Z / (Z.max() + 1e-300), origin="lower", extent=[lb, 1.0, lb, 1.0], aspect="auto", cmap=REF_CMAP,
              vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.contour(X, Y, Z, levels=hpd_levels(Z), colors=[REF_LINE], linestyles=[REF_DASH], linewidths=0.7)
    X, Y, Z = density_2d(x_vae[:, dc], x_vae[:, dr], lb)
    ax.contour(X, Y, Z, levels=hpd_levels(Z), colors=[VAE_LINE], linewidths=1.0)
    ax.plot(z_true[dc], z_true[dr], marker="*", color=TRUE_COLOR, ms=8, mec=TRUE_EDGE, mew=0.4, ls="none", zorder=10,
            clip_on=False)
    ax.set_ylim(lb, 1.0)


@pub_style
def corner_overlay(x_ref, x_vae, z_true, dims, lb, save_path, label_vae=VAE_LABEL):
    """
    Overlay of the two posteriors in the parameters `dims`, at print size (no separate reference figure is needed):
    below the diagonal, the relative joint density of the reference (gold scale, colour bar) with the 50 % and 90 %
    highest-density contours of both posteriors; on the diagonal, the two marginals (reference in grey).
    """
    n = len(dims)
    width = min(FIG_WIDTH, 1.3 * n + 0.6)
    left, right, bottom, top, gap = 0.62, 0.13, 0.46, 0.25, 0.09     # margins [in], gap between panels [panel widths]
    height = width - left - right + bottom + top                     # square panels
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(n, n, left=left / width, right=1.0 - right / width, bottom=bottom / height,
                          top=1.0 - top / height, wspace=gap, hspace=gap)
    grid = np.linspace(lb, 1.0, 300)
    for r in range(n):
        for c in range(r + 1):
            ax = fig.add_subplot(gs[r, c])
            dr, dc = dims[r], dims[c]
            if r == c:
                k_ref, k_q = kde_1d(x_ref[:, dr], grid, lb), kde_1d(x_vae[:, dr], grid, lb)
                ax.fill_between(grid, k_ref, color=REF_FILL, lw=0)
                ax.plot(grid, k_ref, color=REF_LINE, ls=REF_DASH, lw=0.8)
                ax.plot(grid, k_q, color=VAE_LINE, lw=1.2)
                ax.axvline(z_true[dr], color=TRUE_COLOR, lw=0.9)
                ax.set_ylim(0, 1.05 * max(k_ref.max(), k_q.max()))
                ax.set_yticks([])
                despine(ax, ("top", "left", "right"))
                ax.set_title(LABELS_7D[dr], fontsize=10)
            else:
                _overlay_panel(ax, x_ref, x_vae, dc, dr, z_true, lb)
                ax.set_yticks([0.6, 0.8, 1.0])
            ax.set_xlim(lb, 1.0)
            ax.set_xticks([0.6, 0.8, 1.0])
            ax.tick_params(labelbottom=r == n - 1, labelleft=c == 0 and r > 0)
            if r == n - 1:
                ax.set_xlabel(LABELS_7D[dc])
            if c == 0 and r > 0:
                ax.set_ylabel(LABELS_7D[dr], rotation=0, ha="right", va="center")

    handles = [
        mpatches.Patch(facecolor=REF_FILL, edgecolor="none", label=REF_LABEL + ": marginal"),
        mlines.Line2D([], [], color=REF_LINE, ls=REF_DASH, lw=1.0, label=REF_LABEL + ": 50 % and 90 % contours"),
        mlines.Line2D([], [], color=VAE_LINE, lw=1.4, label=label_vae + ": marginal, 50 % and 90 % contours"),
        mlines.Line2D([], [], color=TRUE_COLOR, lw=0.9, marker="*", ms=8, mec=TRUE_EDGE, mew=0.4, label=TRUE_LABEL),
    ]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0 - right / width, 1.0 - 0.06 / height))
    box = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    bar_h = 0.09 / height
    cax = fig.add_axes([box.x0, box.y0 - 0.34 / height - bar_h, box.width, bar_h])
    cb = fig.colorbar(mcm.ScalarMappable(mcolors.Normalize(0.0, 1.0), REF_CMAP), cax=cax, orientation="horizontal",
                      ticks=[0.0, 0.5, 1.0])
    cb.set_label("Reference posterior: joint density / peak of each panel", fontsize=9.5, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cax.xaxis.set_label_position("top")
    save_figure(fig, save_path, tight=False)
    plt.close(fig)


# =============================================================================
# 1-D MARGINALS STRIP
# =============================================================================
@pub_style
def strip_1d(x_ref, x_vae, z_true, lb, save_path):
    """Overlay of the marginals of the two posteriors, one panel per parameter (each with its own density scale)."""
    n = x_ref.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(FIG_WIDTH, 1.9))
    fig.subplots_adjust(left=0.035, right=0.978, bottom=0.25, top=0.84, wspace=0.14)
    grid = np.linspace(lb, 1.0, 300)
    for d, ax in enumerate(axes):
        k_ref, k_q = kde_1d(x_ref[:, d], grid, lb), kde_1d(x_vae[:, d], grid, lb)
        ax.fill_between(grid, k_ref, color=REF_FILL, lw=0)
        ax.plot(grid, k_ref, color=REF_LINE, ls=REF_DASH, lw=0.9)
        ax.plot(grid, k_q, color=VAE_LINE, lw=1.3)
        ax.axvline(z_true[d], color=TRUE_COLOR, lw=1.0)
        ax.set_xlim(lb, 1.0)
        ax.set_ylim(0, 1.05 * max(k_ref.max(), k_q.max()))
        ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1.0"])
        ax.set_yticks([])
        ax.set_xlabel(LABELS_7D[d])
        despine(ax, ("top", "left", "right"))
    axes[0].set_ylabel("Density")
    handles = [mpatches.Patch(facecolor=REF_FILL, edgecolor=REF_LINE, ls=REF_DASH, label=REF_LABEL),
               mlines.Line2D([], [], color=VAE_LINE, lw=1.4, label=VAE_LABEL),
               mlines.Line2D([], [], color=TRUE_COLOR, lw=1.0, label=r"True value $z_i^*$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
    save_figure(fig, save_path, tight=False)
    plt.close(fig)


# =============================================================================
# MARGINAL UNCERTAINTY ALONG THE BRIDGE: VIOLINS OF THE POSTERIOR OF THE METHOD
# =============================================================================
@pub_style
def physical_profile(x_post, z_true, lb, save_path, title=None, label="VAE posterior"):
    """
    Marginal uncertainty of every parameter along the structure: the outcome of the method, one distribution per
    parameter where a deterministic approach gives one number. Width proportional to the marginal density (same maximum
    width for every parameter; tails cut where the density falls below 1 % of its peak), central 95 % credible interval
    (line) and true value (red star). The last two parameters are the bearings. `title` is optional (none in paper
    figures).
    """
    n = len(z_true)
    n_deck = n - 2
    x = np.arange(n, dtype=float)
    x[n_deck:] += 0.6                                                # small gap between deck zones and bearings
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.1))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.81 if title else 0.88, bottom=0.185)
    zg = np.linspace(lb, 1.0, 500)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    for i in range(n):
        dens = kde_1d(x_post[:, i], zg, lb)
        keep = np.where(dens >= 0.01 * dens.max())[0]
        zz, ww = zg[keep[0]:keep[-1] + 1], 0.38 * dens[keep[0]:keep[-1] + 1] / dens.max()
        ax.fill_betweenx(zz, x[i] - ww, x[i] + ww, color=VAE_FILL, lw=0, zorder=2)
        ax.plot(np.r_[x[i] - ww, (x[i] + ww)[::-1], x[i] - ww[0]], np.r_[zz, zz[::-1], zz[0]], color=VAE_LINE, lw=0.7,
                zorder=3)
        low, high = np.quantile(x_post[:, i], [0.025, 0.975])
        ax.plot([x[i], x[i]], [low, high], color=VAE_LINE, lw=1.1, solid_capstyle="butt", zorder=4)
        ax.plot(x[i], z_true[i], marker="*", color=TRUE_COLOR, ms=9, mec="white", mew=0.4, ls="none", zorder=5,
                clip_on=False)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(lb - 0.012, 1.012)
    ax.set_xticks(x, LABELS_7D[:n], fontsize=10)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"Stiffness factor $z$")
    despine(ax)
    trans = ax.get_xaxis_transform()                                 # x in data units, y in axes fraction
    for a, b, name in ((x[0], x[n_deck - 1], "Deck zones"), (x[n_deck], x[-1], "Bearings")):
        ax.plot([a - 0.42, b + 0.42], [-0.115, -0.115], color="0.3", lw=0.7, transform=trans, clip_on=False)
        ax.text(0.5 * (a + b), -0.135, name, transform=trans, ha="center", va="top", fontsize=9.5)
    handles = [mpatches.Patch(facecolor=VAE_FILL, edgecolor=VAE_LINE, label=label),
               mlines.Line2D([], [], color=VAE_LINE, lw=1.1, label="95 % credible interval"),
               mlines.Line2D([], [], color=TRUE_COLOR, ls="none", marker="*", ms=9, mec="white", mew=0.4,
                             label=r"True value $z_i^*$")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3)
    if title:
        fig.suptitle(title, fontsize=9, y=0.99)
    save_figure(fig, save_path, tight=False)
    plt.close(fig)
