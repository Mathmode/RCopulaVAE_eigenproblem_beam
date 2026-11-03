# -*- coding: utf-8 -*-
"""
ks28_dataset_plots.py -- Overview figures of an extended dataset folder.

  fig_bridge_layout.png       zones, bearings and sensors + nominal mode shapes
  fig_prior_vs_uniform.png    zone-factor marginals and number of damaged zones:
                              scenario prior (new design) vs uniform box (paper)
  fig_model_form_error.png    per-mode frequency/MAC discrepancy between the truth
                              models of the shifted families and the decoder model,
                              compared with the measurement noise level
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ks28_config import load_configs
from .ks28_fe_model import build_basis, forward, modal_analysis, stiffness_from_theta
from .ks28_likelihood import mac_matrix_diag

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_ZONE = plt.cm.Blues(np.linspace(0.35, 0.8, 12))


def _load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def plot_bridge_layout(cfg, out):
    fine = build_basis(cfg, 16)
    K = stiffness_from_theta(fine, cfg, cfg.healthy_latent[None])
    f, _, Phi = modal_analysis(K, fine.M, cfg.n_modes, fine.sensor_dofs, return_full=True)
    v = Phi[0, 0::2, :]
    x = fine.node_x
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw=dict(height_ratios=[1, 2.2]))
    ax = axs[0]
    for k in range(cfg.n_zones):
        x0, x1 = k * cfg.L / cfg.n_zones, (k + 1) * cfg.L / cfg.n_zones
        ax.fill_between([x0, x1], -0.12, 0.12, color=C_ZONE[k % 12], alpha=0.9)
        ax.text((x0 + x1) / 2, 0.0, f"zone {k + 1}", ha="center", va="center", color="white", fontsize=10)
    for xb, lab in [(0.0, r"$\eta_L$"), (cfg.L, r"$\eta_R$")]:
        zz = np.linspace(-0.55, -0.15, 40)
        ax.plot(xb + 0.18 * np.sin(np.linspace(0, 8 * np.pi, 40)), zz, color="k", lw=1.2)
        ax.plot([xb - 0.5, xb + 0.5], [-0.58, -0.58], color="k", lw=2)
        ax.text(xb, -0.8, f"bearing ({lab})\n$k_v$, $k_r$", ha="center", va="top", fontsize=9)
    sx = np.asarray(cfg.sensor_x_over_L) * cfg.L
    ax.plot(sx, np.full_like(sx, 0.2), "v", color="#d1495b", ms=9, label="vertical accelerometer")
    ax.set_ylim(-1.3, 0.6); ax.set_xlim(-1, cfg.L + 1); ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)
    ax.set_title(f"{cfg.L:.0f} m single-span deck on elastic bearings: {cfg.n_zones} zones, "
                 f"{cfg.n_sensors} sensors")
    ax = axs[1]
    for j in range(cfg.n_modes):
        vj = v[:, j] / np.max(np.abs(v[:, j]))
        ax.plot(x, vj - 2.4 * j, lw=1.8, label=f"mode {j + 1}: {f[0, j]:.2f} Hz")
        ax.plot(sx, vj[fine.sensor_nodes] - 2.4 * j, "o", color="#d1495b", ms=4)
        ax.axhline(-2.4 * j, color="0.85", lw=0.8, zorder=0)
    ax.set_yticks([]); ax.set_xlabel("x [m]")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    ax.set_title("Nominal vertical mode shapes (non-zero displacement at the bearings)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_prior_vs_uniform(folder, cfg, prior, out):
    tr = _load(os.path.join(folder, "train.npz"))
    z = tr["theta"][:, :cfg.n_zones]
    rng = np.random.default_rng(0)
    zu = rng.uniform(cfg.z_lb, cfg.z_ub, size=z.shape)
    fig, axs = plt.subplots(1, 3 if cfg.support_latent else 2, figsize=(13, 3.8))
    bins = np.linspace(cfg.z_lb, 1.0, 51)
    axs[0].hist(zu.ravel(), bins=bins, density=True, color="0.7", label="uniform box (paper)")
    axs[0].hist(z.ravel(), bins=bins, density=True, color="#2e86ab", alpha=0.8, label="new design")
    axs[0].set_xlabel("zone stiffness factor z"); axs[0].set_ylabel("density"); axs[0].set_yscale("log")
    axs[0].legend(frameon=False)
    nd = np.sum(z < 1 - prior.healthy_band, axis=1)
    ndu = np.sum(zu < 1 - prior.healthy_band, axis=1)
    k = np.arange(cfg.n_zones + 1)
    axs[1].bar(k - 0.2, [np.mean(ndu == i) for i in k], 0.4, color="0.7", label="uniform box")
    axs[1].bar(k + 0.2, [np.mean(nd == i) for i in k], 0.4, color="#2e86ab", label="new design")
    axs[1].set_xlabel("number of damaged zones (z < %.2f)" % (1 - prior.healthy_band))
    axs[1].set_ylabel("fraction of scenarios"); axs[1].legend(frameon=False)
    if cfg.support_latent:
        eta = tr["theta"][:, cfg.n_zones:]
        axs[2].hist(eta.ravel(), bins=60, density=True, color="#2e86ab")
        axs[2].set_xlabel(r"bearing factor $\eta$ ($k = 2^{\eta} k_{nom}$)"); axs[2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_model_form_error(folder, cfg, noise, out):
    """Discrepancy truth-model vs decoder-model for the same theta (noise-free)."""
    basis = build_basis(cfg)
    fams = ["fine_mesh", "mass_perturbed", "temperature", "kr_mismatch", "combined_shift"]
    fams = [f for f in fams if os.path.exists(os.path.join(folder, "tests", f + ".npz"))]
    if not fams:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for i, fam in enumerate(fams):
        d = _load(os.path.join(folder, "tests", fam + ".npz"))
        f_dec, p_dec = forward(d["theta"], basis, cfg)
        err = 100 * (d["f_clean"] / f_dec - 1.0)
        mac = mac_matrix_diag(d["phi_clean"], p_dec)
        pos = np.arange(cfg.n_modes) + (i - len(fams) / 2) * 0.15
        axs[0].boxplot(err, positions=pos, widths=0.12, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=plt.cm.tab10(i), alpha=0.7), medianprops=dict(color="k"))
        axs[1].boxplot(100 * (1 - mac), positions=pos, widths=0.12, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=plt.cm.tab10(i), alpha=0.7), medianprops=dict(color="k"))
        axs[0].plot([], [], "s", color=plt.cm.tab10(i), label=fam)
    for s in (-1, 1):
        axs[0].axhline(100 * s * noise.sigma_f, color="#d1495b", ls="--", lw=1)
    axs[0].text(cfg.n_modes - 0.6, 100 * noise.sigma_f, r"$\pm\sigma_f$", color="#d1495b", va="bottom")
    for a in axs:
        a.set_xticks(range(cfg.n_modes)); a.set_xticklabels([f"mode {j + 1}" for j in range(cfg.n_modes)])
    axs[0].set_ylabel("frequency discrepancy truth vs decoder [%]")
    axs[1].set_ylabel("100 (1 - MAC) truth vs decoder")
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Model-form error of the shifted test families (noise-free), vs measurement noise")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make_dataset_figures(folder):
    cfg, noise, prior, _ = load_configs(os.path.join(folder, "config.json"))
    fd = os.path.join(folder, "figures")
    os.makedirs(fd, exist_ok=True)
    plot_bridge_layout(cfg, os.path.join(fd, "fig_bridge_layout.png"))
    plot_prior_vs_uniform(folder, cfg, prior, os.path.join(fd, "fig_prior_vs_uniform.png"))
    plot_model_form_error(folder, cfg, noise, os.path.join(fd, "fig_model_form_error.png"))
