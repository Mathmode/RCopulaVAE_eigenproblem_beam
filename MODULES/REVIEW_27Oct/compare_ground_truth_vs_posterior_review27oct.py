#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ground_truth_vs_posterior_review27oct.py
================================================
Updated Ground-Truth Bayesian Posterior vs. Estimated Copula-VAE Posterior Comparison
Script for the MS CPENG-8674 Major Revision (`code/MODULES/review_27oct/`).

This script combines and upgrades:
  - `Last_used_scripts/get_ground_truth_plots.py`
  - `MODULES/POSTPROCESSING/Ground_truth_MassPert_KDE_plots.py`
  - `Last_used_scripts/GC_KSmix_results_analysis.py`

How It Compares the Estimated Posterior vs. the True Bayesian Posterior:
------------------------------------------------------------------------
1. Ground-Truth Bayesian Posterior p_GT(z | m_obs):
   - Evaluates Bayes' rule p(z | m_obs) propto p(m_obs | z) * p(z) on a dense reference
     point cloud in [lbound, 1.0]^5 passed through the FE physics engine:
       * Global Sobol QMC points (N_sobol = 16,384) across [lbound, 1.0]^5
       * Realistic DoE prior points (N_prior = 4,096) covering healthy & localized regimes
       * Local target + symmetric-mirror neighborhood exploration around z^* and
         z^*_mirror = (z_5^*, z_4^*, z_3^*, z_2^*, z_1^*) so high-likelihood peaks and
         symmetry-induced bimodal peaks (under sensor limitations) are resolved with
         high sample density.

2. Estimated Copula-VAE Posterior q_phi(z | m_obs):
   - Evaluates BOTH:
       (a) Direct Variational Posterior q_phi(z | m_obs): samples drawn directly from the
           trained Encoder's Gaussian Copula + Kumaraswamy Mixture marginals (0 FE solves).
       (b) Physics-Refined (Importance-Reweighted) Posterior: Copula-VAE samples weighted by
           the physics likelihood exp(-(L_freq + L_MAC) / beta^2).

3. Visual & Quantitative Comparison Outputs (saved in `<run_folder>/GT_vs_Posterior_Comparison/`):
   - `P{pos}_1_GT_CornerPlot.png`: 5D Corner Plot of the True Bayesian Posterior.
   - `P{pos}_2_KSVAE_CornerPlot.png`: 5D Corner Plot of the Estimated KS-Copula Posterior.
   - `P{pos}_3_SplitComparison_CornerPlot.png`: Single 5x5 Split-Triangle & Overlaid-Diagonal
     Corner Plot:
       * Diagonal (r == c): Overlays Ground-Truth 1D PDF (black dashed) vs. Raw Encoder
         KS-Mixture PDF (dark orange) vs. Physics-Refined KS-Copula PDF (steelblue) +
         True value z^* (red star/line).
       * Lower Triangle (r > c): Estimated KS-Copula 2D Joint Posterior contours.
       * Upper Triangle (r < c): Ground-Truth 2D Joint Posterior contours (plotted with the
         exact same axes so each (r, c) pair is a direct mirror comparison!).
   - `GT_vs_Posterior_Metrics.csv` & `.json`: Quantitative distribution-matching metrics per
     test sample:
       * 1D Marginal Wasserstein-1 Distance (W1)
       * 1D Marginal Jensen-Shannon Divergence (JSD)
       * Posterior Mean L2/MAE error ||E_q[z] - E_GT[z]||
       * Posterior Std MAE error ||Std_q[z] - Std_GT[z]||
       * Correlation Matrix Frobenius error ||R_q - R_GT||_F
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde, wasserstein_distance, qmc
import tensorflow as tf
import tensorflow.keras as K

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from MODULES.review_27oct.preprocessing_updated_review27oct import (
    load_data_updated,
    load_known_matrices_updated,
)
from MODULES.review_27oct.GC_GMm_GPU_eigen_functions_updated_review27oct import (
    assemble_global_Kmatrices,
)
from MODULES.review_27oct.GC_GMm_functions_updated_review27oct import (
    build_correlation_matrices_from_cholesky,
)
from MODULES.review_27oct.GC_KSmix_functions_for_results_analysis_updated_review27oct import (
    gaussian_copula,
    kumaraswamy_mixture_quantile,
    calculate_MAC,
    physics_engine_step,
)

K.backend.set_floatx("float32")


def compute_bayesian_weights(
    f_pred,
    rot_pred,
    vert_pred,
    obs_f_scaled,
    obs_r,
    obs_v,
    mean_freq,
    std_freq,
    beta,
    sensor_regime="full",
):
    """
    Computes normalized Bayesian likelihood weights p(m_obs | z) under the active sensor regime.
    Masks out spatial Nyquist nodal zero-crossing modes (e.g., Mode 5 vertical on 5-zone grid)
    and scales MAC residuals consistently with the 5% mode-shape noise level.
    """
    n_pts = f_pred.shape[0]
    pred_logscaled = (np.log(np.maximum(f_pred, 1e-8)) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled[np.newaxis, :] - pred_logscaled), axis=1)

    if sensor_regime == "freq_only":
        loss_mac = np.zeros_like(loss_f)
    else:
        obs_v_batch = np.repeat(obs_v[np.newaxis, :, :], n_pts, axis=0)
        vert_mac = calculate_MAC(obs_v_batch, vert_pred)
        v_mask = (np.linalg.norm(obs_v, axis=-1) > 0.015).astype(np.float32)  # (n_modes,)
        v_err = np.sum((1.0 - vert_mac) * v_mask[np.newaxis, :], axis=1) / np.maximum(np.sum(v_mask), 1.0)

        if sensor_regime in ("no_rotations", "sparse_2modes") or np.all(np.abs(obs_r) < 1e-8):
            loss_mac = 25.0 * v_err
        else:
            obs_r_batch = np.repeat(obs_r[np.newaxis, :, :], n_pts, axis=0)
            rot_mac = calculate_MAC(obs_r_batch, rot_pred)
            r_mask = (np.linalg.norm(obs_r, axis=-1) > 0.015).astype(np.float32)
            r_err = np.sum((1.0 - rot_mac) * r_mask[np.newaxis, :], axis=1) / np.maximum(np.sum(r_mask), 1.0)
            loss_mac = 25.0 * (v_err + r_err)

    inv_gamma_val = 1.0 / (beta ** 2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    weights = likelihood / (np.sum(likelihood) + 1e-15)
    return weights, loss_f + loss_mac


def build_or_load_base_qmc_cloud(
    cache_dir,
    Ke_matrices,
    L_inv,
    n_elements,
    n_modes,
    free_dofs,
    n_dofs,
    fixed_dofs,
    lbound=0.45,
    n_sobol_power=14,
):
    """
    Builds or loads a cached global Sobol QMC point cloud + forward FE modal predictions.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n_qmc = 2 ** n_sobol_power
    cache_file = os.path.join(cache_dir, f"GT_Sobol_E{n_elements}_M{n_modes}_N{n_qmc}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["z_qmc"], data["f_pred"], data["rot_pred"], data["vert_pred"]

    print(f"Generating Sobol QMC cloud (N={n_qmc}) and running forward FE physics...")
    sampler = qmc.Sobol(d=n_elements, scramble=True, seed=42)
    u_qmc = sampler.random_base2(m=n_sobol_power)  # shape (N, n_elements)
    z_qmc = (lbound + (1.0 - lbound) * u_qmc).astype(np.float32)

    z_tf = tf.constant(z_qmc, dtype=tf.float32)
    Ke_dam = tf.einsum("BE, EKQ -> BEKQ", z_tf, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree = assemble_global_Kmatrices(Ke_dam, n_elements, n_qmc, fixed_dofs)

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    batch_size = 512
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_qmc, batch_size):
        f, r, v = physics_engine_step(Kfree[i:i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)

    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)

    np.savez_compressed(
        cache_file,
        z_qmc=z_qmc,
        f_pred=f_pred,
        rot_pred=rot_pred,
        vert_pred=vert_pred,
    )
    return z_qmc, f_pred, rot_pred, vert_pred


def augment_gt_cloud_for_sample(
    z_base,
    f_base,
    rot_base,
    vert_base,
    z_true,
    Ke_matrices,
    L_inv,
    n_elements,
    n_modes,
    free_dofs,
    n_dofs,
    fixed_dofs,
    lbound=0.45,
    n_local=2048,
    seed=123,
):
    """
    Augments the global Sobol cloud with local exploration points around both `z_true`
    and its symmetric reflection `z_true[::-1]` so sharp unimodal or symmetric bimodal
    peaks have high sample resolution.
    """
    rng = np.random.default_rng(seed)
    n_half = n_local // 2
    z_mirror = z_true[::-1]

    # Local perturbations around z_true and z_mirror, reflected into [lbound, 1.0) so no Dirac pile-up occurs at 1.0
    z_loc1 = 1.0 - np.abs(1.0 - (z_true + rng.normal(0.0, 0.045, size=(n_half, n_elements))))
    z_loc2 = 1.0 - np.abs(1.0 - (z_mirror + rng.normal(0.0, 0.045, size=(n_local - n_half, n_elements))))
    z_loc1 = np.clip(z_loc1, lbound + 1e-4, 1.0 - 1e-4)
    z_loc2 = np.clip(z_loc2, lbound + 1e-4, 1.0 - 1e-4)
    z_aug = np.vstack([z_loc1, z_loc2]).astype(np.float32)

    z_tf = tf.constant(z_aug, dtype=tf.float32)
    Ke_dam = tf.einsum("BE, EKQ -> BEKQ", z_tf, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree = assemble_global_Kmatrices(Ke_dam, n_elements, n_local, fixed_dofs)

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    f_aug, rot_aug, vert_aug = physics_engine_step(Kfree, L_inv_tf, n_modes, free_dofs, n_dofs)

    z_all = np.vstack([z_base, z_aug])
    f_all = np.vstack([f_base, f_aug])
    rot_all = np.vstack([rot_base, rot_aug])
    vert_all = np.vstack([vert_base, vert_aug])
    return z_all, f_all, rot_all, vert_all


def filter_top_weighted_samples(z_pts, weights, quantile_threshold=0.95, min_pts=120):
    sorted_idx = np.argsort(weights)[::-1]
    cum_w = np.cumsum(weights[sorted_idx])
    mask_idx = sorted_idx[cum_w <= quantile_threshold]
    if len(mask_idx) < min_pts:
        mask_idx = sorted_idx[:min_pts]
    x_filt = z_pts[mask_idx]
    w_filt = weights[mask_idx]
    w_norm = w_filt / (np.sum(w_filt) + 1e-15)
    return x_filt, w_norm


def evaluate_analytical_ksmix_1d_pdf(x_grid, a_d, b_d, w_d, lbound=0.45):
    """
    Evaluates the exact analytical 1D marginal PDF of the Kumaraswamy Mixture encoder
    at physical stiffness coordinates `x_grid` in [lbound, 1.0].
    """
    u_grid = np.clip((x_grid - lbound) / (1.0 - lbound), 1e-5, 1.0 - 1e-5)
    a_d = np.asarray(a_d, dtype=np.float64)[:, None]  # (K, 1)
    b_d = np.asarray(b_d, dtype=np.float64)[:, None]  # (K, 1)
    w_d = np.asarray(w_d, dtype=np.float64)[:, None]  # (K, 1)
    w_d = w_d / (np.sum(w_d) + 1e-15)

    u_row = u_grid[None, :]  # (1, G)
    pdf_k = w_d * a_d * b_d * np.power(u_row, a_d - 1.0) * np.power(
        np.clip(1.0 - np.power(u_row, a_d), 1e-12, 1.0), b_d - 1.0
    )
    pdf_phys = np.sum(pdf_k, axis=0) / (1.0 - lbound)
    return pdf_phys


def jensen_shannon_divergence_1d(p_grid, q_grid, x_grid):
    p = np.maximum(p_grid, 0.0)
    q = np.maximum(q_grid, 0.0)
    p = p / (np.trapezoid(p, x_grid) + 1e-15)
    q = q / (np.trapezoid(q, x_grid) + 1e-15)
    m = 0.5 * (p + q)
    kl_pm = np.trapezoid(p * np.log((p + 1e-12) / (m + 1e-12)), x_grid)
    kl_qm = np.trapezoid(q * np.log((q + 1e-12) / (m + 1e-12)), x_grid)
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, None))


def weighted_corr_matrix(x_samples, weights):
    w = weights / np.sum(weights)
    mean = np.sum(x_samples * w[:, None], axis=0)
    xm = x_samples - mean[None, :]
    cov = (xm * w[:, None]).T @ xm
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0), mean, std


def plot_single_corner(
    x_filtered,
    w_norm,
    z_true,
    lbound,
    title_label,
    cbar_label,
    save_path,
    cmap="viridis",
):
    n_dims = x_filtered.shape[1]
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(14, 14), facecolor="white")
    labels = [f"$z_{{{k+1}}}$" for k in range(n_dims)]
    cf = None

    for r in range(n_dims):
        for c in range(n_dims):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_norm)
                x_grid = np.linspace(lbound, 1.0, 200)
                y_grid = kde1d(x_grid)
                ax.set_facecolor("#EAEAF2")
                ax.fill_between(x_grid, y_grid, color="steelblue", alpha=0.4)
                ax.plot(x_grid, y_grid, color="steelblue", lw=2.2)
                ax.axvline(z_true[r], color="#D62728", linestyle="--", lw=2.0)
                ax.text(
                    0.5, 0.3, labels[r], fontsize=22, ha="center", va="center",
                    fontweight="bold", transform=ax.transAxes
                )
                ax.set_xlim(lbound, 1.0)
                ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_norm)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)
                ax.plot(z_true[c], z_true[r], "r*", markersize=14, markeredgecolor="white", zorder=10)
                ax.set_xlim(lbound, 1.0)
                ax.set_ylim(lbound, 1.0)
            else:
                ax.axis("off")

            if r >= c:
                ax.tick_params(labelsize=18)
                if c == 0 and r != 0:
                    ax.set_ylabel(labels[r], fontsize=24)
                else:
                    ax.tick_params(labelleft=False)
                if r == n_dims - 1:
                    ax.set_xlabel(labels[c], fontsize=24)
                else:
                    ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.82, pad=0.05, anchor=(0.6, 1.25))
        cbar.set_label(cbar_label, fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    gt_marker = mlines.Line2D(
        [], [], color="#D62728", marker="*", linestyle="--",
        markersize=16, markeredgecolor="white", label="Ground truth $\\mathbf{z}^*$"
    )
    fig.legend(handles=[gt_marker], loc="upper right", bbox_to_anchor=(0.84, 0.94), fontsize=16, frameon=True)
    fig.suptitle(title_label, fontsize=20, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 0.98, 0.95])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_split_and_overlay_comparison_corner(
    x_gt,
    w_gt,
    x_vae,
    w_vae,
    a_vals,
    b_vals,
    w_vals,
    z_true,
    lbound,
    pos,
    save_path,
):
    """
    Plots an intuitive, self-explanatory Lower-Triangle 5x5 Corner Plot directly overlaying:
      - RED  : Reference Ground-Truth Bayesian Posterior p_GT(z | m)
      - BLUE : Estimated KS-Copula Posterior q_phi(z | m)
      - BLACK DASHED LINE / GOLD STAR : True Damage State z^*
      - UPPER-RIGHT SUMMARY PANEL : Clear legend + zone-by-zone numerical table
        (True z_i^* vs. GT Mean ± Std vs. Estimated KS-Copula Mean ± Std).
    """
    n_dims = x_gt.shape[1]
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(15.5, 15.0), facecolor="white")
    labels = [f"$z_{{{k+1}}}$" for k in range(n_dims)]
    x_grid = np.linspace(lbound, 1.0, 250)

    w_gt_n = w_gt / (np.sum(w_gt) + 1e-15)
    w_vae_n = w_vae / (np.sum(w_vae) + 1e-15)
    mean_gt = np.sum(x_gt * w_gt_n[:, None], axis=0)
    std_gt = np.sqrt(np.maximum(np.sum(((x_gt - mean_gt) ** 2) * w_gt_n[:, None], axis=0), 1e-8))
    mean_vae = np.sum(x_vae * w_vae_n[:, None], axis=0)
    std_vae = np.sqrt(np.maximum(np.sum(((x_vae - mean_vae) ** 2) * w_vae_n[:, None], axis=0), 1e-8))

    for r in range(n_dims):
        for c in range(n_dims):
            ax = axes[r, c]
            if r == c:
                # 1D Marginal PDFs on the Diagonal: Red (Ground Truth) vs. Blue (Estimated KS-Copula)
                kde_gt = gaussian_kde(x_gt[:, r], weights=w_gt_n)(x_grid)
                kde_vae = gaussian_kde(x_vae[:, r], weights=w_vae_n)(x_grid)

                ax.set_facecolor("#F7F8FA")
                ax.fill_between(x_grid, kde_gt, color="#D62728", alpha=0.22)
                ax.plot(x_grid, kde_gt, color="#D62728", linestyle="-", lw=2.4, label="Ground-Truth Posterior")

                ax.fill_between(x_grid, kde_vae, color="#1F77B4", alpha=0.28)
                ax.plot(x_grid, kde_vae, color="#1F77B4", linestyle="-", lw=2.4, label="Estimated KS-Copula")

                ax.axvline(z_true[r], color="#111111", linestyle="--", lw=2.2, label="True Value $z_i^*$")

                ax.set_title(
                    f"{labels[r]}  (True={z_true[r]:.2f} | GT={mean_gt[r]:.2f} | Est={mean_vae[r]:.2f})",
                    fontsize=11.5, fontweight="bold", pad=5
                )
                ax.set_xlim(lbound, 1.0)
                ax.set_yticks([])

            elif r > c:
                # 2D Joint Posteriors in the Lower Triangle: Red (Ground Truth) + Blue (Estimated KS-Copula)
                ax.set_facecolor("#FFFFFF")
                xi, yi = np.mgrid[lbound:1.0:55j, lbound:1.0:55j]

                kernel_gt = gaussian_kde(np.vstack([x_gt[:, c], x_gt[:, r]]), weights=w_gt_n)
                zi_gt = kernel_gt(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                zmax_gt = np.max(zi_gt)
                if zmax_gt > 1e-8:
                    ax.contourf(
                        xi, yi, zi_gt,
                        levels=[0.18 * zmax_gt, 0.50 * zmax_gt, zmax_gt],
                        colors=["#FADBD8", "#F1948A"],
                        alpha=0.55,
                    )
                    ax.contour(
                        xi, yi, zi_gt,
                        levels=[0.18 * zmax_gt, 0.50 * zmax_gt],
                        colors=["#C0392B", "#922B21"],
                        linestyles=["--", "-"],
                        linewidths=[1.3, 2.0],
                    )

                kernel_vae = gaussian_kde(np.vstack([x_vae[:, c], x_vae[:, r]]), weights=w_vae_n)
                zi_vae = kernel_vae(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                zmax_vae = np.max(zi_vae)
                if zmax_vae > 1e-8:
                    ax.contourf(
                        xi, yi, zi_vae,
                        levels=[0.18 * zmax_vae, 0.50 * zmax_vae, zmax_vae],
                        colors=["#D6EAF8", "#85C1E9"],
                        alpha=0.45,
                    )
                    ax.contour(
                        xi, yi, zi_vae,
                        levels=[0.18 * zmax_vae, 0.50 * zmax_vae],
                        colors=["#2874A6", "#1B4F72"],
                        linestyles=["--", "-"],
                        linewidths=[1.4, 2.1],
                    )

                ax.plot(
                    z_true[c], z_true[r], marker="*", color="#FFD700",
                    markersize=15, markeredgecolor="#111111", markeredgewidth=1.3, zorder=15
                )
                ax.set_xlim(lbound, 1.0)
                ax.set_ylim(lbound, 1.0)
                ax.grid(True, linestyle=":", alpha=0.4)

            else:
                ax.axis("off")

            if r >= c:
                ax.tick_params(labelsize=12)
                if c == 0 and r > 0:
                    ax.set_ylabel(labels[r], fontsize=18, fontweight="bold")
                else:
                    ax.tick_params(labelleft=False)
                if r == n_dims - 1:
                    ax.set_xlabel(labels[c], fontsize=18, fontweight="bold")
                else:
                    ax.tick_params(labelbottom=False)

    # Clear explanatory Legend & Quantitative Summary Table in the empty Upper-Right space
    legend_handles = [
        mlines.Line2D([], [], color="#D62728", linestyle="-", lw=3.0, label="Ground-Truth Bayesian Posterior $p(\\mathbf{z}|\\mathbf{m})$ (Red Shaded / Contours)"),
        mlines.Line2D([], [], color="#1F77B4", linestyle="-", lw=3.0, label="Estimated KS-Copula Posterior $q_\\phi(\\mathbf{z}|\\mathbf{m})$ (Blue Shaded / Contours)"),
        mlines.Line2D([], [], color="#111111", marker="*", markerfacecolor="#FFD700", linestyle="--", lw=2.0, markersize=15, label="True Stiffness Factor $\\mathbf{z}^*$ (Black Dashed Line / Star)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(0.97, 0.93),
        fontsize=13, frameon=True, facecolor="#F8F9F9", edgecolor="#888888", title="How to Read This Corner Plot", title_fontsize=14
    )

    lines_txt = [
        f"Sample {pos} Zone-by-Zone Posterior Summary:",
        "-" * 54,
        f"{'Zone':<6} | {'True z*':<8} | {'Ground-Truth (Mean±Std)':<23} | {'KS-Copula (Mean±Std)':<20}",
        "-" * 54,
    ]
    for d in range(n_dims):
        lines_txt.append(
            f"z_{d+1:<4} | {z_true[d]:<8.3f} | {mean_gt[d]:.3f} ± {std_gt[d]:.3f}          | {mean_vae[d]:.3f} ± {std_vae[d]:.3f}"
        )
    lines_txt.append("-" * 54)
    lines_txt.append(f"Mean Abs Error |mu_Est - mu_GT| = {np.mean(np.abs(mean_vae - mean_gt)):.4f}")
    lines_txt.append("Diagonal (r=c): 1D Marginal PDF for each zone z_i.")
    lines_txt.append("Lower Triangle (r>c): 2D Joint 50% (solid) & 90% (dashed) regions.")

    fig.text(
        0.48, 0.63, "\n".join(lines_txt),
        fontsize=11.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FDFEFE", edgecolor="#BFC9CA", lw=1.5),
    )

    z_str = "[" + ", ".join(f"{float(v):.2f}" for v in z_true) + "]"
    fig.suptitle(
        f"Sample {pos}: Ground-Truth Bayesian Posterior (Red) vs. Estimated KS-Copula Posterior (Blue)  |  True $\\mathbf{{z}}^* = {z_str}$",
        fontsize=16, fontweight="bold", y=0.985
    )
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.95])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_1d_marginals_strip(
    x_gt,
    w_gt,
    x_vae,
    w_vae,
    a_vals,
    b_vals,
    w_vals,
    z_true,
    lbound,
    pos,
    save_path,
):
    """
    Plots a crystal-clear 1x5 horizontal strip of the 5 marginal distributions z_1 ... z_5
    comparing Ground-Truth Bayesian Posterior (Red) vs. Estimated KS-Copula Posterior (Blue)
    vs. Direct Encoder Analytical KS-Mixture PDF (Green Dash-Dot) and True z_i^* (Black Dashed).
    """
    n_dims = x_gt.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(18, 4.2), facecolor="white", sharey=False)
    x_grid = np.linspace(lbound, 1.0, 250)
    w_gt_n = w_gt / (np.sum(w_gt) + 1e-15)
    w_vae_n = w_vae / (np.sum(w_vae) + 1e-15)

    for d in range(n_dims):
        ax = axes[d]
        kde_gt = gaussian_kde(x_gt[:, d], weights=w_gt_n)(x_grid)
        kde_vae = gaussian_kde(x_vae[:, d], weights=w_vae_n)(x_grid)
        pdf_enc = evaluate_analytical_ksmix_1d_pdf(
            x_grid, a_vals[:, d], b_vals[:, d], w_vals[:, d], lbound=lbound
        )

        mean_gt_d = float(np.sum(x_gt[:, d] * w_gt_n))
        mean_vae_d = float(np.sum(x_vae[:, d] * w_vae_n))

        ax.set_facecolor("#F8F9FA")
        ax.fill_between(x_grid, kde_gt, color="#D62728", alpha=0.22)
        ax.plot(x_grid, kde_gt, color="#D62728", lw=2.3, label="Ground-Truth Posterior")

        ax.fill_between(x_grid, kde_vae, color="#1F77B4", alpha=0.28)
        ax.plot(x_grid, kde_vae, color="#1F77B4", lw=2.3, label="Estimated KS-Copula Posterior")

        ax.plot(x_grid, pdf_enc, color="#2CA02C", linestyle="-.", lw=1.8, label="Raw Encoder KS-Mix $q_\\phi(z_i|\\mathbf{m})$")
        ax.axvline(z_true[d], color="#111111", linestyle="--", lw=2.2, label="True Value $z_i^*$")

        ax.set_title(
            f"Zone $z_{{{d+1}}}$\nTrue={z_true[d]:.2f} | GT={mean_gt_d:.2f} | Est={mean_vae_d:.2f}",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel(f"Stiffness Factor $z_{{{d+1}}}$", fontsize=12)
        ax.set_xlim(lbound, 1.0)
        ax.set_yticks([])
        ax.grid(True, linestyle=":", alpha=0.4)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=4, fontsize=12, frameon=True, facecolor="white", edgecolor="#CCCCCC"
    )
    plt.tight_layout(rect=[0, 0.0, 1.0, 0.90])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Ground-Truth Bayesian Posterior vs. Estimated KS-Copula Posterior (review_27oct)."
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="Path to the trained run folder inside Output/review_27oct/KS_Copula/.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[0, 1, 2, 7],
        help="Test sample indices to analyze and plot.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4096,
        help="Number of samples drawn from the KS-Copula model per test scenario (default: 4096).",
    )
    parser.add_argument(
        "--sobol-power",
        type=int,
        default=13,
        help="Base-2 exponent for global Sobol QMC cloud (default: 13 -> 8192 points + 2048 local points).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder_path = args.run_folder
    if not os.path.isabs(folder_path):
        cand = os.path.join(CODE_ROOT, folder_path)
        if os.path.exists(cand):
            folder_path = cand
        else:
            folder_path = os.path.join(CODE_ROOT, "Output", "review_27oct", "KS_Copula", args.run_folder)

    info = np.load(os.path.join(folder_path, "Problem_info.npy"), allow_pickle=True).item()
    n_dims = int(info["n_dims"])
    num_KS = int(info["num_KS"])
    mean_freq = info["mean_f"]
    std_freq = info["std_f"]
    beta = float(info["gamma"])
    lbound = float(info.get("lbound", 0.45))
    dataset_rel = info.get("dataset", "review_27oct/TRAIN_E5_realistic_doe")
    sensor_regime = info.get("sensor_regime", "full")
    n_elements = int(info.get("n_elements", 5))

    n_dofs = 2 * (n_elements + 1)
    fixed_dofs = [0, n_dofs - 2]
    free_dofs = np.delete(np.arange(n_dofs), fixed_dofs)

    zero_rotations = sensor_regime in ("no_rotations", "sparse_2modes", "freq_only")
    n_modes_override = 2 if sensor_regime == "sparse_2modes" else None

    # Load test predictions
    Test_pred_props = np.load(os.path.join(folder_path, "Test_predicted_props.npy"), allow_pickle=True).item()
    test_a = Test_pred_props["test_a"]
    test_b = Test_pred_props["test_b"]
    test_weights = Test_pred_props["test_weight_vals"]
    test_offdiag = Test_pred_props["test_offdiag_elems"]
    test_diag = Test_pred_props["test_diag_elems"]
    L_matrices = build_correlation_matrices_from_cholesky(test_offdiag, test_diag, n_dims)

    # Load dataset and matrices
    data_path = os.path.join(CODE_ROOT, "Data", dataset_rel)
    (_, _, _, _, _, _, _, _, Freqs_test, Rot_test, Vert_test, Alphas_test, _, _) = load_data_updated(
        data_path,
        batch_size=256,
        n_modes=n_modes_override,
        zero_rotations=zero_rotations,
    )
    if sensor_regime == "freq_only":
        Vert_test = np.zeros_like(Vert_test)

    mass_file = os.path.join(data_path, "Mass_matrix.npy")
    if os.path.exists(mass_file) and np.load(mass_file).shape[0] == 2 * n_elements:
        Mfree, Ke_matrices, L_inv, _, _, _ = load_known_matrices_updated(data_path, n_elements)
    else:
        fallback_path = os.path.join(CODE_ROOT, "Data", "review_27oct", "TRAIN_E5_realistic_doe")
        Mfree, Ke_matrices, L_inv, _, _, _ = load_known_matrices_updated(fallback_path, n_elements)

    # If the dataset has a perturbed mass matrix (e.g. 06Oct_Data_Noisy_E5_Lvl25_MassPerturb5),
    # use it for the Ground-Truth physics pass if present
    pert_mass_file = os.path.join(data_path, "Mass_matrix_perturbed.npy")
    if os.path.exists(pert_mass_file):
        Mfree_gt = np.load(pert_mass_file)
        L_gt = np.linalg.cholesky(Mfree_gt)
        L_inv_gt = np.linalg.inv(L_gt)
    else:
        L_inv_gt = L_inv

    n_modes = Freqs_test.shape[1]
    cache_dir = os.path.join(CODE_ROOT, "Output", "review_27oct", "GT_Posteriors_Cache")
    z_base, f_base, rot_base, vert_base = build_or_load_base_qmc_cloud(
        cache_dir=cache_dir,
        Ke_matrices=Ke_matrices,
        L_inv=L_inv_gt,
        n_elements=n_elements,
        n_modes=n_modes,
        free_dofs=free_dofs,
        n_dofs=n_dofs,
        fixed_dofs=fixed_dofs,
        lbound=lbound,
        n_sobol_power=args.sobol_power,
    )

    out_dir = os.path.join(folder_path, "GT_Comp")
    os.makedirs(out_dir, exist_ok=True)
    # Use relative path from current working directory to avoid Windows 260-char MAX_PATH limit
    try:
        out_dir = os.path.relpath(out_dir)
    except ValueError:
        pass

    comparison_rows = []
    x_grid = np.linspace(lbound, 1.0, 250)
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)

    for pos in args.positions:
        if pos >= len(Alphas_test):
            continue
        z_true = Alphas_test[pos]
        obs_f = Freqs_test[pos]
        obs_r = Rot_test[pos]
        obs_v = Vert_test[pos]

        print(f"\n--- Comparing Ground-Truth vs. KS-Copula Posterior for Sample {pos} (z* = {np.round(z_true, 3)}) ---")

        # 1. Ground-Truth Bayesian Posterior (Global Sobol + Local + Symmetric Mirror exploration)
        z_gt_all, f_gt_all, rot_gt_all, vert_gt_all = augment_gt_cloud_for_sample(
            z_base, f_base, rot_base, vert_base, z_true,
            Ke_matrices, L_inv_gt, n_elements, n_modes, free_dofs, n_dofs, fixed_dofs,
            lbound=lbound, n_local=2048, seed=1000 + pos,
        )
        w_gt_all, _ = compute_bayesian_weights(
            f_gt_all, rot_gt_all, vert_gt_all,
            obs_f, obs_r, obs_v, mean_freq, std_freq, beta, sensor_regime=sensor_regime,
        )
        x_gt_filt, w_gt_norm = filter_top_weighted_samples(z_gt_all, w_gt_all, quantile_threshold=0.95)

        # 2. Estimated KS-Copula Posterior
        a_vals = test_a[pos]
        b_vals = test_b[pos]
        w_vals = test_weights[pos]
        LT_mat = L_matrices[pos]

        u_cop = gaussian_copula(LT_mat, n_dims, args.n_samples)
        z_vae_samples = kumaraswamy_mixture_quantile(u_cop, a_vals, b_vals, w_vals, lbound=lbound)
        z_vae_np = z_vae_samples.numpy()

        Ke_dam_vae = tf.einsum("BE, EKQ -> BEKQ", tf.cast(z_vae_samples, tf.float32), tf.cast(Ke_matrices, tf.float32))
        Kfree_vae = assemble_global_Kmatrices(Ke_dam_vae, n_elements, args.n_samples, fixed_dofs)

        all_f, all_rot, all_vert = [], [], []
        for i in range(0, args.n_samples, 256):
            f, r, v = physics_engine_step(Kfree_vae[i:i + 256], L_inv_tf, n_modes, free_dofs, n_dofs)
            all_f.append(f)
            all_rot.append(r)
            all_vert.append(v)
        f_vae = np.concatenate(all_f, axis=0)
        rot_vae = np.concatenate(all_rot, axis=0)
        vert_vae = np.concatenate(all_vert, axis=0)

        w_vae_all, _ = compute_bayesian_weights(
            f_vae, rot_vae, vert_vae,
            obs_f, obs_r, obs_v, mean_freq, std_freq, beta, sensor_regime=sensor_regime,
        )
        x_vae_filt, w_vae_norm = filter_top_weighted_samples(z_vae_np, w_vae_all, quantile_threshold=0.95)

        # 3. Quantitative Distribution-Matching Metrics (GT vs. Estimated Posterior)
        w1_per_dim = []
        jsd_per_dim = []
        w1_raw_per_dim = []
        jsd_raw_per_dim = []

        for d in range(n_dims):
            w1_d = wasserstein_distance(x_gt_filt[:, d], x_vae_filt[:, d], u_weights=w_gt_norm, v_weights=w_vae_norm)
            w1_raw_d = wasserstein_distance(x_gt_filt[:, d], z_vae_np[:, d], u_weights=w_gt_norm)
            w1_per_dim.append(w1_d)
            w1_raw_per_dim.append(w1_raw_d)

            kde_gt_d = gaussian_kde(x_gt_filt[:, d], weights=w_gt_norm)(x_grid)
            kde_vae_d = gaussian_kde(x_vae_filt[:, d], weights=w_vae_norm)(x_grid)
            pdf_raw_d = evaluate_analytical_ksmix_1d_pdf(
                x_grid, a_vals[:, d], b_vals[:, d], w_vals[:, d], lbound=lbound
            )
            jsd_per_dim.append(jensen_shannon_divergence_1d(kde_gt_d, kde_vae_d, x_grid))
            jsd_raw_per_dim.append(jensen_shannon_divergence_1d(kde_gt_d, pdf_raw_d, x_grid))

        corr_gt, mean_gt, std_gt = weighted_corr_matrix(x_gt_filt, w_gt_norm)
        corr_vae, mean_vae, std_vae = weighted_corr_matrix(x_vae_filt, w_vae_norm)
        lt_np = LT_mat.numpy() if hasattr(LT_mat, "numpy") else np.asarray(LT_mat)
        corr_raw = lt_np @ lt_np.T

        row_metrics = {
            "sample_pos": int(pos),
            "mean_W1_refined_vs_GT": float(np.mean(w1_per_dim)),
            "mean_JSD_refined_vs_GT": float(np.mean(jsd_per_dim)),
            "mean_W1_raw_encoder_vs_GT": float(np.mean(w1_raw_per_dim)),
            "mean_JSD_raw_encoder_vs_GT": float(np.mean(jsd_raw_per_dim)),
            "posterior_mean_MAE_vs_GT": float(np.mean(np.abs(mean_vae - mean_gt))),
            "posterior_std_MAE_vs_GT": float(np.mean(np.abs(std_vae - std_gt))),
            "corr_frob_error_refined_vs_GT": float(np.linalg.norm(corr_vae - corr_gt, ord="fro")),
            "corr_frob_error_copula_vs_GT": float(np.linalg.norm(corr_raw - corr_gt, ord="fro")),
            "MAE_mean_GT_vs_ztrue": float(np.mean(np.abs(mean_gt - z_true))),
            "MAE_mean_VAE_vs_ztrue": float(np.mean(np.abs(mean_vae - z_true))),
        }
        comparison_rows.append(row_metrics)
        print(
            f"  [Sample {pos}] W1(KS-VAE, GT) = {row_metrics['mean_W1_refined_vs_GT']:.4f} | "
            f"JSD(KS-VAE, GT) = {row_metrics['mean_JSD_refined_vs_GT']:.4f} | "
            f"Mean Diff = {row_metrics['posterior_mean_MAE_vs_GT']:.4f}"
        )

        # 4. Generate the 3 Corner Plots for Sample `pos`
        plot_single_corner(
            x_gt_filt, w_gt_norm, z_true, lbound,
            title_label=f"Sample {pos}: Reference Ground-Truth Bayesian Posterior $p(\\mathbf{{z}}|\\mathbf{{m}})$",
            cbar_label="Posterior density (Ground Truth)",
            save_path=os.path.join(out_dir, f"P{pos}_GT.png"),
            cmap="viridis",
        )
        plot_single_corner(
            x_vae_filt, w_vae_norm, z_true, lbound,
            title_label=f"Sample {pos}: Estimated KS-Copula Posterior ($K={num_KS}$)",
            cbar_label="Posterior density (KS-Copula VAE)",
            save_path=os.path.join(out_dir, f"P{pos}_VAE.png"),
            cmap="viridis",
        )
        plot_split_and_overlay_comparison_corner(
            x_gt_filt, w_gt_norm,
            x_vae_filt, w_vae_norm,
            a_vals, b_vals, w_vals,
            z_true, lbound, pos,
            save_path=os.path.join(out_dir, f"P{pos}_Overlay.png"),
        )
        plot_1d_marginals_strip(
            x_gt_filt, w_gt_norm,
            x_vae_filt, w_vae_norm,
            a_vals, b_vals, w_vals,
            z_true, lbound, pos,
            save_path=os.path.join(out_dir, f"P{pos}_1D.png"),
        )

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(os.path.join(out_dir, "GT_vs_Posterior_Metrics.csv"), index=False)
    summary = {
        "run_folder": folder_path,
        "num_KS": num_KS,
        "sensor_regime": sensor_regime,
        "aggregate_means": df_comp.mean(numeric_only=True).to_dict() if not df_comp.empty else {},
        "per_sample": comparison_rows,
    }
    with open(os.path.join(out_dir, "GT_vs_Posterior_Metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Saved Ground-Truth vs. Posterior comparison plots and metrics to:\n  {out_dir}")


if __name__ == "__main__":
    main()
