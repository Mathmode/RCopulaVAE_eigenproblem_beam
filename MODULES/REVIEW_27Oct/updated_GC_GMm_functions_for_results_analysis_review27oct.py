# -*- coding: utf-8 -*-
"""
updated_GC_GMm_functions_for_results_analysis_review27oct.py
============================================================
Full Updated Version of `MODULES/COPULAS/updated_GC_GMm_functions_for_results_analysis.py`
for the MS CPENG-8674 Major Revision (saved in `code/MODULES/review_27oct/`).

Supported & Verified Changes Relative to Original `updated_GC_GMm_functions_for_results_analysis.py`:
-----------------------------------------------------------------------------------------------------
1. [FIX 1 -- Exact `TruncatedNormal` Quantiles, Exact Posterior Mean, and Exact Log-PDF in
             `calculate_testing_metrics` (Addresses Reviewer 1 Comment 3 & Editor Comment 3)]
   - In the original script (lines 121-126 and 136):
     (a) The 95% credible intervals were computed as symmetric untruncated intervals
         `z_lower = np.maximum(means - 1.96 * scales, lbound)` and
         `z_upper = np.minimum(means + 1.96 * scales, 1.0)`.
         Meanwhile, `GC_Gm_uncertainty_quantification.py` (line 39) evaluated coverage using
         `tfd.TruncatedNormal.cdf`. Because a truncated normal distribution near `z = 1.0`
         or `z = lbound` is skewed/asymmetric, symmetric clipping differed from the exact
         equal-tailed quantiles `[F_TN^-1(0.025), F_TN^-1(0.975)]`, causing the 77.2% vs.
         0.789 discrepancy flagged by Reviewer 1!
     (b) For `TruncatedNormal(loc=mu, scale=sigma, low=lbound, high=1.0)`, `loc = mu` is the
         mode/mean of the *untruncated* parent Gaussian, whereas the exact posterior expectation
         is `E[z|m] = truncnorm.mean(a, b, loc=mu, scale=sigma)`.
     (c) In line 136, `norm.logpdf` was used instead of `truncnorm.logpdf`, omitting the
         truncation normalization constant `log(Phi(b) - Phi(a))`.
   - Fix: `calculate_testing_metrics` now computes and returns BOTH the exact `TruncatedNormal`
     metrics (`95% Coverage (Exact TN)`, `Mean Interval Width (Exact TN)`, `MSE (Exact TN Mean)`,
     `Avg Log-Likelihood (Exact TN)`) AND the legacy clipped metrics side-by-side so you can
     transparently show the reviewers the exact difference!

2. [FIX 2 -- Added Healthy-Element False-Positive Rate (FPR) Metrics (Reviewer 1 Comment 6)]
   - `calculate_testing_metrics` also evaluates the False-Positive Rate on all intact elements
     (`z_true >= 0.98`), reporting how often the upper 95% credible bound falls below `0.95`
     (`FPR_95CI_at_0.95`), directly answering Reviewer 1 Comment 6.

3. [FIX 3 -- Restored Normalized Posterior Weights in `plot_results_PDF_uncertainty` (Line 220 Bug)]
   - In the original script (line 219-220):
         `# w_filtered /= np.sum(w_filtered)`
         `w_filtered /= w_filtered`
     Dividing `w_filtered` by itself set every sample's likelihood weight to `1.0`, erasing
     the Bayesian importance weights in the KDE contour plot!
   - Fix: Restored `w_filtered = w_filtered / np.sum(w_filtered)` (with fallback to uniform weights
     only if `np.sum(w_filtered) <= 0`).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, truncnorm
import tensorflow as tf

tf.config.optimizer.set_jit(False)
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from .GC_GMm_GPU_eigen_functions_updated_review27oct import assemble_global_Kmatrices

K.backend.set_floatx("float32")


def plot_configuration():
    plt.rc("font", size=28)
    plt.rc("axes", titlesize=28, labelsize=24)
    plt.rc("xtick", labelsize=28)
    plt.rc("ytick", labelsize=28)
    plt.rc("legend", fontsize=28)
    plt.rc("figure", titlesize=28)


sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style="dark")
plot_configuration()


def gaussian_copula(LT_matrix, n_dims, n_samples):
    mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
    mvn_samples = mvn_model.sample(n_samples)
    copula_samples = tf.clip_by_value(tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples), 1e-6, 1.0 - 1e-6)
    return copula_samples, mvn_samples, mvn_model


def gaussian_marginal_samples(locs, scales, copula_samples, lbound):
    marginal_samples = tfd.TruncatedNormal(
        loc=locs, scale=scales, low=lbound, high=1.0
    ).quantile(copula_samples)
    return marginal_samples


def calculate_MAC(modes_true, modes_pred):
    """
    Aligned with GT script calculate_MAC. Expects (Batch, Modes, Nodes).
    """
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return tf.linalg.diag_part(mac_matrix).numpy()


def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    """
    Standard Physics Engine - Matches Ground Truth script exactly.
    """
    with tf.device("/CPU:0"):
        A_batch = tf.matmul(L_inv_tf, tf.matmul(K_batch, tf.transpose(L_inv_tf)))
        vals, vecs = tf.linalg.eigh(A_batch)
        vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e20)
        f_Hz = tf.math.sqrt(vals_trunc) / (2 * np.pi)
        phi_free = tf.matmul(tf.transpose(L_inv_tf), vecs[:, :, :n_modes])

        batch_size = tf.shape(K_batch)[0]
        indices = tf.expand_dims(free_dofs, axis=-1)
        phi_free_t = tf.transpose(phi_free, perm=[1, 0, 2])
        full_modes_t = tf.scatter_nd(indices, phi_free_t, shape=[n_dofs, batch_size, n_modes])
        full_modes = tf.transpose(full_modes_t, perm=[1, 0, 2])

        vert_modes = full_modes[:, 0::2, :]
        rot_modes = full_modes[:, 1::2, :]

        abs_vert = tf.abs(vert_modes)
        max_idx = tf.argmax(abs_vert, axis=1)
        batch_idx = tf.range(batch_size)[:, tf.newaxis]
        mode_idx = tf.range(n_modes)[tf.newaxis, :]
        gather_coords = tf.stack(
            [
                tf.broadcast_to(batch_idx, [batch_size, n_modes]),
                tf.cast(max_idx, tf.int32),
                tf.broadcast_to(mode_idx, [batch_size, n_modes]),
            ],
            axis=-1,
        )
        peak_vals = tf.gather_nd(vert_modes, gather_coords)
        signs = tf.sign(peak_vals)
        signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
        vert_modes = vert_modes * signs[:, tf.newaxis, :]
        rot_modes = rot_modes * signs[:, tf.newaxis, :]

        return (
            f_Hz.numpy(),
            tf.transpose(rot_modes, perm=[0, 2, 1]).numpy(),
            tf.transpose(vert_modes, perm=[0, 2, 1]).numpy(),
        )


def calculate_testing_metrics(predicted_stats, test_datasets, lbound=0.45, healthy_threshold=0.95):
    """
    Calculates comprehensive point-estimate, uncertainty calibration, and false-positive
    metrics using BOTH exact `TruncatedNormal` quantiles/moments and legacy clipped formulas.
    """
    means = np.squeeze(np.asarray(predicted_stats["test_means"], dtype=np.float64))
    scales = np.maximum(np.squeeze(np.asarray(predicted_stats["test_scales"], dtype=np.float64)), 1e-6)
    z_true = np.asarray(test_datasets["alpha_factors_true_test"], dtype=np.float64)

    # Standardized truncation limits for scipy.stats.truncnorm
    a = (lbound - means) / scales
    b = (1.0 - means) / scales

    # 1. Exact TruncatedNormal Posterior Mean, Median, and Equal-Tailed 95% Credible Intervals
    trunc_mean = truncnorm.mean(a, b, loc=means, scale=scales)
    ci_low_exact = truncnorm.ppf(0.025, a, b, loc=means, scale=scales)
    ci_high_exact = truncnorm.ppf(0.975, a, b, loc=means, scale=scales)

    covered_exact = (z_true >= ci_low_exact) & (z_true <= ci_high_exact)
    coverage_exact = float(np.mean(covered_exact))
    width_exact = float(np.mean(ci_high_exact - ci_low_exact))

    mse_exact_mean = float(np.mean(np.square(trunc_mean - z_true)))
    rmse_exact_mean = float(np.sqrt(mse_exact_mean))
    mae_exact_mean = float(np.mean(np.abs(trunc_mean - z_true)))

    # Exact TruncatedNormal log-likelihood (clipped slightly inside [lbound, 1.0] for boundary z=1.0)
    z_true_safe = np.clip(z_true, lbound + 1e-6, 1.0 - 1e-6)
    log_prob_exact = float(np.mean(truncnorm.logpdf(z_true_safe, a, b, loc=means, scale=scales)))

    # 2. Legacy Untruncated / Clipped Metrics (for transparent comparison on R1.3)
    mse_loc = float(np.mean(np.square(means - z_true)))
    mae_loc = float(np.mean(np.abs(means - z_true)))
    z_lower_clip = np.maximum(means - 1.96 * scales, lbound)
    z_upper_clip = np.minimum(means + 1.96 * scales, 1.0)
    covered_clip = (z_true >= z_lower_clip) & (z_true <= z_upper_clip)
    coverage_clip = float(np.mean(covered_clip))
    width_clip = float(np.mean(z_upper_clip - z_lower_clip))

    # 3. Healthy-Element False-Positive Rate (FPR) for Reviewer 1 Comment 6 (R1.6)
    intact_mask = z_true >= 0.98
    if np.any(intact_mask):
        # Flagged as damaged if the upper 95% credible bound excludes `healthy_threshold` (0.95)
        fpr_95ci = float(np.mean(ci_high_exact[intact_mask] < healthy_threshold))
    else:
        fpr_95ci = np.nan

    metrics = {
        "MSE": mse_exact_mean,
        "RMSE": rmse_exact_mean,
        "MAE": mae_exact_mean,
        "95% Coverage": coverage_exact,
        "Calibration Error |Cov - 0.95|": abs(coverage_exact - 0.95),
        "Mean Interval Width": width_exact,
        "Avg Log-Likelihood": log_prob_exact,
        "Healthy Element FPR (95% CI < 0.95)": fpr_95ci,
        # Legacy values kept for R1.3 comparison table
        "Legacy MSE (loc)": mse_loc,
        "Legacy MAE (loc)": mae_loc,
        "Legacy 95% Coverage (Clipped)": coverage_clip,
        "Legacy Mean Interval Width (Clipped)": width_clip,
    }

    return metrics, covered_exact


def plot_results_PDF_uncertainty(
    fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
    predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path,
):
    """
    Generates joint posterior plots using VAE-predicted distribution + Bayesian re-weighting.
    """
    locs = predicted_stats["test_means"][pos, :]
    scales = predicted_stats["test_scales"][pos, :]
    LT_matrix = predicted_stats["test_L_matrices"][pos, :]

    Freqs_true = test_datasets["Freqs_true_test"]
    Rotmodes_true = test_datasets["Rotmodes_true_test"]
    Vertmodes_true = test_datasets["Vertmodes_true_test"]
    Alphas_true = test_datasets["alpha_factors_true_test"]
    z_true = Alphas_true[pos, :]

    n_dims = locs.shape[-1]
    n_elements = n_dims

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, _, _ = gaussian_copula(LT_matrix, n_dims, n_samples)
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

    Ke_matrices_dam = tf.einsum("BE, EKQ -> BEKQ", z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)

    batch_size_pe = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size_pe):
        f, r, v = physics_engine_step(
            Kfree_matrices[i : i + batch_size_pe], L_inv_tf, n_modes, free_dofs, n_dofs
        )
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)

    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)

    obs_f_scaled = Freqs_true[pos]
    pred_logscaled = (np.log(np.maximum(f_pred, 1e-7)) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

    obs_r_batch = np.repeat(Rotmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    obs_v_batch = np.repeat(Vertmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)

    rot_mac = calculate_MAC(obs_r_batch, rot_pred)
    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)

    inv_gamma_val = 1.0 / (beta ** 2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood)

    quantile_threshold = 0.95
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]

    if len(mask_indices) < 30:
        mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    # [FIX 3]: Properly normalize filtered weights instead of dividing w_filtered by itself!
    w_sum = np.sum(w_filtered)
    w_filtered = (w_filtered / w_sum) if w_sum > 0 else (np.ones_like(w_filtered) / len(w_filtered))

    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor="white")
    labels = [f"$z_{{{k+1}}}$" for k in range(n_elements)]
    cf = None

    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_filtered)
                x_grid = np.linspace(lbound, 1.0, 200)
                y_grid = kde1d(x_grid)
                ax.fill_between(x_grid, y_grid, color="steelblue", alpha=0.4)
                ax.plot(x_grid, y_grid, color="steelblue", lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha="center", va="center",
                        fontweight="bold", transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0)
                ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_filtered)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap="viridis")
                ax.plot(z_true[c], z_true[r], "r*", markersize=14, markeredgecolor="white", label="Truth")
                ax.set_xlim(lbound, 1.0)
                ax.set_ylim(lbound, 1.0)
            else:
                ax.axis("off")

            if r >= c:
                ax.tick_params(labelsize=22)
                if c == 0 and r != 0:
                    ax.set_ylabel(labels[r], fontsize=30)
                else:
                    ax.tick_params(labelleft=False)
                if r == n_elements - 1:
                    ax.set_xlabel(labels[c], fontsize=30)
                else:
                    ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label("Posterior density (VAE)", fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    import matplotlib.lines as mlines
    gt_marker = mlines.Line2D([], [], color="red", marker="*", linestyle="None",
                              markersize=18, markeredgecolor="white", label="Ground truth")
    fig.legend(handles=[gt_marker], loc="upper right", bbox_to_anchor=(0.82, 0.93),
               fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors", f"Sample_{pos}_Matched.png"), dpi=150)
    plt.close()


def calculate_posterior_PDF_info(
    fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
    predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path,
):
    locs = predicted_stats["test_means"][pos, :]
    scales = predicted_stats["test_scales"][pos, :]
    LT_matrix = predicted_stats["test_L_matrices"][pos, :]

    Freqs_true = test_datasets["Freqs_true_test"]
    Rotmodes_true = test_datasets["Rotmodes_true_test"]
    Vertmodes_true = test_datasets["Vertmodes_true_test"]
    Alphas_true = test_datasets["alpha_factors_true_test"]
    z_true = Alphas_true[pos, :]

    n_dims = locs.shape[-1]
    n_elements = n_dims

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, _, _ = gaussian_copula(LT_matrix, n_dims, n_samples)
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

    Ke_matrices_dam = tf.einsum("BE, EKQ -> BEKQ", z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)

    batch_size = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size):
        f, r, v = physics_engine_step(
            Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs
        )
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)

    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)

    log_obs_f = Freqs_true[pos]
    obs_r = Rotmodes_true[pos]
    obs_v = Vertmodes_true[pos]

    pred_log = np.log(np.maximum(f_pred, 1e-7))
    pred_logscaled = (pred_log - mean_freq) / std_freq
    loss_f = np.mean(np.square(log_obs_f - pred_logscaled), axis=1)

    obs_r_batch = tf.convert_to_tensor(np.repeat(obs_r[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)
    obs_v_batch = tf.convert_to_tensor(np.repeat(obs_v[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)

    best_r = calculate_MAC(obs_r_batch, tf.convert_to_tensor(rot_pred))
    best_v = calculate_MAC(obs_v_batch, tf.convert_to_tensor(vert_pred))
    loss_mac = np.mean((1.0 - best_r) + (1.0 - best_v), axis=1)

    inv_gamma_val = 1.0 / (beta ** 2)
    total_loss = loss_f + loss_mac
    exponent = -total_loss * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood)

    return z_true, z_samples, posterior_weights


def plot_physical_pdf_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45):
    """
    Visualizes the physical uncertainty by plotting the marginal PDF for each element.
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 28,
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
    })

    fig, (ax_pdf, ax_beam) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [4, 1]}, sharex=True, facecolor="white"
    )

    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)

    pdf_color = "steelblue"
    truth_color = "#D62728"
    cmap = cm.get_cmap("YlOrRd_r")
    norm_color = mcolors.Normalize(vmin=lbound, vmax=1.0)
    width_scale = 0.4

    for i in range(n_elements):
        kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
        pdf_values = kde(z_grid)
        pdf_visual = (pdf_values / np.max(pdf_values)) * width_scale
        x_center = i + 1

        ax_pdf.fill_betweenx(
            z_grid, x_center - pdf_visual, x_center + pdf_visual,
            color=pdf_color, alpha=0.4, label="Marginal PDF" if i == 0 else "",
        )
        ax_pdf.plot(x_center - pdf_visual, z_grid, color=pdf_color, lw=2.0, alpha=0.6)
        ax_pdf.plot(x_center + pdf_visual, z_grid, color=pdf_color, lw=2.0, alpha=0.6)

        if z_true is not None:
            ax_pdf.plot(
                x_center, z_true[i], marker="*", color=truth_color, markersize=18,
                markeredgecolor="white", markeredgewidth=1.5, zorder=10,
                label="Ground Truth" if i == 0 else "",
            )

    if z_true is not None:
        x_step = np.arange(0.5, n_elements + 1.5, 1)
        y_step = np.concatenate([z_true, [z_true[-1]]])
        ax_pdf.step(x_step, y_step, where="post", color=truth_color, linestyle="--", alpha=0.5, lw=2.5, zorder=1)

    ax_pdf.set_ylabel(r"Stiffness reduction factor ($\mathbf{z}$)", labelpad=15)
    ax_pdf.set_ylim([lbound - 0.02, 1.02])
    ax_pdf.set_xticks(elements)
    ax_pdf.set_xticklabels([f"$e_{{{k}}}$" for k in elements])
    ax_pdf.grid(axis="y", alpha=0.2, linestyle="-", color="gray")

    for spine in ["top", "right"]:
        ax_pdf.spines[spine].set_visible(False)
    ax_pdf.spines["left"].set_linewidth(1.5)
    ax_pdf.spines["bottom"].set_linewidth(1.5)

    ax_pdf.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=True, fancybox=True, shadow=True, borderpad=0.8)

    beam_viz = z_true.reshape(1, -1)
    im = ax_beam.imshow(beam_viz, cmap=cmap, norm=norm_color, aspect="auto", extent=[0.5, n_elements + 0.5, 0, 1])
    ax_beam.set_yticks([])
    ax_beam.set_xticks(elements)
    ax_beam.tick_params(axis="x", length=0)

    for j in range(n_elements):
        gt_val = z_true[j]
        text_color = "white" if gt_val < 0.6 else "black"
        ax_beam.text(j + 1, 0.5, f"{gt_val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=22)

    for spine in ax_beam.spines.values():
        spine.set_linewidth(2.5)

    ax_beam.plot(0.5, 0, marker="^", markersize=25, color="black", clip_on=False, zorder=15)
    ax_beam.plot(n_elements + 0.5, 0, marker="^", markersize=25, color="black", clip_on=False, zorder=15)

    cbar_ax = fig.add_axes([0.93, 0.09, 0.015, 0.14])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\mathbf{z}$", fontsize=26)
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.subplots_adjust(hspace=0.1)

    save_dir = os.path.join(folder_path, "Physical_PDF_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Sample_{pos}_UQ_Violin.png"), dpi=500, bbox_inches="tight", transparent=False)
    plt.close()
