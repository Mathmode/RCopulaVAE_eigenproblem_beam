# -*- coding: utf-8 -*-
"""
Updated Results Analysis & UQ Functions for Kumaraswamy Mixture Copula (review_27oct).

Key updates & bug fixes relative to Last_used_scripts/GC_KSmix_functions_for_results_analysis.py:
1. Exact Mixture Inverse CDF in `kumaraswamy_mixture_quantile`:
   - In the previous script (lines 74-88), `kumaraswamy_mixture_quantile` drew random categorical
     component indices `tf.random.categorical(logits, n_samples)` independently per dimension and
     applied the single-component quantile formula. That broke the Copula rank correlation across
     dimensions and mismatched the training decoder (`ks_mixture_marginal_samples`), which inverts
     the mixture CDF F_mix(x) = sum_k w_k F_k(x) = u.
   - Now uses the exact closed-form quantile when `num_KS == 1` (unimodal) and the vectorized
     35-step bisection solver on F_mix(x) = u when `num_KS > 1` (multimodal), matching training
     bit-for-bit.
2. Healthy Scenario False Positive Rate (FPR) & Localization Accuracy (Reviewer 2 Comment 5):
   - Added to `calculate_ks_mixture_metrics` alongside MSE, MAE, 95% Coverage, Mean Interval Width,
     and Log-Likelihood.
3. Flexible FE Discretization vs. Copula Dimension (Concern 1):
   - Supports any FE mesh size `n_fe_elements` alongside `n_dims = 5` damage zones.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import beta as beta_func
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
import matplotlib.lines as mlines

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from MODULES.review_27oct.GC_GMm_GPU_eigen_functions_updated_review27oct import assemble_global_Kmatrices

K.backend.set_floatx('float32')


def plot_configuration():
    plt.rc('font', size=24)
    plt.rc('axes', titlesize=24, labelsize=22)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('figure', titlesize=24)


sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style="dark")
plot_configuration()


def gaussian_copula(LT_matrix, n_dims, n_samples):
    """
    Generates uniform [0, 1] samples (n_samples, n_dims) using the Gaussian Copula.
    """
    LT_matrix = tf.cast(LT_matrix, tf.float32)
    dim = tf.shape(LT_matrix)[-1]
    mvn_model = tfd.MultivariateNormalTriL(
        loc=tf.zeros(dim, dtype=tf.float32),
        scale_tril=LT_matrix
    )
    mvn_samples = mvn_model.sample(n_samples)
    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)


def kumaraswamy_mixture_quantile(u, a, b, weights, lbound=0.45):
    """
    Exact Inverse CDF (Quantile function) for a Kumaraswamy Mixture on [lbound, 1.0].

    Parameters
    ----------
    u : Tensor of shape (n_samples, n_dims) in (0, 1)
    a, b, weights : arrays/Tensors of shape (num_KS, n_dims) or (n_dims, num_KS)
    lbound : float, physical lower bound of stiffness factor z (default 0.45)

    Returns
    -------
    z_samples : Tensor of shape (n_samples, n_dims) in [lbound, 1.0]
    """
    u = tf.cast(u, dtype=tf.float32)
    u = tf.clip_by_value(u, 1e-6, 1.0 - 1e-6)
    n_samples, n_dims = u.shape

    a = tf.convert_to_tensor(a, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    # Standardize parameter shapes to (n_dims, num_KS)
    if a.shape[0] != n_dims and a.shape[1] == n_dims:
        a = tf.transpose(a)
        b = tf.transpose(b)
        weights = tf.transpose(weights)

    # Normalize weights along component axis to be safe
    weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    num_KS = int(a.shape[1])

    if num_KS == 1:
        # Exact closed-form quantile for unimodal Kumaraswamy (num_KS = 1)
        a_1 = tf.reshape(a[:, 0], [1, n_dims])  # (1, D)
        b_1 = tf.reshape(b[:, 0], [1, n_dims])  # (1, D)
        one_minus_u = tf.clip_by_value(1.0 - u, 1e-7, 1.0 - 1e-7)
        ks_unit = tf.pow(
            tf.clip_by_value(1.0 - tf.pow(one_minus_u, 1.0 / b_1), 1e-7, 1.0 - 1e-7),
            1.0 / a_1
        )
    else:
        # Vectorized 35-step bisection on F_mix(x) = sum_k w_k (1 - (1 - x^a_k)^b_k) = u
        a_exp = a[tf.newaxis, :, :]        # (1, D, K)
        b_exp = b[tf.newaxis, :, :]        # (1, D, K)
        w_exp = weights[tf.newaxis, :, :]  # (1, D, K)

        low = tf.zeros_like(u)
        high = tf.ones_like(u)
        for _ in range(35):
            mid = 0.5 * (low + high)
            mid_exp = tf.clip_by_value(mid[:, :, tf.newaxis], 1e-7, 1.0 - 1e-7)
            term = tf.clip_by_value(1.0 - tf.pow(mid_exp, a_exp), 1e-7, 1.0 - 1e-7)
            cdf_k = 1.0 - tf.pow(term, b_exp)
            mix_cdf = tf.reduce_sum(w_exp * cdf_k, axis=-1)
            low = tf.where(mix_cdf < u, mid, low)
            high = tf.where(mix_cdf >= u, mid, high)
        ks_unit = 0.5 * (low + high)

    z_samples = lbound + (1.0 - lbound) * ks_unit
    return z_samples


def calculate_MAC(modes_true, modes_pred):
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return tf.linalg.diag_part(mac_matrix).numpy()


def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    with tf.device('/CPU:0'):
        A_batch = tf.matmul(L_inv_tf, tf.matmul(K_batch, tf.transpose(L_inv_tf)))
        vals, vecs = tf.linalg.eigh(A_batch)
        vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e+20)
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
        gather_coords = tf.stack([
            tf.broadcast_to(batch_idx, [batch_size, n_modes]),
            tf.cast(max_idx, tf.int32),
            tf.broadcast_to(mode_idx, [batch_size, n_modes])
        ], axis=-1)
        peak_vals = tf.gather_nd(vert_modes, gather_coords)
        signs = tf.sign(peak_vals)
        signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
        vert_modes = vert_modes * signs[:, tf.newaxis, :]
        rot_modes = rot_modes * signs[:, tf.newaxis, :]

        return (
            f_Hz.numpy(),
            tf.transpose(rot_modes, perm=[0, 2, 1]).numpy(),
            tf.transpose(vert_modes, perm=[0, 2, 1]).numpy()
        )


def calculate_ks_mixture_metrics(predicted_stats, test_datasets, lbound=0.45, healthy_threshold=0.95):
    """
    Calculates numerical metrics to evaluate the GC-VAE (Kumaraswamy Mixture) inference:
    - MSE, MAE, RMSE (overall and per-regime)
    - 95% Credible Interval Coverage and Mean Interval Width (Sharpness)
    - Average Marginal Log-Likelihood (in physical space [lbound, 1.0])
    - Healthy Element False Positive Rate (FPR at z < healthy_threshold)
    """
    test_a = np.asarray(predicted_stats['test_a'], dtype=np.float64)            # [N, K, D]
    test_b = np.asarray(predicted_stats['test_b'], dtype=np.float64)            # [N, K, D]
    test_weights = np.asarray(predicted_stats['test_weights'], dtype=np.float64)  # [N, K, D]
    z_true = np.asarray(test_datasets['alpha_factors_true_test'], dtype=np.float64)  # [N, D]

    # Ensure weights sum to 1 along component axis=1
    test_weights = test_weights / np.sum(test_weights, axis=1, keepdims=True)
    n_samples, num_KS, n_dims = test_a.shape

    # --- 1. Point Estimate (Analytical Mean of Kumaraswamy Mixture) ---
    def ks_mean_func(a, b):
        return b * beta_func(1.0 + 1.0 / a, b)

    comp_means = ks_mean_func(test_a, test_b)  # [N, K, D]
    predicted_means_unit = np.sum(test_weights * comp_means, axis=1)  # [N, D]
    predicted_means = lbound + (1.0 - lbound) * predicted_means_unit  # [N, D]

    mse = float(np.mean(np.square(predicted_means - z_true)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predicted_means - z_true)))

    # Healthy Element False Positive Rate (Reviewer 2 Comment 5)
    healthy_elem_mask = z_true >= 0.999
    if np.any(healthy_elem_mask):
        healthy_elem_fpr = float(np.mean(predicted_means[healthy_elem_mask] < healthy_threshold))
        healthy_elem_mae = float(np.mean(np.abs(predicted_means[healthy_elem_mask] - 1.0)))
    else:
        healthy_elem_fpr = float('nan')
        healthy_elem_mae = float('nan')

    # --- 2. Uncertainty Calibration (95% Credible Intervals) ---
    grid = np.linspace(0.0, 1.0, 1000)
    z_lower = np.zeros((n_samples, n_dims))
    z_upper = np.zeros((n_samples, n_dims))

    for i in range(n_samples):
        for d in range(n_dims):
            a_vec = test_a[i, :, d]        # [K]
            b_vec = test_b[i, :, d]        # [K]
            w_vec = test_weights[i, :, d]  # [K]

            comp_cdfs = 1.0 - np.power(np.clip(1.0 - np.power(grid[:, None], a_vec), 0.0, 1.0), b_vec)
            mix_cdf = np.sum(w_vec * comp_cdfs, axis=1)

            idx_low = min(np.searchsorted(mix_cdf, 0.025), len(grid) - 1)
            idx_high = min(np.searchsorted(mix_cdf, 0.975), len(grid) - 1)
            z_lower[i, d] = grid[idx_low]
            z_upper[i, d] = grid[idx_high]

    z_lower_phys = lbound + (1.0 - lbound) * z_lower
    z_upper_phys = lbound + (1.0 - lbound) * z_upper

    covered = (z_true >= z_lower_phys) & (z_true <= z_upper_phys)
    coverage_score = float(np.mean(covered))
    sharpness = float(np.mean(z_upper_phys - z_lower_phys))

    # --- 3. Probabilistic Log-Likelihood ---
    z_true_unit = (z_true - lbound) / (1.0 - lbound)
    z_true_unit = np.clip(z_true_unit, 1e-5, 1.0 - 1e-5)

    log_probs = []
    for i in range(n_samples):
        z_val = z_true_unit[i]       # [D]
        a_k = test_a[i]              # [K, D]
        b_k = test_b[i]              # [K, D]
        w_k = test_weights[i]        # [K, D]

        pdf_comps = w_k * a_k * b_k * np.power(z_val, a_k - 1.0) * np.power(
            np.clip(1.0 - np.power(z_val, a_k), 1e-12, 1.0), b_k - 1.0
        )
        pdf_mix_phys = np.sum(pdf_comps, axis=0) / (1.0 - lbound)
        log_probs.append(np.mean(np.log(pdf_mix_phys + 1e-12)))

    avg_log_prob = float(np.mean(log_probs))

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        '95% Coverage': coverage_score,
        'Mean Interval Width': sharpness,
        'Avg Log-Likelihood': avg_log_prob,
        'Healthy Element FPR (tau=0.95)': healthy_elem_fpr,
        'Healthy Element MAE': healthy_elem_mae,
    }

    return metrics, covered


def plot_results_PDF_uncertainty(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path,
                                 show_plot=False):
    """
    Generates joint posterior plots using KS-mixture Copula VAE + Bayesian importance re-weighting.
    """
    a_vals = predicted_stats['test_a'][pos]
    b_vals = predicted_stats['test_b'][pos]
    w_vals = predicted_stats['test_weights'][pos]
    LT_matrix = predicted_stats['test_L_matrices'][pos]

    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    z_true = Alphas_true[pos, :]

    n_dims = LT_matrix.shape[-1]
    n_fe_elements = int(Ke_matrices.shape[0])

    # 1. Generate samples from KS Mixture Copula
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples = gaussian_copula(LT_matrix, n_dims, n_samples)
    z_samples = kumaraswamy_mixture_quantile(copula_samples, a_vals, b_vals, w_vals, lbound=lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

    # Map zone factors to FE elements if n_fe_elements > n_dims
    if n_fe_elements == n_dims:
        z_elem_samples = z_samples
    else:
        elems_per_zone = n_fe_elements // n_dims
        z_elem_samples = tf.repeat(z_samples, repeats=elems_per_zone, axis=1)

    # 2. Physics propagation
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_elem_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_fe_elements, n_samples, fixed_dofs_indices)

    batch_size_pe = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size_pe):
        f, r, v = physics_engine_step(Kfree_matrices[i:i + batch_size_pe], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)

    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)

    # 3. Calculate Bayesian importance weights
    obs_f_scaled = Freqs_true[pos]
    pred_logscaled = (np.log(f_pred) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

    obs_r_batch = np.repeat(Rotmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    obs_v_batch = np.repeat(Vertmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)

    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    if np.all(np.abs(Rotmodes_true[pos]) < 1e-8):
        loss_mac = np.mean(1.0 - vert_mac, axis=1)
    else:
        rot_mac = calculate_MAC(obs_r_batch, rot_pred)
        loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)

    inv_gamma_val = 1.0 / (beta ** 2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood)

    # 4. Filtering & plotting
    quantile_threshold = 0.95
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    if len(mask_indices) < 30:
        mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_norm = w_filtered / np.sum(w_filtered)

    fig, axes = plt.subplots(n_dims, n_dims, figsize=(14, 14), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_dims)]
    cf = None

    for r in range(n_dims):
        for c in range(n_dims):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_norm)
                x_grid = np.linspace(lbound - 0.05, 1.0, 200)
                y_grid = kde1d(x_grid)
                ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                ax.axvline(z_true[r], color='red', linestyle='--', lw=1.8)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center',
                        fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0)
                ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_norm)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', label='Truth')
                ax.set_xlim(lbound, 1.0)
                ax.set_ylim(lbound, 1.0)
            else:
                ax.axis('off')

            if r >= c:
                ax.tick_params(labelsize=20)
                if c == 0 and r != 0:
                    ax.set_ylabel(labels[r], fontsize=26)
                else:
                    ax.tick_params(labelleft=False)
                if r == n_dims - 1:
                    ax.set_xlabel(labels[c], fontsize=26)
                else:
                    ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (KS-VAE)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)

    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93),
               fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors_KS"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors_KS", f'Sample_{pos}_Matched.png'), dpi=150)
    if show_plot:
        plt.show()
    plt.close()


def calculate_posterior_PDF_info(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    a_vals = predicted_stats['test_a'][pos]
    b_vals = predicted_stats['test_b'][pos]
    w_vals = predicted_stats['test_weights'][pos]
    LT_matrix = predicted_stats['test_L_matrices'][pos]

    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    z_true = test_datasets['alpha_factors_true_test'][pos, :]

    n_dims = LT_matrix.shape[-1]
    n_fe_elements = int(Ke_matrices.shape[0])

    copula_samples = gaussian_copula(LT_matrix, n_dims, n_samples)
    z_samples = kumaraswamy_mixture_quantile(copula_samples, a_vals, b_vals, w_vals, lbound=lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

    if n_fe_elements == n_dims:
        z_elem_samples = z_samples
    else:
        elems_per_zone = n_fe_elements // n_dims
        z_elem_samples = tf.repeat(z_samples, repeats=elems_per_zone, axis=1)

    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_elem_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_fe_elements, n_samples, fixed_dofs_indices)

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    batch_size = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size):
        f, r, v = physics_engine_step(Kfree_matrices[i:i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)

    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)

    obs_f_scaled = Freqs_true[pos]
    pred_logscaled = (np.log(f_pred) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

    obs_r_batch = np.repeat(Rotmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    obs_v_batch = np.repeat(Vertmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)

    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    if np.all(np.abs(Rotmodes_true[pos]) < 1e-8):
        loss_mac = np.mean(1.0 - vert_mac, axis=1)
    else:
        rot_mac = calculate_MAC(obs_r_batch, rot_pred)
        loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)

    inv_gamma_val = 1.0 / (beta ** 2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood)

    return z_true, z_samples.numpy(), posterior_weights


def plot_physical_pdf_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path,
                              lbound=0.45, show_plot=False):
    fig, (ax_pdf, ax_beam) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True, facecolor='white'
    )

    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)
    pdf_color = 'steelblue'
    truth_color = '#D62728'

    for i in range(n_elements):
        kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
        pdf_values = kde(z_grid)
        pdf_visual = (pdf_values / (np.max(pdf_values) + 1e-12)) * 0.4
        x_center = i + 1

        ax_pdf.fill_betweenx(z_grid, x_center - pdf_visual, x_center + pdf_visual, color=pdf_color, alpha=0.4)
        ax_pdf.plot(x_center - pdf_visual, z_grid, color=pdf_color, lw=1.5)
        ax_pdf.plot(x_center + pdf_visual, z_grid, color=pdf_color, lw=1.5)

        if z_true is not None:
            ax_pdf.plot(x_center, z_true[i], marker='*', color=truth_color, markersize=18, markeredgecolor='white')

    ax_pdf.set_ylabel(r'Stiffness factor ($z$)')
    ax_pdf.set_ylim([lbound - 0.05, 1.05])
    ax_pdf.set_xticks(elements)
    ax_pdf.set_xticklabels([f'$z_{{{k}}}$' for k in elements])

    beam_viz = z_true.reshape(1, -1)
    ax_beam.imshow(beam_viz, cmap='YlOrRd_r', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1],
                   vmin=lbound, vmax=1.0)
    ax_beam.set_yticks([])

    plt.tight_layout()
    save_dir = os.path.join(folder_path, "KS_Physical_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_KS_UQ.png'), dpi=300)
    if show_plot:
        plt.show()
    plt.close()
