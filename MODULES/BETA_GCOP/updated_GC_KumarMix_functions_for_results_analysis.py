# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:31:59 2026
Updated Plotting Utilities for Kumaraswamy Mixture GC-VAE
@author: anafd
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
K.backend.set_floatx('float32') 

def plot_configuration():
    plt.rc('font', size=16)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=16)

sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style="dark")
plot_configuration()

def gaussian_copula(LT_matrix, n_dims, n_samples):
    mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
    mvn_samples = mvn_model.sample(n_samples)
    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return copula_samples, mvn_samples, mvn_model

def kumarswamy_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Inference-ready Bisection solver for Kumaraswamy Mixture Marginals.
    Robustly handles arbitrary dimensional rank (Batch, K, Dims) cleanly matching mixture shapes.
    """
    u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)
    original_u_shape = u.shape
    
    # Align parameters: We expect inputs as (..., K, Dims)
    # The mixture model components require them to be (..., Dims, K)
    rank = len(a_params.shape)
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1] # Cleanly swap K and Dims axes
    
    a_t = tf.transpose(a_params, perm=perm)
    b_t = tf.transpose(b_params, perm=perm)
    weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=perm), axis=-1)

    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights_t),
        components_distribution=tfd.Kumaraswamy(concentration1=a_t, concentration0=b_t)
    )

    # Ensure u matches the mixture batch shape exactly.
    # If u has a dummy dimension (e.g. [Batch, 1, Dims]), we squeeze it prior to comparing.
    if len(u.shape) > len(mixture_dist.batch_shape):
        u = tf.squeeze(u, axis=-2)

    low = tf.zeros_like(u) + 1e-6
    high = tf.ones_like(u) - 1e-6

    # Bisection search
    for _ in range(35):
        mid = (low + high) / 2.0
        cdf_mid = mixture_dist.cdf(mid)
        is_less = cdf_mid < u
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    x = (low + high) / 2.0
    z = lbound + (1.0 - lbound) * x
    
    # Restore original dimension footprint of U if we squeezed it
    if len(original_u_shape) > len(z.shape):
        z = tf.expand_dims(z, axis=-2)
        
    return z

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
        gather_coords = tf.stack([tf.broadcast_to(batch_idx, [batch_size, n_modes]),
                                  tf.cast(max_idx, tf.int32),
                                  tf.broadcast_to(mode_idx, [batch_size, n_modes])], axis=-1)
        peak_vals = tf.gather_nd(vert_modes, gather_coords)
        signs = tf.sign(peak_vals)
        vert_modes = vert_modes * signs[:, tf.newaxis, :]
        rot_modes = rot_modes * signs[:, tf.newaxis, :]

        return f_Hz.numpy(), tf.transpose(rot_modes, perm=[0, 2, 1]).numpy(), tf.transpose(vert_modes, perm=[0, 2, 1]).numpy()

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices

def calculate_testing_metrics(predicted_stats, test_datasets, lbound=0.45):
    """
    Calculates exact metrics using Kumaraswamy mixtures via Bisection.
    """
    a_params = predicted_stats['test_alphas']
    b_params = predicted_stats['test_betas']
    weights = predicted_stats['test_weights']
    z_true = test_datasets['z_factors_true_test']
    
    n_samples, n_elements = z_true.shape
    
    # 1. Point Estimates (Using the Median = 50% Quantile)
    u_median = tf.fill([n_samples, n_elements], 0.5)
    z_median = kumarswamy_marginal_samples(a_params, b_params, weights, u_median, lbound).numpy()
    
    mse = np.mean(np.square(z_median - z_true))
    mae = np.mean(np.abs(z_median - z_true))
    
    # 2. Uncertainty Calibration (Exact 95% Credible Intervals via Quantiles)
    u_lower = tf.fill([n_samples, n_elements], 0.025)
    u_upper = tf.fill([n_samples, n_elements], 0.975)
    
    z_lower = kumarswamy_marginal_samples(a_params, b_params, weights, u_lower, lbound).numpy()
    z_upper = kumarswamy_marginal_samples(a_params, b_params, weights, u_upper, lbound).numpy()
    
    covered = (z_true >= z_lower) & (z_true <= z_upper)
    coverage_score = np.mean(covered)
    sharpness = np.mean(z_upper - z_lower)
    
    # 3. Probabilistic Log-Likelihood (Marginal)
    x_true = (z_true - lbound) / (1.0 - lbound)
    x_true = np.clip(x_true, 1e-6, 1.0 - 1e-6)
    
    rank = len(a_params.shape)
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    
    a_t = tf.transpose(a_params, perm=perm)
    b_t = tf.transpose(b_params, perm=perm)
    weights_t = tf.nn.softmax(tf.transpose(weights, perm=perm), axis=-1)

    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights_t),
        components_distribution=tfd.Kumaraswamy(concentration1=a_t, concentration0=b_t)
    )
    
    log_prob_kuma = mixture_dist.log_prob(x_true)
    adjustment = tf.math.log(1.0 - lbound)
    log_probs = (log_prob_kuma - adjustment).numpy()
    avg_log_prob = np.mean(log_probs)

    metrics = {
        'MSE (Median)': mse,
        'MAE (Median)': mae,
        '95% Coverage': coverage_score,
        'Mean Interval Width': sharpness,
        'Avg Log-Likelihood': avg_log_prob
    }
    
    return metrics, covered

def plot_results_PDF_uncertainty(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    
    a_params = predicted_stats['test_alphas'][pos:pos+1]
    b_params = predicted_stats['test_betas'][pos:pos+1]
    weights = predicted_stats['test_weights'][pos:pos+1]
    
    LT_matrix = predicted_stats['test_L_matrices'][pos] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    z_true = test_datasets['z_factors_true_test'][pos,:]
    
    n_dims = a_params.shape[2]
    n_elements = n_dims

    # 1. GENERATE SAMPLES FROM VAE COPULA
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, _, _ = gaussian_copula(LT_matrix, n_dims, n_samples) 
    
    copula_samples_expanded = tf.expand_dims(copula_samples, axis=1) # (n_samples, 1, n_dims)
    
    a_params_tiled = tf.tile(a_params, multiples=[n_samples, 1, 1]) 
    b_params_tiled = tf.tile(b_params, multiples=[n_samples, 1, 1])
    weights_tiled = tf.tile(weights, multiples=[n_samples, 1, 1])

    z_samples = kumarswamy_marginal_samples(a_params_tiled, b_params_tiled, weights_tiled, copula_samples_expanded, lbound)
    z_samples = tf.squeeze(z_samples, axis=1) # -> (n_samples, n_dims)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

    # 2. PHYSICS PROPAGATION
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)
    
    batch_size_pe = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size_pe):
        f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size_pe], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)
        
    f_pred = np.concatenate(all_f, axis=0)
    rot_pred = np.concatenate(all_rot, axis=0)
    vert_pred = np.concatenate(all_vert, axis=0)
    
    # 3. CALCULATE BAYESIAN WEIGHTS (Likelihood)
    obs_f_scaled = Freqs_true[pos]
    pred_logscaled = (np.log(f_pred) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

    obs_r_batch = np.repeat(Rotmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    obs_v_batch = np.repeat(Vertmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)
    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)
    
    inv_gamma_val = 1.0 / (beta**2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood) 
    
    # 4. FILTERING & PLOTTING
    quantile_threshold = 0.95 
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    
    # Guard against completely collapsed samples avoiding scipys Singular Matrix Error in KDE
    jitter = np.random.normal(0, 1e-6, size=x_filtered.shape)
    x_filtered += jitter
    x_filtered = np.clip(x_filtered, lbound, 1.0)
    
    # Normalize weights for KDE plotting safely
    if np.sum(w_filtered) == 0:
        w_filtered = np.ones_like(w_filtered) / len(w_filtered)
    else:
        w_filtered /= np.sum(w_filtered)
    
    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    cf = None

    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_filtered)
                x_grid = np.linspace(lbound, 1.0, 200)
                y_grid = kde1d(x_grid)
                ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0); ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_filtered)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', label='Truth')
                ax.set_xlim(lbound, 1.0); ax.set_ylim(lbound, 1.0)
            else:
                ax.axis('off')

            if r >= c:
                ax.tick_params(labelsize=18)
                if c == 0 and r != 0: ax.set_ylabel(labels[r], fontsize=18)
                else: ax.tick_params(labelleft=False)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=18)
                else: ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (VAE)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    import matplotlib.lines as mlines
    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors", f'Sample_{pos}_Matched.png'), dpi=150)
    plt.show()
    plt.close()