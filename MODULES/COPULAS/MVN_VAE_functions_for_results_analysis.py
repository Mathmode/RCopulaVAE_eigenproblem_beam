# -*- coding: utf-8 -*-
"""
Updated Functions for Results Analysis - TMVN Baseline Version
Focuses on Truncated Multivariate Normal (Independent) latent space.
Includes comprehensive probabilistic metrics for GC-VAE comparison.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Set Float precision
K.backend.set_floatx('float32') 

def plot_configuration():
    plt.rc('font', size=28)
    plt.rc('axes', titlesize=28, labelsize=24)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('figure', titlesize=28)

sns.set(style="dark")
plot_configuration()

def tmvn_sampling(means, scales, n_samples, lbound):
    """
    Samples from the Truncated Normal latent space (Independent dimensions).
    """
    dist = tfd.TruncatedNormal(loc=means, scale=scales, low=lbound, high=1.0)
    samples = dist.sample(n_samples)
    return samples

def calculate_MAC(modes_true, modes_pred):
    """
    Standard MAC calculation for modal alignment.
    """
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return tf.linalg.diag_part(mac_matrix).numpy()

def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    """
    Standard Physics Engine for Eigenproblem solution.
    """
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

        # Sign Consistency
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
    Calculates numerical metrics for TMVN baseline evaluation.
    
    Includes:
    - MSE/MAE: Mean Point Accuracy
    - MRE: Mean Relative Error (Point estimate)
    - 95% Coverage: Calibration of uncertainty
    - Sharpness: Width of credible intervals
    - Avg Log-Likelihood: Probability density of truth under predicted TMVN
    """
    means = predicted_stats['test_means']
    scales = predicted_stats['test_scales']
    z_true = test_datasets['alpha_factors_true_test']
    
    # 1. Point Estimate Metrics
    mse = np.mean(np.square(means - z_true))
    mae = np.mean(np.abs(means - z_true))
    mre = np.mean(np.abs(means - z_true) / z_true) * 100 # In percentage
    
    # 2. Uncertainty Calibration (95% Credible Intervals)
    # Using the independent Truncated Normal properties
    z_lower = means - 1.96 * scales
    z_upper = means + 1.96 * scales
    
    # Clip to physical bounds for realistic interval assessment
    z_lower = np.maximum(z_lower, lbound)
    z_upper = np.minimum(z_upper, 1.0)
    
    covered = (z_true >= z_lower) & (z_true <= z_upper)
    coverage_score = np.mean(covered) # Ideal: 0.95
    sharpness = np.mean(z_upper - z_lower) # Lower is better (more confident)
    
    # 3. Probabilistic Log-Likelihood (The comparison gold standard)
    # We use tfd.TruncatedNormal to calculate exactly how the model 'sees' the truth
    dist = tfd.TruncatedNormal(loc=means, scale=scales, low=lbound, high=1.0)
    log_probs = dist.log_prob(z_true).numpy() # (N_test, n_elements)
    
    # Average log-likelihood across all samples and elements
    # Comparison point: GC-VAE should be higher if correlations help localize truth
    avg_log_prob = np.mean(log_probs)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'MRE (%)': mre,
        '95% Coverage': coverage_score,
        'Mean Interval Width': sharpness,
        'Avg Log-Likelihood': avg_log_prob
    }
    
    return metrics, covered

def plot_results_PDF_uncertainty_tmvn(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                     predicted_stats, L_inv, Ke_matrices, mean_freq, std_freq, lbound, folder_path):
    """
    Joint posterior plots for TMVN (Independent dimensions).
    """
    locs = predicted_stats['test_means'][pos, :]
    scales = predicted_stats['test_scales'][pos, :]
    z_true = test_datasets['alpha_factors_true_test'][pos, :]
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    
    n_elements = locs.shape[0]

    # 1. GENERATE SAMPLES (TMVN)
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    z_samples = tmvn_sampling(locs, scales, n_samples, lbound)

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
    
    # 3. WEIGHTING (Likelihood)
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

    # 4. PLOTTING
    quantile_threshold = 0.95
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_filtered /= w_filtered # Uniform for visualization of density peaks

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
                ax.fill_between(x_grid, y_grid, color='gray', alpha=0.4)
                ax.plot(x_grid, y_grid, color='black', lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0); ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_filtered)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='Greys')
                ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white')
                ax.set_xlim(lbound, 1.0); ax.set_ylim(lbound, 1.0)
            else:
                ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(folder_path, "TMVN_Posteriors"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "TMVN_Posteriors", f'Sample_{pos}_TMVN.png'), dpi=150)
    plt.close()

def plot_physical_pdf_profile_tmvn(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45):
    """
    Journal-style Profile plot for TMVN Baseline.
    """
    plt.rcParams.update({"font.family": "serif", "font.size": 28})
    fig, (ax_pdf, ax_beam) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)
    width_scale = 0.4 

    for i in range(n_elements):
        kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
        pdf_visual = (kde(z_grid) / np.max(kde(z_grid))) * width_scale
        x_center = i + 1
        ax_pdf.fill_betweenx(z_grid, x_center - pdf_visual, x_center + pdf_visual, color='gray', alpha=0.3)
        ax_pdf.plot(x_center - pdf_visual, z_grid, color='black', lw=1.5, alpha=0.5)
        ax_pdf.plot(x_center + pdf_visual, z_grid, color='black', lw=1.5, alpha=0.5)
        ax_pdf.plot(x_center, z_true[i], marker='*', color='red', markersize=18, markeredgecolor='white')

    ax_pdf.set_ylabel('Stiffness Factor (z)')
    ax_pdf.set_ylim([lbound - 0.05, 1.05])
    
    beam_viz = z_true.reshape(1, -1)
    ax_beam.imshow(beam_viz, cmap='YlOrRd_r', extent=[0.5, n_elements + 0.5, 0, 1])
    ax_beam.set_yticks([])
    ax_beam.set_xticks(elements)
    
    plt.tight_layout()
    save_dir = os.path.join(folder_path, "TMVN_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_TMVN_Profile.png'), dpi=300)
    plt.close()