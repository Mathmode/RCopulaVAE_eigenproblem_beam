# -*- coding: utf-8 -*-
"""
Functions for Results Analysis - GMM VAE Posterior
Adapts Bayesian likelihood weighting, physics engine evaluations, and plotting
to the Full Covariance Gaussian Mixture Model latent space.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import seaborn as sns
import matplotlib.lines as mlines
import tensorflow.keras as K

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices

K.backend.set_floatx('float32') 

def plot_configuration():
    plt.rc('font', size=28)
    plt.rc('axes', titlesize=28, labelsize=24)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.rc('legend', fontsize=28)
    plt.rc('figure', titlesize=28)

sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style="dark")
plot_configuration()

def build_cholesky_matrices_numpy(offdiag, diag, n_dims, num_components):
    """Reconstructs the Lower Triangular covariance matrices in NumPy for post-processing."""
    batch_size = offdiag.shape[0]
    L = np.zeros((batch_size, num_components, n_dims, n_dims), dtype=np.float32)
    tril_indices = np.tril_indices(n_dims, k=-1)
    
    for i in range(n_dims):
        L[:, :, i, i] = diag[:, :, i]
        
    for idx, (row, col) in enumerate(zip(tril_indices[0], tril_indices[1])):
        L[:, :, row, col] = offdiag[:, :, idx]
        
    return L

def generate_gmm_physical_samples(logits, locs, offdiag, diag, n_samples, lbound=0.45):
    """
    Generates samples in the physical domain [lbound, 1.0] from the GMM parameters.
    """
    n_dims = locs.shape[-1]
    num_components = locs.shape[-2]
    
    # Needs a batch dimension if missing for a single sample
    if len(logits.shape) == 1:
        logits = np.expand_dims(logits, axis=0)
        locs = np.expand_dims(locs, axis=0)
        offdiag = np.expand_dims(offdiag, axis=0)
        diag = np.expand_dims(diag, axis=0)

    L = build_cholesky_matrices_numpy(offdiag, diag, n_dims, num_components)
    
    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=tfd.MultivariateNormalTriL(loc=locs, scale_tril=L)
    )
    
    z_raw = mixture_dist.sample(n_samples)
    # Output Shape is usually [n_samples, batch, dims] from TFP
    z_phys = lbound + (1.0 - lbound) * tf.math.sigmoid(z_raw)
    return z_phys.numpy()

def calculate_gmm_metrics(predicted_stats, test_datasets, lbound=0.45, n_mc_samples=5000):
    logits = predicted_stats['test_logits'] 
    locs = predicted_stats['test_locs']     
    offdiag = predicted_stats['test_offdiag'] 
    diag = predicted_stats['test_diag']
    z_true = test_datasets['alpha_factors_true_test'] 
    
    n_dims = locs.shape[-1]
    num_components = locs.shape[-2]
    N = logits.shape[0]
    batch_size = 200 # Chunk size to prevent GPU Out-of-Memory (OOM)
    
    all_log_probs, all_emp_log_probs = [], []
    all_mse, all_mae = [], []
    all_coverage, all_sharpness = [], []
    
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        
        # 1. Extract Batch
        b_logits = logits[i:end_idx]
        b_locs = locs[i:end_idx]
        b_offdiag = offdiag[i:end_idx]
        b_diag = diag[i:end_idx]
        b_z_true = z_true[i:end_idx]
        
        b_L = build_cholesky_matrices_numpy(b_offdiag, b_diag, n_dims, num_components)
        
        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=b_logits),
            components_distribution=tfd.MultivariateNormalTriL(loc=b_locs, scale_tril=b_L)
        )
        
        # 2. Analytical Log-Likelihood (Clipped for Boundary Stability)
        b_z_true_unit = (b_z_true - lbound) / (1.0 - lbound)
        b_z_true_unit_safe = np.clip(b_z_true_unit, 0.018, 0.982)
        b_z_true_raw = np.log(b_z_true_unit_safe / (1.0 - b_z_true_unit_safe))
        
        log_prob_raw = mixture_dist.log_prob(b_z_true_raw).numpy()
        
        log_dz_phys = np.log(1.0 - lbound) + np.log(b_z_true_unit_safe) + np.log(1.0 - b_z_true_unit_safe)
        log_det_jacobian = np.sum(log_dz_phys, axis=-1)
        
        b_log_probs = log_prob_raw - log_det_jacobian
        all_log_probs.extend(b_log_probs)
        
        # 3. MC Sampling for Point Estimates & Empirical Metrics
        z_raw_samples = mixture_dist.sample(n_mc_samples)
        z_phys_samples = lbound + (1.0 - lbound) * tf.math.sigmoid(z_raw_samples)
        z_phys_samples = z_phys_samples.numpy()
        
        b_pred_means = np.mean(z_phys_samples, axis=0) 
        b_mse = np.mean(np.square(b_pred_means - b_z_true), axis=-1)
        b_mae = np.mean(np.abs(b_pred_means - b_z_true), axis=-1)
        all_mse.extend(b_mse)
        all_mae.extend(b_mae)
        
        b_pred_vars = np.var(z_phys_samples, axis=0) + 1e-6
        b_emp_ll = np.sum(-0.5 * np.log(2 * np.pi * b_pred_vars) - 0.5 * np.square(b_z_true - b_pred_means) / b_pred_vars, axis=-1)
        all_emp_log_probs.extend(b_emp_ll)
        
        z_lower = np.percentile(z_phys_samples, 2.5, axis=0) 
        z_upper = np.percentile(z_phys_samples, 97.5, axis=0) 
        
        b_covered = np.mean((b_z_true >= z_lower) & (b_z_true <= z_upper), axis=-1)
        b_sharpness = np.mean(z_upper - z_lower, axis=-1)
        
        all_coverage.extend(b_covered)
        all_sharpness.extend(b_sharpness)
    
    return {
        'MSE': float(np.mean(all_mse)),
        'MAE': float(np.mean(all_mae)),
        '95% Coverage': float(np.mean(all_coverage)),
        'Mean Interval Width': float(np.mean(all_sharpness)),
        'Avg Log-Likelihood (Analytical)': float(np.mean(all_log_probs)),
        'Avg Log-Likelihood (Empirical)': float(np.mean(all_emp_log_probs))
    }

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

def calculate_posterior_PDF_info(fixed_dofs_indices, n_modes, gamma, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    logits = predicted_stats['test_logits'][pos]      
    locs = predicted_stats['test_locs'][pos]      
    offdiag = predicted_stats['test_offdiag'][pos]
    diag = predicted_stats['test_diag'][pos]
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    z_true = test_datasets['alpha_factors_true_test'][pos,:]
    
    n_elements = z_true.shape[0]
    
    # 1. Sample directly from the FullCov GMM
    z_samples_batch = generate_gmm_physical_samples(logits, locs, offdiag, diag, n_samples, lbound)
    # TFP usually returns [Samples, Batch, Dims]. Since batch=1 here, grab the first batch index.
    z_samples_np = z_samples_batch[:, 0, :] 
    z_samples = tf.cast(z_samples_np, dtype=tf.float32)
    
    # 2. Reconstruct stiffness and pass through physics engine
    if len(Ke_matrices.shape) == 2:
        Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    else:
        Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
        
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)
     
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
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
    
    # 3. Compute Bayesian Likelihoods
    obs_f_scaled = Freqs_true[pos]
    pred_logscaled = (np.log(f_pred) - mean_freq) / std_freq
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

    obs_r_batch = np.repeat(Rotmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    obs_v_batch = np.repeat(Vertmodes_true[pos][np.newaxis, :, :], n_samples, axis=0)
    
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)
    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)
    
    inv_gamma_val = 1.0 / (gamma**2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood) 
    
    return z_true, z_samples.numpy(), posterior_weights

def plot_results_PDF_uncertainty(fixed_dofs_indices, n_modes, gamma, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    """
    Generates joint posterior plots using GMM-VAE distribution + Bayesian re-weighting.
    """
    z_true, z_samples_np, posterior_weights = calculate_posterior_PDF_info(
        fixed_dofs_indices, n_modes, gamma, n_samples, pos, n_dofs, free_dofs, test_datasets,
        predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path
    )
    n_elements = z_true.shape[0]

    # 4. FILTERING & PLOTTING based on Bayesian weights
    quantile_threshold = 0.95 
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples_np[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_norm = w_filtered / np.sum(w_filtered)

    fig_size = max(14, 2 * n_elements)
    fig, axes = plt.subplots(n_elements, n_elements, figsize=(fig_size, fig_size), facecolor='white')
    if n_elements == 1: axes = np.array([[axes]])
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    cf = None

    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_norm)
                x_grid = np.linspace(lbound-0.05, 1.05, 200)
                y_grid = kde1d(x_grid)
                ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0); ax.set_yticks([])
            elif r > c:
                try:
                    xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                    kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_norm)
                    zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                    cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                except np.linalg.LinAlgError:
                    ax.scatter(x_filtered[:, c], x_filtered[:, r], alpha=0.1, s=2, color='steelblue')
                ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', label='Truth')
                ax.set_xlim(lbound, 1.0); ax.set_ylim(lbound, 1.0)
            else:
                ax.axis('off')

            if r >= c:
                ax.tick_params(labelsize=18)
                if c == 0 and r != 0: ax.set_ylabel(labels[r], fontsize=24)
                else: ax.tick_params(labelleft=False)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=24)
                else: ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (GMM-VAE + Physics)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    
    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors_GMM"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors_GMM", f'Sample_{pos}_Matched.png'), dpi=150)
    plt.close()

def plot_physical_pdf_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45):
    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 24,
        "axes.labelsize": 24, "xtick.labelsize": 22, "ytick.labelsize": 22
    })
    
    fig, (ax_pdf, ax_beam) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True, facecolor='white')
    
    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)
    pdf_color = 'steelblue'
    truth_color = '#D62728'
    
    for i in range(n_elements):
        try:
            kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
            pdf_values = kde(z_grid)
            pdf_visual = (pdf_values / np.max(pdf_values)) * 0.4
            x_center = i + 1
            
            ax_pdf.fill_betweenx(z_grid, x_center - pdf_visual, x_center + pdf_visual, color=pdf_color, alpha=0.4)
            ax_pdf.plot(x_center - pdf_visual, z_grid, color=pdf_color, lw=1.5)
            ax_pdf.plot(x_center + pdf_visual, z_grid, color=pdf_color, lw=1.5)
        except np.linalg.LinAlgError:
            pass # Skip flat dimensions
            
        if z_true is not None:
            ax_pdf.plot(i + 1, z_true[i], marker='*', color=truth_color, markersize=18, markeredgecolor='white')

    ax_pdf.set_ylabel(r'Stiffness factor ($z$)')
    ax_pdf.set_ylim([lbound - 0.05, 1.05])
    ax_pdf.set_xticks(elements)
    ax_pdf.set_xticklabels([f'$e_{{{k}}}$' for k in elements])
    
    if z_true is not None:
        beam_viz = z_true.reshape(1, -1)
        ax_beam.imshow(beam_viz, cmap='YlOrRd_r', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1])
    ax_beam.set_yticks([])
    
    plt.tight_layout()
    save_dir = os.path.join(folder_path, "GMM_Physical_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_GMM_UQ.png'), dpi=300)
    plt.close()