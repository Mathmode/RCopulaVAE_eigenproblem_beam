# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:53:56 2026

@author: anafd
"""

import os
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
from scipy.special import beta as beta_func

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

def gaussian_copula(LT_matrix, n_dims, n_samples):
    """
    Generates uniform [0,1] samples using the Gaussian Copula.
    Derives dimensions directly from LT_matrix to prevent broadcasting errors.
    """
    LT_matrix = tf.cast(LT_matrix, tf.float32)
    dim = tf.shape(LT_matrix)[-1]
    
    mvn_model = tfd.MultivariateNormalTriL(
        loc=tf.zeros(dim, dtype=tf.float32), 
        scale_tril=LT_matrix
    )
    mvn_samples = mvn_model.sample(n_samples)
    if len(mvn_samples.shape) == 3:
        mvn_samples = tf.transpose(mvn_samples, [1, 0, 2]) # Convert to [Batch, Samples, Dims]
        
    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return copula_samples

def kumaraswamy_mixture_quantile(u, a, b, weights, lbound=0.45):
    """
    Inverse CDF (Quantile function) for a mixture of Kumaraswamy distributions.
    Dynamically handles Batched/Unbatched parameters and varying MC sample dimensions.
    """
    u = tf.cast(u, tf.float32)
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    w = tf.cast(weights, tf.float32)
    
    original_u_shape = tf.shape(u)
    
    if len(a.shape) == 2: 
        a = tf.expand_dims(a, axis=0)
        b = tf.expand_dims(b, axis=0)
        w = tf.expand_dims(w, axis=0)
        if len(u.shape) == 2:
            u = tf.expand_dims(u, axis=0)
    else: 
        if len(u.shape) == 2:
            u = tf.expand_dims(u, axis=1)
            
    u = tf.clip_by_value(u, 1e-7, 1.0 - 1e-7)
    
    a_t = tf.transpose(a, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    w_t = tf.transpose(w, perm=[0, 2, 1])[:, tf.newaxis, :, :]

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(35):
        mid = (low + high) / 2.0
        mid_expand = tf.expand_dims(mid, axis=-1)
        
        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0) 
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)
        
        cdf_mid = tf.reduce_sum(w_t * cdf_components, axis=-1)
        is_less = cdf_mid < u
        
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    res = lbound + (1.0 - lbound) * ((low + high) / 2.0)
    res = tf.reshape(res, original_u_shape)
    return res

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

def calculate_ks_mixture_metrics(predicted_stats, test_datasets, lbound=0.45, physics_kwargs=None):
    """
    Calculates numerical metrics to evaluate the GC-VAE (Kumaraswamy Mixture) inference.
    Correctly aligns the bounded prior log-density contribution without redundant cancellations.
    """
    test_a = predicted_stats['test_a']           
    test_b = predicted_stats['test_b']           
    test_weights = predicted_stats['test_weights'] 
    LT_matrix = predicted_stats['test_L_matrices'] 
    z_true = test_datasets['alpha_factors_true_test'] 
    
    N = test_a.shape[0]
    num_KS = test_a.shape[1]
    n_dims = test_a.shape[2]
    
    batch_size = 200
    n_mc_samples = 5000
    
    all_true_log_probs = []
    all_samp_log_probs = []
    all_mse, all_mae = [], []
    all_coverage, all_sharpness = [], []
    all_z_phys_samp = []
    all_covered = [] 
    
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        b_a = test_a[i:end_idx]
        b_b = test_b[i:end_idx]
        b_w = test_weights[i:end_idx]
        b_L = LT_matrix[i:end_idx]
        b_z_true = z_true[i:end_idx]
        
        # --- 1. Point Estimate (Analytical Mean of Mixture) ---
        b_comp_means = b_b * beta_func(1 + 1/b_a, b_b) 
        b_pred_means_unit = np.sum(b_w * b_comp_means, axis=1)
        b_pred_means = lbound + (1.0 - lbound) * b_pred_means_unit
        
        all_mse.extend(np.mean(np.square(b_pred_means - b_z_true), axis=-1))
        all_mae.extend(np.mean(np.abs(b_pred_means - b_z_true), axis=-1))

        # --- 2. Probabilistic Log-Likelihood & True z evaluation ---
        b_z_true_unit = (b_z_true - lbound) / (1.0 - lbound)
        b_z_true_unit = np.clip(b_z_true_unit, 1e-6, 1.0 - 1e-6)
        
        z_exp = np.expand_dims(b_z_true_unit, axis=1) 
        comp_cdfs = 1.0 - np.power(1.0 - np.power(z_exp, b_a), b_b)
        u_d_true = np.sum(b_w * comp_cdfs, axis=1) 
        
        comp_pdfs = b_w * b_a * b_b * np.power(z_exp, b_a - 1) * np.power(1.0 - np.power(z_exp, b_a), b_b - 1)
        log_q_d_true = np.log(np.sum(comp_pdfs, axis=1) + 1e-12)
        
        u_d_true = np.clip(u_d_true, 1e-6, 1.0 - 1e-6)
        w_true = stats.norm.ppf(u_d_true)
        
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims, dtype=tf.float32), scale_tril=tf.cast(b_L, tf.float32))
        log_c_true = mvn.log_prob(tf.cast(w_true, tf.float32)).numpy() - np.sum(stats.norm.logpdf(w_true), axis=1)
        
        # Explicit marginal volume density normalization without redundant inverse offsetting
        joint_log_prob_true = log_c_true + np.sum(log_q_d_true, axis=1) - n_dims * np.log(1.0 - lbound)
        all_true_log_probs.extend(joint_log_prob_true)

        # --- 3. MC Sampling for Calibration & KL ---
        w_samps = mvn.sample(n_mc_samples)
        w_samps = tf.transpose(w_samps, [1, 0, 2]) 
        u_samps = tfd.Normal(loc=0.0, scale=1.0).cdf(w_samps)
        
        z_samps_phys = kumaraswamy_mixture_quantile(u_samps, b_a, b_b, b_w, lbound).numpy()
        z_samps_unit = (z_samps_phys - lbound) / (1.0 - lbound)
        
        z_lower = np.percentile(z_samps_phys, 2.5, axis=1)
        z_upper = np.percentile(z_samps_phys, 97.5, axis=1)
        
        b_covered = (b_z_true >= z_lower) & (b_z_true <= z_upper)
        all_covered.append(b_covered)
        all_coverage.extend(np.mean(b_covered, axis=-1))
        all_sharpness.extend(np.mean(z_upper - z_lower, axis=-1))
        
        z_unit_samp1 = z_samps_unit[:, 0, :] 
        all_z_phys_samp.append(z_samps_phys[:, 0, :])
        
        z_exp_samp = np.expand_dims(z_unit_samp1, axis=1)
        comp_cdfs_s = 1.0 - np.power(1.0 - np.power(z_exp_samp, b_a), b_b)
        u_d_samp = np.sum(b_w * comp_cdfs_s, axis=1)
        
        comp_pdfs_s = b_w * b_a * b_b * np.power(z_exp_samp, b_a - 1) * np.power(1.0 - np.power(z_exp_samp, b_a), b_b - 1)
        log_q_d_samp = np.log(np.sum(comp_pdfs_s, axis=1) + 1e-12)
        
        u_d_samp = np.clip(u_d_samp, 1e-6, 1.0 - 1e-6)
        w_samp = stats.norm.ppf(u_d_samp)
        
        log_c_samp = mvn.log_prob(tf.cast(w_samp, tf.float32)).numpy() - np.sum(stats.norm.logpdf(w_samp), axis=1)
        joint_log_prob_samp = log_c_samp + np.sum(log_q_d_samp, axis=1) - n_dims * np.log(1.0 - lbound)
        all_samp_log_probs.extend(joint_log_prob_samp)

    covered = np.concatenate(all_covered, axis=0)
    avg_log_prob = float(np.mean(all_true_log_probs))
    
    p_params = (3 * num_KS - 1) * n_dims + (n_dims * (n_dims - 1)) / 2
    BIC = -2 * avg_log_prob + (p_params * np.log(N)) / N

    # Correct analytical prior log density matched directly to domain volume bounds
    log_prior = -n_dims * np.log(1.0 - lbound)
    KLtest = float(np.mean(np.array(all_samp_log_probs) - log_prior))
    
    metrics = {
        'MSE': float(np.mean(all_mse)),
        'MAE': float(np.mean(all_mae)),
        '95% Coverage': float(np.mean(all_coverage)),
        'Mean Interval Width': float(np.mean(all_sharpness)),
        'Avg Log-Likelihood (True Z)': avg_log_prob,
        'KLtest': KLtest,
        'BICtest': float(BIC)
    }
    
    if physics_kwargs is not None:
        Ke_matrices = physics_kwargs['Ke_matrices']
        L_inv = tf.cast(physics_kwargs['L_inv'], tf.float32)
        n_modes = physics_kwargs['n_modes']
        free_dofs = physics_kwargs['free_dofs']
        fixed_dofs = physics_kwargs['fixed_dofs']
        n_dofs = physics_kwargs['n_dofs']
        mean_freq = physics_kwargs['mean_freq']
        std_freq = physics_kwargs['std_freq']
        gamma = physics_kwargs['gamma']
        
        z_all_samp_phys = np.concatenate(all_z_phys_samp, axis=0)
        z_tensor = tf.cast(z_all_samp_phys, dtype=tf.float32)
        if len(Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', z_tensor, tf.cast(Ke_matrices, dtype=tf.float32))
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_tensor, tf.cast(Ke_matrices, dtype=tf.float32))
            
        Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_dims, N, fixed_dofs)
        
        batch_size_pe = 256
        all_f, all_rot, all_vert = [], [], []
        for i in range(0, N, batch_size_pe):
            f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size_pe], L_inv, n_modes, free_dofs, n_dofs)
            all_f.append(f)
            all_rot.append(r)
            all_vert.append(v)
            
        f_pred = np.concatenate(all_f, axis=0)
        rot_pred = np.concatenate(all_rot, axis=0)
        vert_pred = np.concatenate(all_vert, axis=0)
        
        obs_f_scaled = test_datasets['Freqs_true_test']
        pred_logscaled = (np.log(f_pred) - mean_freq) / std_freq
        loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)

        obs_r_batch = test_datasets['Rotmodes_true_test']
        obs_v_batch = test_datasets['Vertmodes_true_test']
        
        rot_mac = calculate_MAC(obs_r_batch, rot_pred)
        vert_mac = calculate_MAC(obs_v_batch, vert_pred)
        loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)
        
        inv_gamma_val = 1.0 / (gamma**2)
        data_log_lik = -(loss_f + loss_mac) * inv_gamma_val
        KL_divergences = np.array(all_samp_log_probs) - log_prior
        
        Avg_ELBOtest = float(np.mean(data_log_lik - KL_divergences))
        metrics['Avg. ELBOtest'] = Avg_ELBOtest

    return metrics, covered

def plot_results_PDF_uncertainty(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    a_vals = predicted_stats['test_a'][pos]      
    b_vals = predicted_stats['test_b'][pos]      
    w_vals = predicted_stats['test_weights'][pos] 
    LT_matrix = predicted_stats['test_L_matrices'][pos] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    z_true = Alphas_true[pos,:]
    
    n_elements = LT_matrix.shape[-1]

    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples = gaussian_copula(LT_matrix, n_elements, n_samples) 
    z_samples = kumaraswamy_mixture_quantile(copula_samples, a_vals, b_vals, w_vals, lbound=lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)

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
    
    quantile_threshold = 0.95 
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_norm = w_filtered / np.sum(w_filtered)

    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    cf = None

    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            if r == c:
                vals = x_filtered[:, r]
                kde1d = gaussian_kde(vals, weights=w_norm)
                x_grid = np.linspace(lbound-0.05, 1.0, 200)
                y_grid = kde1d(x_grid)
                ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0); ax.set_yticks([])
            elif r > c:
                xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
                kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_norm)
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', label='Truth')
                ax.set_xlim(lbound, 1.0); ax.set_ylim(lbound, 1.0)
            else:
                ax.axis('off')

            if r >= c:
                ax.tick_params(labelsize=22)
                if c == 0 and r != 0: ax.set_ylabel(labels[r], fontsize=30)
                else: ax.tick_params(labelleft=False)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=30)
                else: ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (KS-VAE)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    
    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors_KS"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors_KS", f'Sample_{pos}_Matched.png'), dpi=150)
    plt.show()
    plt.close()
    


def calculate_posterior_PDF_info(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq,  lbound, folder_path):
    a_vals = predicted_stats['test_a'][pos]      
    b_vals = predicted_stats['test_b'][pos]      
    w_vals = predicted_stats['test_weights'][pos] 
    LT_matrix = predicted_stats['test_L_matrices'][pos] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    z_true = test_datasets['alpha_factors_true_test'][pos,:]
    
    n_elements = LT_matrix.shape[-1]
    
    copula_samples = gaussian_copula(LT_matrix, n_elements, n_samples) 
    z_samples = kumaraswamy_mixture_quantile(copula_samples, a_vals, b_vals, w_vals, lbound=lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)
    
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)
     
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    batch_size = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size):
        f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs)
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
    
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)
    vert_mac = calculate_MAC(obs_v_batch, vert_pred)
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)
    
    inv_gamma_val = 1.0 / (beta**2)
    exponent = -(loss_f + loss_mac) * inv_gamma_val
    likelihood = np.exp(exponent - np.max(exponent))
    posterior_weights = likelihood / np.sum(likelihood) 
    
    return z_true, z_samples.numpy(), posterior_weights

def plot_physical_pdf_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45):
    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 28,
        "axes.labelsize": 28, "xtick.labelsize": 26, "ytick.labelsize": 26
    })
    
    fig, (ax_pdf, ax_beam) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1]}, sharex=True, facecolor='white')
    
    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)
    pdf_color = 'steelblue'
    truth_color = '#D62728'
    
    for i in range(n_elements):
        kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
        pdf_values = kde(z_grid)
        pdf_visual = (pdf_values / np.max(pdf_values)) * 0.4
        x_center = i + 1
        
        ax_pdf.fill_betweenx(z_grid, x_center - pdf_visual, x_center + pdf_visual, color=pdf_color, alpha=0.4)
        ax_pdf.plot(x_center - pdf_visual, z_grid, color=pdf_color, lw=1.5)
        ax_pdf.plot(x_center + pdf_visual, z_grid, color=pdf_color, lw=1.5)
        
        if z_true is not None:
            ax_pdf.plot(x_center, z_true[i], marker='*', color=truth_color, markersize=18, markeredgecolor='white')

    ax_pdf.set_ylabel(r'Stiffness factor ($z$)')
    ax_pdf.set_ylim([lbound - 0.05, 1.05])
    ax_pdf.set_xticks(elements)
    ax_pdf.set_xticklabels([f'$e_{{{k}}}$' for k in elements])
    
    beam_viz = z_true.reshape(1, -1)
    ax_beam.imshow(beam_viz, cmap='YlOrRd_r', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1])
    ax_beam.set_yticks([])
    
    plt.tight_layout()
    save_dir = os.path.join(folder_path, "KS_Physical_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_KS_UQ.png'), dpi=300)
    plt.show()
    plt.close()

def plot_random_2d_posteriors(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45, J=10):
    """
    Plots J randomly chosen 2D slices of the posterior PDF, effectively displaying
    a manageable subset of the corner plot to avoid unreadable high-dimensional visualizations.
    """
    # 1. Filtering & Weight Normalization
    quantile_threshold = 0.95 
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_norm = w_filtered / np.sum(w_filtered)
    
    # 2. Randomly select J pairs of unique dimensions
    max_possible_pairs = (n_elements * (n_elements - 1)) // 2
    J_actual = min(J, max_possible_pairs)
    
    # Collect all possible lower-triangle pairs (r > c)
    all_pairs = [(r, c) for r in range(n_elements) for c in range(r)]
    chosen_pairs = random.sample(all_pairs, J_actual)
    
    # 3. Dynamic layout configuration
    cols = min(J_actual, 5) # Max 5 columns per row for readability
    rows = int(np.ceil(J_actual / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), facecolor='white')
    
    if J_actual == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 4. Iterating and plotting KDE contours
    for idx, (r, c) in enumerate(chosen_pairs):
        ax = axes[idx]
        xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
        
        kernel = gaussian_kde(np.vstack([x_filtered[:, c], x_filtered[:, r]]), weights=w_norm)
        zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        
        cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
        ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', label='Truth' if idx == 0 else "")
        
        ax.set_xlim(lbound, 1.0)
        ax.set_ylim(lbound, 1.0)
        ax.set_xlabel(f'$z_{{{c+1}}}$', fontsize=22)
        ax.set_ylabel(f'$z_{{{r+1}}}$', fontsize=22)
        ax.tick_params(labelsize=18)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=16)

    # 5. Hide any unused grid axes
    for idx in range(J_actual, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    save_dir = os.path.join(folder_path, "Random_2D_Posteriors")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_Random_{J_actual}_2D_Posteriors.png'), dpi=150)
    plt.show()
    plt.close()