# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:03:56 2026

@author: anafd
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf

# Disable JIT globally via the TensorFlow API as a secondary safety measure
tf.config.optimizer.set_jit(False)
from tensorflow_probability import distributions as tfd
import seaborn as sns
import tensorflow.keras as K
from scipy.stats import norm
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# def main():
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
    mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
    mvn_samples = mvn_model.sample(n_samples)
    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return copula_samples, mvn_samples, mvn_model

def gaussian_marginal_samples(locs, scales, copula_samples, lbound):
    # Match the TruncatedNormal bounds used in training
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low=lbound + 0.0001, high=0.9999).quantile(copula_samples)
    return marginal_samples

def calculate_MAC(modes_true, modes_pred):
    """
    Aligned with GT script calculate_MAC.
    Expects (Batch, Modes, Nodes)
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
    Calculates numerical metrics to evaluate the GC-VAE inference performance.
    
    Metrics included:
    - MSE/MAE: Point estimate accuracy of the mean.
    - Coverage: % of true values within the predicted 95% credible interval.
    - Sharpness: Average width of the uncertainty bands.
    - Log-Prob: Log-likelihood of the true parameters under the predicted marginals.
    """
    means = predicted_stats['test_means']
    scales = predicted_stats['test_scales']
    z_true = test_datasets['alpha_factors_true_test']
    
    n_samples, n_elements = z_true.shape
    
    # 1. Point Estimate Metrics
    mse = np.mean(np.square(means - z_true))
    mae = np.mean(np.abs(means - z_true))
    
    # 2. Uncertainty Calibration (95% Credible Intervals)
    # Using the marginal Gaussian properties
    z_lower = means - 1.96 * scales
    z_upper = means + 1.96 * scales
    
    # Clip to physical bounds for realistic interval assessment
    z_lower = np.maximum(z_lower, lbound)
    z_upper = np.minimum(z_upper, 1.0)
    
    covered = (z_true >= z_lower) & (z_true <= z_upper)
    coverage_score = np.mean(covered) # Ideal: 0.95
    sharpness = np.mean(z_upper - z_lower) # Lower is better (more confident)
    
    # 3. Probabilistic Log-Likelihood (Marginal)
    log_probs = []
    for i in range(n_samples):
        # Calculate log-pdf of true Z given predicted mu and sigma
        p_z = norm.logpdf(z_true[i], loc=means[i], scale=scales[i])
        log_probs.append(np.mean(p_z))
    
    avg_log_prob = np.mean(log_probs)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        '95% Coverage': coverage_score,
        'Mean Interval Width': sharpness,
        'Avg Log-Likelihood': avg_log_prob
    }
    
    return metrics, covered

    
def plot_results_PDF_uncertainty(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    """
    Generates joint posterior plots using VAE-predicted distribution + Bayesian re-weighting.
    """
    locs = predicted_stats['test_means'][pos,:]
    scales = predicted_stats['test_scales'][pos,:] 
    LT_matrix = predicted_stats['test_L_matrices'][pos,:] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    z_true = Alphas_true[pos,:]
    
    n_dims = locs.shape[1]
    n_elements = n_dims

    # 1. GENERATE SAMPLES FROM VAE COPULA
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, _, _ = gaussian_copula(LT_matrix, n_dims, n_samples) 
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
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
    
    # 4. FILTERING & PLOTTING (Aligned with GT script logic)
    quantile_threshold = 0.95 # Slightly wider for VAE samples
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    # w_filtered /= np.sum(w_filtered)
    w_filtered /= w_filtered
    

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
                ax.tick_params(labelsize=22)
                if c == 0 and r != 0: ax.set_ylabel(labels[r], fontsize=30)
                else: ax.tick_params(labelleft=False)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=30)
                else: ax.tick_params(labelbottom=False)

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (VAE)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    import matplotlib.lines as mlines
    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
    os.makedirs(os.path.join(folder_path, "Matched_Posteriors"), exist_ok=True)
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors", f'Sample_{pos}_Matched.png'), dpi=150)
    plt.show()
    plt.close()
    

def calculate_posterior_PDF_info(fixed_dofs_indices, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq,  lbound, folder_path):
    
    locs = predicted_stats['test_means'][pos,:]
    scales = predicted_stats['test_scales'][pos,:] 
    LT_matrix = predicted_stats['test_L_matrices'][pos,:] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    z_true = Alphas_true[pos,:]
    
    n_dims = locs.shape[1]
    n_elements = n_dims
    
    # 1. GENERATE SAMPLES FROM VAE COPULA
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, _, _ = gaussian_copula(LT_matrix, n_dims, n_samples) 
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)
    
    # 2. PHYSICS PROPAGATION
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, fixed_dofs_indices)
     
    # C. Solve Eigenproblem (Using the Model's Layer for consistency)
    batch_size = 256
    all_f, all_rot, all_vert = [], [], []
    for i in range(0, n_samples, batch_size):
        f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes, free_dofs, n_dofs)
        all_f.append(f)
        all_rot.append(r)
        all_vert.append(v)
        
    f_pred = tf.concat(all_f, axis=0).numpy()      # (n_samples, n_modes)
    rot_pred = tf.concat(all_rot, axis=0).numpy()  # (n_samples, n_modes, n_nodes)
    vert_pred = tf.concat(all_vert, axis=0).numpy() # (n_samples, n_modes, n_nodes)
    
    # 3. CALCULATE POSTERIOR (DATA MISFIT)
    log_obs_f = Freqs_true[pos]     # (n_modes,)
    obs_r = Rotmodes_true[pos]  # (n_modes, n_nodes)
    obs_v = Vertmodes_true[pos] # (n_modes, n_nodes)
    
    # A. Frequency Loss
    pred_log = np.log(f_pred)
    pred_logscaled = (pred_log - mean_freq) / std_freq
    f_err = np.square(log_obs_f -pred_logscaled)
    loss_f = np.mean(f_err, axis=1) # (n_samples,)

    
    # B. MAC Loss
    obs_r_batch = tf.convert_to_tensor(np.repeat(obs_r[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)
    obs_v_batch = tf.convert_to_tensor(np.repeat(obs_v[np.newaxis, :, :], n_samples, axis=0), dtype=tf.float32)
    
    # Note: calculate_MAC is assumed to return MAC per sample per mode
    mac_mat_r = calculate_MAC(obs_r_batch, tf.convert_to_tensor(rot_pred)) # (n_samples, n_modes)
    mac_mat_v = calculate_MAC(obs_v_batch, tf.convert_to_tensor(vert_pred)) # (n_samples, n_modes)
    
    # FIXED: mac_mat is already (n_samples, n_modes). We take the mean across modes (axis=1)
    # If calculate_MAC returned (samples, modes_true, modes_pred), you'd use axis=2, but your shapes show (1000, 6).
    best_r = mac_mat_r if isinstance(mac_mat_r, np.ndarray) else mac_mat_r.numpy()
    best_v = mac_mat_v if isinstance(mac_mat_v, np.ndarray) else mac_mat_v.numpy()
    
    loss_mac = np.mean((1.0 - best_r) + (1.0 - best_v), axis=1) 
    
    # C. Total Loss & Likelihood
    inv_gamma_val = 1.0 / (beta**2)
    total_loss = loss_f + loss_mac 
    
    exponent = -total_loss * inv_gamma_val
    max_exp = np.max(exponent)
    likelihood = np.exp(exponent - max_exp)
    posterior_weights = likelihood / np.sum(likelihood) 
    
    return z_true, z_samples, posterior_weights
    
def plot_physical_pdf_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path, lbound=0.45):
    """
    Visualizes the physical uncertainty by plotting the marginal PDF for each element.
    Optimized for Journal-level quality: focus on Ground Truth reference, removing prediction means.
    """
    # 1. Setup Journal Style via rcParams
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
        "legend.fontsize": 24
    })
    
    # 2. Setup Figure
    fig, (ax_pdf, ax_beam) = plt.subplots(
        2, 1, 
        figsize=(14, 10), 
        gridspec_kw={'height_ratios': [4, 1]}, 
        sharex=True,
        facecolor='white'
    )
    
    elements = np.arange(1, n_elements + 1)
    z_grid = np.linspace(lbound, 1.0, 500)
    
    # Color palettes
    pdf_color = 'steelblue' # SteelBlue for PDF
    truth_color = '#D62728' # Professional Crimson for Truth
    cmap = cm.get_cmap('YlOrRd_r') 
    norm_color = mcolors.Normalize(vmin=lbound, vmax=1.0)
    
    width_scale = 0.4 
    
    # ==========================================
    # TOP PLOT: Marginal PDFs per Element
    # ==========================================
    for i in range(n_elements):
        # KDE Calculation using posterior weights
        kde = gaussian_kde(z_samples[:, i], weights=posterior_weights)
        pdf_values = kde(z_grid)
        
        # Scale PDF to fit the element "slot"
        pdf_visual = (pdf_values / np.max(pdf_values)) * width_scale
        x_center = i + 1
        
        # Plot PDF Area
        ax_pdf.fill_betweenx(z_grid, x_center - pdf_visual, x_center + pdf_visual, 
                             color=pdf_color, alpha=0.4, label='Marginal PDF' if i==0 else "")
        ax_pdf.plot(x_center - pdf_visual, z_grid, color=pdf_color, lw=2.0, alpha=0.6)
        ax_pdf.plot(x_center + pdf_visual, z_grid, color=pdf_color, lw=2.0, alpha=0.6)
        
        # Plot Ground Truth Marker
        if z_true is not None:
            ax_pdf.plot(x_center, z_true[i], marker='*', color=truth_color, markersize=18, 
                        markeredgecolor='white', markeredgewidth=1.5, zorder=10, 
                        label='Ground Truth' if i==0 else "")
            
    # Professional Step-line for truth profile alignment
    if z_true is not None:
        x_step = np.arange(0.5, n_elements + 1.5, 1)
        y_step = np.concatenate([z_true, [z_true[-1]]])
        ax_pdf.step(x_step, y_step, where='post', color=truth_color, linestyle='--', 
                    alpha=0.5, lw=2.5, zorder=1)

    # Formatting Top Plot
    ax_pdf.set_ylabel(r'Stiffness reduction factor ($\mathbf{z}$)', labelpad=15)
    ax_pdf.set_ylim([lbound - 0.02, 1.02])
    ax_pdf.set_xticks(elements)
    ax_pdf.set_xticklabels([f'$e_{{{k}}}$' for k in elements])
    ax_pdf.grid(axis='y', alpha=0.2, linestyle='-', color='gray')
    
    # Remove clutter
    for spine in ['top', 'right']:
        ax_pdf.spines[spine].set_visible(False)
    ax_pdf.spines['left'].set_linewidth(1.5)
    ax_pdf.spines['bottom'].set_linewidth(1.5)
    
    ax_pdf.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2, 
                  frameon=True, fancybox=True, shadow=True, borderpad=0.8)

    # ==========================================
    # BOTTOM PLOT: Physical Beam (Target State)
    # ==========================================
    # Visualization based on Ground Truth to show the target physical state
    beam_viz = z_true.reshape(1, -1)
    im = ax_beam.imshow(
        beam_viz, cmap=cmap, norm=norm_color, aspect='auto', 
        extent=[0.5, n_elements + 0.5, 0, 1]
    )
    
    ax_beam.set_yticks([])
    ax_beam.set_xticks(elements)
    ax_beam.tick_params(axis='x', length=0)
    
    # Add Ground Truth values inside the beam
    for j in range(n_elements):
        gt_val = z_true[j]
        text_color = 'white' if gt_val < 0.6 else 'black'
        ax_beam.text(j + 1, 0.5, f"{gt_val:.2f}", ha='center', va='center', 
                     color=text_color, fontweight='bold', fontsize=22)
    
    # Beam boundaries and support styling
    for spine in ax_beam.spines.values():
        spine.set_linewidth(2.5)
    
    # Realistic support triangles
    ax_beam.plot(0.5, 0, marker='^', markersize=25, color='black', clip_on=False, zorder=15)
    ax_beam.plot(n_elements + 0.5, 0, marker='^', markersize=25, color='black', clip_on=False, zorder=15)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.09, 0.015, 0.14])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\mathbf{z}$', fontsize=26)
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.subplots_adjust(hspace=0.1)

    # Save logic
    save_dir = os.path.join(folder_path, "Physical_PDF_Profiles")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'Sample_{pos}_UQ_Violin.png'), 
                dpi=500, bbox_inches='tight', transparent=False)
    plt.show()
    plt.close()