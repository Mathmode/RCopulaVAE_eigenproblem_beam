# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:03:57 2026

@author: anafd
"""
import tensorflow as tf
import numpy as np
import os
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy import stats
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm

#Graph configuration function (font sizes)
def plot_configuration():
    plt.rc('font', size = 16)          # controls default text sizes
    plt.rc('axes', titlesize = 16)     # fontsize of the axes title
    plt.rc('axes', labelsize = 16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 16)   # legend fontsize
    plt.rc('figure', titlesize= 16)   # fontsize of the figure title
    

sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style = "dark")
plot_configuration()

#%% Functions to generate samples from the model 
def gaussian_copula(LT_matrix, n_dims, n_samples):
    """
    Generates samples from a batch of Gaussian copulas.
    """
    mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
    mvn_samples = mvn_model.sample(n_samples)

    copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return copula_samples, mvn_samples, mvn_model


def gaussian_marginal_samples(locs, scales, copula_samples, lbound):
    """
    Builds marginal samples given copula samples and Gaussian marginal parameters.
    Returns marginal_samples with shape (batch_size, num_samples, n_dims).
    """
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low = lbound +0.0001, high = 0.999).quantile(copula_samples)
    # marginal_samples: (num_samples, n_dims)
    return marginal_samples



# %% 2. MAC and Physics Engine
def calculate_MAC(phi_true, phi_pred):
    """
    Computes MAC: (phi_t^T * phi_p)^2 / ((phi_t^T * phi_t) * (phi_p^T * phi_p))
    phi_true/pred shapes: (Batch, N_dofs, N_modes)
    """
    # Numerator: dot product squared
    inner_prod = np.sum(phi_true * phi_pred, axis=1) # (Batch, N_modes)
    numerator = np.square(inner_prod)
    
    # Denominator: norms squared
    denom_true = np.sum(np.square(phi_true), axis=1)
    denom_pred = np.sum(np.square(phi_pred), axis=1)
    denominator = denom_true * denom_pred + 1e-10
    
    return numerator / denominator


def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    """
    K_batch: (Batch, Free_DOF, Free_DOF)
    L_inv_tf: (Free_DOF, Free_DOF) - L^-1 where M_free = L @ L.T
    n_modes: Number of modes to truncate
    free_dofs: List/Array of indices of free degrees of freedom
    n_dofs: Total number of DOFs (2 * n_nodes)
    """
    with tf.device('/CPU:0'):
        # 1. Solve Eigenproblem in Modal Space: A = L^-1 * K * L^-T
        # Note: Script uses L_inv @ K @ L_inv.T. 
        # Since L_inv_tf is L^-1, we use L_inv on left and transpose on right.
        A_batch = tf.matmul(L_inv_tf, tf.matmul(K_batch, tf.transpose(L_inv_tf)))
        
        vals, vecs = tf.linalg.eigh(A_batch)
        
        # 2. Extract and Truncate
        vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e+20)
        f_Hz = tf.math.sqrt(vals_trunc) / (2 * np.pi)
        
        # 3. Recover Physical Modes for Free DOFs: phi_free = L^-T * vecs
        phi_free = tf.matmul(tf.transpose(L_inv_tf), vecs[:, :, :n_modes])
        
        # 4. FULL RECONSTRUCTION (Include Boundary Conditions)
        # We need to map (Batch, Free_DOF, Modes) -> (Batch, Total_DOF, Modes)
        batch_size = tf.shape(K_batch)[0]
        
        # Create a scatter mask to put free DOFs back into their global positions
        indices = tf.expand_dims(free_dofs, axis=-1) # Shape (Free_DOF, 1)
        
        # We transpose phi_free to (Free_DOF, Batch, Modes) for easier scattering
        phi_free_t = tf.transpose(phi_free, perm=[1, 0, 2])
        full_modes_t = tf.scatter_nd(indices, phi_free_t, shape=[n_dofs, batch_size, n_modes])
        
        # Transpose back to (Batch, Total_DOF, Modes)
        full_modes = tf.transpose(full_modes_t, perm=[1, 0, 2])
        
        # 5. SLICING (0, 2, 4... for Vertical; 1, 3, 5... for Rotational)
        # Shape: (Batch, n_nodes, n_modes)
        vert_modes = full_modes[:, 0::2, :]
        rot_modes = full_modes[:, 1::2, :]

        # 6. SIGN CONSISTENCY (Matching the peak of vertical displacement)
        # Find the index of the maximum absolute vertical displacement for each mode
        # vert_modes shape: (Batch, Node, Mode)
        abs_vert = tf.abs(vert_modes)
        max_idx = tf.argmax(abs_vert, axis=1) # (Batch, Mode)
        
        # Gather the actual values at those indices to find the sign
        # We use batch_gather logic via gather_nd
        batch_idx = tf.range(batch_size)[:, tf.newaxis]
        mode_idx = tf.range(n_modes)[tf.newaxis, :]
        
        # Create coordinates for gather_nd: [batch, max_idx_for_that_mode, mode]
        gather_coords = tf.stack([
            tf.broadcast_to(batch_idx, [batch_size, n_modes]),
            tf.cast(max_idx, tf.int32),
            tf.broadcast_to(mode_idx, [batch_size, n_modes])
        ], axis=-1)
        
        peak_vals = tf.gather_nd(vert_modes, gather_coords)
        signs = tf.sign(peak_vals) # (Batch, Mode)
        
        # Apply signs: (Batch, Node, Mode) * (Batch, 1, Mode)
        vert_modes = vert_modes * signs[:, tf.newaxis, :]
        rot_modes = rot_modes * signs[:, tf.newaxis, :]

        # 7. Final Transpose to match (Batch, n_modes, n_nodes)
        vert_final = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_final = tf.transpose(rot_modes, perm=[0, 2, 1])

        return f_Hz.numpy(), rot_final.numpy(), vert_final.numpy()
    
from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices
  
def calculate_posterior_PDF_info(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq,  lbound, folder_path):
    
    """
    Generates uncertainty plots for a specific test sample 'pos'.
    """
    # Unpack stats for this specific sample
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
    
    
    print(f"\n--- Analyzing Sample #{pos} ---")
    
    # 1. GENERATE SAMPLES
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, mvn_samples, mvn_model = gaussian_copula(LT_matrix, n_dims, n_samples) 
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)
    z_samples = z_samples.numpy()
    
    # 2. PHYSICS PROPAGATION (Vectorized on GPU/CPU)
    print(f"Propagating {n_samples} samples through Physics Engine...")
    # B. Assemble Global K
    print("Assembling Global Stiffness...")
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, model.fixed_dofs_indices)
    
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

def plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path):
    """
    Generates uncertainty plots for a specific test sample 'pos'.
    """
    # Unpack stats for this specific sample
    locs = predicted_stats['test_means'][pos,:]
    scales = predicted_stats['test_scales'][pos,:] 
    LT_matrix = predicted_stats['test_L_matrices'][pos,:] 
    
    Freqs_true = test_datasets['Freqs_true_test']
    Rotmodes_true = test_datasets['Rotmodes_true_test']
    Vertmodes_true = test_datasets['Vertmodes_true_test']
    Alphas_true = test_datasets['alpha_factors_true_test']
    
    n_dims = locs.shape[1]
    n_elements = n_dims
    
    
    print(f"\n--- Analyzing Sample #{pos} ---")
    
    # 1. GENERATE SAMPLES
    L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
    copula_samples, mvn_samples, mvn_model = gaussian_copula(LT_matrix, n_dims, n_samples) 
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples, lbound)
    z_samples = tf.cast(z_samples, dtype=tf.float32)
    z_samples = z_samples.numpy()
    
    # 2. PHYSICS PROPAGATION (Vectorized on GPU/CPU)
    print(f"Propagating {n_samples} samples through Physics Engine...")
    # B. Assemble Global K
    print("Assembling Global Stiffness...")
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples, tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, model.fixed_dofs_indices)
    
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
           
    
    
    ### PLOTTING prueba
    print("Generating KDE Plots...")
    z_true = Alphas_true[pos,:]
    std_z = np.std(z_samples, axis=0)
    # Thresholding to ignore low-probability noise for KDE stability
    # We use a more robust mask or just use all samples if weights are well-distributed
    threshold = np.max(posterior_weights) * 1e-6
    mask = posterior_weights > threshold
    
    # If the mask is too restrictive, relax it
    if np.sum(mask) < 10:
        mask = np.ones(len(posterior_weights), dtype=bool)

    x_filtered = z_samples[mask]
    w_filtered = posterior_weights[mask]
    w_filtered /= np.sum(w_filtered) # Re-normalize

    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z{{{k+1}}}$' for k in range(n_elements)]
    
    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            
            if r == c: # Diagonal: 1D Marginal Distribution
                try:
                    vals = x_filtered[:, r]
                    # Generate 1D KDE
                    kde1d = gaussian_kde(vals, weights=w_filtered)
                    x_grid = np.linspace(lbound, 1, 100)
                    y_grid = kde1d(x_grid)
                    
                    ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                    ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                    
                    # # Ground truth line
                    # ax.axvline(z_true[r], color='red', linestyle='--', lw=2, label='True')
                    ax.text(0.5, 0.3, labels[r], fontsize=20, ha='center', va='center', fontweight='bold', transform=ax.transAxes)

                    # Formatting
                    # ax.set_title(f"{labels[r]}\nTrue: {z_true[r]:.2f}", fontsize=10)
                    ax.set_xlim(lbound, 1)
                    ax.set_yticks([]) # Remove density scale for cleaner look
                except Exception as e:
                    ax.text(0.5, 0.5, "KDE Fail", ha='center')
                
            elif r > c: # Lower Triangle: 2D Joint Distribution
                x_vals = x_filtered[:, c]
                y_vals = x_filtered[:, r]
                
                try:
                    # Stacking for KDE
                    values = np.vstack([x_vals, y_vals])
                    kernel = gaussian_kde(values, weights=w_filtered)
                    
                    # Create grid for evaluation
                    # Use 40-50 for performance, higher for smoothness
                    xi, yi = np.mgrid[lbound:1:50j, lbound:1:50j]
                    coords = np.vstack([xi.flatten(), yi.flatten()])
                    zi = kernel(coords).reshape(xi.shape)
                    
                    # Plot filled contours
                    cf = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
                    
                    # Add Ground Truth point
                    ax.plot(z_true[c], z_true[r], color='red', marker='*', 
                            markersize=14, markeredgecolor='white', label='Truth')
                    
                except Exception as e:
                    # Fallback to scatter if KDE fails (e.g. singular matrix)
                    ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
                    ax.plot(z_true[c], z_true[r], 'r*', markersize=18)

                # Axis Labels
                if c == 0: ax.set_ylabel(labels[r], fontsize=18)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=18)
                
                ax.set_xlim(lbound, 1)
                ax.set_ylim(lbound, 1)
                ax.grid(True, linestyle=':', alpha=0.3)
                
            else: # Upper Triangle: Empty or hidden
                ax.axis('off')
                
            # Refine axis ticks: hide inner labels to make a true corner plot
            if r >= c:
                ax.tick_params(labelsize=18)
                if c == 0: 
                    if r != 0: ax.set_ylabel(labels[r], fontsize=18)
                else:
                    ax.tick_params(labelleft=False)
                    
                if r == n_elements - 1: 
                    ax.set_xlabel(labels[c], fontsize=18)
                else:
                    ax.tick_params(labelbottom=False)        
    # 1. Add Global Colorbar using the captured contour (`cf`)
    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density', fontsize=18)

    # 2. Add Ground Truth Legend in the empty upper triangle
    
    import matplotlib.lines as mlines
    gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=18, markeredgecolor='white', label='Ground truth')
    # Placed nicely in the upper right
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), 
               fontsize=18, frameon=True, facecolor='white', edgecolor='silver', shadow=True)

    plt.tight_layout()
    # plt.suptitle(f"Sample {pos}: Bayesian Posterior Uncertainty ($\\beta={beta}$)\nGround Truth marked in Red", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Logic
    save_dir = os.path.join(folder_path, "Predicted_Posteriors")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'Sample_{pos}_Posterior.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
    
    

def plot_physical_damage_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path):
    """
    Visualizes the physical uncertainty of the damage estimates along a beam.
    Corrected Logic: 
        - z = 1.0 (Yellow/Light) -> Healthy
        - z = 0.5 (Dark Red)     -> Severe Damage
    """
    # Force default matplotlib style
    plt.style.use('default')
    
    # 1. Calculate Expected Value and Standard Deviation (Weighted by Posterior)
    weights_norm = posterior_weights / np.sum(posterior_weights)
    
    expected_z = np.average(z_samples, weights=weights_norm, axis=0)
    variance_z = np.average((z_samples - expected_z)**2, weights=weights_norm, axis=0)
    std_z = np.sqrt(variance_z)
    
    # 2. Setup Figure and Axes
    fig, (ax_bar, ax_beam) = plt.subplots(
        2, 1, 
        figsize=(11, 7), 
        gridspec_kw={'height_ratios': [3.5, 1]}, 
        sharex=True,
        facecolor='white'
    )
    
    elements = np.arange(1, n_elements + 1)
    
    # ==========================================
    # COLOR LOGIC ADJUSTMENT
    # ==========================================
    # Using 'YlOrRd_r' (reversed) so 0 is Red and 1 is Yellow.
    # Or keep 'YlOrRd' but use a normalization that maps 0->1 and 1->0.
    # Best approach: Use YlOrRd_r so high values (1.0) are Yellow and low (0.0) are Red.
    cmap = cm.get_cmap('YlOrRd_r') 
    norm = mcolors.Normalize(vmin=0.5, vmax=1)
    bar_colors = [cmap(norm(val)) for val in expected_z]
    
    color_true = '#111111'   # Sharp black for truth
    # color_error = '#555555'  # Soft charcoal for error bars

    color_error = 'blue'  # Soft charcoal for error bars
    
    # ==========================================
    # TOP PLOT: Bar Chart with Uncertainty
    # ==========================================
    ax_bar.bar(
        elements, expected_z, yerr=std_z, 
        color=bar_colors, edgecolor='none',
        label='Average value', 
        capsize=6, error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': color_error}
    )
    
    # Dummy plot for legend
    ax_bar.errorbar([], [], yerr=[], ecolor=color_error, capsize=6, elinewidth=2, 
                    linestyle='None', label='$\pm 1\sigma$ interval')

    if z_true is not None:
        x_step = np.arange(0.5, n_elements + 1.5, 1)
        y_step = np.concatenate([z_true, [z_true[-1]]])
        ax_bar.step(
            x_step, y_step, where='post', 
            color=color_true, label='True value', 
            linestyle='--', linewidth=2, zorder=5
        )
        ax_bar.plot(elements, z_true, marker='s', linestyle='none', color=color_true, markersize=6, zorder=6)

    # Formatting
    ax_bar.set_ylabel('Stiffness reduction factor ($z$)', fontsize=18, fontweight='medium', labelpad=10)
    ax_bar.set_ylim([0.5, 1.0]) # Reduction factor usually doesn't exceed 1.0 significantly
    ax_bar.tick_params(axis='y', labelsize=15)
    ax_bar.grid(axis='y', alpha=0.4, linestyle=':')
    
    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    
    ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=True, 
                  facecolor='white', edgecolor='#dddddd', fontsize=15, framealpha=1)
    
    # ==========================================
    # BOTTOM PLOT: Physical Beam Heatmap
    # ==========================================
    beam_viz = expected_z.reshape(1, -1)
    
    im = ax_beam.imshow(
        beam_viz, cmap=cmap, norm=norm, aspect='auto', 
        extent=[0.5, n_elements + 0.5, 0, 1]
    )
    
    ax_beam.set_yticks([])
    ax_beam.set_xticks(elements)
    ax_beam.set_xticklabels([f'El {k}' for k in elements], fontsize=15)
    ax_beam.tick_params(axis='x', length=0, pad=10)
    
    for j in range(n_elements):
        mu = expected_z[j]
        sig = std_z[j]
        
        # In YlOrRd_r, low values (near 0) are Dark Red.
        # So if mu < 0.45, use white text for contrast against dark red.
        text_color = 'white' if mu < 0.45 else 'black'
        
        annotation_text = f"{mu:.2f}\n($\pm${sig:.2f})"
        ax_beam.text(
            j + 1, 0.5, annotation_text, 
            ha='center', va='center', color=text_color, 
            fontweight='bold', fontsize=15, linespacing=1.4
        )
    
    # Physical beam boundaries
    for spine in ax_beam.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
        
    # Support triangles
    ax_beam.plot(0.5, 0, marker='^', markersize=20, color='black', clip_on=False, zorder=10)
    ax_beam.plot(n_elements + 0.5, 0, marker='^', markersize=20, color='black', clip_on=False, zorder=10)

    # Colorbar Adjustment
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('reduction factor', fontsize=15)
    # Adding tick labels to clarify severity
    cbar.set_ticks([0.5, 1])
    cbar.set_ticklabels(['z = 0.5', 'z = 1.0  (Healthy)'])
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    fig.subplots_adjust(hspace=0.05) 
    
    # Save logic
    save_dir = os.path.join(folder_path, "Physical_uncertainty_pred")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'Sample_{pos}_Diagnosis_Profile.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
        

def plot_copula_posterior_insights(z_samples, posterior_weights, z_true, n_elements, pos, save_dir):
    """
    Visualizes the joint dependencies and true probabilistic nature of the Copula posterior.
    Panel 1: Sample Trajectories (Spaghetti Plot) weighted by likelihood.
    Panel 2: Spatial Correlation Matrix of the damage state.
    """
    plt.style.use('default')
    
    # 1. Statistical Calculations (Weighted by Posterior)
    weights_norm = posterior_weights / np.sum(posterior_weights)
    
    # Weighted Covariance and Correlation Matrix
    cov_matrix = np.cov(z_samples, aweights=weights_norm, rowvar=False)
    std_z = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_z, std_z)
    
    # 2. Setup Figure
    fig, (ax_traj, ax_corr) = plt.subplots(
        1, 2, 
        figsize=(14, 6), 
        gridspec_kw={'width_ratios': [1.5, 1]},
        facecolor='white'
    )
    
    elements = np.arange(1, n_elements + 1)
    
    # ==========================================
    # LEFT PLOT: Sample Trajectories (Parallel Coordinates)
    # ==========================================
    # Sort samples so the highest likelihood samples are plotted last (on top)
    idx_sorted = np.argsort(posterior_weights)
    
    # Take top 500 samples to prevent visual clutter, but show the distribution
    n_plot = min(500, len(z_samples))
    top_idx = idx_sorted[-n_plot:]
    top_samples = z_samples[top_idx]
    top_weights = weights_norm[top_idx]
    
    # Normalize alpha transparency based on posterior weights
    alpha_min, alpha_max = 0.05, 0.8
    if np.max(top_weights) > np.min(top_weights):
        alphas = (top_weights - np.min(top_weights)) / (np.max(top_weights) - np.min(top_weights))
        alphas = alpha_min + (alpha_max - alpha_min) * alphas
    else:
        alphas = np.full(n_plot, 0.2)

    # Plot the trajectories
    for i in range(n_plot):
        ax_traj.plot(elements, top_samples[i], color='#4575b4', alpha=alphas[i], linewidth=1.5)
        
    # Dummy line for legend
    ax_traj.plot([], [], color='#4575b4', alpha=0.6, linewidth=2, label='Posterior Samples (Opacity $\propto$ Likelihood)')

    # Overlay True Damage state
    if z_true is not None:
        ax_traj.plot(elements, z_true, color='#d73027', marker='s', linestyle='--', 
                     linewidth=2.5, markersize=8, label='True Damage State ($z^*$)', zorder=10)

    # Formatting Left Plot
    ax_traj.set_title("Joint Posterior Trajectories", fontsize=14, fontweight='bold', pad=15)
    ax_traj.set_ylabel('Stiffness Reduction Factor ($z$)', fontsize=13)
    ax_traj.set_xlabel('Beam Element', fontsize=13)
    ax_traj.set_xticks(elements)
    ax_traj.set_ylim([-0.05, 1.05])
    ax_traj.grid(True, alpha=0.3, linestyle='--')
    ax_traj.spines['top'].set_visible(False)
    ax_traj.spines['right'].set_visible(False)
    ax_traj.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#dddddd')

    # ==========================================
    # RIGHT PLOT: Weighted Correlation Matrix
    # ==========================================
    # RdBu colormap: Blue = Negative Correlation, Red = Positive Correlation
    cmap = cm.get_cmap('RdBu_r') 
    
    im = ax_corr.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    
    # Annotate the correlation values inside the heatmap
    for i in range(n_elements):
        for j in range(n_elements):
            val = corr_matrix[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            # Only show off-diagonal or format diagonal differently if desired
            if i == j:
                ax_corr.text(j, i, f"1.00", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            else:
                ax_corr.text(j, i, f"{val:+.2f}", ha='center', va='center', color=text_color, fontsize=10)

    # Formatting Right Plot
    ax_corr.set_title("Spatial Correlation Structure", fontsize=14, fontweight='bold', pad=15)
    ax_corr.set_xticks(np.arange(n_elements))
    ax_corr.set_yticks(np.arange(n_elements))
    ax_corr.set_xticklabels([f'El {k}' for k in elements])
    ax_corr.set_yticklabels([f'El {k}' for k in elements])
    
    # Add Colorbar for Correlation
    cbar = fig.colorbar(im, ax=ax_corr, shrink=0.8, pad=0.05)
    cbar.set_label('Pearson Correlation Coefficient ($r$)', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    
    # Save the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'Sample_{pos}_Copula_Insights.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
        
    plt.show()
    plt.close(fig)






    
    
    
    
    
    
    
    
# def plot_physical_damage_profile(z_samples, posterior_weights, z_true, n_elements, pos, save_dir):
#     """
#     Visualizes the physical uncertainty of the damage estimates along a beam.
#     Generates a high-quality two-panel plot: 
#       1. A bar chart with 1-sigma credible intervals.
#       2. A physical 1D heatmap representing the beam's stiffness reduction.
    
#     Args:
#         z_samples: (n_samples, n_elements) array of sampled stiffness reduction factors.
#         posterior_weights: (n_samples,) array of calculated likelihood/posterior weights.
#         z_true: (n_elements,) array of ground truth stiffness reduction factors.
#         n_elements: Number of elements in the beam.
#         pos: Sample ID/index (used for saving the file).
#         save_dir: Directory path where the plot will be saved.
#     """
    
#     # 1. Calculate Expected Value and Standard Deviation (Weighted by Posterior)
#     # Ensure weights sum exactly to 1 for accurate statistical calculation
#     weights_norm = posterior_weights / np.sum(posterior_weights)
    
#     expected_z = np.average(z_samples, weights=weights_norm, axis=0)
#     variance_z = np.average((z_samples - expected_z)**2, weights=weights_norm, axis=0)
#     std_z = np.sqrt(variance_z)
    
#     # 2. Setup Figure and Axes
#     fig, (ax_bar, ax_beam) = plt.subplots(
#         2, 1, 
#         figsize=(10, 6.5), 
#         gridspec_kw={'height_ratios': [4, 1.2]}, 
#         sharex=True,
#         facecolor='white'
#     )
    
#     elements = np.arange(1, n_elements + 1)
    
#     # Sophisticated Color Palette
#     color_estimate = '#4575b4' # Calm Blue for probabilistic estimates
#     color_true = '#d73027'     # Strong Red for the absolute truth
    
#     # ==========================================
#     # TOP PLOT: Bar Chart with Uncertainty
#     # ==========================================
#     ax_bar.bar(
#         elements, expected_z, yerr=std_z, 
#         color=color_estimate, alpha=0.75, edgecolor='none',
#         label='Estimated Stiffness Reduction ($\mu \pm 1\sigma$)', 
#         capsize=6, error_kw={'elinewidth': 2.5, 'capthick': 2.5, 'ecolor': '#1a1a1a'}
#     )
    
#     # Overlay True Damage state as a continuous step line
#     if z_true is not None:
#         x_step = np.arange(0.5, n_elements + 1.5, 1)
#         y_step = np.concatenate([z_true, [z_true[-1]]])
#         ax_bar.step(
#             x_step, y_step, where='post', 
#             color=color_true, label='True Damage State ($z^*$)', 
#             linestyle='--', linewidth=2.5, zorder=5
#         )
#         # Optional: dots at the center of the elements for absolute clarity
#         ax_bar.plot(elements, z_true, marker='o', linestyle='none', color=color_true, markersize=5, zorder=6)

#     # Top Plot Formatting
#     ax_bar.set_ylabel('Stiffness Reduction Factor', fontsize=13, fontweight='medium')
    
#     # Dynamically set Y-limit so error bars aren't cut off (Minimum 1.1 height)
#     max_y = max(1.1, np.max(expected_z + std_z) * 1.15)
#     ax_bar.set_ylim([0, max_y])
    
#     ax_bar.tick_params(axis='y', labelsize=11)
#     ax_bar.grid(axis='y', alpha=0.25, linestyle='--')
#     ax_bar.spines['top'].set_visible(False)
#     ax_bar.spines['right'].set_visible(False)
    
#     ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=12)
    
#     # Explanatory text box for the uncertainty
#     props = dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6')
#     ax_bar.text(
#         0.02, 0.95, "Error bars denote the 1$\sigma$ credible interval\n(derived from VAE posterior density)", 
#         transform=ax_bar.transAxes, fontsize=10, verticalalignment='top', bbox=props, color='#495057'
#     )
    
#     # ==========================================
#     # BOTTOM PLOT: Physical Beam Heatmap
#     # ==========================================
#     beam_viz = expected_z.reshape(1, -1)
    
#     im = ax_beam.imshow(
#         beam_viz, cmap='YlGnBu', aspect='auto', 
#         extent=[0.5, n_elements + 0.5, 0, 1], vmin=0, vmax=1
#     )
    
#     # Clean up bottom axes
#     ax_beam.set_yticks([])
#     ax_beam.set_xticks(elements)
#     ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=12, fontweight='medium')
#     ax_beam.tick_params(axis='x', length=0, pad=8)
    
#     # Print the exact estimated numbers inside the beam
#     for j, val in enumerate(expected_z):
#         text_color = 'white' if val > 0.55 else 'black'
#         ax_beam.text(
#             j + 1, 0.5, f'{val:.2f}', 
#             ha='center', va='center', color=text_color, 
#             fontweight='bold', fontsize=12
#         )
    
#     # Draw physical beam boundaries
#     for spine in ax_beam.spines.values():
#         spine.set_linewidth(1.5)
#         spine.set_color('#111111')
        
#     # Draw simply-supported triangles at the ends (Visual grounding)
#     ax_beam.plot(0.5, 0, marker='^', markersize=14, color='#333333', clip_on=False, zorder=10)
#     ax_beam.plot(n_elements + 0.5, 0, marker='^', markersize=14, color='#333333', clip_on=False, zorder=10)

#     plt.tight_layout()
    
#     # Save the plot
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f'Sample_{pos}_PhysicalDamage_Profile.png')
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
#     plt.show()
#     plt.close(fig)
    
    
# def physical_uncertainty_plot(z_samples, posterior_pdf,alpha_factors_true_test, n_elements, pos, save_dir):
#     expected_z = np.sum(z_samples * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf)
#     std_z = np.sqrt(np.sum(np.square(z_samples - expected_z) * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf))
#     z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None
    
#     # 5.2 Main Posterior Plot (Pairwise Matrices)
#     fig, axes = plt.subplots(n_elements, n_elements, figsize=(10, 10), facecolor='white')
#     labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
#     mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)
    
#     for r in range(n_elements):
#         for c in range(n_elements):
#             ax = axes[r, c]
#             if r == c: # Diagonal: Physical Labels
#                 ax.text(0.5, 0.5, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', color='#333333')
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.axis('off')
#             elif r > c: # Lower Triangle: Smooth 2D Joint PDF
#                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
#                 kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
#                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
#                 zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
#                 ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.9)
#                 ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.3, alpha=0.2)
                
#                 if z_true is not None:
#                     ax.plot(z_true[c], z_true[r], 'ro', markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=10)
                
#                 if c == 0: ax.set_ylabel(labels[r], fontsize=12)
#                 if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.tick_params(labelsize=8)
#                 ax.grid(True, linestyle=':', alpha=0.3)
#             else:
#                 ax.axis('off')
    
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#     plt.savefig(os.path.join(save_dir, f'P{pos}_JointPosterior.png'), dpi=500, bbox_inches='tight')
#     plt.show()
    
#     # 5.3 Enhanced Physical Damage Profile (Fancier Visuals)
#     fig, (ax_bar, ax_beam) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
#     elements = np.arange(1, n_elements + 1)
#     color_estimate = '#4575b4' # Sophisticated Blue
#     color_true = '#d73027'     # Strong Red
    
#     # Top Plot: Bar chart with Uncertainty
#     ax_bar.bar(elements, expected_z, yerr=std_z, color=color_estimate, alpha=0.65, 
#                label='Estimated Stiffness Reduction ($E[z|\mathbf{m}]$)', 
#                capsize=8, error_kw={'elinewidth':2, 'capthick':2, 'ecolor': '#1a1a1a'})
    
#     if z_true is not None:
#         # Step boundaries for 5 elements
#         x_step = np.arange(0.5, n_elements + 1.5, 1)
#         y_step = np.concatenate([z_true, [z_true[-1]]])
#         ax_bar.step(x_step, y_step, where='post', color=color_true, 
#                     label='True Damage State ($\mathbf{z}^*$)', linestyle='--', lw=2.5, zorder=5)
    
#     ax_bar.set_ylabel('Stiffness Reduction ($z$)', fontsize=13, fontweight='medium')
#     ax_bar.set_ylim([0, 1.2])
#     ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=11)
#     ax_bar.grid(axis='y', alpha=0.2, linestyle='-')
    
#     # Add explanatory note for Uncertainty
#     props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
#     ax_bar.text(0.02, 0.95, "Note: Error bars denote the 1$\sigma$ \ncredible interval (posterior std. dev.)", 
#                 transform=ax_bar.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
#     # Bottom Plot: Physical Beam Heatmap
#     beam_viz = expected_z.reshape(1, -1)
#     im = ax_beam.imshow(beam_viz, cmap='YlGnBu', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1], vmin=0, vmax=1)
    
#     # Clean up beam visualization
#     ax_beam.set_yticks([])
#     ax_beam.set_xticks(elements)
#     ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=11)
#     ax_beam.tick_params(axis='x', length=0)
    
#     # Add values and physical frame
#     for j, val in enumerate(expected_z):
#         text_color = 'white' if val > 0.6 else 'black'
#         ax_beam.text(j + 1, 0.5, f'{val:.2f}', ha='center', va='center', 
#                      color=text_color, fontweight='bold', fontsize=11)
    
#     # Outer beam border
#     for spine in ax_beam.spines.values():
#         spine.set_linewidth(1.2)
#         spine.set_color('#333333')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'P{pos}_PhysicalProfile_Enhanced.png'), dpi=500, bbox_inches='tight')
#     plt.show()