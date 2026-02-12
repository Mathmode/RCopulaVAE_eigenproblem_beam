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


def gaussian_marginal_samples(locs, scales, copula_samples):
    """
    Builds marginal samples given copula samples and Gaussian marginal parameters.
    Returns marginal_samples with shape (batch_size, num_samples, n_dims).
    """
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low = 0.00005, high = 0.999).quantile(copula_samples)
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

# =============================================================================
# 2. MAIN VISUALIZATION FUNCTION (Using your provided structure)
# =============================================================================
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
  
def plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                 predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, folder_path):
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
    z_samples = gaussian_marginal_samples(locs, scales, copula_samples)
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
    labels = [f'$\\alpha_{{{k+1}}}$' for k in range(n_elements)]
    
    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            
            if r == c: # Diagonal: 1D Marginal Distribution
                try:
                    vals = x_filtered[:, r]
                    # Generate 1D KDE
                    kde1d = gaussian_kde(vals, weights=w_filtered)
                    x_grid = np.linspace(0, 1, 100)
                    y_grid = kde1d(x_grid)
                    
                    ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                    ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                    
                    # # Ground truth line
                    # ax.axvline(z_true[r], color='red', linestyle='--', lw=2, label='True')
                    
                    # Formatting
                    # ax.set_title(f"{labels[r]}\nTrue: {z_true[r]:.2f}", fontsize=10)
                    ax.set_xlim(0, 1)
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
                    xi, yi = np.mgrid[0:1:50j, 0:1:50j]
                    coords = np.vstack([xi.flatten(), yi.flatten()])
                    zi = kernel(coords).reshape(xi.shape)
                    
                    # Plot filled contours
                    cf = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
                    
                    # Add Ground Truth point
                    ax.plot(z_true[c], z_true[r], color='red', marker='*', 
                            markersize=12, markeredgecolor='white', label='Truth')
                    
                except Exception as e:
                    # Fallback to scatter if KDE fails (e.g. singular matrix)
                    ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
                    ax.plot(z_true[c], z_true[r], 'r*', markersize=10)

                # Axis Labels
                if c == 0: ax.set_ylabel(labels[r], fontsize=12)
                if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle=':', alpha=0.3)
                
            else: # Upper Triangle: Empty or hidden
                ax.axis('off')

    plt.suptitle(f"Sample {pos}: Bayesian Posterior Uncertainty ($\\beta={beta}$)\nGround Truth marked in Red", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Logic
    save_dir = os.path.join(folder_path, "Predicted_Posteriors")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'Sample_{pos}_Posterior.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

