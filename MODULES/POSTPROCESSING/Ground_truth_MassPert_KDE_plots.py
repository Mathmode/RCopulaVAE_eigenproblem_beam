# -*- coding: utf-8 -*-
"""
Ground Truth KDE Plots (Perturbed Mass Model)
Updated to incorporate the modeling error, proper MAC boundaries,
robust likelihood masking, and saving outputs to a specific perturbed folder.
"""

# %% 1. Initialization and Data Loading
import os

# Fix for XLA / libdevice path in Conda environments
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.path.join(conda_prefix, 'Library')}"

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_eigen_functions import assemble_global_Kmatrices
from MODULES.POSTPROCESSING.QMC_sampling import QuadratureMethod

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

K.utils.set_random_seed(1234)
K.backend.set_floatx('float32')

# Data Loading configuration
data_folder = "06Oct_Data_Noisy_E5_Lvl25_MassPerturb5"
data_path = os.path.join("Data", data_folder)

batch_size = 256
n_elements = 5
n_modes = 5
n_dofs = 2 * (n_elements + 1) 
num_dofs = n_dofs - 2 
lbound = 0.49

# Unpack all 14 variables returned by load_data
(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
 Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
 Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test,
 mean_f, std_f) = load_data(data_path, batch_size)

# Load baseline matrices and the PERTURBED mass matrix
_, Ke_matrices, _ = load_known_matrices(data_path, n_elements)
Mfree_perturbed = np.load(os.path.join(data_path, "Mass_matrix_perturbed.npy"))

# Recalculate Cholesky for the perturbed physics
L = np.linalg.cholesky(Mfree_perturbed)
L_inv_tf = tf.cast(np.linalg.inv(L), dtype=tf.float32)

# Set beta to 0.4 to match the gamma=0.4 used in the VAE training
beta = 0.35
inv_gamma_val = 1.0 / (beta**2)

# %% 2. Physics Engine & MAC Calculation
def calculate_MAC(modes_true, modes_pred):
    """Robust MAC calculation matching the KSmix analysis script."""
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return tf.linalg.diag_part(mac_matrix).numpy()

def physics_engine_step(K_batch, L_inv_tf, n_modes):
    """Computes frequencies and modes, properly padding boundaries and transposing shapes."""
    with tf.device('/CPU:0'):
        # CORRECTED: Match the data generation script (L_inv @ K @ L_inv_T)
        A_batch = tf.matmul(L_inv_tf, tf.matmul(K_batch, tf.transpose(L_inv_tf)))
        vals, vecs = tf.linalg.eigh(A_batch)
        vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e+20)
        
        f_Hz = tf.math.sqrt(vals_trunc) / (2 * np.pi)
        phi = tf.matmul(tf.transpose(L_inv_tf), vecs)[:, :, :n_modes]
        
        eig_cut = phi[:, 0:-1, :]
        rot = tf.concat([eig_cut[:, 0::2, :], phi[:, -1:, :]], axis=1) 
        vert_inner = eig_cut[:, 1::2, :] 
        
        # Transpose to shape (Batch, Modes, Nodes)
        rot = tf.transpose(rot, perm=[0, 2, 1])
        vert_inner = tf.transpose(vert_inner, perm=[0, 2, 1])
        
        # Pad vertical boundaries to match the 6-node structure of the ground truth
        batch_sz = tf.shape(vert_inner)[0]
        zeros_bound = tf.zeros((batch_sz, n_modes, 1), dtype=tf.float32)
        vert = tf.concat([zeros_bound, vert_inner, zeros_bound], axis=2)
        
        return f_Hz.numpy(), rot.numpy(), vert.numpy()

# %% 3. QMC Grid & Physics Pass
saving_path = os.path.join("MODULES", "POSTPROCESSING", "MassPert_Arrays")
if not os.path.exists(saving_path): 
    os.makedirs(saving_path)

# REGENERATE GRIDS: Since bounds and mass changed, we MUST recalculate the evaluation grid
print("Generating new QMC grids for the perturbed physics...")
qm = QuadratureMethod(gdim=5)
N_qmc = 2**13 
z_qmc_raw, _ = qm.QMC(N_qmc)

# Scale sample space using updated lbound
z_samples = lbound + (1.0 - lbound) * z_qmc_raw.T 
z_samples_tf = tf.cast(z_samples, dtype=tf.float32)
np.save(os.path.join(saving_path, "Z_pointcloud_MassPert.npy"), z_samples_tf, allow_pickle = True)

print("Assembling Perturbed Global Stiffness...")
Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples_tf, tf.cast(Ke_matrices, dtype=tf.float32))
Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, N_qmc)
np.save(os.path.join(saving_path, "Kfree_matrices_MassPert.npy"), Kfree_matrices, allow_pickle = True)

# To reuse later without recomputing, comment the generation block above and uncomment below:
# z_samples = np.load(os.path.join(saving_path, "Z_pointcloud_MassPert.npy"), allow_pickle = True)
# Kfree_matrices = np.load(os.path.join(saving_path, "Kfree_matrices_MassPert.npy"), allow_pickle = True)

print(f"Forward Pass for {N_qmc} points...")
batch_size = 256
all_f, all_rot, all_vert = [], [], []
for i in range(0, N_qmc, batch_size):
    f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes)
    all_f.append(f)
    all_rot.append(r)
    all_vert.append(v)

f_pred = np.concatenate(all_f, axis=0)      
rot_pred = np.concatenate(all_rot, axis=0)  
vert_pred = np.concatenate(all_vert, axis=0) 

# %% 4. Likelihood Calculation (Matching Training Loss Logic)
# positions = [34]

positions = [0, 1, 2, 3, 7,  9, 11, 17,25, 34, 45, 100, 138, 219, 234, 300, 343, 456, 555, 612, 690, 761]
# positions = [10,15,20,21,101,102,103,104,105,106,107,108,200,201,201,202,203,204]
for i in range(len(positions)):
    pos = positions[i]
    
    # 1. Frequency Loss (Logarithmic and Scaled)
    obs_f_scaled = Freqs_true_test[pos] 
    pred_logscaled = (np.log(f_pred) - mean_f) / std_f
    loss_f = np.mean(np.square(obs_f_scaled - pred_logscaled), axis=1)
    
    # 2. Mode Shape Loss (MAC-based)
    obs_r = Rotmodes_true_test[pos]  
    obs_v = Vertmodes_true_test[pos] 
    
    # Expand obs to match batch size
    obs_r_batch = np.repeat(obs_r[np.newaxis, :, :], N_qmc, axis=0)
    obs_v_batch = np.repeat(obs_v[np.newaxis, :, :], N_qmc, axis=0)
    
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)   
    vert_mac = calculate_MAC(obs_v_batch, vert_pred) 
    
    mac_combined = np.concatenate([rot_mac, vert_mac], axis=1) 
    loss_mac = np.mean(np.square(1 - mac_combined), axis=1)     
    
    # Total Data Misfit (matching ELBO logic)
    total_loss = loss_f + loss_mac
    exponent = -total_loss * inv_gamma_val 
    
    # Normalization
    max_exp = np.max(exponent)
    likelihood = np.exp(exponent - max_exp)
    posterior_pdf = likelihood / np.mean(likelihood)

# %% 5. Advanced Visualization & Uncertainty Quantification
    print(f"Analyzing and Plotting Position {pos}...")
    
    expected_z = np.sum(z_samples * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf)
    std_z = np.sqrt(np.sum(np.square(z_samples - expected_z) * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf))
    z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None

    # Save Directory for the plots
    save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_MassPert_plots")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # 5.1 Main Posterior Plot (Matching Manuscript Corner Plot Style)
    import matplotlib.lines as mlines
    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    
    # ROBUST MASK: Ensure at least 50 points survive to prevent KDE crash
    mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)
    if np.sum(mask) < 10:
        top_indices = np.argsort(posterior_pdf)[-50:]
        mask = np.zeros_like(posterior_pdf, dtype=bool)
        mask[top_indices] = True

    # Normalize weights for the filtered subset
    w_norm = posterior_pdf[mask] / np.sum(posterior_pdf[mask])
    cf = None

    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            
            if r == c: 
                # 1D Marginal Distribution Plot
                vals = z_samples[mask, r]
                kde1d = gaussian_kde(vals, weights=w_norm)
                x_grid = np.linspace(lbound - 0.05, 1.0, 200)
                y_grid = kde1d(x_grid)
                
                # Style matches the target image (light gray background, steelblue density)
                ax.set_facecolor('#EAEAF2') 
                ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(lbound, 1.0)
                ax.set_yticks([])
                
            elif r > c: 
                # 2D Joint Distribution Plot
                xi, yi = np.mgrid[lbound:1:100j, lbound:1:100j]
                kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
                kde = gaussian_kde(kde_coords, weights=w_norm)
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                
                if z_true is not None:
                    ax.plot(z_true[c], z_true[r], 'r*', markersize=14, markeredgecolor='white', zorder=10)
                
                ax.set_xlim(lbound, 1.0)
                ax.set_ylim(lbound, 1.0)
                
            else:
                ax.axis('off')

            # Axis formatting logic
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

    # Generate Shared Colorbar
    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density (true)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    
    # Generate Shared Legend
    gt_marker = mlines.Line2D([], [], color='#D62728', marker='*', linestyle='None', markersize=18, markeredgecolor='white', label='Ground truth')
    fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), fontsize=18, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])
    plt.savefig(os.path.join(save_dir, f'P{pos}_GT_JointPosterior.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # # 5.2 Enhanced Physical Damage Profile
    # fig, (ax_bar, ax_beam) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    # elements = np.arange(1, n_elements + 1)
    # color_estimate = '#4575b4' 
    # color_true = '#d73027'     
    
    # ax_bar.bar(elements, expected_z, yerr=std_z, color=color_estimate, alpha=0.65, 
    #            label='Estimated Stiffness Reduction ($E[z|\mathbf{m}]$)', 
    #            capsize=8, error_kw={'elinewidth':2, 'capthick':2, 'ecolor': '#1a1a1a'})
    
    # if z_true is not None:
    #     x_step = np.arange(0.5, n_elements + 1.5, 1)
    #     y_step = np.concatenate([z_true, [z_true[-1]]])
    #     ax_bar.step(x_step, y_step, where='post', color=color_true, 
    #                 label='True Damage State ($\mathbf{z}^*$)', linestyle='--', lw=2.5, zorder=5)
    
    # ax_bar.set_ylabel('Stiffness Reduction ($z$)', fontsize=13, fontweight='medium')
    # ax_bar.set_ylim([lbound - 0.05, 1.2])
    # ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=11)
    # ax_bar.grid(axis='y', alpha=0.2, linestyle='-')
    
    # props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
    # ax_bar.text(0.02, 0.95, "Note: Error bars denote the 1$\sigma$ \ncredible interval (posterior std. dev.)", 
    #             transform=ax_bar.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    # beam_viz = expected_z.reshape(1, -1)
    # im = ax_beam.imshow(beam_viz, cmap='YlGnBu', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1], vmin=lbound, vmax=1)
    
    # ax_beam.set_yticks([])
    # ax_beam.set_xticks(elements)
    # ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=11)
    # ax_beam.tick_params(axis='x', length=0)
    
    # for j, val in enumerate(expected_z):
    #     text_color = 'white' if val > 0.6 else 'black'
    #     ax_beam.text(j + 1, 0.5, f'{val:.2f}', ha='center', va='center', 
    #                  color=text_color, fontweight='bold', fontsize=11)
    
    # for spine in ax_beam.spines.values():
    #     spine.set_linewidth(1.2)
    #     spine.set_color('#333333')

    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'P{pos}_PhysicalProfile_Enhanced.png'), dpi=500, bbox_inches='tight')
    # plt.close()

print("Ground truth generation and plotting finished.")