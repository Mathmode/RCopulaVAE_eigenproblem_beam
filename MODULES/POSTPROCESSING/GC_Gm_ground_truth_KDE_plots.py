# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:08:57 2026

@author: anafdeznavamuel
"""

# -*- coding: utf-8 -*-
"""
Updated Script for plotting Ground Truth posterior PDFs
Includes Mode Shape Dimensionality [Batch, Modes, Nodes] and Sign Consistency
"""

import os
import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_eigen_functions import assemble_global_Kmatrices
from MODULES.POSTPROCESSING.QMC_sampling import QuadratureMethod
import seaborn as sns
sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style = "dark")

# %% 1. Initialization and Data Loading
K.utils.set_random_seed(1234)
data_path = os.path.join("Data", "11Feb2026_Corrected_Randomdata5elements")
batch_size = 256
(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
 Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
 Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
 mean_f, std_f) = load_data(data_path, batch_size)



batch_size = 48
n_elements = 5
n_modes = 5
n_dofs = 2 * (n_elements + 1)  # 12 total DOFs
num_dofs_total = n_dofs
num_dofs = n_dofs - 2          # 10 free DOFs

Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
L_inv_tf = tf.cast(L_inv, dtype=tf.float32)


# Scaling/noise parameters
beta = 0.26
inv_gamma_val = 1.0 / (beta**2)

# %% 2. Physics Engine & MAC Calculation (Updated for Sign Consistency & Shape)

def calculate_MAC(modes_true, modes_pred):
    """
    Computes Modal Assurance Criterion (MAC) between corresponding modes.
    modes_true, modes_pred: shapes (Batch, Modes, Nodes)
    Returns: Diagonal MAC values of shape (Batch, Modes)
    """
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    
    # Normalize along the Node axis (axis 2)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    
    # Compute MAC Matrix: (Batch, Modes, Nodes) @ (Batch, Nodes, Modes) -> (Batch, Modes, Modes)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    
    # We only need the corresponding True vs Pred mode MACs (the diagonal)
    mac_diag = tf.linalg.diag_part(mac_matrix)
    return mac_diag.numpy()

def physics_engine_step(K_batch, L_inv_tf, n_modes, num_dofs_total=12, fixed_dofs=np.array([0, 10])):
    """
    Forward physics solver identical to the Solve_eigenproblem Layer during training.
    """
    batch_size = tf.shape(K_batch)[0]
    
    with tf.device('/CPU:0'):
        # --- Solve Generalized Eigenproblem ---
        L_inv_T = tf.transpose(L_inv_tf)
        half_transformed = tf.matmul(K_batch, L_inv_tf)
        A = tf.matmul(L_inv_T, half_transformed)

        # Jitter for stability
        jitter = 1e-6 * tf.eye(tf.shape(A)[1], batch_shape=[batch_size])
        A = A + jitter
        
        eigvals, eigvecs_transformed = tf.linalg.eigh(A)
        
        # --- Filter & Frequencies ---
        eigvals = eigvals[:, :n_modes]
        eigvecs_transformed = eigvecs_transformed[:, :, :n_modes]
        
        eigvals = tf.nn.relu(eigvals) 
        freqs_rad = tf.sqrt(eigvals + 1e-12)
        freqs_hz = freqs_rad / (2.0 * np.pi)
        
        # --- Recover Physical Modes (Free DOFs) ---
        physical_modes_free = tf.matmul(L_inv_tf, eigvecs_transformed)

        # --- Reconstruct Full System ---
        all_indices = np.arange(num_dofs_total)
        free_dofs = np.delete(all_indices, fixed_dofs)

        flat_modes = tf.reshape(physical_modes_free, [-1, n_modes])
        batch_indices = tf.repeat(tf.range(batch_size), len(free_dofs))
        dof_indices = tf.tile(tf.constant(free_dofs, dtype=tf.int32), [batch_size])
        scatter_indices = tf.stack([batch_indices, dof_indices], axis=1) 
        
        physical_modes_full = tf.scatter_nd(
            indices=scatter_indices,
            updates=flat_modes,
            shape=[batch_size, num_dofs_total, n_modes]
        )
        
        # --- Separate Components ---
        vert_modes = physical_modes_full[:, 0::2, :]
        rot_modes = physical_modes_full[:, 1::2, :]
        
        # --- Sign Consistency ---
        max_indices = tf.argmax(tf.abs(vert_modes), axis=1, output_type=tf.int32) 
        
        batch_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), 1), tf.shape(max_indices))
        mode_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(n_modes), 0), tf.shape(max_indices))
        
        gather_indices = tf.stack([batch_idx_grid, max_indices, mode_idx_grid], axis=2)
        
        peak_vals = tf.gather_nd(vert_modes, gather_indices) 
        signs = tf.sign(peak_vals)
        signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
        signs_expanded = tf.expand_dims(signs, axis=1)
        
        vert_modes = vert_modes * signs_expanded
        rot_modes = rot_modes * signs_expanded

        # --- Transpose to [Batch, Modes, Nodes] ---
        vert_modes = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_modes = tf.transpose(rot_modes, perm=[0, 2, 1])
        
        # Slice the zero boundary nodes for vertical modes to match the 4 free coordinates
        vert_modes = vert_modes[:, :, 1:-1]
        
        return freqs_hz.numpy(), rot_modes.numpy(), vert_modes.numpy()

# %% 3. QMC Grid & Physics Pass
saving_path = os.path.join("Output")
if not os.path.exists(saving_path): os.makedirs(saving_path)

qm = QuadratureMethod(gdim=5)
N_qmc = 2**13 
# z_qmc_raw, _ = qm.QMC(N_qmc)
# z_samples = 0.05 + 0.90 * z_qmc_raw.T 
# z_samples_tf = tf.cast(z_samples, dtype=tf.float32)
# np.save(os.path.join(saving_path, "Z_pointcloud_samples.npy"), z_samples_tf, allow_pickle=True)

# print("Assembling Global Stiffness...")
# Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples_tf, tf.cast(Ke_matrices, dtype=tf.float32))
# Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, N_qmc)
# np.save(os.path.join(saving_path, "Kfree_matrices_03Feb.npy"), Kfree_matrices, allow_pickle=True)

z_samples = np.load(os.path.join(saving_path, "Z_pointcloud_samples.npy"), allow_pickle=True)
Kfree_matrices = np.load(os.path.join(saving_path, "Kfree_matrices_03Feb.npy"), allow_pickle = True)


print(f"Forward Pass for {N_qmc} points...")
batch_size_fwd = 512
all_f, all_rot, all_vert = [], [], []

for i in range(0, N_qmc, batch_size_fwd):
    f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size_fwd], L_inv_tf, n_modes)
    all_f.append(f)
    all_rot.append(r)
    all_vert.append(v)

f_pred = np.concatenate(all_f, axis=0)       # (16384, 5)
rot_pred = np.concatenate(all_rot, axis=0)   # (16384, 5, 6)
vert_pred = np.concatenate(all_vert, axis=0) # (16384, 5, 4)

# Apply padding to vertical predictions
paddings = tf.constant([[0, 0], [0, 0], [1, 1]])
vert_pred = tf.pad(vert_pred, paddings, mode="CONSTANT", constant_values=0)




# %% 4. Likelihood Calculation (Matching Training Loss Logic)
# positions = [0, 1, 7,  9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]

for pos in positions:
    # 1. Frequency Loss (Matched exactly to Copula format)
    obs_f_scaled = Freqs_true_test[pos] # Assumed to be already standardized log-freqs
    pred_log = np.log(f_pred)
    pred_logscaled = (pred_log - mean_f) / std_f
    f_err = np.square(obs_f_scaled - pred_logscaled)
    loss_f = np.mean(f_err, axis=1) # (16384,)
    
    # 2. Mode Shape Loss (MAC-based)
    obs_r = Rotmodes_true_test[pos]  
    obs_v = Vertmodes_true_test[pos] 
    
    # Fallback/Safety Check: Ensure shapes are [Modes, Nodes]
    if obs_r.shape[0] != n_modes: obs_r = obs_r.T
    if obs_v.shape[0] != n_modes: obs_v = obs_v.T
    
    # Expand obs to match batch size
    obs_r_batch = np.repeat(obs_r[np.newaxis, :, :], N_qmc, axis=0)
    obs_v_batch = np.repeat(obs_v[np.newaxis, :, :], N_qmc, axis=0)
    
    # MAC calculation returns batched diagonal (16384, 5)
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)   
    vert_mac = calculate_MAC(obs_v_batch, vert_pred) 
    
    # Match EXACTLY the logic from the Copula script 
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1) # (16384,)
    
    # 3. Total Data Misfit & Exponent
    total_loss = loss_f + loss_mac
    exponent = -total_loss * inv_gamma_val 
    
    # Normalization
    max_exp = np.max(exponent)
    likelihood = np.exp(exponent - max_exp)
    posterior_pdf = likelihood / np.mean(likelihood)
    
    # %% 5. Advanced Visualization
    print(f"Generating KDE Plots for Position {pos}...")
    
    alpha_true = alpha_factors_true_test[pos, :] if 'alpha_factors_true_test' in locals() else None

    # Thresholding to ignore low-probability noise for KDE stability
    threshold = np.max(posterior_pdf) * 1e-6
    mask = posterior_pdf > threshold
    
    # If the mask is too restrictive, relax it
    if np.sum(mask) < 10:
        mask = np.ones(len(posterior_pdf), dtype=bool)

    x_filtered = z_samples[mask]
    w_filtered = posterior_pdf[mask]
    w_filtered /= np.sum(w_filtered) # Re-normalize

    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z{{{k+1}}}$' for k in range(n_elements)]
    
    cf = None # Variable to capture contour mapping for colorbar

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
                    ax.text(0.5, 0.3, labels[r], fontsize=20, ha='center', va='center', fontweight='bold', transform=ax.transAxes)

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
                    kde_coords = np.vstack([x_vals, y_vals])
                    kernel = gaussian_kde(kde_coords, weights=w_filtered)
                    
                    # Create grid for evaluation (Using 50j for matching style performance)
                    xi, yi = np.mgrid[0:1:50j, 0:1:50j]
                    coords = np.vstack([xi.flatten(), yi.flatten()])
                    zi = kernel(coords).reshape(xi.shape)
                    
                    # Plot filled contours (matched to 15 levels)
                    cf = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
                    
                    # Add Ground Truth point
                    if alpha_true is not None:
                        ax.plot(alpha_true[c], alpha_true[r], color='red', marker='*', 
                                markersize=16, markeredgecolor='white', label='Truth')
                except Exception as e:
                    # Fallback to scatter if KDE fails (e.g., singular matrix due to identical points)
                    ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
                    if alpha_true is not None:
                        ax.plot(alpha_true[c], alpha_true[r], 'r*', markersize=12, markeredgecolor='white')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
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
    if alpha_true is not None:
        import matplotlib.lines as mlines
        gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                  markersize=18, markeredgecolor='white', label='Ground truth')
        # Placed nicely in the upper right
        fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93), 
                   fontsize=18, frameon=True, facecolor='white', edgecolor='silver', shadow=True)

    plt.tight_layout()
    
    save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f'P{pos}_JointPosterior_22Feb_beta026.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

print("Ground truth generation finished.")
    












# %% 5. Advanced Visualization & Uncertainty Quantification
print(f"Analyzing and Plotting Position {pos}...")

# 5.1 Uncertainty Quantification
expected_z = np.sum(z_samples * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf)
std_z = np.sqrt(np.sum(np.square(z_samples - expected_z) * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf))
z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None

# 5.2 Main Posterior Plot (Pairwise Matrices)
fig, axes = plt.subplots(n_elements, n_elements, figsize=(10, 10), facecolor='white')
labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)

for r in range(n_elements):
    for c in range(n_elements):
        ax = axes[r, c]
        if r == c: # Diagonal: Physical Labels
            ax.text(0.5, 0.5, labels[r], fontsize=22, ha='center', va='center', fontweight='bold', color='#333333')
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.axis('off')
        elif r > c: # Lower Triangle: Smooth 2D Joint PDF
            xi, yi = np.mgrid[0:1:100j, 0:1:100j]
            kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
            kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            
            ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.9)
            ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.3, alpha=0.2)
            
            if z_true is not None:
                ax.plot(z_true[c], z_true[r], 'ro', markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=10)
            
            if c == 0: ax.set_ylabel(labels[r], fontsize=12)
            if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=12)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.3)
        else:
            ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
if not os.path.exists(save_dir): os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, f'P{pos}_JointPosterior.png'), dpi=500, bbox_inches='tight')
plt.show()

# 5.3 Enhanced Physical Damage Profile (Fancier Visuals)
fig, (ax_bar, ax_beam) = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

elements = np.arange(1, n_elements + 1)
color_estimate = '#4575b4' # Sophisticated Blue
color_true = '#d73027'     # Strong Red

# Top Plot: Bar chart with Uncertainty
ax_bar.bar(elements, expected_z, yerr=std_z, color=color_estimate, alpha=0.65, 
           label='Estimated Stiffness Reduction ($E[z|\mathbf{m}]$)', 
           capsize=8, error_kw={'elinewidth':2, 'capthick':2, 'ecolor': '#1a1a1a'})

if z_true is not None:
    # Step boundaries for 5 elements
    x_step = np.arange(0.5, n_elements + 1.5, 1)
    y_step = np.concatenate([z_true, [z_true[-1]]])
    ax_bar.step(x_step, y_step, where='post', color=color_true, 
                label='True Damage State ($\mathbf{z}^*$)', linestyle='--', lw=2.5, zorder=5)

ax_bar.set_ylabel('Stiffness Reduction ($z$)', fontsize=13, fontweight='medium')
ax_bar.set_ylim([0, 1.2])
ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=11)
ax_bar.grid(axis='y', alpha=0.2, linestyle='-')

# Add explanatory note for Uncertainty
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='silver')
ax_bar.text(0.02, 0.95, "Note: Error bars denote the 1$\sigma$ \ncredible interval (posterior std. dev.)", 
            transform=ax_bar.transAxes, fontsize=9, verticalalignment='top', bbox=props)

# Bottom Plot: Physical Beam Heatmap
beam_viz = expected_z.reshape(1, -1)
im = ax_beam.imshow(beam_viz, cmap='YlGnBu', aspect='auto', extent=[0.5, n_elements + 0.5, 0, 1], vmin=0, vmax=1)

# Clean up beam visualization
ax_beam.set_yticks([])
ax_beam.set_xticks(elements)
ax_beam.set_xticklabels([f'Element {k}' for k in elements], fontsize=11)
ax_beam.tick_params(axis='x', length=0)

# Add values and physical frame
for j, val in enumerate(expected_z):
    text_color = 'white' if val > 0.6 else 'black'
    ax_beam.text(j + 1, 0.5, f'{val:.2f}', ha='center', va='center', 
                 color=text_color, fontweight='bold', fontsize=11)

# Outer beam border
for spine in ax_beam.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'P{pos}_PhysicalProfile_Enhanced.png'), dpi=500, bbox_inches='tight')
plt.show()


######################################3    





# # %% 4. Likelihood Calculation (Matching Training Loss Logic)
# positions = [0, 1, 7,  9, 11, 17,25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# # positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
# # positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
# for i in range(len(positions)):
#     pos = positions[i]
#     # 1. Frequency Loss (Logarithmic)
#     obs_f = Freqs_true_test[pos] # (5,)
#     f_err = np.square(np.log(obs_f) - np.log(f_pred))
#     loss_f = np.mean(f_err, axis=1) # (16384,)
    
#     # 2. Mode Shape Loss (MAC-based)
#     obs_r = Rotmodes_true_test[pos]  # (6, 5)
#     obs_v = Vertmodes_true_test[pos] # (4, 5)
    
#     # Expand obs to match batch size
#     obs_r_batch = np.repeat(obs_r[np.newaxis, :, :], N_qmc, axis=0)
#     obs_v_batch = np.repeat(obs_v[np.newaxis, :, :], N_qmc, axis=0)
    
#     rot_mac = calculate_MAC(obs_r_batch, rot_pred)   # (16384, 5)
#     vert_mac = calculate_MAC(obs_v_batch, vert_pred) # (16384, 5)
    
#     mac_combined = np.concatenate([rot_mac, vert_mac], axis=1) # (16384, 10)
#     loss_mac = np.mean(np.square(1 - mac_combined), axis=1)     # (16384,)
    
#     # Total Data Misfit (matching ELBO logic)
#     # Note: We treat the combined loss as the negative log-likelihood scaled by beta^2
#     total_loss = loss_f + loss_mac
#     exponent = -total_loss * inv_gamma_val 
    
#     # Normalization
#     max_exp = np.max(exponent)
#     likelihood = np.exp(exponent - max_exp)
#     posterior_pdf = likelihood / np.mean(likelihood)

    
# # %% 5.1 Advanced Visualization & Uncertainty Quantification
#     print(f"Analyzing and Plotting Position {pos}...")
    
#     # 5.1 Uncertainty Quantification
#     expected_z = np.sum(z_samples * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf)
#     std_z = np.sqrt(np.sum(np.square(z_samples - expected_z) * posterior_pdf[:, np.newaxis], axis=0) / np.sum(posterior_pdf))
#     z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None

#     # 5.2 Main Posterior Plot
#     fig, axes = plt.subplots(n_elements, n_elements, figsize=(12, 12), facecolor='white')
#     labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
#     mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)

#     for r in range(n_elements):
#         for c in range(n_elements):
#             ax = axes[r, c]
#             if r == c: # Diagonal: Labels & Marginal Summary
#                 ax.text(0.5, 0.6, labels[r], fontsize=24, ha='center', va='center', fontweight='bold')
#                 if z_true is not None:
#                     ax.text(0.5, 0.3, f'True: {z_true[r]:.2f}\n$\pm${std_z[r]:.2f}', 
#                             fontsize=12, ha='center', va='center', color='red')
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.axis('off')
#             elif r > c: # Lower Triangle: Smooth 2D Joint PDF
#                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
#                 kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
#                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
#                 zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
#                 ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.9)
#                 ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.4, alpha=0.2)
                
#                 if z_true is not None:
#                     ax.plot(z_true[c], z_true[r], 'ro', markersize=8, markeredgecolor='white', zorder=10)
                
#                 if c == 0: ax.set_ylabel(labels[r], fontsize=14)
#                 if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=14)
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.grid(True, linestyle=':', alpha=0.3)
#             else:
#                 ax.axis('off')

#     plt.tight_layout()
#     save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#     plt.savefig(os.path.join(save_dir, f'P{pos}_GT_JointPosterior.png'), dpi=400, bbox_inches='tight')
#     plt.show()
#     print(f"Generating Advanced 2D Pair Plots for Position {pos}...")
#     fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
#     labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
#     z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None

#     # Threshold for speed: only use points that carry non-negligible mass
#     mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)

#     for r in range(n_elements):
#         for c in range(n_elements):
#             ax = axes[r, c]
            
#             if r == c: # --- Diagonal: Marginal Labels (No Histograms) ---
#                 ax.text(0.5, 0.5, labels[r], fontsize=22, ha='center', va='center', 
#                         fontweight='bold', color='#333333')
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.set_xticks([]); ax.set_yticks([])
#                 for spine in ax.spines.values(): spine.set_visible(False)
                
#             elif r > c: # --- Lower Triangle: Smooth 2D Joint PDF ---
#                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                
#                 # Weighted KDE evaluation for specific pair (c, r)
#                 kde_coords = np.vstack([z_samples[mask, c], z_samples[mask, r]])
#                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
#                 zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
#                 # 1. Filled Contours
#                 ax.contourf(xi, yi, zi, levels=30, cmap='viridis', alpha=0.85)
                
#                 # 2. Contour Outlines for structural clarity
#                 ax.contour(xi, yi, zi, levels=5, colors='white', linewidths=0.5, alpha=0.3)
                
#                 # 3. Mark Ground Truth Solution
#                 if z_true is not None:
#                     ax.plot(z_true[c], z_true[r], 'ro', markersize=10, 
#                             markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                
#                 # Styling
#                 if c == 0: ax.set_ylabel(labels[r], fontsize=14)
#                 if r == n_elements - 1: ax.set_xlabel(labels[c], fontsize=14)
#                 ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
#                 ax.grid(True, linestyle=':', alpha=0.4)

#             else: # --- Upper Triangle: Blank ---
#                 ax.axis('off')

#     plt.suptitle(f'Ground Truth Posterior - Position {pos}\nMetric: Log-Freq + MAC (beta={beta})', 
#                  fontsize=20, y=0.98, fontweight='bold')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save Path
#     save_dir = os.path.join("MODULES", "POSTPROCESSING", "Ground_truth_plots")
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
    
#     save_path = os.path.join(save_dir, f'GroundTruth_PairComparison_P{pos}.png')
#     plt.savefig(save_path, dpi=500, bbox_inches='tight')
#     plt.show()

#     print("Ground truth generation finished.")
    
#     # %% 5. Focused 2D Joint Visualization
#     print(f"Generating Smooth 2D Pair Plots for Scenario {pos}...")
#     fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
#     labels = [f'$z_{{{i+1}}}$' for i in range(n_elements)]
#     z_true = alpha_factors_true_test[pos] if 'alpha_factors_true_test' in locals() else None

#     mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)

#     for i in range(n_elements):
#         for j in range(n_elements):
#             ax = axes[i, j]
            
#             if i == j: # --- Diagonal: Marginal Labels (No Histograms) ---
#                 ax.text(0.5, 0.5, labels[i], fontsize=20, ha='center', va='center', fontweight='bold')
#                 ax.set_xlim([0, 1])
#                 ax.set_ylim([0, 1])
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 # Optional: keep frame or remove it
#                 for spine in ax.spines.values(): spine.set_visible(False)
                
#             elif i > j: # --- Lower Triangle: Smooth 2D Joint PDF ---
#                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
#                 kde_coords = np.vstack([z_samples[mask, j], z_samples[mask, i]])
#                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
#                 zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                
#                 ax.contourf(xi, yi, zi.reshape(xi.shape), levels=30, cmap='viridis')
                
#                 if z_true is not None:
#                     ax.plot(z_true[j], z_true[i], 'ro', markersize=9, markeredgecolor='white', label='True' if i==1 and j==0 else "")
                
#                 if j == 0: ax.set_ylabel(labels[i], fontsize=12)
#                 if i == n_elements - 1: ax.set_xlabel(labels[j], fontsize=12)
#                 ax.set_xlim([0, 1])
#                 ax.set_ylim([0, 1])

#             else: # --- Upper Triangle: Empty ---
#                 ax.axis('off')

#     plt.suptitle(f'Scenario {pos}: Ground Truth 2D Damage Posteriors (MAC & Log-Freq Likelihood)', fontsize=18, y=0.98)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     save_path = os.path.join("MODULES", "POSTPROCESSING","Ground_truth_plots",  f'GroundTruth_PairComparison_S{pos}.png')
#     plt.savefig(save_path, dpi=500, bbox_inches='tight')
#     plt.show()

# print("Ground truth generation finished.")



# import os
# import tensorflow as tf
# import tensorflow.keras as K 
# import numpy as np 
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

# from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
# from MODULES.COPULAS.GC_GMm_eigen_functions import assemble_global_Kmatrices
# from MODULES.POSTPROCESSING.QMC_sampling import QuadratureMethod

# # %% 1. Initialization and Data Loading
# K.utils.set_random_seed(1234)
# data_path = os.path.join("Data", "17MARRandomdata5elements5nmodes")
# batch_size = 48
# n_elements = 5
# n_modes = 5
# n_dofs = 2*(n_elements +1) 
# num_dofs = n_dofs -2 
# Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
# L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
# Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)

# # Assume Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test are already in memory
# # Example scaling/noise parameters
# beta = 0.5
# inv_gamma_val = 1.0 / (beta**2)

# # %% 2. Physics Step Definition (Optimized for Memory)
# def physics_engine_step(K_batch, L_inv_tf, n_modes):
#     """
#     Computes frequencies and modes for a batch. 
#     Forced to CPU to avoid CUDA/XLA 'libdevice' errors.
#     """
#     with tf.device('/CPU:0'):
#         # A = L_inv^T * K * L_inv
#         A_batch = tf.matmul(tf.transpose(L_inv_tf), tf.matmul(K_batch, L_inv_tf))
        
#         # Solve Eigenproblem
#         vals, vecs = tf.linalg.eigh(A_batch)
#         vals_trunc = tf.clip_by_value(vals[:, :n_modes], 1e-08, 1e+20)
        
#         f_Hz = tf.math.sqrt(vals_trunc) / (2 * np.pi)
#         phi = tf.matmul(tf.transpose(L_inv_tf), vecs)[:, :, :n_modes]
        
#         # Mode extraction logic
#         eig_cut = phi[:, 0:-1, :]
#         rot = tf.concat([eig_cut[:, 0::2, :], phi[:, -1:, :]], axis=1) # (Batch, 6, 5)
#         vert = eig_cut[:, 1::2, :] # (Batch, 4, 5)
        
#         return tf.concat([
#             f_Hz, 
#             tf.reshape(rot, [tf.shape(rot)[0], -1]), # 30 features
#             tf.reshape(vert, [tf.shape(vert)[0], -1]) # 20 features
#         ], axis=1)
    
# # %% 3. Creating the QMC random points (The Latent Grid)
# qm = QuadratureMethod(gdim=5)
# N_qmc = 2**14
# z_qmc_raw, _ = qm.QMC(N_qmc)

# # Scale damage parameters to [0.05, 0.95]
# z_min, z_max = 0.05, 0.95
# z_samples = z_min + (z_max - z_min) * z_qmc_raw.T 
# z_samples_tf = tf.cast(z_samples, dtype=tf.float32)

# # %% 4. Physics Forward Pass (Batched to prevent OOM)
# # Pre-build stiffness matrices for all QMC points
# Kpath = os.path.join("MODULES", "POSTPROCESSING")
# # Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples_tf, tf.cast(Ke_matrices, dtype=tf.float32))
# # Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, N_qmc)
# # np.save(os.path.join(Kpath, "Kfree_matrices_03Feb.npy"), Kfree_matrices, allow_pickle = True)
# Kfree_matrices = np.load(os.path.join(Kpath, "Kfree_matrices_03Feb.npy"), allow_pickle = True)


# # %% 5. Run Physics Forward Pass in Batches
# print(f"Running Physics Forward Pass for {N_qmc} points...")
# batch_size = 512
# all_m_hat = []
# for i in range(0, N_qmc, batch_size):
#     m_hat_batch = physics_engine_step(Kfree_matrices[i : i + batch_size], L_inv_tf, n_modes)
#     all_m_hat.append(m_hat_batch.numpy())

# m_hat_all = np.concatenate(all_m_hat, axis=0) # Shape: (16384, 55)
# print("Forward pass complete.")

# # %% 6. Posterior Calculation & Plotting for 10 Scenarios
# num_scenarios = 10
# M_total = 55 # 5 + 30 + 20

# # Diagonal precision matrix inverse(Gamma)
# inv_gamma_mat = np.ones(M_total) * inv_gamma_val

# for s in range(num_scenarios):
#     # Construct m_obs for this scenario
#     obs_f = Freqs_true_test[s]            # (5,)
#     obs_r = Rotmodes_true_test[s].flatten()  # (30,)
#     obs_v = Vertmodes_true_test[s].flatten() # (20,)
#     m_obs = np.concatenate([obs_f, obs_r, obs_v])
    
#     # Calculate Residuals (m_obs - m_hat)
#     diff = m_hat_all - m_obs # (16384, 55)
    
#     # Exponent: -0.5 * sum( residual^2 / beta^2 )
#     # This is equivalent to (m-m_hat).T @ Gamma_inv @ (m-m_hat)
#     exponent = -0.5 * np.sum(np.square(diff) * inv_gamma_mat, axis=1)
    
#     # Log-Sum-Exp trick for stability
#     max_exp = np.max(exponent)
#     likelihood = np.exp(exponent - max_exp)
    
#     # B calculation (The proportionality constant from your LaTeX)
#     # B = (1/N) * sum(likelihood)
#     B = np.mean(likelihood)
    
#     # Normalized Posterior PDF
#     posterior_pdf = likelihood / B

#     # %% 6. Plotting - All Pairs for Scenario s
#     print(f"Generating Smooth Plot for Scenario {s+1}...")
#     fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
#     labels = [f'$z_{{{i+1}}}$' for i in range(n_elements)]
    
#     # Extract true damage for markers
#     # Using alpha_factors_true_test as specified in your query
#     z_true = alpha_factors_true_test[s] if 'alpha_factors_true_test' in locals() else None

#     # Filter mask for speed and stability
#     mask = posterior_pdf > (np.max(posterior_pdf) * 0.0001)

#     for i in range(n_elements):
#         for j in range(n_elements):
#             ax = axes[i, j]
            
#             if i == j: # --- Diagonal: Marginal Distribution (KDE instead of Hist) ---
#                 kde_1d = gaussian_kde(z_samples[mask, i], weights=posterior_pdf[mask])
#                 x_grid = np.linspace(0, 1, 200)
#                 ax.plot(x_grid, kde_1d(x_grid), color='#2c7bb6', lw=2)
#                 ax.fill_between(x_grid, kde_1d(x_grid), color='#2c7bb6', alpha=0.3)
                
#                 if z_true is not None:
#                     ax.axvline(z_true[i], color='red', linestyle='--', lw=1.5, label='True')
#                 ax.set_title(labels[i], fontsize=12)
#                 ax.set_xlim([0, 1])
#                 ax.set_yticks([])
                
#             elif i > j: # --- Lower Triangle: Smooth 2D Joint PDF ---
#                 # Evaluation grid
#                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                
#                 # Gaussian KDE with weights for specific pair j and i
#                 kde_coords = np.vstack([z_samples[mask, j], z_samples[mask, i]])
#                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
                
#                 zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                
#                 # Smooth Contour
#                 ax.contourf(xi, yi, zi.reshape(xi.shape), levels=30, cmap='viridis')
                
#                 # MARK GROUND TRUTH
#                 if z_true is not None:
#                     ax.plot(z_true[j], z_true[i], 'ro', markersize=8, markeredgecolor='white', label='True' if i==1 and j==0 else "")
                
#                 if j == 0: ax.set_ylabel(labels[i], fontsize=12)
#                 if i == n_elements - 1: ax.set_xlabel(labels[j], fontsize=12)
#                 ax.set_xlim([0, 1])
#                 ax.set_ylim([0, 1])

#             else: # --- Upper Triangle: Empty ---
#                 ax.axis('off')

#     plt.suptitle(f'Scenario {s+1}: Ground Truth 5D Posterior (KDE Smoothed)', fontsize=18, y=0.98)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Save the result
#     save_path = os.path.join("MODULES", "POSTPROCESSING", f'GroundTruth_Scenario_{s+1}.png')
#     plt.savefig(save_path, dpi=500, bbox_inches='tight')
#     plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
# #     print(f"Generating Smooth Plot for Scenario {s+1}...")
# #     fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
# #     labels = [f'$z_{{{i+1}}}$' for i in range(n_elements)]
    
# #     # Extract true damage for markers (Assuming z_true_test exists)
# #     z_true = alpha_factors_true_test[s] if 'z_true_test' in locals() else None

# #     for i in range(n_elements):
# #         for j in range(n_elements):
# #             ax = axes[i, j]
            
# #             if i == j: # --- Diagonal: Marginal Distribution ---
# #                 ax.hist(z_samples[:, i], weights=posterior_pdf, bins=40, 
# #                         density=True, color='#2c7bb6', alpha=0.7, edgecolor='white')
# #                 if z_true is not None:
# #                     ax.axvline(z_true[i], color='red', linestyle='--', lw=1.5)
# #                 ax.set_title(labels[i], fontsize=12)
                
# #             elif i > j: # --- Lower Triangle: Smooth 2D Joint PDF ---
# #                 # We use a subset of points for KDE speed if N_qmc is huge, 
# #                 # but with 16k points, we can filter for points with non-negligible weight
# #                 mask = posterior_pdf > (np.max(posterior_pdf) * 0.001)
                
# #                 # Evaluation grid
# #                 xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                
# #                 # Gaussian KDE with weights
# #                 kde = gaussian_kde(z_samples[mask, j:j+2].T, weights=posterior_pdf[mask])
# #                 # Note: Above is for adjacent, we need exactly j and i
# #                 kde_coords = np.vstack([z_samples[mask, j], z_samples[mask, i]])
# #                 kde = gaussian_kde(kde_coords, weights=posterior_pdf[mask])
                
# #                 zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                
# #                 # Smooth Contour
# #                 ax.contourf(xi, yi, zi.reshape(xi.shape), levels=30, cmap='viridis')
                
# #                 if z_true is not None:
# #                     ax.plot(z_true[j], z_true[i], 'ro', markersize=6, label='True' if i==1 and j==0 else "")
                
# #                 if j == 0: ax.set_ylabel(labels[i], fontsize=12)
# #                 if i == n_elements - 1: ax.set_xlabel(labels[j], fontsize=12)
# #                 ax.set_xlim([0, 1])
# #                 ax.set_ylim([0, 1])

# #             else: # --- Upper Triangle: Empty ---
# #                 ax.axis('off')

# #     plt.suptitle(f'Scenario {s+1}: Ground Truth Posterior (KDE Smoothed)', fontsize=18, y=0.98)
# #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# #     plt.savefig(os.path.join("MODULES", "POSTPROCESSING", 'Prueba03Feb_KDE'+str(s+1)+'.png'),dpi = 500, bbox_inches='tight')
# #     plt.show()

# # print("Processing complete.")





#     # # We will plot a subset or a corner-plot style grid
#     # fig, axes = plt.subplots(n_elements, n_elements, figsize=(15, 15))
#     # for i in range(n_elements):
#     #     for j in range(n_elements):
#     #         ax = axes[i, j]
#     #         if i == j: # Marginal
#     #             ax.hist(z_samples[:, i], weights=posterior_pdf, bins=30, color='skyblue', density=True)
#     #             ax.set_title(f'Element {i+1}')
#     #         elif i > j: # Pairwise Joint PDF
#     #             ax.tricontourf(z_samples[:, j], z_samples[:, i], posterior_pdf, levels=50, cmap='viridis')
#     #             ax.set_xlabel(f'$z_{j+1}$')
#     #             ax.set_ylabel(f'$z_{i+1}$')
#     #         else:
#     #             ax.axis('off')
    
#     # plt.suptitle(f'Scenario {s+1}: 5D Ground Truth Posterior (M={M_total} features)', fontsize=16)
#     # plt.tight_layout(rect=[0, 0, 1, 0.97])
#     # plt.show()




# # num_scenarios = 2

# # # Ensure your test data is available (Freqs_true_test, etc.)
# # for s in range(num_scenarios):
# #     # Construct Observation Vector
# #     obs_f = tf.cast(Freqs_true_test[s], tf.float32)
# #     obs_r = tf.cast(tf.reshape(Rotmodes_true_test[s], [-1]), tf.float32)
# #     obs_v = tf.cast(tf.reshape(Vertmodes_true_test[s], [-1]), tf.float32)
# #     m_obs = tf.concat([obs_f, obs_r, obs_v], axis=0)
    
# #     # Likelihood: exp(-0.5 * sum((obs - pred)^2 * inv_gamma))
# #     diff = m_obs - m_hat_all
# #     exponent = -0.5 * np.sum(np.square(diff) * inv_gamma, axis=1)
    
# #     # Stability trick: Subtract max exponent before exp()
# #     likelihood = np.exp(exponent - np.max(exponent))
    
# #     # Normalize (Constant B)
# #     B = np.mean(likelihood)
# #     posterior_pdf = likelihood / B

# #     # %% 7. Visualization
# #     plt.figure(figsize=(6, 5))
# #     # Plotting z1 vs z2 (elements 0 and 1)
# #     # Using tricontourf because QMC points are a cloud
# #     cntr = plt.tricontourf(z_samples[:, 1], z_samples[:, 3], posterior_pdf, levels=50, cmap='viridis')
# #     plt.colorbar(cntr, label='Posterior Density')
# #     plt.xlabel('$z_1$ (Element 1)')
# #     plt.ylabel('$z_2$ (Element 2)')
# #     plt.title(f'Ground Truth PDF - Scenario {s+1}')
# #     plt.tight_layout()
# #     plt.show()

# # print("All 10 scenarios processed successfully.")