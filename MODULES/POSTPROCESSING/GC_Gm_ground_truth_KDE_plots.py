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
import seaborn as sns

# Keep your local imports intact
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_eigen_functions import assemble_global_Kmatrices
from MODULES.POSTPROCESSING.QMC_sampling import QuadratureMethod

# Setup plotting style
sns.set(style="whitegrid", rc={"axes.facecolor": "#f0f0f0", "grid.color": "gray", "grid.linestyle": "--"})
sns.set(style="dark")

# %% 1. Initialization and Data Loading
K.utils.set_random_seed(1234)

# Define path (adjust if necessary)
data_path = os.path.join("Data", "01Mar2026_Noisy_E5_level25")

batch_size = 256
(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
 Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
 Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
 mean_f, std_f) = load_data(data_path, batch_size)

n_elements = 5
n_modes = 5
n_dofs = 2 * (n_elements + 1)  # 12 total DOFs
num_dofs_total = n_dofs
num_dofs = n_dofs - 2          # 10 free DOFs

# 2. Identify fixed indices (Simply Supported)
# Node 0 vertical is index 0; Node N vertical is index 10
fixed_dofs = [0, n_dofs - 2] 
# 3. Create the list of free DOFs
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)


Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
L_inv_tf = tf.cast(L_inv, dtype=tf.float32)

# Scaling/noise parameters
beta = 0.25
inv_gamma_val = 1.0 / (beta**2)
lbound = 0.45  # Note: ensure this matches the paper's damage bounds bounds


# %% 2. Physics Engine & MAC Calculation
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

def physics_engine_step(K_batch, L_inv_tf, n_modes, free_dofs, n_dofs):
    """
    K_batch: (Batch, Free_DOF, Free_DOF)
    L_inv_tf: (Free_DOF, Free_DOF) - L^-1 where M_free = L @ L.T
    n_modes: Number of modes to truncate
    free_dofs: List/Array of indices of free degrees of freedom
    n_dofs: Total number of DOFs (2 * n_nodes)
    
    Outpus: solves the eigenvalue problem and produces the natural frquencies and mode shapes
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
    
    

# %% 3. QMC Grid & Physics Pass
saving_path = os.path.join("Output", "Ground_truth_plots", "Noisy_GT_plots", "Beta025")
if not os.path.exists(saving_path): os.makedirs(saving_path)

qm = QuadratureMethod(gdim=5)
N_qmc = 2**12 # 8192 samples
z_qmc_raw, _ = qm.QMC(N_qmc)
z_samples = lbound + (1.0 - lbound) * z_qmc_raw.T 
z_samples_tf = tf.cast(z_samples, dtype=tf.float32)
np.save(os.path.join(saving_path, "Z_pointcloud_samples_03Mar_noisy25.npy"), z_samples_tf, allow_pickle=True)

print("Assembling Global Stiffness...")
Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', z_samples_tf, tf.cast(Ke_matrices, dtype=tf.float32))
Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, N_qmc)
np.save(os.path.join(saving_path, "Kfree_matrices_02Mar_25noisy.npy"), Kfree_matrices, allow_pickle=True)

print(f"Forward Pass for {N_qmc} points...")
batch_size_fwd = 512
all_f, all_rot, all_vert = [], [], []

for i in range(0, N_qmc, batch_size_fwd):
    f, r, v = physics_engine_step(Kfree_matrices[i : i + batch_size_fwd], L_inv_tf, n_modes, free_dofs, n_dofs)
    all_f.append(f)
    all_rot.append(r)
    all_vert.append(v)

f_pred = np.concatenate(all_f, axis=0)       # (N_qmc, 5)
rot_pred = np.concatenate(all_rot, axis=0)   # (N_qmc, 5, 6)
vert_pred = np.concatenate(all_vert, axis=0) # (N_qmc, 5, 4)

# # Apply padding to vertical predictions
# paddings = tf.constant([[0, 0], [0, 0], [1, 1]])
# vert_pred = tf.pad(vert_pred, paddings, mode="CONSTANT", constant_values=0)

# %% 4. Likelihood Calculation
# positions = [0, 1, 7,  9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
# positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
# positions = [0,1,48,63,77,219,300,313,612,1000]
positions  = [1,17,77,78,219,313,315, 63, 246,383 ,455,459, 1000,1182,1396,1489]
# positions = [ 815,  723, 1318, 1077, 1228, 1396,  664, 1679,  689,  279, 1257,
#        1178,   30, 1707, 1182, 1772, 1398,  442,  120, 1500, 1349, 1360,
#         969,  383,  246,  510, 1455, 1586, 1776, 1787, 1100,  293, 1530,
#        1219,  743, 1163,  640,  745,  336,    3, 1282, 1299,  908,  459,
#         371, 1643, 1489, 1038, 1267,  455]
# positions = [219]
for pos in positions:
    # 1. Frequency Loss
    obs_f_scaled = Freqs_true_test[pos]
    pred_log = np.log(f_pred)
    pred_logscaled = (pred_log - mean_f) / std_f
    f_err = np.square(obs_f_scaled - pred_logscaled)
    loss_f = np.mean(f_err, axis=1) # (16384,)
    
    # 2. Mode Shape Loss (MAC-based)
    obs_r = Rotmodes_true_test[pos]  
    obs_v = Vertmodes_true_test[pos] 
    
    if obs_r.shape[0] != n_modes: obs_r = obs_r.T
    if obs_v.shape[0] != n_modes: obs_v = obs_v.T
    
    # Expand obs to match batch size
    obs_r_batch = np.repeat(obs_r[np.newaxis, :, :], N_qmc, axis=0)
    obs_v_batch = np.repeat(obs_v[np.newaxis, :, :], N_qmc, axis=0)
    
    # MAC calculation returns batched diagonal (16384, 5)
    rot_mac = calculate_MAC(obs_r_batch, rot_pred)   
    vert_mac = calculate_MAC(obs_v_batch, vert_pred) 
    
    # Equal contribution from rot and vert modes error
    loss_mac = np.mean((1.0 - rot_mac) + (1.0 - vert_mac), axis=1)
    
    # 3. Total Data Misfit & Likelihood Evaluation (Bayesian Updates)
    total_loss = loss_f + loss_mac
    exponent = -total_loss * inv_gamma_val 
    
    # Normalization (Log-Sum-Exp Trick for numerical stability)
    max_exp = np.max(exponent)
    likelihood = np.exp(exponent - max_exp)
    posterior_pdf = likelihood / np.mean(likelihood)
    
    
    quantile_threshold  = 0.95
    
    alpha_true = alpha_factors_true_test[pos, :] if alpha_factors_true_test is not None else None
    
    # --- 1. Filter by Likelihood/Posterior Threshold ---
    sorted_indices = np.argsort(posterior_pdf)[::-1]
    sorted_weights = posterior_pdf[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= cumulative_weights[-1]
    
    mass_mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mass_mask_indices) < 20:
        threshold = np.max(posterior_pdf) * 1e-4
        mask = posterior_pdf > threshold
    else:
        mask = np.zeros(len(posterior_pdf), dtype=bool)
        mask[mass_mask_indices] = True

    x_filtered = z_samples[mask]
    w_filtered = posterior_pdf[mask]
    w_filtered /= np.sum(w_filtered) 
    
    # --- 2. Standardized Grid Setup ---
    # We use fixed limits [lbound, 1.0] to ensure all subplots have identical dimensions
    # and match the reference figure scale perfectly.
    fixed_min = lbound
    fixed_max = 1.0
    
    fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    cf = None  
    
    for r in range(n_elements):
        for c in range(n_elements):
            ax = axes[r, c]
            
            if r == c:  # Diagonal: 1D Marginal Distribution
                try:
                    vals = x_filtered[:, r]
                    kde1d = gaussian_kde(vals, weights=w_filtered)
                    x_grid = np.linspace(fixed_min, fixed_max, 200)
                    y_grid = kde1d(x_grid)
                    
                    ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
                    ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                    
                    # Centered Label
                    ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', 
                            fontweight='bold', transform=ax.transAxes)
                    
                    ax.set_xlim(fixed_min, fixed_max)
                    ax.set_yticks([]) 
                except Exception:
                    ax.text(0.5, 0.5, "KDE Fail", ha='center')
            
            elif r > c:  # Lower Triangle: 2D Joint Distribution
                x_vals = x_filtered[:, c]
                y_vals = x_filtered[:, r]
                
                try:
                    kde_coords = np.vstack([x_vals, y_vals])
                    kernel = gaussian_kde(kde_coords, weights=w_filtered)
                    
                    # Use fixed grid ranges
                    xi, yi = np.mgrid[fixed_min:fixed_max:60j, fixed_min:fixed_max:60j]
                    coords = np.vstack([xi.flatten(), yi.flatten()])
                    zi = kernel(coords).reshape(xi.shape)
                    
                    cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                    
                    if alpha_true is not None:
                        ax.plot(alpha_true[c], alpha_true[r], color='red', marker='*',
                                markersize=14, markeredgecolor='white', label='Truth')
                except Exception:
                    ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
                    if alpha_true is not None:
                        ax.plot(alpha_true[c], alpha_true[r], 'r*', markersize=18, markeredgecolor='white')
                
                ax.set_xlim(fixed_min, fixed_max)
                ax.set_ylim(fixed_min, fixed_max)
                ax.grid(True, linestyle=':', alpha=0.3)
            
            else:  # Upper Triangle
                ax.axis('off')
            
            # --- 3. Axis Formatting ---
            if r >= c:
                ax.tick_params(labelsize=18)
                
                # Y-axis labels: Left column
                if c == 0 and r != 0:
                    ax.set_ylabel(labels[r], fontsize=18)
                else:
                    ax.tick_params(labelleft=False)
                
                # X-axis labels: Bottom row
                if r == n_elements - 1:
                    ax.set_xlabel(labels[c], fontsize=18)
                else:
                    ax.tick_params(labelbottom=False)

    # Global Colorbar - Using constrained layout principles to maintain plot sizes
    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
        cbar.set_label('Posterior density', fontsize=18)
    
    # Global Legend
    if alpha_true is not None:
        import matplotlib.lines as mlines
        gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                  markersize=18, markeredgecolor='white', label='Ground truth')
        fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93),
                   fontsize=18, frameon=True, facecolor='white', edgecolor='silver', shadow=True)
    
    # Strict layout control to ensure 1:1 aspect and matching dimensions
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    save_path = os.path.join(saving_path, f'PRUEBA_GT{pos}_JointPosterior_Matched_beta{beta}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    # alpha_true = alpha_factors_true_test[pos, :] if alpha_factors_true_test is not None else None
    
    # # --- 1. Filter by Likelihood/Posterior Threshold ---
    # # Sort weights to find the cutoff for the top quantile of probability mass
    # sorted_indices = np.argsort(posterior_pdf)[::-1]
    # sorted_weights = posterior_pdf[sorted_indices]
    # cumulative_weights = np.cumsum(sorted_weights)
    # cumulative_weights /= cumulative_weights[-1]
    
    # # Identify samples that fall within the desired high-probability mass
    # mass_mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    # # Stability fallback: if quantile is too aggressive, ensure we have enough points
    # if len(mass_mask_indices) < 20:
    #     # Fall back to a simple relative threshold if mass filtering is too tight
    #     threshold = np.max(posterior_pdf) * 1e-4
    #     mask = posterior_pdf > threshold
    # else:
    #     mask = np.zeros(len(posterior_pdf), dtype=bool)
    #     mask[mass_mask_indices] = True

    # x_filtered = z_samples[mask]
    # w_filtered = posterior_pdf[mask]
    # w_filtered /= np.sum(w_filtered)  # Re-normalize weights for KDE
    
    # # --- 2. Setup Figure ---
    # fig, axes = plt.subplots(n_elements, n_elements, figsize=(14, 14), facecolor='white')
    # labels = [f'$z_{{{k+1}}}$' for k in range(n_elements)]
    # cf = None  
    
    # for r in range(n_elements):
    #     for c in range(n_elements):
    #         ax = axes[r, c]
            
    #         if r == c:  # Diagonal: 1D Marginal Distribution
    #             try:
    #                 vals = x_filtered[:, r]
    #                 kde1d = gaussian_kde(vals, weights=w_filtered)
    #                 # Use actual data range for the grid to keep it tight, or lbound/1.0
    #                 x_grid = np.linspace(lbound, 1.0, 200)
    #                 y_grid = kde1d(x_grid)
                    
    #                 ax.fill_between(x_grid, y_grid, color='steelblue', alpha=0.4)
    #                 ax.plot(x_grid, y_grid, color='steelblue', lw=2)
                    
    #                 # Label inside the plot
    #                 ax.text(0.5, 0.3, labels[r], fontsize=22, ha='center', va='center', 
    #                         fontweight='bold', transform=ax.transAxes)
                    
    #                 ax.set_xlim(lbound, 1.0)
    #                 ax.set_yticks([]) 
    #             except Exception:
    #                 ax.text(0.5, 0.5, "KDE Fail", ha='center')
            
    #         elif r > c:  # Lower Triangle: 2D Joint Distribution
    #             x_vals = x_filtered[:, c]
    #             y_vals = x_filtered[:, r]
                
    #             try:
    #                 kde_coords = np.vstack([x_vals, y_vals])
    #                 kernel = gaussian_kde(kde_coords, weights=w_filtered)
                    
    #                 # Grid restricted to the bound range
    #                 xi, yi = np.mgrid[lbound:1.0:60j, lbound:1.0:60j]
    #                 coords = np.vstack([xi.flatten(), yi.flatten()])
    #                 zi = kernel(coords).reshape(xi.shape)
                    
    #                 # Normalizing contour levels to the density in the filtered region
    #                 cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                    
    #                 if alpha_true is not None:
    #                     ax.plot(alpha_true[c], alpha_true[r], color='red', marker='*',
    #                             markersize=14, markeredgecolor='white', label='Truth')
    #             except Exception:
    #                 ax.scatter(x_vals, y_vals, c=w_filtered, s=2, cmap='viridis', alpha=0.3)
    #                 if alpha_true is not None:
    #                     ax.plot(alpha_true[c], alpha_true[r], 'r*', markersize=18, markeredgecolor='white')
                
    #             ax.set_xlim(lbound, 1.0)
    #             ax.set_ylim(lbound, 1.0)
    #             ax.grid(True, linestyle=':', alpha=0.3)
            
    #         else:  # Upper Triangle
    #             ax.axis('off')
            
    #         # --- 3. Axis Formatting ---
    #         if r >= c:
    #             ax.tick_params(labelsize=18)
                
    #             # Y-axis labels: Left column, skip z1
    #             if c == 0 and r != 0:
    #                 ax.set_ylabel(labels[r], fontsize=18)
    #             else:
    #                 ax.tick_params(labelleft=False)
                
    #             # X-axis labels: Bottom row, skip last (the logic you requested earlier)
    #             if r == n_elements - 1 and c < n_elements - 1:
    #                 ax.set_xlabel(labels[c], fontsize=18)
    #             else:
    #                 ax.tick_params(labelbottom=False)

    # # Colorbar
    # if cf is not None:
    #     cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, pad=0.06, anchor=(0.6, 1.3))
    #     cbar.set_label('Posterior density (Filtered)', fontsize=18)
    
    # # Legend
    # if alpha_true is not None:
    #     import matplotlib.lines as mlines
    #     gt_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
    #                               markersize=18, markeredgecolor='white', label='Ground truth')
    #     fig.legend(handles=[gt_marker], loc='upper right', bbox_to_anchor=(0.82, 0.93),
    #                fontsize=18, frameon=True, facecolor='white', edgecolor='silver', shadow=True)
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # save_path = os.path.join(saving_path, f'GT{pos}_JointPosterior_Q{quantile_threshold}_beta{beta}.png')
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.close()
    
   
    