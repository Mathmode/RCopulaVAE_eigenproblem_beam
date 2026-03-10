# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:38:01 2026

@author: anafd
"""

# -*- coding: utf-8 -*-
"""
Updated Plotting Utilities for GC-VAE
Aligned with Ground Truth KDE logic for matching results.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow as tf

# Disable JIT globally via the TensorFlow API as a secondary safety measure
tf.config.optimizer.set_jit(False)


import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd 
import matplotlib.lines as mlines

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

    
def plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
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
    Kfree_matrices = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, n_samples, model.fixed_dofs_indices)
    
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
    quantile_threshold = 0.85 # Slightly wider for VAE samples
    sorted_indices = np.argsort(posterior_weights)[::-1]
    cumulative_weights = np.cumsum(posterior_weights[sorted_indices])
    mask_indices = sorted_indices[cumulative_weights <= quantile_threshold]
    
    if len(mask_indices) < 30: mask_indices = sorted_indices[:100]

    x_filtered = z_samples.numpy()[mask_indices]
    w_filtered = posterior_weights[mask_indices]
    w_filtered /= np.sum(w_filtered)
    # w_filtered /= w_filtered
    

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
    plt.savefig(os.path.join(folder_path, "Matched_Posteriors", f'Borrar_Sample_{pos}_Matched.png'), dpi=150)
    plt.show()
    plt.close()
    

    

import tensorflow.keras as K
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow_probability as tfp
from scipy.stats import norm
# def main():
K.backend.set_floatx('float32') 
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.COPULAS.GC_plot_posteriors import calculate_posterior_PDF_info, plot_physical_damage_profile

# --- Config ---
filename = "time12MarOLDARCH_Nosiy2.5Mild50_0.45lbound_Bayesian_MACloss_Beta0.2_Samples1_LR1e-05_Epochs7"
folder_path = os.path.join('Output', 'Gaussian_Copula', filename)
lbound = 0.45
# Load Problem Info
info_path = os.path.join(folder_path, 'Problem_info.npy')
if not os.path.exists(info_path):
    raise FileNotFoundError(f"Problem_info.npy not found at {info_path}")

info = np.load(info_path, allow_pickle=True).item()

# Extract parameters
input_dim = info['input_dim_enc']
n_dims = info['n_dims']
beta = info['beta']
mean_freq = info['mean_f']
std_freq = info['std_f']
n_elements = 5
n_dofs = 2 * (n_elements + 1)
batch_size = 256

# 2. Identify fixed indices (Simply Supported)
# Node 0 vertical is index 0; Node N vertical is index 10
fixed_dofs = [0, n_dofs - 2] 
# 3. Create the list of free DOFs
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)



history_ = np.load(os.path.join(folder_path, "model_history.npy"),allow_pickle = True)
def plot_trainval_loss(history, folder_path):
    """
    Plots the training and validation evolution for the GC-VAE.
    Uses Matplotlib's internal mathtext for labels to avoid external LaTeX dependencies.
    """
    # Use a style suitable for academic publishing without requiring external LaTeX
    plt.rcParams.update({
        "text.usetex": False, # Set to False to resolve 'latex could not be found' error
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "cm" # Use Computer Modern for a LaTeX-like look
    })
    
    trainvalloss_plot_fig = plt.figure(figsize=(8, 6))
    
    # Axis scales
    plt.yscale('log')
    plt.xscale('log')
    
    # Plotting data
    plt.plot(history.item()['loss'], 
             color='black', 
             linewidth=1.5, 
             linestyle='--', 
             label=r'Train loss ($\mathcal{L}_{\mathrm{ELBO}}^{\text{train}}$)')
    
    plt.plot(history.item()['val_loss'], 
             color='darkgray', 
             linewidth=1.5, 
             label=r'Validation loss ($\mathcal{L}_{\mathrm{ELBO}}^{\text{val}}$)')
    
    # Labels and Legend
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{ELBO}}$ (log scale)', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Grid and layout
    plt.grid(True, which="both", linestyle='-.', alpha=0.5)
    plt.tight_layout()
    
    # Save with high resolution
    trainvalloss_plot_fig.savefig(
        os.path.join(folder_path, 'TrainVal_lossevolution.png'),
        dpi=500, 
        bbox_inches='tight'
    )
    # plt.close()
plot_trainval_loss(history_, folder_path)
   


# %% 1. Initialization and Data Loading
K.utils.set_random_seed(1234)
# data_path = os.path.join("Data", "11Feb2026_Corrected_Randomdata5elements")
# data_path = os.path.join("Data", "28Feb2026_MildDam50_Randomdata5elements")
data_path = os.path.join("Data", "01Mar2026_Noisy_E5_level25")

print(f"Loading data from {data_path}...")

(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
 Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
 Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
 mean_f, std_f) = load_data(data_path, batch_size)

# Load Physics Matrices
Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

# Fix input dim calculation based on your previous messages
n_modes = Freqs_true_train.shape[1]

Test_pred_props = np.load(os.path.join(folder_path, "Test_predicted_props.npy"),allow_pickle= True).item()
test_means, test_scales, test_offdiag_elems, test_diag_elems, test_weight_vals  = Test_pred_props['test_means'],Test_pred_props['test_scales'], Test_pred_props['test_offdiag_elems'],Test_pred_props['test_diag_elems'],Test_pred_props['test_weight_vals'],
test_datasets = {
    'Freqs_true_test': Freqs_true_test,
    'Rotmodes_true_test': Rotmodes_true_test,
    'Vertmodes_true_test':Vertmodes_true_test,
    'alpha_factors_true_test': alpha_factors_true_test
}

# Build L matrices
L_matrices = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems, n_dims)

predicted_stats = {
    'test_means': test_means,
    'test_scales': test_scales,
    'test_weights':test_weight_vals,
    'test_L_matrices': L_matrices
}


print("\n--- Calculating Global Performance Metrics ---")

# Execute metric calculation
metrics_results, coverage_mask = calculate_testing_metrics(predicted_stats, test_datasets, lbound=lbound)

# Display Results
print("-" * 30)
for metric, value in metrics_results.items():
    print(f"{metric:25}: {value:.6f}")
print("-" * 30)

# Save to CSV for reporting
metrics_df = pd.DataFrame([metrics_results])
metrics_save_path = os.path.join(folder_path, "testing_performance_metrics.csv")
metrics_df.to_csv(metrics_save_path, index=False)
print(f"Metrics saved to: {metrics_save_path}")

# # Optional: Element-wise Analysis (Which elements are hardest to predict?)
# element_mae = np.mean(np.abs(predicted_stats['test_means'] - test_datasets['alpha_factors_true_test']), axis=0)
# element_coverage = np.mean(coverage_mask, axis=0)

# elem_report = pd.DataFrame({
#     'Element': [f'z{i+1}' for i in range(n_elements)],
#     'MAE': element_mae,
#     'Coverage': element_coverage
# })
# print("\nElement-wise breakdown:")
# print(elem_report.to_string(index=False))

# # Save element-wise report
# elem_report.to_csv(os.path.join(folder_path, "element_wise_metrics.csv"), index=False)



# # --- Visualization Loop 1---
# positions = [0, 1, 7, 9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# # positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
# # positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
# # positions = [1,48,63,77,219,300,313,314,612,1000]
# # positions  = [1,17,25,41,43,48,77,78,219,313,315, 63, 246,383 ,455,459, 1000,1182,1396,1489]
# # positions = [ 815,  723, 1318, 1077, 1228, 1396,  664, 1679,  689,  279, 1257,
# #        1178,   30, 1707, 1182, 1772, 1398,  442,  120, 1500, 1349, 1360,
# #         969,  383,  246,  510, 1455, 1586, 1776, 1787, 1100,  293, 1530,
# #        1219,  743, 1163,  640,  745,  336,    3, 1282, 1299,  908,  459,
# #         371, 1643, 1489, 1038, 1267,  455]
# # 
# n_samples = 4096
# N_test_samples = len(Freqs_true_test)

# for pos in positions:
#     if pos < N_test_samples:
#         # plot_vae_copula_pdf(pos, predicted_stats, test_datasets, lbound, folder_path)
        
#         plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
#                                          predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path)
        
        
#         # z_true, z_samples, posterior_weights = calculate_posterior_PDF_info(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
#         #                                  predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path)

#         # plot_physical_damage_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path)
        










########################################## loading the trained model 

# # %% 2. Model Re-instantiation and Loading Weights
# print("Initializing model...")
# # Force CPU context for initialization
# with tf.device('/CPU:0'):    
#     model = My_CopulaVAE_withEigen(
#         input_dim=input_dim,
#         num_dofs=n_dofs,
#         n_elements=n_elements,
#         n_modes=n_modes,
#         Ke_matrices=Ke_matrices,
#         Mfree=Mfree,
#         L_inv=L_inv,
#         epsi=0.0,
#         n_dims=n_elements,
#         num_gaussians=1,
#         num_samples=1,
#         beta=beta,
#         mean_f=mean_f,
#         std_f=std_f,
#         lbound = lbound,
#         fixed_dofs_indices=[0, n_dofs-2]
#     )

    # # MANDATORY: We must build the model before loading HDF5 weights.
    # # We do this by passing a dummy batch through the model.
    # print("Building model variables...")
    # dummy_f = tf.zeros((1, n_modes))
    # dummy_r = tf.zeros((1, n_modes, 6))
    # dummy_v = tf.zeros((1, n_modes, 6))
    # dummy_z = tf.zeros((1, n_dims))
    
    # # This call creates the internal weights/variables
    # _ = model([dummy_f, dummy_r, dummy_v, dummy_z])

# # --- UPDATED WEIGHT LOADING ---
# weights_path = os.path.join(folder_path, "final_model_weights.weights.h5")
# if os.path.exists(weights_path):
#     print("Loading weights...")
#     # Since you saved using model.save_weights(path), we should load into the full model object.
#     target_layer = model.Encoder_model.FC_encoder
    
#     # Checksum of initialized weights in the specific encoder
#     pre_sum = np.sum([np.sum(w) for w in target_layer.get_weights()])
#     print(f"Checksum before load: {pre_sum:.6f}")
#     # model.load_weights(weights_path, by_name=True)


#     try:
#         # Strategy 1: Load into the whole model (matching the save method)
#         model.load_weights(weights_path, by_name=True)
#         print("Standard load into full model executed (by_name=True).")
#     except Exception as e:
#         print(f"Full model load failed: {e}. Trying direct load into FC_encoder...")
#         try:
#             target_layer.load_weights(weights_path, by_name=True, skip_mismatch=True)
#             print("Direct load into FC_encoder executed.")
#         except Exception as e2:
#             print(f"Direct load also failed: {e2}")

#     post_sum = np.sum([np.sum(w) for w in target_layer.get_weights()])
#     print(f"Checksum after load:  {post_sum:.6f}")

#     if np.isclose(pre_sum, post_sum):
#         print("!!! ERROR: Weights did not change.")
#         print("The weight file likely contains names that do not match the current session's layer names.")
#     else:
#         print("SUCCESS: Weight verification passed.")
# else:
#     print(f"CRITICAL: Weights file not found at {weights_path}")

# # %% --- DIRECT PREDICTION PATH ---
# print("\n--- Running Direct Encoder Prediction ---")
# # Manually flatten modal data to match FC_encoder input exactly
# flat_rot = tf.reshape(Rotmodes_true_test, [Rotmodes_true_test.shape[0], -1])
# flat_vert = tf.reshape(Vertmodes_true_test, [Vertmodes_true_test.shape[0], -1])
# # Join: [Frequencies, Vertical Modes, Rotational Modes]
# modal_input = tf.concat([Freqs_true_test, flat_vert, flat_rot], axis=1)

# # Predict directly from the FC_encoder which now has verified weights
# raw_params = target_layer.predict(modal_input, batch_size=batch_size)

# # Manual Unpacking based on Encoder Architecture
# n_gauss = 1
# n_correlations = n_dims * (n_dims - 1) // 2
# split_indices = [
#     n_dims * n_gauss,  # means
#     n_dims * n_gauss,  # sigmas
#     n_dims * n_gauss,  # weights
#     n_correlations,    # off_diag
#     n_dims             # diag
# ]

# splits = np.split(raw_params, np.cumsum(split_indices)[:-1], axis=1)

# # Apply Post-Architecture Scaling (Replicating logic from Fully_connected_enc_GC)
# t_means_raw = splits[0]
# t_means = (lbound + 0.001) + (0.998 - lbound) * t_means_raw

# t_sigmas_raw = splits[1]
# t_scales = 1e-6 + 0.999 * t_sigmas_raw

# t_weights = splits[2] 
# t_offdiag = splits[3]
# t_diag = splits[4] + 1e-6

# # SANITY CHECK: Verify variation across test set
# unique_vals = len(np.unique(np.round(t_means[:, 0], 5)))
# if unique_vals <= 1:
#     stuck_val = t_means.flat[0].item()
#     print(f"!!! ALERT: test_means are still stuck at {stuck_val:.4f} !!!")
# else:
#     print(f"SUCCESS: test_means are varying correctly. Found {unique_vals} unique values in z1.")
#     print(f"Range of z1: [{np.min(t_means[:,0]):.4f}, {np.max(t_means[:,0]):.4f}]")



   
# # --- Prediction Step ---
# print("Running Encoder Prediction...")
# inverse_model = model.Encoder_model
# # Note: inputs are passed as a list
# test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
#     [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
# )

