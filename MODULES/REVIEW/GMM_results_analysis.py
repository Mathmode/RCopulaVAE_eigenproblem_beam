# -*- coding: utf-8 -*-
"""
Main Results Analysis for Gaussian Mixture Model (GMM) VAE
Adapted for Full Covariance Output Keys.
"""
import os
import numpy as np
import tensorflow as tf
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.REVIEW.GMM_uncertainty_quantification import calculate_and_plot_gmm_calibration, enhanced_metrics_comparison 
from MODULES.POSTPROCESSING.plot_losses import plot_trainval_loss, plot_freqsMACs_loss
from MODULES.REVIEW.GMM_functions_for_results_analysis import calculate_gmm_metrics, plot_results_PDF_uncertainty

# --- Config ---
filename = "Full07Sept26_GMM_10Els_2Mix_Gamma0.4_LR0.0001_10000epoch" # Ensure this matches your training output
model_type_name = 'GMM_VAE'
folder_path = os.path.join('Output', model_type_name, filename)
lbound = 0.45

# 1. Load Problem Info
info_path = os.path.join(folder_path, 'Problem_info.npy')
info = np.load(info_path, allow_pickle=True).item()

n_dims = info['n_dims']
num_components = info['num_components']
mean_freq = info['mean_f']
std_freq = info['std_f']
gamma = info['gamma']

n_elements = n_dims 
n_dofs = 2 * (n_elements + 1)
fixed_dofs = [0, n_dofs - 2] 
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)
batch_size = 256

# 1.2 Plot the loss functions
try:
    history_ = np.load(os.path.join(folder_path, "model_history.npy"), allow_pickle=True).item()
    plot_trainval_loss(history_, folder_path)
    plot_freqsMACs_loss(history_, folder_path)
except Exception as e:
    print(f"Loss Plotting failed: {e}")


#1.2. plot the loss functions
from MODULES.POSTPROCESSING.plot_losses import plot_trainval_loss, plot_freqsMACs_loss
history_ = np.load(os.path.join(folder_path, "model_history.npy"),allow_pickle = True)
plot_trainval_loss(history_, folder_path)
plot_freqsMACs_loss(history_, folder_path)

# 2. Load Test Results (Updated keys for Full Covariance architecture)
Test_pred_props = np.load(os.path.join(folder_path, "Test_predicted_props.npy"), allow_pickle=True).item()

predicted_stats = {
    'test_logits': Test_pred_props['test_logits'],
    'test_locs': Test_pred_props['test_locs'],
    'test_offdiag': Test_pred_props['test_offdiag'],
    'test_diag': Test_pred_props['test_diag']
}

# 3. Load Physics and Ground Truth Data
data_folder = "06Sept2026_Noisy_E50_level25_50modes" 
data_path = os.path.join("Data", data_folder)

(Freqs_train, _, _, _, _, _, _, _, Freqs_test, Rot_test, Vert_test, Alphas_test, _, _) = load_data(data_path, batch_size)
Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
n_modes = Freqs_test.shape[1]

test_datasets = {
    'Freqs_true_test': Freqs_test,
    'Rotmodes_true_test': Rot_test,
    'Vertmodes_true_test': Vert_test,
    'alpha_factors_true_test': Alphas_test
}

# Add physics configurations for Table 4 ELBO & Probabilistic evaluations
physics_kwargs = {
    'Ke_matrices': Ke_matrices,
    'L_inv': L_inv,
    'n_modes': n_modes,
    'free_dofs': free_dofs,
    'fixed_dofs': fixed_dofs,
    'n_dofs': n_dofs,
    'mean_freq': mean_freq,
    'std_freq': std_freq,
    'gamma': gamma
}



metrics, covered = calculate_gmm_metrics(predicted_stats, test_datasets, lbound, n_mc_samples=5000, physics_kwargs=physics_kwargs)
for key, value in metrics.items():
    print(f"{key}: {value:.5f}")
    
with open(os.path.join(folder_path, 'GMM_Metrics.txt'), 'w') as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
        
        
# # --- Analysis Execution ---
# print("\n--- GMM-VAE Performance Metrics ---")
# metrics = calculate_gmm_metrics(predicted_stats, test_datasets, lbound, physics_kwargs=physics_kwargs)
# print("Metrics summary:")
# for k, v in metrics.items():
#     print(f"  {k}: {v}")

# print("\n--- Computing Enhanced UQ Metrics ---")
# enhanced_metrics = enhanced_metrics_comparison(predicted_stats, test_datasets, lbound)
# for k, v in enhanced_metrics.items():
#     print(f"{k}: {v}")

# --- Visualizing Specific Samples ---
positions = [ 815,  723, 1318, 1077, 1228, 1396,  664, 1679,  689,  279, 1257,
       1178,   30, 1707, 1182, 1772, 1398,  442,  120, 1500, 1349, 1360,
        969,  383,  246,  510, 1455, 1586, 1776, 1787, 1100,  293, 1530,
       1219,  743, 1163,  640,  745,  336,    3, 1282, 1299,  908,  459,
        371, 1643, 1489, 1038, 1267,  455]

n_samples = 4096

# for pos in positions:
#     print(f"Processing Sample {pos}...")
#     z_true, z_samples, post_weights = calculate_posterior_PDF_info(
#         fixed_dofs, n_modes, gamma, n_samples, pos, n_dofs, free_dofs, 
#         test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree, 
#         mean_freq, std_freq, lbound, folder_path
#     )
#     print(z_true)
    
#     plot_results_PDF_uncertainty(
#         fixed_dofs, n_modes, gamma, n_samples, pos, n_dofs, free_dofs, 
#         test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree, 
#         mean_freq, std_freq, lbound, folder_path
#     )
    
#     # plot_physical_pdf_profile(z_samples, post_weights, z_true, n_elements, pos, folder_path, lbound=lbound)

print("Analysis Complete.")

# UNCERTAINTY QUANTIFICATION METRICS (MACE)
print("\n--- Generating Uncertainty Quantification (UQ) Plots ---")
calculate_and_plot_gmm_calibration(predicted_stats, test_datasets, lbound, folder_path) 
print(f"All UQ plots saved to: {os.path.join(folder_path, 'UQ_Metrics')}")