# -*- coding: utf-8 -*-
"""
Main Results Analysis for Kumaraswamy Mixture Copula VAE
@author: anafd

"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K

# Import new KS functions, including the newly added plot_random_2d_posteriors
from MODULES.KUMARSWAMY.GC_KSmix_functions_for_results_analysis import (
    calculate_posterior_PDF_info, plot_results_PDF_uncertainty,
    plot_physical_pdf_profile, calculate_ks_mixture_metrics,
    plot_random_2d_posteriors
)
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.KUMARSWAMY.GC_KSmix_uncertainty_quantification import calculate_and_plot_ks_calibration, enhanced_metrics_comparison 

# --- Config ---
filename = "u06Sept26_GCopKS_10Els_3KSmix_Gamma0.4_LR0.0001_10000epoch" # Update to folder name
model_type_name = 'KS_Copula'
folder_path = os.path.join('Output', model_type_name, filename)
lbound = 0.45

# 1. Load Problem Info
info_path = os.path.join(folder_path, 'Problem_info.npy')
info = np.load(info_path, allow_pickle=True).item()

n_dims = info['n_dims']
num_KS = info['num_KS']
mean_freq = info['mean_f']
std_freq = info['std_f']
beta = info['gamma']

n_elements = 10
n_dofs = 2 * (n_elements + 1)
fixed_dofs = [0, n_dofs - 2] 
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)
batch_size = 256

#1.2. plot the loss functions
from MODULES.POSTPROCESSING.plot_losses import plot_trainval_loss, plot_freqsMACs_loss
history_ = np.load(os.path.join(folder_path, "model_history.npy"),allow_pickle = True)
plot_trainval_loss(history_, folder_path)
plot_freqsMACs_loss(history_, folder_path)

# 2. Load Test Results
Test_pred_props = np.load(os.path.join(folder_path, "Test_predicted_props.npy"), allow_pickle=True).item()
# Unpack KS parameters
# Expected keys: 'test_a', 'test_b', 'test_weight_vals', 'test_offdiag_elems', 'test_diag_elems'
test_a = Test_pred_props['test_a']
test_b = Test_pred_props['test_b']
test_weights = Test_pred_props['test_weight_vals']
test_offdiag = Test_pred_props['test_offdiag_elems']
test_diag = Test_pred_props['test_diag_elems']

# 3. Load Physics and Ground Truth Data
data_folder = "16Mar2026_Noisy_E10_level25_10modes"
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

# 4. Reconstruct Copula Correlation Matrices
L_matrices = build_correlation_matrices_from_cholesky(test_offdiag, test_diag, n_dims)

predicted_stats = {
    'test_a': test_a,
    'test_b': test_b,
    'test_weights': test_weights,
    'test_L_matrices': L_matrices
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
    'gamma': beta 
}

# # --- Analysis Execution ---
print("\n--- KS-VAE Performance Metrics ---")
metrics, covered = calculate_ks_mixture_metrics(predicted_stats, test_datasets, lbound, physics_kwargs=physics_kwargs)
print("Metrics summary:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# %%  --- Visualizing Specific Samples ---

positions = [ 815,  723, 1318, 1077, 1228, 1396,  664, 1679,  689,  279, 1257,
       1178,   30, 1707, 1182, 1772, 1398,  442,  120, 1500, 1349, 1360,
        969,  383,  246,  510, 1455, 1586, 1776, 1787, 1100,  293, 1530,
       1219,  743, 1163,  640,  745,  336,    3, 1282, 1299,  908,  459,
        371, 1643, 1489, 1038, 1267,  455]

n_samples = 4096

for pos in positions:
    print(f"Processing Sample {pos}...")
    
    # Gather necessary data using the calculation function
    z_true, z_samples, post_weights = calculate_posterior_PDF_info(
        fixed_dofs, n_modes, beta, n_samples, pos, n_dofs, free_dofs, 
        test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree, 
        mean_freq, std_freq, lbound, folder_path
    )
    
    print(f"Ground Truth for Sample {pos}: {z_true}")
    
    # -------------------------------------------------------------
    # Plot J randomly chosen 2D slices instead of the full corner plot 
    # to maintain legibility for high dimensional outputs.
    # We choose J=10 subplots arbitrarily here.
    # -------------------------------------------------------------
    print(f"Plotting 10 Random 2D Posteriors for sample {pos}...")
    plot_random_2d_posteriors(
        z_samples, post_weights, z_true, n_elements, pos, 
        folder_path, lbound=lbound, J=10
    )
    
    # Optionally, you can also keep the physical profile graph
    # print(f"Plotting Physical Profile for sample {pos}...")
    # plot_physical_pdf_profile(z_samples, post_weights, z_true, n_elements, pos, folder_path, lbound=lbound)

print("Analysis Complete.")

# %% UNCERTAINTY QUANTIFICATION METRICS (MACE)
print("\n--- Generating Uncertainty Quantification (UQ) Plots ---")

# 1. Marginal Calibration Curve
calculate_and_plot_ks_calibration(predicted_stats, test_datasets, lbound, folder_path)