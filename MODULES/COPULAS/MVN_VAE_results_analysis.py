# -*- coding: utf-8 -*-
"""
Main Results Analysis Script for TMVN Baseline Model
@author: anafd

"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as K

# Setup context
K.backend.set_floatx('float32')

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.MVN_VAE_functions_for_results_analysis import (
    calculate_testing_metrics, 
    plot_results_PDF_uncertainty_tmvn, 
    plot_physical_pdf_profile_tmvn,
    tmvn_sampling
)

# --- Configuration ---
filename = "22MarMVNlatent_5Els_5modes_Nosiy2.5_0.45lbound_Beta0.3_10000Epochs" # Update to your TMVN folder name
folder_path = os.path.join('Output', 'Gaussian_Copula', filename)
lbound = 0.45 

# Load Problem Info
info = np.load(os.path.join(folder_path, 'Problem_info.npy'), allow_pickle=True).item()
n_dims = info['n_dims']
beta = info['beta']
mean_freq = info['mean_f']
std_freq = info['std_f']
n_elements = 5
n_dofs = 2 * (n_elements + 1)
fixed_dofs = [0, n_dofs - 2]
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)

# --- Data Loading ---
data_path = os.path.join("Data", "01Mar2026_Noisy_E5_level25")
(Freqs_train, _, _, _, _, _, _, _, 
 Freqs_test, Rot_test, Vert_test, Alphas_test, 
 mean_f, std_f) = load_data(data_path, 256)
Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
n_modes = Freqs_test.shape[1]

# --- Load Prediction Results ---
# Note: You need to run model.predict() on your TMVN model first and save these.
# Based on MVN_VAE.py, the encoder returns (means, scales)
pred_path = os.path.join(folder_path, "Test_predicted_props.npy")
if os.path.exists(pred_path):
    Test_pred_props = np.load(pred_path, allow_pickle=True).item()
    test_means = Test_pred_props['test_means']
    test_scales = Test_pred_props['test_scales']
else:
    print("Warning: Pred file not found. Ensure you saved 'test_means' and 'test_scales'.")
    # Placeholder for logic to generate them if model is loaded here
    test_means = np.zeros_like(Alphas_test)
    test_scales = np.ones_like(Alphas_test)

test_datasets = {
    'Freqs_true_test': Freqs_test,
    'Rotmodes_true_test': Rot_test,
    'Vertmodes_true_test': Vert_test,
    'alpha_factors_true_test': Alphas_test
}

predicted_stats = {
    'test_means': test_means,
    'test_scales': test_scales
}

# --- 1. Global Performance Metrics ---
print("\n--- TMVN Global Performance ---")
metrics, coverage_mask = calculate_testing_metrics(predicted_stats, test_datasets, lbound=lbound)
for k, v in metrics.items(): print(f"{k:25}: {v:.6f}")

# --- 2. Visualization ---
positions = [1, 17, 25, 48, 63, 77] # Example samples
n_samples = 2048

for pos in positions:
    print(f"Processing Sample {pos}...")
    # Generate joint posterior plots
    plot_results_PDF_uncertainty_tmvn(
        fixed_dofs, n_modes, beta, n_samples, pos, n_dofs, free_dofs, 
        test_datasets, predicted_stats, L_inv, Ke_matrices, 
        mean_freq, std_freq, lbound, folder_path
    )
    
    # Generate profile UQ plots
    locs = test_means[pos, :]
    scales = test_scales[pos, :]
    z_samples = tmvn_sampling(locs, scales, n_samples, lbound)
    
    # Simple Weighting for profile plot
    plot_physical_pdf_profile_tmvn(
        z_samples.numpy(), np.ones(n_samples)/n_samples, 
        Alphas_test[pos], n_elements, pos, folder_path, lbound
    )

print("\nAnalysis Complete.")