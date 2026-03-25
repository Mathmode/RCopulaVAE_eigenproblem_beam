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

# Import new KS functions
from MODULES.KUMARSWAMY.GC_KSmix_functions_for_results_analysis import (
    calculate_testing_metrics, 
    calculate_posterior_PDF_info, plot_results_PDF_uncertainty,
    plot_physical_pdf_profile
)
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky

# --- Config ---
filename = "24Mar_KS_Copula_5Els_3KSmix_Gamma0.45_LR1e-05_800epoch" # Update to your folder name
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

n_elements = 5
n_dofs = 2 * (n_elements + 1)
fixed_dofs = [0, n_dofs - 2] 
all_dofs = np.arange(n_dofs)
free_dofs = np.delete(all_dofs, fixed_dofs)
batch_size = 256

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

data_folder = "01Mar2026_Noisy_E5_level25"
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

# # --- Analysis Execution ---
# print("\n--- KS-VAE Performance Metrics ---")
# metrics = calculate_testing_metrics(predicted_stats, test_datasets, lbound=lbound)
# for k, v in metrics.items():
#     print(f"{k}: {v:.6f}")

# --- Visualizing Specific Samples ---
# positions = [63,219]
positions = [34, 63, 77, 219,1178] # Example indices
n_samples = 4096

for pos in positions:
    print(f"Processing Sample {pos}...")
    z_true, z_samples, post_weights = calculate_posterior_PDF_info(
        fixed_dofs, n_modes, beta, n_samples, pos, n_dofs, free_dofs, 
        test_datasets, predicted_stats, L_inv, Ke_matrices, Mfree, 
        mean_freq, std_freq, lbound, folder_path
    )
    print(z_true)
    plot_results_PDF_uncertainty(fixed_dofs, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                     predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path)
    
    
    
    
    # plot_physical_pdf_profile(z_samples, post_weights, z_true, n_elements, pos, folder_path, lbound=lbound)

print("Analysis Complete.")