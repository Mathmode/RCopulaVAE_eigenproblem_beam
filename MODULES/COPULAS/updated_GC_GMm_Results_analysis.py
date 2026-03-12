# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:38:01 2026
@author: anafd
Updated Plotting Utilities for GC-VAE
Aligned with Ground Truth KDE logic for matching results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Disable JIT globally via the TensorFlow API as a secondary safety measure
tf.config.optimizer.set_jit(False)
import pandas as pd 
import tensorflow.keras as K
# def main():
K.backend.set_floatx('float32') 
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.COPULAS.updated_GC_GMm_functions_for_results_analysis import calculate_testing_metrics, plot_results_PDF_uncertainty
from MODULES.COPULAS.GC_plot_posteriors import calculate_posterior_PDF_info, plot_physical_damage_profile

# --- Config ---
filename = "time12MarOLDARCH_Nosiy2.5Mild50_0.45lbound_Bayesian_MACloss_Beta0.4_Samples1_LR1e-05_Epochs50000"
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

training_time = np.load(os.path.join(folder_path,"training_time.npy"),allow_pickle = True).item()['training_time']
print("training time (hours):", training_time/3600.)


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



# --- Visualization Loop 1---
# positions = [0, 1, 7, 9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
# positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
positions = [1,17,48,63,77,219,300, 305,313,314,612,1489]
# positions  = [1,17,25,41,43,48,77,78,219,313,315, 63, 246,383 ,455,459, 1000,1182,1396,1489]
# positions = [ 815,  723, 1318, 1077, 1228, 1396,  664, 1679,  689,  279, 1257,
#        1178,   30, 1707, 1182, 1772, 1398,  442,  120, 1500, 1349, 1360,
#         969,  383,  246,  510, 1455, 1586, 1776, 1787, 1100,  293, 1530,
#        1219,  743, 1163,  640,  745,  336,    3, 1282, 1299,  908,  459,
#         371, 1643, 1489, 1038, 1267,  455]
# 
n_samples = 4096
N_test_samples = len(Freqs_true_test)

for pos in positions:
    if pos < N_test_samples:
        # plot_vae_copula_pdf(pos, predicted_stats, test_datasets, lbound, folder_path)
        
        plot_results_PDF_uncertainty(fixed_dofs, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                         predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path)
        
        
        # z_true, z_samples, posterior_weights = calculate_posterior_PDF_info(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
        #                                  predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, lbound, folder_path)

        # plot_physical_damage_profile(z_samples, posterior_weights, z_true, n_elements, pos, folder_path)
        










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

