# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:06:42 2026

@author: anafd
"""

import os
import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
# Custom module imports
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.COPULAS.GC_postprocessing_copulas import plot_configuration, plot_results_PDF_uncertainty ,plot_KDE_pdf
plot_configuration()
# %% 1. Path and Problem Configuration
# Define the path to your trained model folder
filename = "GemFix_LRdecay_09Feb_NewLosses_5els5modes_test_Data17MarCopula_0.1Beta_1Samples_1e-05LR_12000epochs_256batchs"
folder_path = os.path.join('Output', 'Gaussian_Copula', filename)

# Load Problem Info
info_path = os.path.join(folder_path, 'Problem_info.npy')
if not os.path.exists(info_path):
    raise FileNotFoundError(f"Problem_info.npy not found at {info_path}")

info = np.load(info_path, allow_pickle=True).item()

# Extract parameters from saved info
input_dim = info['input_dim_enc']
n_dims = info['n_dims']
num_gaussians = info['n_gaussians']
num_samples = info['n_samples'] # This is the H samples used during training
beta = info['beta']
epsi = info['epsi']

# %% 1. Initialization and Data Loading
K.utils.set_random_seed(1234)
data_path = os.path.join("Data", "17MARRandomdata5elements5nmodes")
batch_size = 48
n_elements = 5
n_modes = 5
n_dofs = 2*(n_elements +1) 
num_dofs = n_dofs -2 
Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
L_inv_tf = tf.cast(L_inv, dtype=tf.float32)
Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)

# Assume Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test are already in memory
# Example scaling/noise parameters
inv_gamma_val = 1.0 / (beta**2)


# %% 2. Model Re-instantiation and Loading Weights
print("Initializing model and loading weights...")
# Force CPU context for initialization to prevent CUDA/XLA 'libdevice' errors
with tf.device('/CPU:0'):
    model = My_CopulaVAE_withEigen(
        input_dim, num_dofs, n_elements, n_modes, 
        Ke_matrices, Mfree, L_inv, epsi, 
        n_dims, num_gaussians, num_samples, beta
    )

    # MANDATORY: We must build the model before loading HDF5 weights.
    # We do this by passing a dummy batch through the model.
    print("Building model variables...")
    dummy_f = tf.zeros((1, n_modes))
    dummy_r = tf.zeros((1, 6, n_modes))
    dummy_v = tf.zeros((1, 4, n_modes))
    dummy_z = tf.zeros((1, n_dims))
    
    # This call creates the internal weights/variables
    _ = model([dummy_f, dummy_r, dummy_v, dummy_z])

# Now that variables are created, we load the weights
weights_path = os.path.join(folder_path, "model_weights.weights.h5")
if os.path.exists(weights_path):
    # FIX: Use by_name=True to resolve layer count mismatch in subclassed models
    model.load_weights(weights_path, by_name=True)
    print("Model weights loaded successfully.")
else:
    print(f"CRITICAL: Weights file not found at {weights_path}")

# model = K.models.load_model(os.path.join(folder_path, "Full_Model_Export"), compile=False)
inverse_model = model.Encoder_model
test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems  = inverse_model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
# test_LT_matrices  = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems,n_dims)

# With the suggestions of Gemini to improve the code: 
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
bijector = tfb.CorrelationCholesky()
test_LT_matrices = bijector.forward(test_offdiag_elems)

positions = [0, 1, 7,  9, 11, 17,25, 34,45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
# positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
# positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
# 

n_samples = 800
for i in range(len(positions)):
        pos = positions[i]    
        locs = test_means[pos,:]    
        scales = test_scales[pos,:]
        weight_vals = test_weight_vals[pos,:]
        LT_matrix = test_LT_matrices[pos,:]
        plot_results_PDF_uncertainty(model, beta, n_elements, n_modes, locs, scales, weight_vals, LT_matrix, n_dims, n_samples, pos, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, Mfree, Ke_matrices, L_inv, folder_path)
        

