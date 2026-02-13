#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2026
@author: afernandez

Visualization of Joint Posterior Distributions (Uncertainty Quantification).
- Loads Trained Model.
- Generates Monte Carlo samples from the predicted Copula distribution.
- Propagates samples through the Physics Engine (Eigen Solver).
- Computes Bayesian Posterior based on Data Misfit.
- Plots 2D KDE of the damage parameters.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import tensorflow_probability as tfp
from scipy.stats import norm

# -----------------------------------------
# --- CRITICAL CONFIGURATION: FORCE CPU & DISABLE JIT ---
# This bypasses the 'libdevice not found' XLA/GPU error by hiding the GPU.
try:
    tf.config.set_visible_devices([], 'GPU')
    # Disable JIT to prevent XLA from trying to compile ops that miss libdevice
    tf.config.optimizer.set_jit(False) 
    print("🖥️ GPU and JIT disabled for visualization to prevent XLA/libdevice errors.")
except:
    pass

tfd = tfp.distributions

# --- LOCAL IMPORTS (Preserving your structure) ---
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
from MODULES.COPULAS.GC_plot_posteriors import plot_results_PDF_uncertainty


def main():
    K.backend.set_floatx('float32') 

    # --- Config ---
    filename = "Bayesian_MACloss_Beta0.25_Samples1_LR0.0001_Epochs10000"
    folder_path = os.path.join('Output', 'Gaussian_Copula', filename)
    
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

    # %% 1. Initialization and Data Loading
    K.utils.set_random_seed(1234)
    data_path = os.path.join("Data", "11Feb2026_Corrected_Randomdata5elements")
    print(f"Loading data from {data_path}...")
    
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    # Load Physics Matrices
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
    
    # Fix input dim calculation based on your previous messages
    n_modes = Freqs_true_train.shape[1]
    
    # %% 2. Model Re-instantiation and Loading Weights
    print("Initializing model...")
    # Force CPU context for initialization
    with tf.device('/CPU:0'):    
        model = My_CopulaVAE_withEigen(
            input_dim=input_dim,
            num_dofs=n_dofs,
            n_elements=n_elements,
            n_modes=n_modes,
            Ke_matrices=Ke_matrices,
            Mfree=Mfree,
            L_inv=L_inv,
            epsi=0.0,
            n_dims=n_elements,
            num_gaussians=1,
            num_samples=1,
            beta=beta,
            mean_f=mean_f,
            std_f=std_f,
            fixed_dofs_indices=[0, n_dofs-2]
        )
    
        # MANDATORY: We must build the model before loading HDF5 weights.
        # We do this by passing a dummy batch through the model.
        print("Building model variables...")
        dummy_f = tf.zeros((1, n_modes))
        dummy_r = tf.zeros((1, n_modes, 6))
        dummy_v = tf.zeros((1, n_modes, 6))
        dummy_z = tf.zeros((1, n_dims))
        
        # This call creates the internal weights/variables
        _ = model([dummy_f, dummy_r, dummy_v, dummy_z])

    # Now that variables are created, we load the weights
    weights_path = os.path.join(folder_path, "final_model_weights.weights.h5")
    if os.path.exists(weights_path):
        # FIX: Use by_name=True to resolve layer count mismatch in subclassed models
        model.load_weights(weights_path, by_name=True)
        print("Model weights loaded successfully.")
    else:
        print(f"CRITICAL: Weights file not found at {weights_path}")
    
    
   
    # --- Prediction Step ---
    print("Running Encoder Prediction...")
    inverse_model = model.Encoder_model
    # Note: inputs are passed as a list
    test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    
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
    
    # --- Visualization Loop ---
    # positions = [0, 1, 7, 9, 11, 17, 25, 34, 45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
    # positions  = [2,20,21,41,43,48,50,63, 64, 65, 77, 78, 91,92,98,99,102,560,576]
    positions = [300,301,302,303,304,305,310,311,312,313,314,315,321,322,323]
    # 
    
    n_samples = 1000 
    N_test_samples = len(Freqs_true_test)
    
    for pos in positions:
        if pos < N_test_samples:
            plot_results_PDF_uncertainty(model, n_modes, beta, n_samples, pos, n_dofs, free_dofs, test_datasets,
                                             predicted_stats, L_inv, Ke_matrices, Mfree, mean_freq, std_freq, folder_path)






if __name__ == "__main__":
    main()