#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:16:13 2025

@author: afernandez

Cleaned and Optimized Main Script for GC-GMM Copula Training.
Updated to validate robustness against modeling error (Mass Matrix Perturbation).
"""

# %% IMPORTS & SETUP
# =============================================================================
import os
import time 
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

# --- GPU CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization Error: {e}")

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.KUMARSWAMY.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss


# %% CUSTOM CALLBACKS
# =============================================================================
class DelayedEarlyStopping(K.callbacks.Callback):
    """
    Robust Early Stopping. 
    It saves the best weights during training, and will only increment the counter 
    AFTER the start_epoch is reached. It guarantees best model retention.
    """
    def __init__(self, patience=1500, start_epoch=6000, restore_best_weights=True):
        super(DelayedEarlyStopping, self).__init__()
        self.patience = patience
        self.start_epoch = start_epoch
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is None: return
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            if epoch >= self.start_epoch:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n✅ Early stopping triggered at epoch {epoch}.")
                    self.model.stop_training = True
                    if self.restore_best_weights and self.best_weights is not None:
                        print("Restoring best model weights.")
                        self.model.set_weights(self.best_weights)


# %% MAIN FUNCTION
# =============================================================================
def main():
    # --- 1. GLOBAL SETTINGS ---
    # Set to True for a fast 3-epoch test with a tiny dataset subset.
    DUMMY_RUN = True 
    
    # Fix the random seed for reproducibility
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    # --- 2. DATA LOADING ---
    data_folder = "06Oct_Data_Noisy_E5_Lvl25_MassPerturb5"
    data_path = os.path.join("Data", data_folder)
    
    n_elements = 5
    n_dofs = 2 * (n_elements + 1) 
    lbound = 0.45 
    batch_size = 256
    fixed_dofs_indices = [0, n_dofs - 2]
    
    print(f"Loading data from {data_path}...")
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train = Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test
    Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val = Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test
    
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

    # --- 3. MODEL CONFIGURATION ---
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    n_nodes = Rotmodes_true_train.shape[2]
    n_epochs = 10000
    base_lr = 1e-5
    num_KS = 3
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = 1 
    gamma = 0.4
    
    # --- 4. DUMMY RUN ADJUSTMENTS ---
    if DUMMY_RUN:
        print("\n" + "="*50)
        print("🚀 DUMMY RUN ENABLED 🚀")
        print("Reducing  dataset size for quick validation.")
        print("="*50 + "\n")
        n_epochs = 5000
        batch_size = 32
        dummy_limit = 64
        
        Freqs_true_train = Freqs_true_train[:dummy_limit]
        Rotmodes_true_train = Rotmodes_true_train[:dummy_limit]
        Vertmodes_true_train = Vertmodes_true_train[:dummy_limit]
        alpha_factors_true_train = alpha_factors_true_train[:dummy_limit]
        
        Freqs_true_val = Freqs_true_val[:dummy_limit]
        Rotmodes_true_val = Rotmodes_true_val[:dummy_limit]
        Vertmodes_true_val = Vertmodes_true_val[:dummy_limit]
        alpha_factors_true_val = alpha_factors_true_val[:dummy_limit]
        
        Freqs_true_test = Freqs_true_test[:dummy_limit]
        Rotmodes_true_test = Rotmodes_true_test[:dummy_limit]
        Vertmodes_true_test = Vertmodes_true_test[:dummy_limit]
        alpha_factors_true_test = alpha_factors_true_test[:dummy_limit]
    
    # Output Directory Configuration
    filename = f"T09Oct_GCop_{num_KS}KSmx_{n_elements}Els_{n_modes}modes_MassPert5_{lbound}lb_Gamma{gamma}_{n_epochs}ep"
    folder_path = os.path.join('Output', "REV_GCopulaKS_MULTIMODALITY", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # --- 5. MODEL INITIALIZATION & COMPILATION ---
    model = My_CopulaKSVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_elements=n_elements,
        n_modes=n_modes, Ke_matrices=Ke_matrices, Mfree=Mfree, L_inv=L_inv, 
        n_dims=n_dims, num_KS=num_KS, num_samples=num_samples, 
        gamma=gamma, mean_f=mean_f, std_f=std_f, lbound=lbound, 
        fixed_dofs_indices=fixed_dofs_indices
    )
    
    optim = K.optimizers.Adam(learning_rate=base_lr, clipnorm=0.5)
    
    # Force build to initialize layers
    _ = model([Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]])
    
    model.compile(
        optimizer=optim, 
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term]
    )
    
    # --- 6. TRAINING ---
    start_time = time.time()
    print("Starting Kumaraswamy Copula Training...")
    
    # Allow the model to train deeper into the epoch limit based on mode
    if DUMMY_RUN:
        delayed_stop = DelayedEarlyStopping(patience=200, start_epoch=1000, restore_best_weights=True)
    else:
        delayed_stop = DelayedEarlyStopping(patience=1500, start_epoch=6000, restore_best_weights=True)
    
    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
        y=[alpha_factors_true_train], # Dummy y
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], 
            [alpha_factors_true_val], 
        ), callbacks=[delayed_stop]
    )
    
    # Save base results
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Retrieve Inference Properties
    inverse_model = model.Encoder_model
    test_a, test_b, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    
    # --- 7. RESULTS POST-PROCESSING & SAVING ---    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'num_KS': num_KS, 
                    'n_samples': num_samples, 'epochs': n_epochs, 'gamma': gamma, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
    Test_predicted_properties = {'test_a': test_a, 'test_b': test_b, 'test_offdiag_elems': test_offdiag_elems, 
                    'test_diag_elems': test_diag_elems, 'test_weight_vals': test_weight_vals}
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), Test_predicted_properties, allow_pickle=True)
        
    test_datasets = {
        'Freqs_true_test': Freqs_true_test,
        'Rotmodes_true_test': Rotmodes_true_test,
        'Vertmodes_true_test': Vertmodes_true_test,
        'alpha_factors_true_test': alpha_factors_true_test
    }
    
    from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
    L_matrices = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems, n_dims)
    
    predicted_stats = {
        'test_a': test_a,
        'test_b': test_b,
        'test_weights': test_weight_vals,
        'test_L_matrices': L_matrices
    }
    
    fixed_dofs = [0, n_dofs - 2] 
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)
    
    physics_kwargs = {
        'Ke_matrices': Ke_matrices,
        'L_inv': L_inv,
        'n_modes': n_modes,
        'free_dofs': free_dofs,
        'fixed_dofs': fixed_dofs,
        'n_dofs': n_dofs,
        'mean_freq': mean_f,
        'std_freq': std_f,
        'gamma': gamma 
    }
    
    # Calculate metrics
    from MODULES.KUMARSWAMY.GC_KSmix_functions_for_results_analysis import (
        calculate_ks_mixture_metrics,
    )
    
    print("\n--- KS-VAE Performance Metrics ---")
    metrics, covered = calculate_ks_mixture_metrics(predicted_stats, test_datasets, lbound, physics_kwargs=physics_kwargs)
    print("Metrics summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Plotting
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


# %% SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    main()