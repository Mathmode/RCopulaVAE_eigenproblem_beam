#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:16:13 2025

@author: afernandez

Cleaned and Optimized Main Script for GC-GMM Copula Training.
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

# --- GPU CONFIGURATION (Must be first) ---
# Prevents allocating all memory and crashes on some setups
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization Error: {e}")

# --- IMPORTS ---
# Assuming 'MODULES' contains your data loading utils
try:
    from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
    from MODULES.COPULAS.GC_postprocessing_copulas import (
        plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss
    )
except ImportError:
    print("⚠️ Warning: Could not import preprocessing/postprocessing modules. Ensure paths are correct.")

# Import local optimized models
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen


def main():
    # --- 1. GLOBAL SETTINGS ---
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    # --- 2. DATA LOADING ---
    # Path Configuration
    data_folder = "11Feb2026_Corrected_Randomdata5elements"
    data_path = os.path.join("Data", data_folder)
    
    # System Parameters
    n_elements = 5
    # Total DOFs: 2 per node, (N+1) nodes. 
    # For 5 elements: 6 nodes * 2 = 12 DOFs.
    n_dofs = 2 * (n_elements + 1) 
    
    # Boundary Conditions: Simply Supported (Pin-Pin)
    # Fix Vertical displacement at first node (Index 0) and last node (Index 2*N)
    # Note: 2*N corresponds to n_dofs - 2
    fixed_dofs_indices = [0, n_dofs - 2]
    
    batch_size = 256
    
    print(f"Loading data from {data_path}...")
    # Load dataset
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    # Load Physical Matrices (Mass, Stiffness, Cholesky Inverse)
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

    # --- 3. MODEL CONFIGURATION ---
    # Input dimension: Freqs + Flattened RotModes + Flattened VertModes
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    
    # Training Hyperparameters
    n_epochs = 10000
    base_lr = 1e-4
    epsi = 0.0 # Regularizer weight
    
    # Copula/GMM Hyperparameters
    num_gaussians = 1
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = 1 # Samples for training (Monte Carlo integration in loss)
    beta = 0.5   # Weight for the Joint Copula Loss term

    print(f"Initializing Model with Total DOFs: {n_dofs}, Fixed Indices: {fixed_dofs_indices}")
    
    # Instantiate Model
    # IMPORTANT: We pass 'n_dofs' (Total) not 'num_dofs' (Free), 
    # because the solver needs to reconstruct the full shape.
    model = My_CopulaVAE_withEigen(
        input_dim=input_dim,
        num_dofs=n_dofs,  # Pass TOTAL DOFs here
        n_elements=n_elements,
        n_modes=n_modes,
        Ke_matrices=Ke_matrices,
        Mfree=Mfree,
        L_inv=L_inv,
        epsi=epsi,
        n_dims=n_dims,
        num_gaussians=num_gaussians,
        num_samples=num_samples,
        beta=beta,
        mean_f=mean_f,
        std_f=std_f,
        fixed_dofs_indices=fixed_dofs_indices
    )
    
    # --- 4. COMPILATION & SETUP ---
    run_eagerly = False # Set True only for debugging
    
    # Output Directory
    filename = f"Bayesian_MACloss_Beta{beta}_Samples{num_samples}_LR{base_lr}_Epochs{n_epochs}"
    folder_path = os.path.join('Output', "Gaussian_Copula", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[1000, 8000], 
        values=[1e-5, 1e-4, 1e-5] 
    )
    
    # Compile
    optimizer = K.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer, 
        loss=model.ELBO_Copula_loss, 
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term], 
        run_eagerly=run_eagerly
    )

    # --- 5. TRAINING ---
    # Checkpoints
    checkpoint_path = os.path.join(folder_path, "checkpoints", "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=0, 
        save_weights_only=True,
        save_freq='epoch',
        period=100 # Save every 100 epochs
    )
    
    # TensorBoard (Optional, useful for monitoring)
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=folder_path)

    print("Starting Training...")
    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
        y=alpha_factors_true_train,
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], 
            alpha_factors_true_val
        ),
        callbacks=[cp_callback]
    )
    
    # --- 6. SAVING RESULTS ---
    print("Saving Results...")
    
    # Save Metadata
    Problem_info = {
        'input_dim_enc': input_dim,
        'n_dims': n_dims, 
        'n_gaussians': num_gaussians, 
        'n_samples': num_samples, 
        'epochs': n_epochs,
        'epsi': epsi,
        'beta': beta,
        'std_f':std_f,
        'mean_f':mean_f
        
    }
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
    # Save History
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
    # Save Weights
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    
    # --- 7. PLOTTING & PREDICTION ---
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Generate predictions on Test Set
    
    # Unpack the four outputs returned by the model
    pred_alphas = model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )

    print(f"Predicted Alphas: {pred_alphas[0,:]}")
    # pred_alphas = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
    np.save(os.path.join(folder_path, "pred_alphas.npy"), pred_alphas, allow_pickle=True)
   
    print(f"✅ Run Complete. Results saved to: {folder_path}")

if __name__ == "__main__":
    main()