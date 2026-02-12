# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:19:33 2026

@author: anafd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 2026
@author: afernandez

Testing Script for Physics-Informed VAE (Gaussian Copula).
- Generates NEW unseen data.
- Loads the trained model.
- Predicts Damage Factors (Alphas) using the Encoder.
- Visualizes True vs Predicted results.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as K

# --- IMPORTS ---
# Import the Dataset Generator to create new "measurements"
from random_dataset_generation import generate_dataset

# Import the VAE Model Architecture
from GC_GMm_GPU_eigen_functions import My_CopulaVAE_withEigen

def main():
    # --- 1. SETTINGS & CONFIGURATION ---
    # Must match the training configuration
    n_elements = 5
    n_modes = 5
    n_dofs = 2 * (n_elements + 1)
    
    # Testing Parameters
    N_test_samples = 100 # Number of new "measured" beams
    folder_name = f"10Feb2026_Randomdata{n_elements}elements{n_modes}nmodes"
    data_path = os.path.join("Data", folder_name)
    
    # Path to trained weights (Ensure this matches your training output folder)
    # We assume standard hyperparameters from the previous script
    beta = 0.01
    num_samples = 1
    base_lr = 1e-4
    n_epochs = 10000
    
    model_folder = f"Run_Beta{beta}_Samples{num_samples}_LR{base_lr}_Epochs{n_epochs}"
    weights_path = os.path.join('Output', "Gaussian_Copula", model_folder, "final_model_weights.weights.h5")
    
    # --- 2. GENERATE NEW "MEASUREMENT" DATA ---
    print(f"--- Generating {N_test_samples} New Test Samples ---")
    
    # We create a temporary folder for test data or just generate in memory
    # Here we reuse the generation function but don't strictly need to save to disk if we use returns
    # But the function requires a path, so we use a temp one.
    test_data_path = os.path.join("Data", "Test_Temp")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    # Generate Data
    freqs_new, rot_new, vert_new, alphas_true_new = generate_dataset(
        N_test_samples, n_elements, n_modes, test_data_path, zero_rotations=False
    )
    
    print("New data shapes:")
    print(f" Frequencies: {freqs_new.shape}")
    print(f" Modes:       {vert_new.shape}") # (Samples, Modes, Nodes)
    print(f" True Alphas: {alphas_true_new.shape}")

    # --- 3. LOAD MATRICES FOR MODEL INIT ---
    # The model needs the physics matrices (Mass, Stiffness, L_inv) to be initialized
    # We load the base ones from the original training folder
    print("--- Loading Physics Matrices ---")
    try:
        Ke_base = np.load(os.path.join(data_path, "Ke_matrix.npy"))
        M_free = np.load(os.path.join(data_path, "Mass_matrix.npy"))
        
        # Re-compute L_inv (Cholesky Inverse) just to be safe/independent
        L = np.linalg.cholesky(M_free)
        L_inv = np.linalg.inv(L)
    except FileNotFoundError:
        print(f"❌ Error: Base matrices not found in {data_path}. Run training setup first.")
        return

    # --- 4. INSTANTIATE MODEL ---
    print("--- Initializing Model Architecture ---")
    
    # Calculate input dimensions for the Encoder
    # Freqs (N_modes) + Flattened Rot (N_modes*N_nodes) + Flattened Vert (N_modes*N_nodes)
    n_nodes = n_elements + 1
    input_dim = n_modes + (n_modes * n_nodes) + (n_modes * n_nodes)
    
    # Load Stats (Mean/Std Freqs) used during training
    # Ideally, load 'Problem_info.npy' or 'scaler.npy'. 
    # Here we approximate or load if available.
    try:
        # Try loading metadata if you saved it in training
        # problem_info = np.load(os.path.join('Output', "Gaussian_Copula", model_folder, 'Problem_info.npy'), allow_pickle=True).item()
        pass
    except:
        pass

    # For this test, we recalculate mean/std from the NEW data just to populate the init 
    # (Note: In strict inference, you should use the TRAINING mean/std, but for the Encoder 
    # prediction path, these values are used in the LOSS function, not the Encoder inference).
    mean_f = tf.math.reduce_mean(tf.math.log(freqs_new), axis=0)
    std_f = tf.math.reduce_std(tf.math.log(freqs_new), axis=0)
    
    # Fixed DOFs (Simply Supported)
    fixed_dofs_indices = [0, n_dofs - 2]
    
    # Model Init
    model = My_CopulaVAE_withEigen(
        input_dim=input_dim,
        num_dofs=n_dofs,
        n_elements=n_elements,
        n_modes=n_modes,
        Ke_matrices=Ke_base,
        Mfree=M_free,
        L_inv=L_inv,
        epsi=0.0,
        n_dims=n_elements, # Alpha dimension
        num_gaussians=1,   # Assuming Single Gaussian for now
        num_samples=1,
        beta=beta,
        mean_f=mean_f,
        std_f=std_f,
        fixed_dofs_indices=fixed_dofs_indices
    )
    
    # --- 5. LOAD WEIGHTS ---
    print(f"--- Loading Weights from: {weights_path} ---")
    # We need to run a dummy call to build the graph before loading weights
    # or use build(). Keras often likes a forward pass.
    dummy_input = [
        freqs_new[0:1], 
        rot_new[0:1], 
        vert_new[0:1], 
        alphas_true_new[0:1]
    ]
    _ = model(dummy_input) # Build graph
    
    model.load_weights(weights_path)
    print("✅ Weights Loaded Successfully.")

    # --- 6. PREDICTION (INFERENCE) ---
    print("--- Running Prediction ---")
    
    # Prepare inputs
    # The Encoder expects: [Freqs, Rot, Vert, Alphas]
    # Note: Alphas are passed only because the unpacking signature expects 4 items.
    # The Encoder DOES NOT use the alphas for prediction (that would be cheating).
    # It constructs 'modal_data' from Freqs/Modes only.
    inputs_test = [freqs_new, rot_new, vert_new, alphas_true_new]
    
    # Get Encoder Outputs
    # returns: means, scales, weight_vals, offdiag_elems, diag_elems
    means, scales, weights, _, _ = model.Encoder_model(inputs_test)
    
    # EXTRACT DETERMINISTIC ESTIMATE
    # shape of means: (Batch, Num_Gaussians, N_dims)
    # Since Num_Gaussians = 1, we take index 0.
    # If Num_Gaussians > 1, we would take the weighted average or the max weight component.
    pred_alphas = means[:, 0, :]
    
    # Numpy conversion
    pred_alphas_np = pred_alphas.numpy()
    
    # --- 7. VISUALIZATION & METRICS ---
    print("--- Visualizing Results ---")
    
    # A. Global Correlation Plot (All Elements)
    plt.figure(figsize=(8, 8))
    plt.scatter(alphas_true_new.flatten(), pred_alphas_np.flatten(), alpha=0.5, c='blue', s=10)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    plt.title(f"True vs Predicted Stiffness Reduction (Alpha)\n{N_test_samples} Test Samples")
    plt.xlabel("True Alpha (Damage Factor)")
    plt.ylabel("Predicted Alpha")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(test_data_path, "Test_Correlation_Plot.png"))
    plt.show()
    
    # B. Single Beam Profile Comparison
    # Pick a random sample index
    sample_idx = np.random.randint(0, N_test_samples)
    
    true_profile = alphas_true_new[sample_idx]
    pred_profile = pred_alphas_np[sample_idx]
    
    x_elems = np.arange(1, n_elements + 1)
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.bar(x_elems - width/2, true_profile, width, label='True Damage', color='navy', alpha=0.7)
    plt.bar(x_elems + width/2, pred_profile, width, label='Predicted Damage', color='orange', alpha=0.7)
    
    plt.xlabel("Element Index")
    plt.ylabel("Alpha (1 - Damage)")
    plt.title(f"Damage Detection: Sample #{sample_idx}")
    plt.xticks(x_elems)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(test_data_path, f"Test_Sample_{sample_idx}_Profile.png"))
    plt.show()
    
    # C. Calculate Metrics
    mse = np.mean((alphas_true_new - pred_alphas_np)**2)
    print(f"\n--- TEST RESULTS ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    
    # Calculate R2 Score manually
    ss_res = np.sum((alphas_true_new - pred_alphas_np)**2)
    ss_tot = np.sum((alphas_true_new - np.mean(alphas_true_new))**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()