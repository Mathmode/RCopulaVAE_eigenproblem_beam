#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:48:23 2025

@author: afernandez
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import time 

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
from MODULES.BETA_GCOP.Beta_models import My_CopulaVAE_Kumarmarg_withEigen
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss


def main():
    # --- 1. GLOBAL SETTINGS ---
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    # --- 2. DATA LOADING ---
    data_folder = "01Mar2026_Noisy_E5_level25"
    data_path = os.path.join("Data", data_folder)
    
    n_elements = 5
    n_dofs = 2 * (n_elements + 1) 
    lbound = 0.45 
    fixed_dofs_indices = [0, n_dofs - 2]
    batch_size = 256
    
    print(f"Loading data from {data_path}...")
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, z_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, z_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, z_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

    # --- 3. MODEL CONFIGURATION ---
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    n_epochs = 8000
    base_lr = 1e-6
    num_kumar_mix = 2
    n_dims = z_factors_true_train.shape[1]
    num_samples = 1
    regu_weight = 0.3
    
    
    # Output Directory
    filename = f"Kumar_Nosiy2.5_{num_kumar_mix}mixtures_{regu_weight}RWeight_LR{base_lr}_Epochs{n_epochs}"
    folder_path = os.path.join('Output', "GCop_Betamixture", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        

    print(f"Initializing Model...")
    model = My_CopulaVAE_Kumarmarg_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_elements=n_elements,
        n_modes=n_modes, Ke_matrices=Ke_matrices, Mfree=Mfree,
        L_inv=L_inv, regu_weight=regu_weight, n_dims=n_dims, num_kumar_mix=num_kumar_mix,
        num_samples=num_samples, mean_f=mean_f,
        std_f=std_f, lbound=lbound, fixed_dofs_indices=fixed_dofs_indices
    )

         
    # --- 4. COMPILATION ---
    optim = K.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)
    run_eagerly = False
    model.compile(
        optimizer=optim, 
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term],
        run_eagerly= run_eagerly
    )

    # --- 5. TRAINING ---
    start_time = time.time()
    print("Starting Training...")
    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, z_factors_true_train],
        y=[z_factors_true_train, z_factors_true_train],
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, z_factors_true_val], 
            [z_factors_true_val, z_factors_true_val]
        )
    )
    
    ## with these lines we prevent having to load the trained model in the postprocessing, which was supisciously not working well (FIX)
    inverse_model = model.Encoder_model
    test_alphas, test_betas, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, z_factors_true_test]
    )
    print(f"test alphas check", test_alphas)
    
    end_time = time.time()  # Record the ending time
    training_time = end_time - start_time  # Calculate the training time
    print(f"Total training time: {training_time:.2f} seconds")
    # Save the training time
    time_info = {'training_time': training_time}
    np.save(os.path.join(folder_path, 'training_time.npy'), time_info, allow_pickle=True)
    
    
    # --- 7. SAVING RESULTS ---    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'num_beta_mix': num_beta_mix, 
                    'n_samples': num_samples, 'epochs': n_epochs, 'regu_weight': regu_weight, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)   
    Test_predicted_properties = {'test_alphas': test_alphas, 'test_betas': test_betas, 'test_offdiag_elems': test_offdiag_elems, 
                    'test_diag_elems': test_diag_elems, 'test_weight_vals': test_weight_vals}
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), Test_predicted_properties, allow_pickle=True)


    
    # # --- 6. WEIGHT VERIFICATION AFTER TRAINING ---
    # trained_weights_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    # print(f"\n--- Weight Integrity Check ---")
    # print(f"Pre-training Sum:  {initial_weights_sum:.6f}")
    # print(f"Post-training Sum: {trained_weights_sum:.6f}")
    
    # if np.isclose(initial_weights_sum, trained_weights_sum):
    #     print("❌ CRITICAL ERROR: Weights did not change during training! Gradients are not flowing.")
    # else:
    #     print("✅ SUCCESS: Weights updated during training.")
    # weights_file = os.path.join(folder_path, "final_model_weights.weights.h5")
    # model.save_weights(weights_file)
    # print(f"Weights saved to: {weights_file}")
           
    # # --- 8. TEST RELOAD IN SAME SESSION ---
    # print("\n--- Verifying Saved File (Same Session Reload) ---")
    # # Reset model to initial state temporarily
    # # (Optional: just check if re-loading into the same object changes nothing)
    # model.load_weights(weights_file, by_name=True)
    # post_load_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    # print(f"Sum after re-loading saved file: {post_load_sum:.6f}")
    
    # --- 9. PREDICTION VARIATION TEST ---
    print("\n--- Testing Prediction Variation ---")
    preds = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, z_factors_true_test])
    

    # --- 7. PLOTTING & PREDICTION ---
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()

