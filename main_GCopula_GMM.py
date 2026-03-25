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
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
# Assuming plot functions are updated to handle the new return structure
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss

## ADDITIONAL WORK FOR PAPER: COMPARING RESULTS AGAINST A FULLY CONNECTED NN DECODER
from MODULES.COPULAS.GC_GMm_model_surrogatedecoder import My_CopulaVAE_Surrogate 

# --- CUSTOM CALLBACK DEFINITION ---
class DelayedEarlyStopping(K.callbacks.Callback):
    def __init__(self, patience=1000, start_epoch=10000):
        super(DelayedEarlyStopping, self).__init__()
        self.patience = patience
        self.start_epoch = start_epoch
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is None: return
        if epoch >= self.start_epoch:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
                print(f" - Epoch {epoch}: New best val_loss: {current_loss:.6f}")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n✅ Early stopping triggered at epoch {epoch}.")
                    self.model.stop_training = True
        elif current_loss < self.best_loss:
            self.best_loss = current_loss


def main():
    # --- 1. GLOBAL SETTINGS ---
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    # --- 2. DATA LOADING ---
    data_folder = "01Mar2026_Noisy_E5_level25"
    # data_folder = "16Mar2026_Noisy_E10_level25_5modes"

    data_path = os.path.join("Data", data_folder)
    
    n_elements = 5
    n_dofs = 2 * (n_elements + 1) 
    lbound = 0.45 
    fixed_dofs_indices = [0, n_dofs - 2]
    batch_size = 256
    
    
    print(f"Loading data from {data_path}...")
    (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
     mean_f, std_f) = load_data(data_path, batch_size)
    
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

    # --- 3. MODEL CONFIGURATION ---
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    n_nodes = Rotmodes_true_train.shape[2]
    n_epochs = 10000
    base_lr = 1e-5
    num_gaussians = 1
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = 1 
    gamma = 0.3
    
    # Output Directory
    filename = f"F22MarHD_{n_elements}Els_{n_modes}modes_Nosiy2.5_{lbound}lbound_Gamma{gamma}_{n_epochs}Epochs_lr{base_lr}"
    folder_path = os.path.join('Output', "Gaussian_Copula", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    

    # --- 6. TRAINING ---
    print(f"Initializing Model...")
    model = My_CopulaVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_elements=n_elements,
        n_modes=n_modes, Ke_matrices=Ke_matrices, Mfree=Mfree,
        L_inv=L_inv, n_dims=n_dims, num_gaussians=num_gaussians,
        num_samples=num_samples, gamma=gamma, mean_f=mean_f,
        std_f=std_f, lbound=lbound, fixed_dofs_indices=fixed_dofs_indices
    )
    
    # --- 4. OPTIMIZER & COMPILATION ---
    # Using a learning rate schedule helps with convergence in high dimensions
    lr_schedule = K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=10000,
        decay_rate=0.9
    )
    optim = K.optimizers.Adam(learning_rate=base_lr, clipnorm=0.5)
    
        
    # Capture INITIAL weights for verification
    # We trigger a build first
    _ = model([Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]])
    initial_weights_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    print(f"Initial Encoder Weight Sum: {initial_weights_sum:.6f}")
         
    # --- 4. COMPILATION ---
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
    # ---  CALLBACKS SETUP ---
    # Define the delayed stopping: start after 10k epochs, wait 1k epochs for improvement
    delayed_stop = DelayedEarlyStopping(patience=5000, start_epoch=10000)
    
    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
        y=[alpha_factors_true_train, alpha_factors_true_train],
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], 
            [alpha_factors_true_val, alpha_factors_true_val], 
        ), callbacks=[delayed_stop]
    )
    
    ## with these lines we prevent having to load the trained model in the postprocessing, which was supisciously not working well (FIX)
    inverse_model = model.Encoder_model
    test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    print(f"test means check", test_means)
    
    end_time = time.time()  # Record the ending time
    training_time = end_time - start_time  # Calculate the training time
    print(f"Total training time: {training_time:.2f} seconds")
    # Save the training time
    time_info = {'training_time': training_time}
    np.save(os.path.join(folder_path, 'training_time.npy'), time_info, allow_pickle=True)
    
    
    # --- 7. SAVING RESULTS ---    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'n_gaussians': num_gaussians, 
                    'n_samples': num_samples, 'epochs': n_epochs, 'gamma': gamma, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)   
    Test_predicted_properties = {'test_means': test_means, 'test_scales': test_scales, 'test_offdiag_elems': test_offdiag_elems, 
                    'test_diag_elems': test_diag_elems, 'test_weight_vals': test_weight_vals}
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), Test_predicted_properties, allow_pickle=True)


    
    # --- 6. WEIGHT VERIFICATION AFTER TRAINING ---
    trained_weights_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    print(f"\n--- Weight Integrity Check ---")
    print(f"Pre-training Sum:  {initial_weights_sum:.6f}")
    print(f"Post-training Sum: {trained_weights_sum:.6f}")
    
    if np.isclose(initial_weights_sum, trained_weights_sum):
        print("❌ CRITICAL ERROR: Weights did not change during training! Gradients are not flowing.")
    else:
        print("✅ SUCCESS: Weights updated during training.")
    weights_file = os.path.join(folder_path, "final_model_weights.weights.h5")
    model.save_weights(weights_file)
    print(f"Weights saved to: {weights_file}")
           
    # --- 8. TEST RELOAD IN SAME SESSION ---
    print("\n--- Verifying Saved File (Same Session Reload) ---")
    # Reset model to initial state temporarily
    # (Optional: just check if re-loading into the same object changes nothing)
    model.load_weights(weights_file, by_name=True)
    post_load_sum = np.sum([np.sum(w) for w in model.Encoder_model.FC_encoder.get_weights()])
    print(f"Sum after re-loading saved file: {post_load_sum:.6f}")
    
    # --- 9. PREDICTION VARIATION TEST ---
    print("\n--- Testing Prediction Variation ---")
    preds = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
    

    # --- 7. PLOTTING & PREDICTION ---
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
















# import os
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as K
# import time  # Import the time module

# # --- GPU CONFIGURATION (Must be first) ---
# # Prevents allocating all memory and crashes on some setups
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"✅ GPU Memory Growth Enabled for {len(gpus)} GPU(s)")
#     except RuntimeError as e:
#         print(f"⚠️ GPU Initialization Error: {e}")

# # --- IMPORTS ---
# # Assuming 'MODULES' contains your data loading utils
# try:
#     from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
#     from MODULES.COPULAS.GC_postprocessing_copulas import (
#         plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss
#     )
# except ImportError:
#     print("⚠️ Warning: Could not import preprocessing/postprocessing modules. Ensure paths are correct.")

# # Import local optimized models
# from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen


# def main():
#     # --- 1. GLOBAL SETTINGS ---
#     K.utils.set_random_seed(1234)
#     K.backend.set_floatx('float32')
    
#     # --- 2. DATA LOADING ---
#     # Path Configuration
#     # data_folder = "11Feb2026_Corrected_Randomdata5elements"
#     # data_folder = "28Feb2026_MildDam50_Randomdata5elements"
#     data_folder = "01Mar2026_Noisy_E5_level25"


#     data_path = os.path.join("Data", data_folder)
    
#     # System Parameters
#     n_elements = 5
#     # Total DOFs: 2 per node, (N+1) nodes. 
#     # For 5 elements: 6 nodes * 2 = 12 DOFs.
#     n_dofs = 2 * (n_elements + 1) 
#     lbound = 0.45 # minimum reduction factor to truncate the marginals
    
#     # Boundary Conditions: Simply Supported (Pin-Pin)
#     # Fix Vertical displacement at first node (Index 0) and last node (Index 2*N)
#     # Note: 2*N corresponds to n_dofs - 2
#     fixed_dofs_indices = [0, n_dofs - 2]
    
#     batch_size = 256
    
#     print(f"Loading data from {data_path}...")
#     # Load dataset
#     (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
#      Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
#      Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
#      mean_f, std_f) = load_data(data_path, batch_size)
    
#     # Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train = Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test
#     # Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val = Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test
    
    
#     # Load Physical Matrices (Mass, Stiffness, Cholesky Inverse)
#     Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

#     # --- 3. MODEL CONFIGURATION ---
#     # Input dimension: Freqs + Flattened RotModes + Flattened VertModes
#     input_dim = (Freqs_true_train.shape[1] + 
#                  (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
#                  (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
#     n_modes = Freqs_true_train.shape[1]
    
#     # Training Hyperparameters
#     n_epochs = 30000
#     base_lr = 1e-4
#     epsi = 0.0 # Regularizer weight
    
#     # Copula/GMM Hyperparameters
#     num_gaussians = 1
#     n_dims = alpha_factors_true_train.shape[1]
#     num_samples = 1 # Samples for training (Monte Carlo integration in loss)
#     beta = 0.3 # Weight for the Joint Copula Loss term

#     print(f"Initializing Model with Total DOFs: {n_dofs}, Fixed Indices: {fixed_dofs_indices}")
#     # Instantiate Model
#     # IMPORTANT: We pass 'n_dofs' (Total) not 'num_dofs' (Free), 
#     # because the solver needs to reconstruct the full shape.
#     model = My_CopulaVAE_withEigen(
#         input_dim=input_dim,
#         num_dofs=n_dofs,  # Pass TOTAL DOFs here
#         n_elements=n_elements,
#         n_modes=n_modes,
#         Ke_matrices=Ke_matrices,
#         Mfree=Mfree,
#         L_inv=L_inv,
#         epsi=epsi,
#         n_dims=n_dims,
#         num_gaussians=num_gaussians,
#         num_samples=num_samples,
#         beta=beta,
#         mean_f=mean_f,
#         std_f=std_f,
#         lbound = lbound,
#         fixed_dofs_indices=fixed_dofs_indices
#     )
    
#     # --- 4. COMPILATION & SETUP ---
#     run_eagerly = False # Set True only for debugging
    
#     # Output Directory
#     filename = f"time11Mar_Nosiy2.5Mild50_{lbound}lbound_Bayesian_MACloss_Beta{beta}_Samples{num_samples}_LR{base_lr}_Epochs{n_epochs}"
#     folder_path = os.path.join('Output', "Gaussian_Copula", filename)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
    
#     # Learning Rate Schedule
#     lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#         boundaries=[1000, 8000], 
#         values=[1e-6, 1e-5, 1e-6] 
#     )
    
#     # Compile
#     optimizer = K.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)
    
#     model.compile(
#         optimizer=optimizer, 
#         loss=[model.ELBO_Copula_loss, None],
#         metrics={
#         "output_1": [model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term]
#     },
#         # metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term], 
#         run_eagerly=run_eagerly
#     )

#     # # --- 5. TRAINING ---
#     # # Checkpoints
#     # checkpoint_path = os.path.join(folder_path, "checkpoints", "cp-{epoch:04d}.ckpt")
#     # checkpoint_dir = os.path.dirname(checkpoint_path)
#     # if not os.path.exists(checkpoint_dir):
#     #     os.makedirs(checkpoint_dir)
        
#     # cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     #     filepath=checkpoint_path, 
#     #     verbose=0, 
#     #     save_weights_only=True,
#     #     save_freq='epoch',
#     #     period=100 # Save every 100 epochs
#     # )
    
#     # TensorBoard (Optional, useful for monitoring)
#     # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=folder_path)

#     # ---  Add Time Controller ---
#     start_time = time.time()  # Record the starting time
#     print("Starting Training...")
#     model_history = model.fit(
#         x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
#         y=alpha_factors_true_train,
#         batch_size=batch_size,
#         epochs=n_epochs,
#         shuffle=True,
#         validation_data=(
#             [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], 
#             alpha_factors_true_val
#         ),
#         callbacks=[]
#     )
    
#     # --- 6. SAVING RESULTS ---
#     print("Saving Results...")
    
#     # Save Metadata
#     Problem_info = {
#         'input_dim_enc': input_dim,
#         'n_dims': n_dims, 
#         'n_gaussians': num_gaussians, 
#         'n_samples': num_samples, 
#         'epochs': n_epochs,
#         'epsi': epsi,
#         'beta': beta,
#         'std_f':std_f,
#         'mean_f':mean_f
        
#     }
#     np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
#     # Save History
#     np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
#     # Save Weights
#     model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    
    
#     end_time = time.time()  # Record the ending time
#     training_time = end_time - start_time  # Calculate the training time
    
#     print(f"Total training time: {training_time:.2f} seconds")
    
#     # Save the training time
#     time_info = {'training_time': training_time}
#     np.save(os.path.join(folder_path, 'training_time.npy'), time_info, allow_pickle=True)
    
    
    
#     # --- 7. PLOTTING & PREDICTION ---
#     try:
#         plot_trainval_loss(model, folder_path)
#         plot_JointPDF_loss(model, folder_path)
#         plot_Freqs_loss(model, folder_path)
#     except Exception as e:
#         print(f"Plotting failed: {e}")

#     # Generate predictions on Test Set
    
#     # Unpack the four outputs returned by the model
#     pred_alphas, pred_means, pred_scales, pred_diag_entries, pred_offdiag_entries = model.predict(
#         [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
#     )
    

#     print(f"Predicted Alphas: {pred_alphas[0,:]}")
#     np.save(os.path.join(folder_path, "pred_alphas.npy"), pred_alphas, allow_pickle=True)
   
#     # print(f"verify what are teh test_means after training to verify")
#     # print(f"Predicted means: {pred_means[0,:]}")
#     # np.save(os.path.join(folder_path, "pred_means.npy"), pred_means, allow_pickle=True)

    
#     # np.save(os.path.join(folder_path, "pred_scales.npy"), pred_scales, allow_pickle=True)
#     # np.save(os.path.join(folder_path, "pred_diag_entries.npy"), pred_diag_entries, allow_pickle=True)
#     # np.save(os.path.join(folder_path, "pred_offdiag_entries.npy"), pred_offdiag_entries, allow_pickle=True)


#     print(f"✅ Run Complete. Results saved to: {folder_path}")

# if __name__ == "__main__":
#     main()

    # def count_model_params(model, model_name="Model", input_data=None):
    #     """
    #     Detailed breakdown of trainable and non-trainable parameters.
        
    #     If parameters appear as 0, provide 'input_data' (a sample batch) 
    #     to force the model to build its internal layers.
        
    #     Safety: This function forces execution on CPU to avoid XLA/libdevice 
    #     errors often encountered with jit_compile=True during metadata checks.
    #     """
    #     # Force build if input data is provided, but do it safely on CPU
    #     if input_data is not None:
    #         try:
    #             # Force CPU execution to avoid GPU-specific XLA/libdevice errors
    #             with tf.device('/CPU:0'):
    #                 # We also disable JIT compilation globally for this build pass
    #                 # to prevent searching for libdevice.10.bc
    #                 _ = model(input_data, training=False)
    #         except Exception as e:
    #             print(f"[ERROR] Could not build model safely: {e}")
    
    #     # Helper to sum parameters for a specific component
    #     def get_component_counts(obj):
    #         if obj is None: return 0, 0
    #         try:
    #             trainable = int(np.sum([tf.keras.backend.count_params(w) for w in obj.trainable_weights]))
    #             non_trainable = int(np.sum([tf.keras.backend.count_params(w) for w in obj.non_trainable_weights]))
    #             return trainable, non_trainable
    #         except Exception:
    #             return 0, 0
    
    #     total_trainable, total_non_trainable = get_component_counts(model)
        
    #     print(f"\n{'='*45}")
    #     print(f"REPORT FOR: {model_name}")
    #     print(f"{'='*45}")
    #     print(f"Total Trainable params:     {total_trainable:,}")
    #     print(f"Total Non-trainable params: {total_non_trainable:,}")
    #     print(f"Grand Total:               {total_trainable + total_non_trainable:,}")
        
    #     # Check specific sub-components if they exist
    #     # 1. Encoder
    #     if hasattr(model, 'Encoder_model'):
    #         t, nt = get_component_counts(model.Encoder_model)
    #         print(f"\n[Sub-Model] Encoder")
    #         print(f"  - Trainable:     {t:,}")
    #         if nt > 0: print(f"  - Non-trainable: {nt:,}")
            
    #     # 2. NN Decoder (Surrogate)
    #     if hasattr(model, 'Decoder_NN'):
    #         t, nt = get_component_counts(model.Decoder_NN)
    #         print(f"\n[Sub-Model] NN Decoder")
    #         if t == 0 and nt == 0:
    #             print(f"  - [WARNING] Parameters are 0. Model is likely not 'built'.")
    #         else:
    #             print(f"  - Trainable:     {t:,}")
    #             if nt > 0: print(f"  - Non-trainable: {nt:,}")
                
    #     # 3. Eigen Solver (Physics-based)
    #     if hasattr(model, 'Eigen_solver'):
    #         t, nt = get_component_counts(model.Eigen_solver)
    #         print(f"\n[Sub-Model] Eigen Solver (Physics)")
    #         print(f"  - Trainable:     {t:,} (Constant Physics)")
    
    #     print(f"{'='*45}\n")
    #     return total_trainable

    # # --- EXAMPLE USAGE ---
    # sample_input = [Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]]
    # count_model_params(modelB, "VAE with NN Decoder", input_data=sample_input)
        
    