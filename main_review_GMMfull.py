# -*- coding: utf-8 -*-
"""
Main Training Script for GMM Posterior VAE with Adaptive Stiffness Jitter and Fallbacks.
Full Covariance Implementation (Cholesky Factors).
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import time 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization Error: {e}")

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_Freqs_loss

from MODULES.REVIEW.GMM_models import My_FullCovGMMVAE_withEigen
from MODULES.REVIEW.GMM_functions_for_results_analysis import calculate_gmm_metrics, plot_results_PDF_uncertainty

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
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\n✅ Early stopping triggered at epoch {epoch}.")
                    self.model.stop_training = True
        elif current_loss < self.best_loss:
            self.best_loss = current_loss

def main():
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    # --- DATA LOADING ---
    data_folder = "06Sept2026_Noisy_E50_level25_50modes"
    data_path = os.path.join("Data", data_folder)
    
    n_elements = 50
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
    
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    n_epochs = 10000
    
    base_lr = 1e-4 
    num_components = 2
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = 1 
    gamma = 0.4
    
    filename = f"Full08Sept26_GMM_{n_elements}Els_{num_components}Mix_Gamma{gamma}_LR{base_lr}_{n_epochs}epoch"
    folder_path = os.path.join('Output', "GMM_VAE", filename)
    os.makedirs(folder_path, exist_ok=True)
    
    model = My_FullCovGMMVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_elements=n_elements,
        n_modes=n_modes, Ke_matrices=Ke_matrices, Mfree=Mfree, L_inv=L_inv, 
        n_dims=n_dims, num_components=num_components, num_samples=num_samples, 
        gamma=gamma, mean_f=mean_f, std_f=std_f, lbound=lbound, 
        fixed_dofs_indices=fixed_dofs_indices
    )
    
    optim = K.optimizers.Adam(learning_rate=base_lr, clipnorm=0.5)
    
    # Force build
    _ = model([Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]])
    
    model.compile(
        optimizer=optim, 
        loss=model.ELBO_GMM_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.GMM_KL_Divergence_Loss]
    )
    
    start_time = time.time()
    print("Starting GMM-VAE Training with Full Covariance & Monte Carlo KL...")
    delayed_stop = DelayedEarlyStopping(patience=500, start_epoch=5000)
    
    model_history = model.fit(
        x=[Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
        y=[alpha_factors_true_train],
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        validation_data=(
            [Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], 
            [alpha_factors_true_val], 
        ), callbacks=[delayed_stop]
    )
    
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    inverse_model = model.Encoder_model
    test_logits, test_locs, test_offdiag, test_diag = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'num_components': num_components, 
                    'n_samples': num_samples, 'epochs': n_epochs, 'gamma': gamma, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
    predicted_stats = {
        'test_logits': test_logits, 'test_locs': test_locs, 
        'test_offdiag': test_offdiag, 'test_diag': test_diag
    }
    test_datasets = {
        'alpha_factors_true_test': alpha_factors_true_test,
        'Freqs_true_test': Freqs_true_test,
        'Rotmodes_true_test': Rotmodes_true_test,
        'Vertmodes_true_test': Vertmodes_true_test
    }
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), predicted_stats, allow_pickle=True)
    fixed_dofs = [0, n_dofs - 2] 
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)
    print("\n--- Calculating Quantitative GMM Metrics (Table 4) ---")
    physics_kwargs = {
        'Ke_matrices': Ke_matrices,
        'L_inv': L_inv,
        'n_modes': n_modes,
        'free_dofs': free_dofs,
        'fixed_dofs': fixed_dofs_indices,
        'n_dofs': n_dofs,
        'mean_freq': mean_f,
        'std_freq': std_f,
        'gamma': gamma
    }
    
    metrics, covered = calculate_gmm_metrics(predicted_stats, test_datasets, lbound, n_mc_samples=5000, physics_kwargs=physics_kwargs)
    for key, value in metrics.items():
        print(f"{key}: {value:.5f}")
        
    with open(os.path.join(folder_path, 'GMM_Metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    # print("\n--- Generating Corner Plots ---")
    # posiciones_a_evaluar = [0, 10, 50] 
    # for pos in posiciones_a_evaluar:
    #     print(f"Plotting sample index {pos}...")
    #     try:
    #         plot_results_PDF_uncertainty(
    #             fixed_dofs_indices=fixed_dofs_indices, n_modes=n_modes, gamma=gamma, 
    #             n_samples=4096, pos=pos, n_dofs=n_dofs, free_dofs=model.free_dofs, 
    #             test_datasets=test_datasets, predicted_stats=predicted_stats, 
    #             L_inv=L_inv, Ke_matrices=Ke_matrices, Mfree=Mfree, 
    #             mean_freq=mean_f, std_freq=std_f, lbound=lbound, folder_path=folder_path
    #         )
    #     except Exception as e:
    #         print(f"Plotting failed for position {pos}: {e}")

    # try:
    #     plot_trainval_loss(model, folder_path)
    #     plot_Freqs_loss(model, folder_path)
    # except Exception as e:
    #     print(f"Loss Plotting failed: {e}")

if __name__ == "__main__":
    main()