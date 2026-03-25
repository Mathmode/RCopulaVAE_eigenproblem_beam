#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2026
Updated for Kumaraswamy Mixture Marginals
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
from MODULES.KUMARSWAMY.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss

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
    data_folder = "01Mar2026_Noisy_E5_level25"
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
    
    # --- MODEL CONFIGURATION ---
    input_dim = (Freqs_true_train.shape[1] + 
                 (Rotmodes_true_train.shape[1] * Rotmodes_true_train.shape[2]) + 
                 (Vertmodes_true_train.shape[1] * Vertmodes_true_train.shape[2]))
    
    n_modes = Freqs_true_train.shape[1]
    n_epochs = 800
    base_lr = 1e-5
    num_KS = 2  # Number of Kumaraswamy components
    n_dims = alpha_factors_true_train.shape[1]
    num_samples = 1 
    gamma = 0.45
    
    filename = f"25Mar_KS_Copula_{n_elements}Els_{num_KS}KSmix_Gamma{gamma}_LR{base_lr}_{n_epochs}epoch"
    folder_path = os.path.join('Output', "KS_Copula", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # --- INITIALIZATION ---
    model = My_CopulaKSVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_elements=n_elements,
        n_modes=n_modes, Ke_matrices=Ke_matrices, Mfree = Mfree, L_inv=L_inv, 
        n_dims=n_dims, num_KS=num_KS, num_samples=num_samples, 
        gamma=gamma, mean_f=mean_f, std_f=std_f, lbound=lbound, 
        fixed_dofs_indices=fixed_dofs_indices
    )
    
    optim = K.optimizers.Adam(learning_rate=base_lr, clipnorm=0.5)
    
    # Force build
    _ = model([Freqs_true_train[:1], Rotmodes_true_train[:1], Vertmodes_true_train[:1], alpha_factors_true_train[:1]])
    
    model.compile(
        optimizer=optim, 
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term]
    )
    
    # --- TRAINING ---
    start_time = time.time()
    print("Starting Kumaraswamy Copula Training...")
    delayed_stop = DelayedEarlyStopping(patience=1000, start_epoch=10000)
    
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
    
    # Save Weights and History
    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    ## with these lines we prevent having to load the trained model in the postprocessing, which was supisciously not working well (FIX)
    inverse_model = model.Encoder_model
    test_a, test_b, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test]
    )
    
    # --- 7. SAVING RESULTS ---    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'num_KS': num_KS, 
                    'n_samples': num_samples, 'epochs': n_epochs, 'gamma': gamma, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)   
    Test_predicted_properties = {'test_a': test_a, 'test_b': test_b, 'test_offdiag_elems': test_offdiag_elems, 
                    'test_diag_elems': test_diag_elems, 'test_weight_vals': test_weight_vals}
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), Test_predicted_properties, allow_pickle=True)

    
    # --- PLOTTING ---
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()