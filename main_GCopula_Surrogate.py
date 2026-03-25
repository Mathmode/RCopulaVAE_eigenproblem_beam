#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 2026

@author: afernandez
Main Script for Two-Step Training of GC-GMM Copula with Surrogate Decoder.
Step 1: Pre-train Decoder (Forward Physics)
Step 2: Train VAE (Inverse Mapping)
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
from MODULES.COPULAS.GC_GMm_model_surrogatedecoder import My_CopulaVAE_Surrogate, calculate_MAC
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_JointPDF_loss, plot_Freqs_loss

# --- CUSTOM CALLBACK ---
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
    
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')
    
    data_folder = "01Mar2026_Noisy_E5_level25"
    # data_folder = "16Mar2026_Noisy_E10_level25_5modes"

    data_path = os.path.join("Data", data_folder)
    
    n_elements = 5
    n_dofs = 2 * (n_elements + 1) 
    lbound = 0.45 
    fixed_dofs_indices = [0, n_dofs - 2]
    batch_size = 256
    
    print(f"Loading data from {data_path}...")
    (Freq_train, Rot_train, Vert_train, Alpha_train, 
     Freq_val, Rot_val, Vert_val, Alpha_val, 
     Freq_test, Rot_test, Vert_test, Alpha_test, 
     mean_f, std_f) = load_data(data_path, batch_size)

    # --- DTYPE CORRECTION ---
    # Convert all datasets to float32 to avoid TypeErrors during manual loops
    Freq_train = Freq_train.astype('float32')
    Rot_train = Rot_train.astype('float32')
    Vert_train = Vert_train.astype('float32')
    Alpha_train = Alpha_train.astype('float32')

    Freq_val = Freq_val.astype('float32')
    Rot_val = Rot_val.astype('float32')
    Vert_val = Vert_val.astype('float32')
    Alpha_val = Alpha_val.astype('float32')
    
    Freq_test = Freq_test.astype('float32')
    Rot_test = Rot_test.astype('float32')
    Vert_test = Vert_test.astype('float32')
    Alpha_test = Alpha_test.astype('float32')
    
    mean_f = np.float32(mean_f)
    std_f = np.float32(std_f)


    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)

    # --- 2. MODEL CONFIG ---
    input_dim = Freq_train.shape[1] + (Rot_train.shape[1]*Rot_train.shape[2]) + (Vert_train.shape[1]*Vert_train.shape[2])
    n_modes = Freq_train.shape[1]
    n_nodes = Rot_train.shape[2]
    n_dims = Alpha_train.shape[1]
    
    # Hyperparameters
    num_gaussians = 1
    num_samples = 1 
    beta = 0.3
    pretrain_epochs = 10000 # Step 1 duration
    vae_epochs = 50000    # Step 2 duration
    
    # Output Directory
    filename = f"22Mar128_2stepSURROGATE_{n_elements}Els_Beta{beta}_{vae_epochs}Epochs"
    folder_path = os.path.join('Output', "Surrogate_Comparison", filename)
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    # --- 3. INITIALIZE MODEL ---
    model = My_CopulaVAE_Surrogate(
        input_dim=input_dim, n_modes=n_modes, n_nodes_per_mode=n_nodes, 
        n_dims=n_dims, num_gaussians=num_gaussians, num_samples=num_samples, 
        beta=beta, mean_f=mean_f, std_f=std_f, lbound=lbound
    )

    # Trigger build
    _ = model([Freq_train[:1], Rot_train[:1], Vert_train[:1], Alpha_train[:1]])

    # --- 4. STEP 1: PRE-TRAIN DECODER (FORWARD PHYSICS) ---
    print("\n" + "="*50)
    print("STEP 1: Training Surrogate Decoder on Ground Truth Alphas")
    print("="*50)

    # We only train the Surrogate_decoder component here
    optimizer_dec = K.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step_decoder(alphas, f_true, r_true, v_true):
        with tf.GradientTape() as tape:
            p_f, p_r, p_v = model.Surrogate_decoder(alphas)
            
            # Loss 1: Frequency MSE
            # Note: We compare in Hz here because that's what Surrogate outputs
            # But the dataset usually comes scaled/logged. 
            # If f_true is already scaled, we should apply scale to p_f for comparison.
            p_f_log = tf.math.log(tf.maximum(p_f, 1e-6))
            p_f_scaled = (p_f_log - mean_f) / std_f
            loss_f = tf.reduce_mean(tf.square(f_true - p_f_scaled))
            
            # Loss 2: MAC Loss
            mac_mat_r = calculate_MAC(r_true, p_r)
            mac_mat_v = calculate_MAC(v_true, p_v)
            loss_m = (tf.reduce_mean(1.0 - tf.reduce_max(mac_mat_r, axis=2)) + 
                      tf.reduce_mean(1.0 - tf.reduce_max(mac_mat_v, axis=2)))
            
            total_dec_loss = loss_f + 10.0 * loss_m
            
        grads = tape.gradient(total_dec_loss, model.Surrogate_decoder.trainable_variables)
        optimizer_dec.apply_gradients(zip(grads, model.Surrogate_decoder.trainable_variables))
        return total_dec_loss, loss_f, loss_m

    # Manual Loop for Step 1
    for epoch in range(pretrain_epochs):
        # Shuffling
        idx = np.random.permutation(len(Alpha_train))
        Alpha_s, Freq_s, Rot_s, Vert_s = Alpha_train[idx], Freq_train[idx], Rot_train[idx], Vert_train[idx]
        
        epoch_loss = 0
        num_batches = len(Alpha_train) // batch_size
        for i in range(num_batches):
            b_a = Alpha_s[i*batch_size:(i+1)*batch_size]
            b_f = Freq_s[i*batch_size:(i+1)*batch_size]
            b_r = Rot_s[i*batch_size:(i+1)*batch_size]
            b_v = Vert_s[i*batch_size:(i+1)*batch_size]
            
            loss_val, lf, lm = train_step_decoder(b_a, b_f, b_r, b_v)
            epoch_loss += loss_val
            
        if (epoch + 1) % 100 == 0:
            print(f"Pre-train Epoch {epoch+1}/{pretrain_epochs} - Loss: {epoch_loss/num_batches:.6f} (F: {lf:.4f}, M: {lm:.4f})")

    # --- 5. STEP 2: VAE TRAINING (FROZEN DECODER) ---
    print("\n" + "="*50)
    print("STEP 2: Training VAE with Frozen Pre-trained Decoder")
    print("="*50)
    
    # Freeze decoder to allow encoder to find stable latent space first
    model.Surrogate_decoder.trainable = False
    
    # --- 4. OPTIMIZER & COMPILATION ---
    # Using a learning rate schedule helps with convergence in high dimensions
    lr_schedule = K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9
    )
    optim_vae = K.optimizers.Adam(learning_rate=1e-05, clipnorm=0.5)
    
    model.compile(
        optimizer=optim_vae, 
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.Joint_copula_dens_term]
    )

    start_time = time.time()
    
    delayed_stop = DelayedEarlyStopping(patience=1000, start_epoch=10000)

    model_history = model.fit(
        x=[Freq_train, Rot_train, Vert_train, Alpha_train],
        y=Alpha_train, # Dummy y, ELBO_Copula_loss uses internal attributes
        batch_size=batch_size,
        epochs=vae_epochs,
        shuffle=True,
        validation_data=(
            [Freq_val, Rot_val, Vert_val, Alpha_val], 
            Alpha_val
        ), 
        callbacks=[delayed_stop]
    )
    
    ## with these lines we prevent having to load the trained model in the postprocessing, which was supisciously not working well (FIX)
    inverse_model = model.Encoder_model
    test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems = inverse_model.predict(
        [Freq_test, Rot_test, Vert_test, Alpha_test]
    )
    print(f"test means check", test_means)
    # --- 6. SAVING & RESULTS ---
    training_time = time.time() - start_time
    print(f"VAE training time: {training_time:.2f} seconds")
    
    # Save the training time
    time_info = {'training_time': training_time}
    np.save(os.path.join(folder_path, 'training_time.npy'), time_info, allow_pickle=True)
    # --- 7. SAVING RESULTS ---    
    Problem_info = {'input_dim_enc': input_dim, 'n_dims': n_dims, 'n_gaussians': num_gaussians, 
                    'n_samples': num_samples, 'epochs': vae_epochs, 'beta': beta, 'std_f':std_f, 'mean_f':mean_f}
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)   
    Test_predicted_properties = {'test_means': test_means, 'test_scales': test_scales, 'test_offdiag_elems': test_offdiag_elems, 
                    'test_diag_elems': test_diag_elems, 'test_weight_vals': test_weight_vals}
    np.save(os.path.join(folder_path, 'Test_predicted_props.npy'), Test_predicted_properties, allow_pickle=True)


    
    # Save weights and metadata
    model.save_weights(os.path.join(folder_path, "two_step_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    
    
    
    
    # Plotting
    try:
        plot_trainval_loss(model, folder_path)
        plot_JointPDF_loss(model, folder_path)
        plot_Freqs_loss(model, folder_path)
    except:
        print("Plotting skipped.")

    print(f"✅ Two-step run complete. Results in: {folder_path}")

if __name__ == "__main__":
    main()