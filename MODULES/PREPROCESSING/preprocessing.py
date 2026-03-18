# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:18:33 2020
Updated: March 2026 for 10-element GC-GMM Scaling
@author: 109457 / afernandez
"""

import os
import numpy as np

def load_known_matrices(data_path, n_elements):
    """
    Loads constant Mass and Stiffness matrices. 
    Improved for 10-element stability.
    """
    # Load Mass Matrix
    M_free = np.load(os.path.join(data_path, "Mass_matrix.npy"))
    
    # Numerical stability check for Cholesky
    # High-dimensional matrices (10+ elements) are more prone to being singular
    jitter = 1e-10
    try:
        L = np.linalg.cholesky(M_free)
    except np.linalg.LinAlgError:
        print("⚠️ Mass matrix not PSD. Applying jitter...")
        L = np.linalg.cholesky(M_free + np.eye(M_free.shape[0]) * jitter)
        
    L_inv = np.linalg.inv(L)

    # Load Base Stiffness Matrix
    # We assume a uniform beam where Ke is the same for all elements
    Ke_base = np.load(os.path.join(data_path, "Ke_matrix.npy"))
    
    # Ensure Ke_matrices is tiled correctly for the assembly layer
    Ke_matrices = np.tile(Ke_base, (n_elements, 1, 1))

    return M_free, Ke_matrices, L_inv

def load_data(data_path, batch_size):
    """
    Loads and standardizes frequency and modal data.
    Ensures standardization is consistent with the VAE Loss function.
    """
    print(f"--- Loading 10-Element Dataset from: {data_path} ---")
    
    Freqs_true = np.load(os.path.join(data_path, "freqs_data_true.npy"))
    Rotmodes_true = np.load(os.path.join(data_path, "rotmodes_data_true.npy"))
    Vertmodes_true = np.load(os.path.join(data_path, "vertmodes_data_true.npy"))
    alpha_factors_true = np.load(os.path.join(data_path, "alpha_factors_true.npy"))  
    
    # -------------------------------------------------------------------------
    # 1. LOG TRANSFORM
    # -------------------------------------------------------------------------
    # In 10 elements, higher modes have much larger frequency values.
    # Log compression is mandatory to keep the network's weights from exploding.
    Freqs_log = np.log(Freqs_true + 1e-7) 
    
    

    # -------------------------------------------------------------------------
    # 2. DATA SPLITTING (Batch-aligned)
    # -------------------------------------------------------------------------
    Nblocks = Freqs_log.shape[0] // batch_size 
    Ntrain = int(np.ceil(0.6 * Nblocks)) * batch_size 
    Nval = int(np.ceil(0.2 * Nblocks)) * batch_size
    
    # Split into Train, Val, Test
    # Using the same slicing logic across all arrays
    def split_set(arr):
        train = arr[0:Ntrain, ...]
        val = arr[Ntrain:Ntrain+Nval, ...]
        test = arr[Ntrain+Nval:, ...]
        return train, val, test

    f_train, f_val, f_test = split_set(Freqs_log)
    r_train, r_val, r_test = split_set(Rotmodes_true)
    v_train, v_val, v_test = split_set(Vertmodes_true)
    a_train, a_val, a_test = split_set(alpha_factors_true)

    # -------------------------------------------------------------------------
    # 3. GLOBAL STANDARDIZATION (Frequency Only)
    # -------------------------------------------------------------------------

    # 2. mode-wise standardization (CRITICAL)
    # Instead of global mean/std, we use mode-wise stats.
    # We calculate stats ONLY on training data to prevent data leakage.
    # For 10 dimensions, calculating per-mode mean/std is more stable than a global scalar.
    mean_f = np.mean(f_train, axis=0) 
    std_f = np.std(f_train, axis=0)
    
    # Safety: avoid division by zero for modes with no variance
    std_f[std_f < 1e-8] = 1.0

    # Apply transform
    f_train_std = (f_train - mean_f) / std_f
    f_val_std = (f_val - mean_f) / std_f
    f_test_std = (f_test - mean_f) / std_f

    # -------------------------------------------------------------------------
    # 4. MODAL NORMALIZATION CHECK
    # -------------------------------------------------------------------------
    # Mode shapes should ideally be unit-norm to keep MAC loss within [0, 1] range.
    # Vertmodes_true and Rotmodes_true are usually already processed, 
    # but we ensure they are float32 here.
    r_train = r_train.astype('float32')
    v_train = v_train.astype('float32')

    print(f"Standardization complete. Freq Mean: {np.mean(mean_f):.4f}, Freq Std: {np.mean(std_f):.4f}")
    print(f"Final Sets: Train({f_train_std.shape[0]}), Val({f_val_std.shape[0]}), Test({f_test_std.shape[0]})")

    return (f_train_std, r_train, v_train, a_train, 
            f_val_std, r_val, v_val, a_val, 
            f_test_std, r_test, v_test, a_test, 
            mean_f, std_f)





























# #import packages
# import os
# import numpy as np
# import tensorflow as tf
# # from MODULES.PREPROCESSING.preprocessing_tools import import_data, rem_zeros, standardization, rescaling, mode_UnitNormalizer

# #############################################################################################################################
# #Solve the preprocessing: obtain std datasets for train/val/test/dam_test
# def load_known_matrices(data_path, n_elements):
#     """
#     Loads constant Mass and Stiffness matrices saved during generation.
#     Returns M_free (Mass) and L_inv (Inverse Cholesky of Mass).
#     """
#     # Load Mass Matrix
#     M_free = np.load(os.path.join(data_path, "Mass_matrix.npy"))
    
#     # Calculate L_inv (Inverse Cholesky) used for Physics Layer
#     # M = L * L.T  ->  L_inv = inv(L)
#     try:
#         L = np.linalg.cholesky(M_free)
#     except np.linalg.LinAlgError:
#         # Stability fallback
#         M_free += np.eye(M_free.shape[0]) * 1e-12
#         L = np.linalg.cholesky(M_free)
        
#     L_inv = np.linalg.inv(L)

#     # Load Base Stiffness Matrix (Element level) if needed, 
#     # though usually the physics layer reconstructs global K from alphas.
#     # Here we return the base element matrices for the Physics Layer to use.
#     Ke_base = np.load(os.path.join(data_path, "Ke_matrix.npy"))
    
#     # Ke_matrices expects shape (n_elements, 4, 4) or similar?
#     # The Physics layer usually takes the base Ke and scales it by alpha.
#     # We return a list or array of base Ke matrices.
#     # Assuming Ke_base is the SAME for all elements (uniform beam):
#     Ke_matrices = np.tile(Ke_base, (n_elements, 1, 1))

#     return M_free, Ke_matrices, L_inv

# def load_data(data_path, batch_size):
#     print(f"Loading data from: {data_path}")
    
#     Freqs_true = np.load(os.path.join(data_path, "freqs_data_true.npy"))
#     Rotmodes_true = np.load(os.path.join(data_path, "rotmodes_data_true.npy"))
#     Vertmodes_true = np.load(os.path.join(data_path, "vertmodes_data_true.npy"))
#     alpha_factors_true = np.load(os.path.join(data_path, "alpha_factors_true.npy"))  
    
#     # -------------------------------------------------------------------------
#     # 1. LOG TRANSFORM (Crucial for VAE dynamics)
#     # -------------------------------------------------------------------------
#     # Compresses dynamic range and linearizes stiffness-frequency relation
#     Freqs_true = np.log(Freqs_true + 1e-6) 

#     # -------------------------------------------------------------------------
#     # 2. SPLITTING
#     # -------------------------------------------------------------------------
#     Nblocks = Freqs_true.shape[0] // batch_size 
#     Ntrain = int(np.ceil(0.6 * Nblocks)) * batch_size 
#     Nval = int(np.ceil(0.2 * Nblocks)) * batch_size
#     Ntest = (Nblocks - (int(np.ceil(0.6 * Nblocks)) + int(np.ceil(0.2 * Nblocks)))) * batch_size
    
#     # Train
#     Freqs_true_train = Freqs_true[0:Ntrain, :]
#     Rotmodes_true_train = Rotmodes_true[0:Ntrain, :]
#     Vertmodes_true_train = Vertmodes_true[0:Ntrain, :]
#     alpha_factors_true_train = alpha_factors_true[0:Ntrain, :]

#     # Val
#     Freqs_true_val = Freqs_true[Ntrain:Ntrain+Nval, :]
#     Rotmodes_true_val = Rotmodes_true[Ntrain:Ntrain+Nval, :]
#     Vertmodes_true_val = Vertmodes_true[Ntrain:Ntrain+Nval, :]
#     alpha_factors_true_val = alpha_factors_true[Ntrain:Ntrain+Nval, :]

#     # Test
#     Freqs_true_test = Freqs_true[Ntrain+Nval:Ntrain+Nval+Ntest, :]
#     Rotmodes_true_test = Rotmodes_true[Ntrain+Nval:Ntrain+Nval+Ntest, :]
#     Vertmodes_true_test = Vertmodes_true[Ntrain+Nval:Ntrain+Nval+Ntest, :]
#     alpha_factors_true_test = alpha_factors_true[Ntrain+Nval:Ntrain+Nval+Ntest, :]
    
#     # -------------------------------------------------------------------------
#     # 3. STANDARD SCALER (Non-negotiable for NNs)
#     # -------------------------------------------------------------------------
#     mean_f = np.mean(Freqs_true_train, axis=0) # Axis 0 is safer
#     std_f = np.std(Freqs_true_train, axis=0)
    
#     # Avoid division by zero
#     std_f[std_f < 1e-08] = 1.0

#     Freqs_true_train = (Freqs_true_train - mean_f) / std_f
#     Freqs_true_val = (Freqs_true_val - mean_f) / std_f
#     Freqs_true_test = (Freqs_true_test - mean_f) / std_f

#     print("Data Loaded.")
#     print(f"Train samples: {Freqs_true_train.shape[0]}")
#     print(f"Val samples: {Freqs_true_val.shape[0]}")
#     print(f"Test samples: {Freqs_true_test.shape[0]}")

#     return (Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
#             Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
#             Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, 
#             mean_f, std_f)


# ############################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&64





























# def load_data(data_path, batch_size):
#     Freqs_true = np.load(os.path.join(data_path, "freqs_data_true.npy"))
#     Rotmodes_true = np.load(os.path.join(data_path, "rotmodes_data_true.npy"))
#     Vertmodes_true = np.load(os.path.join(data_path, "vertmodes_data_true.npy"))
#     alpha_factors_true = np.load(os.path.join(data_path,  "alpha_factors_true.npy"))  
    
    
#     # 1. Take Log of frequencies
#     Freqs_true = np.log(Freqs_true + 1e-6) 

    
#     Nblocks = Freqs_true.shape[0]//batch_size #number of blocks in my dataset
#     Ntrain = int(np.ceil(0.6*Nblocks))*batch_size #samples to be allocated for training
#     Nval = int(np.ceil(0.2*Nblocks))*batch_size
#     Ntest = (Nblocks - ( int(np.ceil(0.6*Nblocks)) +int(np.ceil(0.2*Nblocks))))*batch_size
    
#     Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train = Freqs_true[0:Ntrain,:], Rotmodes_true[0:Ntrain,:], Vertmodes_true[0:Ntrain,:], alpha_factors_true[0:Ntrain,:]
#     Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val  = Freqs_true[Ntrain:Ntrain+Nval,:], Rotmodes_true[Ntrain:Ntrain+Nval,:], Vertmodes_true[Ntrain:Ntrain+Nval,:], alpha_factors_true[Ntrain:Ntrain+Nval,:]
#     Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test = Freqs_true[Ntrain+Nval:Ntrain+Nval+Ntest,:], Rotmodes_true[Ntrain+Nval:Ntrain+Nval+Ntest,:], Vertmodes_true[Ntrain+Nval:Ntrain+Nval+Ntest,:], alpha_factors_true[Ntrain+Nval:Ntrain+Nval+Ntest,:]

    
#     # 2. (Optional but recommended) Standard Scaler
#     # Subtract mean, divide by std
#     mean_f = np.mean(Freqs_true_train)
#     std_f = np.std(Freqs_true_train)
#     Freqs_true_train = (Freqs_true_train - mean_f) / std_f
    
#     Freqs_true_val = (Freqs_true_val - mean_f) / std_f
#     Freqs_true_test = (Freqs_true_test - mean_f) / std_f

    
#     return Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, mean_f, std_f
# #################################################################################################################################


# def load_known_matrices(data_path, n_elements):
#     Mfree = np.load(os.path.join(data_path,"Mass_matrix.npy")) #This matrix remains the same regardless the scenario
#     Ke = np.load(os.path.join(data_path, "Ke_matrix.npy"))
#     Ke = tf.expand_dims(Ke, axis = 0)  # Shape: (1, 4, 4)
#     Ke_matrices = tf.tile(Ke, [n_elements, 1, 1])
#     L_inv = np.load(os.path.join(data_path, "L_inv.npy"))

#     return Mfree, Ke_matrices, L_inv

# def preprocessing_interface(bridge_loc):
#     if bridge_loc == "Z24":
#         #standardization of the data (applied to the input data)
#         Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, Data_real= preprocessing_FEMU_Z24()
#         std_model = rescaling(Ytrain,0.5,1.5)
#         # std_model = standardization(Ytrain)
#         Ytrain_std = std_model.transform(Ytrain)
#         Yval_std = std_model.transform(Yval)
#         Ytest_std = std_model.transform(Ytest) 
#     if bridge_loc == "PORTO":
#         #standardization of the data (applied to the input data)
#         Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, Data_real, Test_Experimental= preprocessing_FEMU_Porto()
#         std_model = rescaling(Ytrain,0 ,1)
#         # std_model = standardization(Ytrain)
#         Ytrain_std = std_model.transform(Ytrain)
#         Yval_std = std_model.transform(Yval)
#         Ytest_std = std_model.transform(Ytest) 
#         # Ytrain_std =Ytrain
#         # Yval_std = Yval
#         # Ytest_std = Ytest

#     return Xtrain, Xval, Xtest, Ytrain_std, Yval_std, Ytest_std, std_model, Data_real, Test_Experimental

