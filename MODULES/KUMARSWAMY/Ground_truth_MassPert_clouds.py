#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth Point Cloud Generator (Perturbed Mass Model)
Updated to incorporate the systemic modeling error (mass perturbation) 
to match the exact physics used to generate the test data.
"""

# %% 1. IMPORTS & SETUP
import os

# Fix for XLA / libdevice path in Conda environments
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.path.join(conda_prefix, 'Library')}"

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
import random

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_eigen_functions import Solve_eigenproblem, assemble_global_Kmatrices

# GPU and Precision Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
K.utils.set_random_seed(1234)
K.backend.set_floatx('float32')

# %% 2. DATA LOADING & PHYSICS INITIALIZATION
folder_path = os.path.join("Output", "Gaussian_Copula_Perturbed")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Pointing to the new perturbed dataset
data_folder = "06Oct_Data_Noisy_E5_Lvl50_MassPerturb5"
data_path = os.path.join("Data", data_folder)

n_elements = 5
n_dofs = 2 * (n_elements + 1) 
num_dofs = n_dofs - 2 
batch_size = 48
lbound = 0.40

# Load test data (UPDATED to unpack all 14 values)
(Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, 
 Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, 
 Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test,
 mean_f, std_f) = load_data(data_path, batch_size)

# Load the unperturbed baseline stiffness matrices 
_, Ke_matrices, _ = load_known_matrices(data_path, n_elements)

# EXPLICITLY load the perturbed mass matrix to match the test data generation
Mfree_perturbed = np.load(os.path.join(data_path, "Mass_matrix_perturbed.npy"))

# Recalculate the Cholesky inverse for the perturbed mass
L = np.linalg.cholesky(Mfree_perturbed)
L_inv_perturbed = np.linalg.inv(L)

# %% 3. METRICS FUNCTIONS
def calculate_MAC(modes_true, modes_pred):
    """Robust MAC calculation matching the KSmix analysis script."""
    modes_true = tf.cast(modes_true, tf.float32)
    modes_pred = tf.cast(modes_pred, tf.float32)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return tf.linalg.diag_part(mac_matrix).numpy()

# %% 4. GROUND TRUTH SAMPLING FUNCTION
def select_gt_pointcloud(Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, 
                         Mfree_pert, Ke_matrices, L_inv_pert, n_elements, num_dofs, folder_path, lbound):
    point_clouds  = []
    N = 20000 # Number of points to form the cloud to be used as Ground Truth (GT)
    num_samples = 1
    n_modes = Freqs_true_test.shape[1] 

    # Generate random damage features within the defined bounds
    ztest = []
    for j in range(N):
        z1 = random.uniform(lbound, 1.0)
        z2 = random.uniform(lbound, 1.0)
        z3 = random.uniform(lbound, 1.0)
        z4 = random.uniform(lbound, 1.0)
        z5 = random.uniform(lbound, 1.0)
        ztest.append([z1, z2, z3, z4, z5])
    
    ztest = np.array(ztest) 
    
    # Apply the corresponding factor to the baseline stiffness
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(ztest, dtype=tf.float32), tf.cast(Ke_matrices, dtype=tf.float32))
    Kfree = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples) 
    
    # Initialize the eigensolver with the PERTURBED mass matrix
    Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree_pert, L_inv_pert)
    
    # Solve forward dynamics (Forced to CPU to bypass XLA compiler)
    with tf.device('/CPU:0'):
        pred_freqs, pred_rotmodes, pred_vertmodes = Eigen_solver(Kfree)
    
    pred_freqs = np.abs(np.array(pred_freqs)) 
    
    # TRANSPOSE to match the (Batch, Modes, Nodes) shape of the true dataset
    pred_rotmodes = np.transpose(np.array(pred_rotmodes), (0, 2, 1))
    pred_vertmodes = np.transpose(np.array(pred_vertmodes), (0, 2, 1))
    
    # PAD the vertical modes with the fixed boundary nodes (zeros at x=0 and x=L)
    # This transforms pred_vertmodes from (Batch, 5, 4) to (Batch, 5, 6)
    batch_sz = pred_vertmodes.shape[0]
    modes_sz = pred_vertmodes.shape[1]
    boundary_zeros = np.zeros((batch_sz, modes_sz, 1), dtype=np.float32)
    pred_vertmodes = np.concatenate([boundary_zeros, pred_vertmodes, boundary_zeros], axis=2)
   
    def Freqs_loss(y_true, y_pred):
        Freqs_sq_error = np.square(np.log(y_true) - np.log(y_pred))
        return np.mean(Freqs_sq_error, axis=1)
    
    def MAC_modes_loss(true_rotmodes, true_vertmodes, pred_rotmodes, pred_vertmodes):
        Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) 
        Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) 
        MACs = np.concatenate([Rot_MACs, Vert_MACs], axis=1)
        return np.mean(1 - MACs, axis=1) 
    
    # Evaluate over all test cases
    for i in range(Freqs_true_test.shape[0]):
        print(f"Processing GT for test sample {i}...")
        Freqs_true = Freqs_true_test[i, :].reshape(1, Freqs_true_test.shape[1]) 
        ftrue = np.array(tf.repeat(Freqs_true, N, axis=0)) 
        
        Mrot_true = Rotmodes_true_test[i, :].reshape(1, Rotmodes_true_test.shape[1], Rotmodes_true_test.shape[2])
        mrot_true  = np.array(tf.repeat(Mrot_true, N, axis=0))
        
        Mvert_true = Vertmodes_true_test[i, :].reshape(1, Vertmodes_true_test.shape[1], Vertmodes_true_test.shape[2])
        mvert_true  = np.array(tf.repeat(Mvert_true, N, axis=0))
        
        # Calculate evaluation losses
        floss = Freqs_loss(ftrue, pred_freqs)
        MAC_loss = MAC_modes_loss(mrot_true, mvert_true, pred_rotmodes, pred_vertmodes)
        total_loss = floss + MAC_loss
    
        # Get the indices of the best N matches (sorting the point cloud by likelihood proxy)
        top_k_indices = np.argsort(total_loss)[:N]
    
        z1_filtered = ztest[top_k_indices, 0]
        z2_filtered = ztest[top_k_indices, 1]
        z3_filtered = ztest[top_k_indices, 2]
        z4_filtered = ztest[top_k_indices, 3]
        z5_filtered = ztest[top_k_indices, 4]
        Loss_data_filtered = total_loss[top_k_indices]            
        
        combined_array = np.column_stack((z1_filtered, z2_filtered, z3_filtered, z4_filtered, z5_filtered, Loss_data_filtered))
        point_clouds.append(combined_array)
    
    point_clouds = np.array(point_clouds)      
    np.save(os.path.join(folder_path, "Test_Point_clouds.npy"), point_clouds)
    print(f"Ground truth point clouds saved to {folder_path}")

# %% 5. MAIN EXECUTION
if __name__ == "__main__":
    select_gt_pointcloud(
        Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, 
        Mfree_perturbed, Ke_matrices, L_inv_perturbed, 
        n_elements, num_dofs, folder_path, lbound
    )