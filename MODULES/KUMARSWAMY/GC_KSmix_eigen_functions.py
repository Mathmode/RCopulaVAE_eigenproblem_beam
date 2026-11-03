#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:04:41 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow.keras as K 
import tensorflow_probability as tfp
import numpy as np 
tfd = tfp.distributions
from tensorflow.keras.layers import Layer


# JIT-compiled function for the eigenvalue decomposition.
# Defining it outside the class is slightly cleaner but functionally equivalent.
@tf.function(jit_compile=True)
def jit_eigh(tensor):
    """JIT-compiled wrapper for tf.linalg.eigh."""
    eigenval, eigenvec = tf.linalg.eigh(tensor)
    return eigenval, eigenvec

# --- STRATEGY 1: Improved and Stabilized Original Layer ---
# This version includes better numerical stability and minor refactoring.

class SolveEigenproblemStable(Layer):
    """
    Solves the generalized eigenvalue problem by converting it to a standard one.
    This version includes a small diagonal shift for improved numerical stability,
    which can prevent issues with non-positive definite matrices that arise
    from numerical precision errors, especially during training.
    """
    def __init__(self, num_dofs, n_modes, Mfree, L_inv, **kwargs):
        super(SolveEigenproblemStable, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_modes = n_modes  # The number of modes to retain
        self.Mfree = Mfree      # Fixed mass matrix (known)
        self.L_inv = tf.cast(L_inv, dtype=tf.float32)

    def call(self, inputs):
        """
        The input is a tensor of Kfree matrices with shape (batch_size, num_dofs, num_dofs).
        """
        Kfree_matrices = tf.cast(inputs, dtype=tf.float32)
        # --- NaN/Inf Debugging ---
        # Add a check to fail fast if the inputs are already invalid.
        Kfree_matrices = tf.debugging.check_numerics(Kfree_matrices, "Input Kfree_matrices contains NaN or Inf")


        # Compute the generalized eigenvalue problem by transforming it into a standard one.
        # A = L_inv^T * Kfree * L_inv
        A = tf.matmul(tf.transpose(self.L_inv, perm=[0, 2, 1]) if len(self.L_inv.shape) > 2 else tf.transpose(self.L_inv), 
                      tf.matmul(Kfree_matrices, self.L_inv))

        # --- Stability Improvement ---
        # Add a small identity matrix (epsilon * I) to A.
        # This ensures the matrix is positive definite, preventing numerical errors
        # that can lead to negative eigenvalues, which is more robust than clipping.
        identity = tf.eye(self.num_dofs, batch_shape=[tf.shape(A)[0]], dtype=A.dtype)
        A_stable = A + identity * 1e-9 # Increased epsilon slightly for more stability

        # Solve the standard eigenvalue problem
        eigenvalues, eigenvectors = jit_eigh(A_stable)

        # Truncate to select the desired number of modes
        # The eigenvalues from eigh are sorted in non-decreasing order.
        eigenvalues_trunc = eigenvalues[:, :self.n_modes]
        
        # --- Prevent NaN from sqrt of negative numbers ---
        # Despite stabilization, numerical precision can still yield tiny negative numbers.
        # Explicitly clip them to a small positive value before the square root.
        eigenvalues_trunc_positive = tf.clip_by_value(eigenvalues_trunc, 1e-12, 1e+24)

        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.math.sqrt(eigenvalues_trunc_positive)

        # Convert to frequencies in Hz (f = ω / 2π)
        frequencies_Hz = frequencies_rad_s / (2.0 * np.pi)

        # Recover the original eigenmodes (v = L_inv * y)
        eigenmodes = tf.matmul(self.L_inv, eigenvectors)
        eigenmodes_trunc = eigenmodes[:, :, :self.n_modes]

        # Deconstruct modes into rotational and vertical components (problem-specific)
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes_trunc[:, :-1, :]
        rot_modes = K.layers.Concatenate(axis=1)([eig_cut[:, 0::2, :], eigenmodes_trunc[:, -1:, :]])
        vert_modes = eig_cut[:, 1::2, :]

        return frequencies_Hz, rot_modes, vert_modes




class Solve_eigenproblem(Layer):
    def __init__(self, num_dofs, n_modes, Mfree, L_inv, **kwargs):
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_modes = n_modes # the number of modes we want to retain
        self.Mfree = Mfree # Fixed mass matrix (known) given a tensor*
        self.L_inv = tf.cast(L_inv,dtype = tf.float32
                             
                             )
        #Intentamos agilizar el entrenamiento: 
        @tf.function(jit_compile = True)
        def jit_eigh(tensor):
            eigenval, eigenvec= tf.linalg.eigh(tensor)
            return eigenval, eigenvec
        self.jit_eigh = jit_eigh
        
    def call(self, inputs):
        #The input is the tensor with the Kfree matrices with shape (Batch_size, num_dofs, num_dofs)
        Kfree_matrices = tf.cast(inputs, dtype = tf.float32)        
        # Compute the generalized eigenvalue problem by transforming into a standard one for each system in the batch
        A = tf.matmul(tf.transpose(self.L_inv), tf.matmul(Kfree_matrices, self.L_inv))
        eigenvalues, eigenvectors = self.jit_eigh(A)
        # Clipping to avoid negative values (do not think it is working as we want)
        eigenvalues = tf.clip_by_value(eigenvalues, clip_value_min=1e-08, clip_value_max=1e+20) #We do this since one of the GPUs whas finding negative values (probably due to intialization issues)
        #Truncate to select the desired number of modes:
        eigenvalues_trunc = eigenvalues[:,0:self.n_modes]
        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.math.sqrt(eigenvalues_trunc)
        # Convert to frequencies in Hz (ω = 2πf)
        frequencies_Hz = frequencies_rad_s / (2 * tf.constant(np.pi, dtype='float32')) # TODO tf.constant translation to KEras
        

        eigenmodes =tf.matmul(tf.transpose(self.L_inv), eigenvectors)
        eigenmodes_trunc = eigenmodes[:,:, 0:self.n_modes]
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes_trunc[:,0:-1,:]
        rot_modes = K.layers.Concatenate(axis=1)([eig_cut[:,0::2,:],eigenmodes_trunc[:,-1:,:]])
        vert_modes = eig_cut[:,1::2,:]
        
        return frequencies_Hz, rot_modes, vert_modes


def new_generate_indices(n_elements, dof_indices):
    # Generate indices for scattering the local element matrices into global matrix
    indices = []
    for i in range(n_elements):
        idx_grid = tf.meshgrid(dof_indices[i], dof_indices[i])
        element_indices = tf.stack([tf.reshape(idx_grid[0], [-1]), tf.reshape(idx_grid[1], [-1])], axis=-1)
        # Append to indices list
        indices.append(element_indices)
    # Concatenate along the elements dimension
    return K.layers.Concatenate(axis=0)(indices)

def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples):
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype = tf.float32)
    ## Since we are drawing H samples before entering the decoder, the input dimension of Ke_matrices_dam is (Batch_size* num_samples, n_elements, 2,2)
    # TODO if num_samples brings problem, i'd avoid it as we will always use H =1 .......
    n_dofs = 2 * (n_elements + 1)  # 2 DOFs per node, n_elements + 1 nodes

    # Prepare DOF indices for each element
    #TODO meter esta variable dof_indices en el self 
    dof_indices = tf.constant([[2*i, 2*i+1, 2*i+2, 2*i+3] for i in range(n_elements)], dtype=tf.int32)

    K_flattened = tf.reshape(Ke_matrices_dam, [-1, n_elements * 16]) # Ke_matrices contains n_elements matrices of shape (4x4) and this for each sample in the batch 
    # after flattening we have the entries that are nonzero in the global assembled matrix. 
    
    nonzero_indices = new_generate_indices(n_elements, dof_indices)
    K_gg = tf.map_fn(
    lambda _: tf.zeros((n_dofs, n_dofs), dtype=tf.float32),
    elems=tf.range(tf.shape(Ke_matrices_dam)[0]),  # Dynamically infer the batch size
    fn_output_signature=tf.TensorSpec((n_dofs, n_dofs), dtype=tf.float32)
)

    def fill_matrix(nonzero_vals, matrix):
        # Assign values to the matrix
        indices = nonzero_indices  # Use the fixed indices
        updates = nonzero_vals  # Use the known nonzero values in K_flattened
        matrix = tf.tensor_scatter_nd_add(matrix, indices, updates)
        return matrix

    # Batch-wise filling
    def fillK_with_batch(batch_data):
        nonzero_vals, matrix = batch_data
        return fill_matrix(nonzero_vals, matrix)

    # Use tf.map_fn to apply the filling function across the batch
    K_global = tf.map_fn(
        fillK_with_batch,
        (K_flattened, K_gg),
        fn_output_signature=tf.TensorSpec((n_dofs, n_dofs), dtype=tf.float32)
    )
    
    fixed_dofs = [0, n_dofs-2]
    free_dofs = tf.constant([i for i in range(n_dofs) if i not in fixed_dofs], dtype= 'int32')
    ## Select the desired dofs (free ones) in the axis 1 and 2, since axis 0 corresponds to batch size. 
    K_free = tf.gather(tf.gather(K_global, free_dofs, axis=1), free_dofs, axis=2) #  transfrom tf.gather to K compatible
    # The final shape of K_free is***** (batch_size*H, n_free,n_free) * review*
    
    return K_free