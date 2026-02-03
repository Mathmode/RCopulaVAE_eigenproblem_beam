
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:04:41 2025

@author: afernandez

Optimized GC_GMm_eigen_functions.py for GPU Performance.

Key Improvements:
1. fully vectorized `assemble_global_Kmatrices` (removes slow tf.map_fn loops).
2. Robust `Solve_eigenproblem` using standard eigenvalue decomposition for symmetric matrices.

@author: afernandez (Optimized)
"""
import tensorflow as tf
import tensorflow.keras as K 
import numpy as np 
from tensorflow.keras.layers import Layer

# -------------------------------------------------------------------------
# 1. VECTORIZED ASSEMBLY (Critical for GPU Speed)
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples):
    """
    Vectorized assembly of global stiffness matrices.
    Removes tf.map_fn to allow massive parallelization on GPU.
    
    Args:
        Ke_matrices_dam: Shape (Batch_Size * H, n_elements, 4, 4) (Assuming 2D beam 4x4)
        n_elements: Number of finite elements
        num_samples: Number of samples (H)
        
    Returns:
        K_free: Shape (Batch_Size * H, n_free_dofs, n_free_dofs)
    """
    # 1. Cast and Shapes
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype=tf.float32)
    batch_size_total = tf.shape(Ke_matrices_dam)[0] # This is Batch * num_samples
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes
    
    # 2. Flatten Element Matrices
    # Current shape: (B, n_elems, 4, 4) -> (B, n_elems * 16)
    # This aligns with how we will scatter them.
    K_values_flat = tf.reshape(Ke_matrices_dam, [batch_size_total, -1]) 

    # 3. Precompute Indices (ONCE, assumed constant structure)
    # We need to map every value in the 4x4 element matrix to its global (row, col) position.
    # DOF mapping: Element i connects nodes i and i+1.
    # Node i has DOFs [2i, 2i+1]. Node i+1 has DOFs [2i+2, 2i+3].
    
    # Create base indices for one single system (no batch dim yet)
    indices_list = []
    
    # 4x4 generic indices (0,1,2,3) mapped to global DOFs
    for i in range(n_elements):
        # Global DOF indices for this element
        global_dofs = [2*i, 2*i+1, 2*i+2, 2*i+3]
        
        # Cartesian product of these DOFs gives the 16 indices for the 4x4 matrix
        # Row indices: repeat each DOF 4 times
        rows = np.repeat(global_dofs, 4)
        # Col indices: tile the DOFs 4 times
        cols = np.tile(global_dofs, 4)
        
        # Stack them
        elem_indices = np.stack([rows, cols], axis=1) # (16, 2)
        indices_list.append(elem_indices)
        
    # Combine all element indices
    # Shape: (n_elements * 16, 2)
    base_indices = np.concatenate(indices_list, axis=0)
    base_indices_tf = tf.constant(base_indices, dtype=tf.int32)
    
    # 4. Vectorized Scatter (The GPU Magic)
    # We want to scatter K_values_flat into K_global.
    # K_values_flat shape: (B, N_vals)
    # Target shape: (B, n_dofs, n_dofs)
    
    # To do this efficiently in TF without a loop, we simply iterate over the scalar values (N_vals)
    # and add them to the correct (row, col) for ALL batches simultaneously.
    # Wait, tf.vectorized_map is essentially a loop. 
    # Better approach: effectively sum the sparse representations.
    
    # Since structure is identical for all batches, we can simply map the "values" dimension
    # to the "indices" dimension.
    
    # However, tf.scatter_nd usually requires indices to include the batch dimension if done flatly.
    # BUT, we can use `tf.map_fn` ONLY if strictly compiled, OR use `tf.vectorized_map`.
    # Actually, the fastest way in TF for fixed topology is often simple matrix addition if N is small,
    # or `tf.scatter_nd` with tiled indices.
    
    # Construct full indices with batch dimension?
    # No, that consumes huge memory.
    
    # Alternative: Reshape to (B, n_dofs, n_dofs) via math operations? No.
    
    # BEST GPU APPROACH for Fixed Topology:
    # 1. Create a "Template" Index tensor of shape (B, N_vals, 3) where 3 is (b, row, col).
    # 2. Use scatter_nd.
    
    num_vals = n_elements * 16
    
    # Create batch indices: [0, 0, ..., 0], [1, 1, ... 1], ...
    batch_indices = tf.repeat(tf.range(batch_size_total), num_vals)
    
    # Tile the base indices for the whole batch
    # Shape: (B * num_vals, 2)
    tiled_base_indices = tf.tile(base_indices_tf, [batch_size_total, 1])
    
    # Combine: (B * num_vals, 3) -> [batch_idx, row, col]
    full_indices = tf.concat([
        tf.expand_dims(batch_indices, 1), 
        tf.cast(tiled_base_indices, tf.int32)
    ], axis=1)
    
    # Flatten values to match indices
    flat_values = tf.reshape(K_values_flat, [-1])
    
    # Scatter!
    K_global = tf.scatter_nd(full_indices, flat_values, shape=[batch_size_total, n_dofs, n_dofs])
    
    # 5. Extract Free DOFs
    # Assuming fixed at start (0) and end (n_dofs-2) based on your snippet [0, n_dofs-2]
    # NOTE: Your snippet had `fixed_dofs = [0, n_dofs-2]`. 
    # Usually fixed DOFs are 0, 1 (left node) and N-2, N-1 (right node)? 
    # I will stick to your logic: removing 0 and -2. 
    # Wait, -2 is second to last. If N=12 (indices 0..11), -2 is 10. 11 is free?
    # Typically beams are fixed at 0 (vertical), 1 (rotation) ??
    # I will strictly follow your code: `fixed_dofs = [0, n_dofs-2]`
    
    # Generate list of free indices
    all_indices = np.arange(n_dofs)
    fixed_dofs_np = [0, n_dofs - 2] # Hardcoded based on your snippet
    free_dofs_np = np.delete(all_indices, fixed_dofs_np)
    free_dofs_tf = tf.constant(free_dofs_np, dtype=tf.int32)
    
    # Slicing is faster than gather
    # But free_dofs might not be contiguous. Gather is fine.
    K_free = tf.gather(K_global, free_dofs_tf, axis=1)
    K_free = tf.gather(K_free, free_dofs_tf, axis=2)
    
    return K_free

# -------------------------------------------------------------------------
# 2. ROBUST EIGENSOLVER
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def safe_eigh(tensor):
    """
    Wrapper for self-adjoint eigendecomposition.
    Much faster/stable than generic eig for symmetric matrices.
    """
    return tf.linalg.eigh(tensor)

class Solve_eigenproblem(Layer):
    """
    Optimized and Robust Eigenproblem Solver Layer.
    Combines stability checks + JIT compilation.
    """
    def __init__(self, num_dofs, n_modes, Mfree, L_inv, **kwargs):
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_modes = n_modes
        self.Mfree = Mfree
        self.L_inv = tf.cast(L_inv, dtype=tf.float32)
        
    def call(self, inputs):
        """
        Input: Kfree_matrices (Batch, D, D)
        Output: Freqs (Hz), RotModes, VertModes
        """
        Kfree = tf.cast(inputs, dtype=tf.float32)
        
        # 1. Transform to Standard Eigenproblem
        # A = L^-T * K * L^-1
        # Transpose(L_inv) * K * L_inv
        # L_inv shape usually (D, D).
        
        # Helper: Ensure L_inv orientation is correct. 
        # Usually L_inv is Lower Triangular from Cholesky of M.
        # M = L * L.T  ->  K x = w^2 M x  -> K x = w^2 L L.T x
        # let y = L.T x  -> x = L.T^-1 y
        # L.T^-1 K L^-1 y = w^2 y
        # A = (L^-1).T * K * (L^-1)
        
        # Efficient batch matmul
        L_inv_T = tf.transpose(self.L_inv)
        
        # A = L_inv_T @ (K @ L_inv)
        # We can use broadcasting since L_inv is constant shared across batch
        temp = tf.matmul(Kfree, self.L_inv) # (B, D, D) * (D, D) -> (B, D, D)
        A = tf.matmul(L_inv_T, temp)        # (D, D) * (B, D, D) -> (B, D, D) via broadcasting? 
        # TF matmul broadcasts automatically if ranks differ, but let's be explicit
        # A = tf.matmul(tf.expand_dims(L_inv_T, 0), temp) # Safest if shape issues arise
        # Actually, standard tf.matmul(A, B) works if A is (B,N,M) and B is (B,M,K).
        # Here L_inv is (D,D). 
        # Correct efficient way:
        A = tf.matmul(tf.matmul(Kfree, self.L_inv), self.L_inv, transpose_b=True)
        # Note: (K * L) * L_T is equivalent to L * K * L_T if symmetric? 
        # Wait, the formula is A = L_inv.T * K * L_inv.
        # Let's stick to the explicit one that worked for you:
        A = tf.matmul(L_inv_T, tf.matmul(Kfree, self.L_inv))

        # 2. Jitter for Stability (CRITICAL)
        # Add epsilon to diagonal to ensure positive definiteness during training
        # prevents NaNs in gradients of sqrt() later.
        jitter = 1e-7 * tf.eye(self.num_dofs, batch_shape=tf.shape(Kfree)[:1])
        A = A + jitter
        
        # 3. Solve Eigenvalues
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = safe_eigh(A)
        
        # 4. Filter & Process
        # Take first n_modes
        eigvals = eigvals[:, :self.n_modes]
        eigvecs = eigvecs[:, :, :self.n_modes]
        
        # Safe Sqrt (ReLu-like clipping for safety against tiny negatives)
        eigvals = tf.nn.relu(eigvals) 
        freqs_rad = tf.sqrt(eigvals + 1e-12) # Add epsilon inside sqrt gradient
        freqs_hz = freqs_rad / (2.0 * np.pi)
        
        # 5. Recover Physical Modes
        # x = L_inv * y
        # physical_modes = L_inv @ eigvecs
        physical_modes = tf.matmul(self.L_inv, eigvecs)
        
        # 6. Split Modes (Rot/Vert)
        # Assuming structure: [R, V, R, V, ... R, R] based on your snippet
        # Slice excluding last row for strict R/V pattern?
        # Your previous code: eig_cut = eigenmodes_trunc[:, :-1, :]
        
        modes_cut = physical_modes[:, :-1, :]
        
        # Rotational: Even indices (0, 2, ...) + Last one
        rot_part_main = modes_cut[:, 0::2, :]
        rot_part_last = physical_modes[:, -1:, :]
        rot_modes = tf.concat([rot_part_main, rot_part_last], axis=1)
        
        # Vertical: Odd indices (1, 3, ...)
        vert_modes = modes_cut[:, 1::2, :]
        
        return freqs_hz, rot_modes, vert_modes

# Alias for compatibility if needed
SolveEigenproblemStable = Solve_eigenproblem