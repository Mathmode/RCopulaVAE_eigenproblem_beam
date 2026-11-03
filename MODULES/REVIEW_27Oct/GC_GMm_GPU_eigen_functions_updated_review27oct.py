# -*- coding: utf-8 -*-
"""
GC_GMm_GPU_eigen_functions_updated_review27oct.py
=================================================
Full Updated Version of `MODULES/COPULAS/GC_GMm_GPU_eigen_functions.py` for the
MS CPENG-8674 Major Revision (saved in `code/MODULES/review_27oct/`).

Supported & Verified Changes Relative to Original `GC_GMm_GPU_eigen_functions.py`:
----------------------------------------------------------------------------------
1. [FIX 1 -- `self.L_inv` Set to `trainable=False` (Critical Physics Integrity Fix)]
   - In the original `GC_GMm_GPU_eigen_functions.py` (line 106), `self.L_inv` was defined
     with `trainable=True`.
   - Why this must be fixed: During training, Adam computed gradients with respect to
     `self.L_inv` and modified the inverse Cholesky factor of the mass matrix at every step!
     This contradicted Section 2.2 of the manuscript (which states `L_inv` is a constant
     precomputed from the undamaged mass matrix `M_G`) and caused a mismatch during
     post-processing when reloading the original `Mass_matrix.npy` from disk.
   - Fix: `trainable=False` (line 138 below).

2. [FIX 2 -- Exact Cholesky Transformation Order `A = L_inv @ K @ L_inv^T` & `phi = L_inv^T @ v`]
   - In `Noisy_random_dataset_generation.py` (lines 78-80, 102-104) and in your October
     `Ground_truth_MassPert_KDE_plots.py` (lines 81, 86), `L = np.linalg.cholesky(M_free)` is
     lower-triangular (`M = L @ L.T`), so the generalized eigenvalue problem `K @ phi = lam * L @ L.T @ phi`
     is transformed via `v = L.T @ phi` (`phi = L_inv.T @ v`) into:
         `A = L_inv @ K @ L_inv.T`,    `phi_free = L_inv.T @ eigvecs`
   - In the original `GC_GMm_GPU_eigen_functions.py` (lines 120-122 and 139), the transpose was
     inverted (`A = L_inv.T @ K @ L_inv` and `phi_free = L_inv @ eigvecs`), which corresponds
     to `M = L.T @ L != L @ L.T`! You had already corrected this in `Ground_truth_MassPert_KDE_plots.py`
     (line 80: `# CORRECTED: Match the data generation script (L_inv @ K @ L_inv_T)`) and in
     `updated_GC_GMm_functions_for_results_analysis.py` (line 67), so we align the training
     layer here to match the exact same physics.

3. [FIX 3 -- Support for Affine Zone & Elastic-Bearing Assembly + Sensor Node Extraction]
   - Added `assemble_affine_Kfree` and optional `sensor_nodes` in `Solve_eigenproblem` so the
     exact same physics layer also works when a 20- or 40-element mesh is parameterized by
     5 or 10 damage zones or includes elastic bearing latents `(s_L, s_R)`.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


# -------------------------------------------------------------------------
# 1. VECTORIZED ELEMENT ASSEMBLY (100% compatible with 5-element & 10-element models)
# -------------------------------------------------------------------------
@tf.function(jit_compile=True)
def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples, fixed_dofs_indices=None):
    """
    Vectorized assembly of global stiffness matrices from scaled element matrices (Batch, n_el, 4, 4).
    """
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype=tf.float32)
    batch_size_total = tf.shape(Ke_matrices_dam)[0]
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes

    K_values_flat = tf.reshape(Ke_matrices_dam, [batch_size_total, -1])

    indices_list = []
    for i in range(n_elements):
        node_i = i
        node_j = i + 1
        global_dofs = [2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1]
        rows = np.repeat(global_dofs, 4)
        cols = np.tile(global_dofs, 4)
        elem_indices = np.stack([rows, cols], axis=1)
        indices_list.append(elem_indices)

    base_indices = np.concatenate(indices_list, axis=0)
    base_indices_tf = tf.constant(base_indices, dtype=tf.int32)

    num_vals = n_elements * 16
    batch_indices = tf.repeat(tf.range(batch_size_total), num_vals)
    tiled_base_indices = tf.tile(base_indices_tf, [batch_size_total, 1])

    full_indices = tf.concat([
        tf.expand_dims(batch_indices, 1),
        tf.cast(tiled_base_indices, tf.int32),
    ], axis=1)

    flat_values = tf.reshape(K_values_flat, [-1])
    K_global = tf.scatter_nd(full_indices, flat_values, shape=[batch_size_total, n_dofs, n_dofs])

    if fixed_dofs_indices is None:
        fixed_list = [0, int(n_dofs) - 2]
    elif isinstance(fixed_dofs_indices, (list, tuple, np.ndarray)):
        fixed_list = [int(x) for x in fixed_dofs_indices]
    else:
        fixed_list = [0, int(n_dofs) - 2]

    if len(fixed_list) > 0:
        free_list = [i for i in range(int(n_dofs)) if i not in set(fixed_list)]
        free_dofs_tf = tf.constant(free_list, dtype=tf.int32)
        K_free = tf.gather(K_global, free_dofs_tf, axis=1)
        K_free = tf.gather(K_free, free_dofs_tf, axis=2)
    else:
        K_free = K_global

    return K_free


@tf.function(jit_compile=True)
def assemble_affine_Kfree(alpha_samples, Kzone_free, Ksupport_free=None, K0_free=None):
    """
    Fast affine stiffness assembly for zone-discretized meshes and elastic bearings:
        K_free(z, s) = K0_free + sum_{z=1}^{n_zones} z_z * Kzone_free[z]
                               + sum_{b=1}^{n_sup}   s_b * Ksupport_free[b]
    """
    alpha_samples = tf.cast(alpha_samples, tf.float32)
    Kzone_free = tf.cast(Kzone_free, tf.float32)
    n_zones = tf.shape(Kzone_free)[0]

    z_zones = alpha_samples[:, :n_zones]
    K_free = tf.einsum("BZ, ZKQ -> BKQ", z_zones, Kzone_free)

    if K0_free is not None:
        K_free = K_free + tf.expand_dims(tf.cast(K0_free, tf.float32), axis=0)

    if Ksupport_free is not None:
        s_sup = alpha_samples[:, n_zones:]
        K_sup = tf.einsum("BS, SKQ -> BKQ", s_sup, tf.cast(Ksupport_free, tf.float32))
        K_free = K_free + K_sup

    return K_free


# -------------------------------------------------------------------------
# 2. ROBUST EIGENSOLVER LAYER (Fixed `L_inv` + Exact Cholesky Order + Sign Consistency)
# -------------------------------------------------------------------------
@tf.function(jit_compile=True)
def safe_eigh(tensor):
    return tf.linalg.eigh(tensor)


class Solve_eigenproblem(Layer):
    def __init__(self, num_dofs_total, n_modes, L_inv, fixed_dofs_indices=None, sensor_nodes=None, **kwargs):
        """
        Args:
            num_dofs_total     : Total DOFs of the FE mesh (2 * n_nodes)
            n_modes            : Number of vibration modes to extract
            L_inv              : Constant inverse lower-Cholesky of nominal M_free
            fixed_dofs_indices : Fixed support DOFs ([0, num_dofs_total - 2] for pinned; [] for elastic)
            sensor_nodes       : Optional list of node indices where sensors are placed
        """
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs_total = num_dofs_total
        self.n_modes = n_modes
        self.sensor_nodes = sensor_nodes

        if fixed_dofs_indices is None:
            fixed_dofs_np = np.array([0, num_dofs_total - 2], dtype=np.int32)
        elif tf.is_tensor(fixed_dofs_indices):
            fixed_dofs_np = fixed_dofs_indices.numpy().astype(np.int32)
        else:
            fixed_dofs_np = np.array(fixed_dofs_indices, dtype=np.int32)

        all_indices = np.arange(num_dofs_total, dtype=np.int32)
        self.free_dofs = np.delete(all_indices, fixed_dofs_np) if len(fixed_dofs_np) > 0 else all_indices
        self.fixed_dofs = fixed_dofs_np

        # [FIX 1]: trainable=False so the mass matrix Cholesky factor remains strictly constant!
        self.L_inv = tf.Variable(
            initial_value=tf.cast(L_inv, dtype=tf.float32),
            trainable=False,
            name="L_inverse_matrix",
        )

    def call(self, inputs):
        """
        Input : Kfree_matrices (Batch, N_free, N_free)
        Output: freqs_hz (Batch, Modes), rot_modes (Batch, Modes, Sensors), vert_modes (Batch, Modes, Sensors)
        """
        Kfree = tf.cast(inputs, dtype=tf.float32)
        batch_size = tf.shape(Kfree)[0]

        # [FIX 2]: Exact Cholesky congruence matching `Noisy_random_dataset_generation.py`:
        # Since M_free = L @ L.T, A = L_inv @ K_free @ L_inv.T and phi_free = L_inv.T @ eigvecs
        L_inv_T = tf.transpose(self.L_inv)
        A = tf.matmul(self.L_inv, tf.matmul(Kfree, L_inv_T))

        jitter = 1e-6 * tf.eye(tf.shape(A)[1], batch_shape=[batch_size])
        A = 0.5 * (A + tf.transpose(A, perm=[0, 2, 1])) + jitter

        eigvals, eigvecs_transformed = safe_eigh(A)

        eigvals = eigvals[:, :self.n_modes]
        eigvecs_transformed = eigvecs_transformed[:, :, :self.n_modes]

        eigvals = tf.nn.relu(eigvals)
        freqs_rad = tf.sqrt(eigvals + 1e-12)
        freqs_hz = freqs_rad / (2.0 * np.pi)

        # [FIX 2]: Recover physical mode shapes on free DOFs: phi_free = L_inv.T @ eigvecs
        physical_modes_free = tf.matmul(L_inv_T, eigvecs_transformed)

        # Reconstruct full DOF vector (inserting zeros at fixed support DOFs if any)
        flat_modes = tf.reshape(physical_modes_free, [-1, self.n_modes])
        batch_indices = tf.repeat(tf.range(batch_size), len(self.free_dofs))
        dof_indices = tf.tile(tf.constant(self.free_dofs, dtype=tf.int32), [batch_size])
        scatter_indices = tf.stack([batch_indices, dof_indices], axis=1)

        physical_modes_full = tf.scatter_nd(
            indices=scatter_indices,
            updates=flat_modes,
            shape=[batch_size, self.num_dofs_total, self.n_modes],
        )

        # Separate Vertical and Rotational nodal DOFs: shape (Batch, Nodes, Modes)
        vert_modes = physical_modes_full[:, 0::2, :]
        rot_modes = physical_modes_full[:, 1::2, :]

        # If a fine FE mesh is used, extract mode shapes at the physical sensor nodes
        if self.sensor_nodes is not None:
            s_idx = tf.constant(self.sensor_nodes, dtype=tf.int32)
            vert_modes = tf.gather(vert_modes, s_idx, axis=1)
            rot_modes = tf.gather(rot_modes, s_idx, axis=1)

        # Sign Consistency: positive peak displacement per mode.
        # Note: For Mode 5 on a 5-element pinned beam, vertical nodal displacements are at a
        # spatial Nyquist zero-crossing (sin(5*pi*k/5) = 0), so we use rot_modes whenever
        # max(|vert_modes|) < 1e-3.
        abs_vert = tf.abs(vert_modes)
        max_vert_val = tf.reduce_max(abs_vert, axis=1, keepdims=True)  # (Batch, 1, Modes)
        ref_modes = tf.where(max_vert_val > 1e-3, vert_modes, rot_modes)

        max_indices = tf.cast(tf.argmax(tf.abs(ref_modes), axis=1), dtype=tf.int32)
        batch_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), 1), tf.shape(max_indices))
        mode_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(self.n_modes), 0), tf.shape(max_indices))
        gather_indices = tf.stack([batch_idx_grid, max_indices, mode_idx_grid], axis=2)

        peak_vals = tf.gather_nd(ref_modes, gather_indices)
        signs = tf.sign(peak_vals)
        signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
        signs_expanded = tf.expand_dims(signs, axis=1)

        vert_modes = vert_modes * signs_expanded
        rot_modes = rot_modes * signs_expanded

        # Transpose to [Batch, Modes, Sensors]
        vert_modes = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_modes = tf.transpose(rot_modes, perm=[0, 2, 1])

        return freqs_hz, rot_modes, vert_modes
