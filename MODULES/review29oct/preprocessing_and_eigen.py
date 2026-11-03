#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/preprocessing_and_eigen.py

Fast Vectorized Global Stiffness Assembly & Differentiable Generalized Eigensolver
for the 7D Bridge-with-Bearings Model (100% XLA-safe, zero dynamic shape ops).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def load_bridge_data(data_dir):
    """Loads pre-split train, validation, and fine-mesh test datasets."""
    npz_path = os.path.join(data_dir, "bridge_7d_splits.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Dataset split file not found at {npz_path}. Run dataset_generation_7D_bridge.py first."
        )
    d = np.load(npz_path, allow_pickle=True)
    return (
        d["f_train"], d["r_train"], d["v_train"], d["a_train"],
        d["f_val"],   d["r_val"],   d["v_val"],   d["a_val"],
        d["f_test"],  d["r_test"],  d["v_test"],  d["a_test"],
        d["mean_f"],  d["std_f"]
    )


def load_baseline_matrices(data_dir):
    """Loads baseline mass matrix Mfree, K_basis (7, n_dofs, n_dofs), and Cholesky inverse L_inv."""
    Mfree = np.load(os.path.join(data_dir, "Mass_matrix.npy")).astype(np.float32)
    K_basis = np.load(os.path.join(data_dir, "K_basis_matrices.npy")).astype(np.float32)
    L_inv = np.load(os.path.join(data_dir, "L_inv_matrix.npy")).astype(np.float32)
    return Mfree, K_basis, L_inv


@tf.function(jit_compile=True)
def assemble_global_Kmatrices_bearings(z_samples, K_basis):
    """
    Fast O(1) vectorized assembly of the global stiffness matrix for a batch of 7D stiffness factors.
      z_samples : Tensor of shape (Batch, 7)
      K_basis   : Tensor of shape (7, n_dofs, n_dofs)
      Returns   : K_global of shape (Batch, n_dofs, n_dofs)
    """
    z_samples = tf.cast(z_samples, tf.float32)
    K_basis = tf.cast(K_basis, tf.float32)
    return tf.einsum("bd,dij->bij", z_samples, K_basis)


class Solve_Bridge_Eigenproblem(Layer):
    """
    Differentiable Generalized Eigensolver for the 7D Bridge with Elastic Bearings.
    Uses static one-hot peak sign extraction (zero `tf.shape` or `tf.gather_nd` calls).
    """
    def __init__(self, num_dofs, n_modes, L_inv, n_sensor_nodes=6, eig_device=None, **kwargs):
        super(Solve_Bridge_Eigenproblem, self).__init__(**kwargs)
        # eig_device="/CPU:0" runs tf.linalg.eigh with LAPACK on the CPU. For thousands of small
        # (22x22) matrices this is much faster than the GPU (per-matrix cuSolver loop) or the
        # XLA Jacobi eigh. Requires the train step NOT to be XLA-compiled (jit_compile=False).
        self.eig_device = eig_device
        self.num_dofs = int(num_dofs)
        self.n_modes = int(n_modes)
        self.n_sensor_nodes = int(n_sensor_nodes)
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)
        self.L_inv_T = tf.transpose(self.L_inv)

        n_elements = (self.num_dofs // 2) - 1
        sensor_nodes = np.linspace(0, n_elements, self.n_sensor_nodes, dtype=np.int32)
        self.vert_dofs = tf.constant(2 * sensor_nodes, dtype=tf.int32)
        self.rot_dofs = tf.constant(2 * sensor_nodes + 1, dtype=tf.int32)

        # Static diagonal stabilization matrix (1, num_dofs, num_dofs)
        jitter_vec = np.linspace(1e-7, 5e-7, self.num_dofs, dtype=np.float32)
        self.diag_jitter_mat = tf.constant(np.diag(jitter_vec)[np.newaxis, :, :], dtype=tf.float32)

    def call(self, K_matrices):
        K_matrices = tf.cast(K_matrices, dtype=tf.float32)

        # Standardize generalized eigenproblem: A = L_inv * K * L_inv^T
        KL = tf.matmul(K_matrices, self.L_inv_T)
        A = tf.matmul(self.L_inv, KL)
        A_sym = 0.5 * (A + tf.transpose(A, perm=[0, 2, 1]))
        A_stable = A_sym + self.diag_jitter_mat

        # Solve symmetric eigenvalue problem (native GPU cusolver / XLA safe)
        if self.eig_device is None:
            eigvals, eigvecs = tf.linalg.eigh(A_stable)
        else:
            with tf.device(self.eig_device):
                eigvals, eigvecs = tf.linalg.eigh(A_stable)

        eigvals_trunc = tf.clip_by_value(eigvals[:, :self.n_modes], 1e-8, 1e12)
        frequencies_Hz = tf.math.sqrt(eigvals_trunc) / (2.0 * np.pi)

        # Recover physical DOFs: phi = L_inv^T * y -> shape (Batch, num_dofs, n_modes)
        modes_full = tf.matmul(self.L_inv_T, eigvecs[:, :, :self.n_modes])

        # Gather sensor stations: shape (Batch, n_sensor_nodes, n_modes)
        vert_modes = tf.gather(modes_full, self.vert_dofs, axis=1)
        rot_modes = tf.gather(modes_full, self.rot_dofs, axis=1)

        # Static one-hot peak sign extraction (avoids dynamic tf.shape & gather_nd!)
        abs_vert = tf.abs(vert_modes)
        max_node_idx = tf.argmax(abs_vert, axis=1)  # (Batch, n_modes)
        one_hot_mask = tf.one_hot(max_node_idx, depth=self.n_sensor_nodes, axis=1, dtype=tf.float32)
        peak_vals = tf.reduce_sum(vert_modes * one_hot_mask, axis=1, keepdims=True)  # (Batch, 1, n_modes)
        signs = tf.where(peak_vals >= 0.0, 1.0, -1.0)

        vert_modes = vert_modes * signs
        rot_modes = rot_modes * signs

        # Transpose to (Batch, n_modes, n_sensor_nodes) and L2-normalize
        vert_modes = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_modes = tf.transpose(rot_modes, perm=[0, 2, 1])

        vert_modes = tf.math.l2_normalize(vert_modes, axis=-1, epsilon=1e-8)
        rot_modes = tf.math.l2_normalize(rot_modes, axis=-1, epsilon=1e-8)

        return frequencies_Hz, rot_modes, vert_modes