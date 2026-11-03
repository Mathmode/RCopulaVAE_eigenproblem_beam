#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/preprocessing_and_eigen.py

Fast Data Loading and Differentiable Eigensolver for the 7-Parameter Bridge with Bearings.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K


def load_bridge_data(data_path):
    """Loads train, val, and fine-mesh test datasets with training-only log-frequency scaling."""
    f_train = np.load(os.path.join(data_path, "freqs_train.npy"))
    r_train = np.load(os.path.join(data_path, "rotmodes_train.npy"))
    v_train = np.load(os.path.join(data_path, "vertmodes_train.npy"))
    a_train = np.load(os.path.join(data_path, "alphas_train.npy"))

    f_val = np.load(os.path.join(data_path, "freqs_val.npy"))
    r_val = np.load(os.path.join(data_path, "rotmodes_val.npy"))
    v_val = np.load(os.path.join(data_path, "vertmodes_val.npy"))
    a_val = np.load(os.path.join(data_path, "alphas_val.npy"))

    f_test = np.load(os.path.join(data_path, "freqs_test.npy"))
    r_test = np.load(os.path.join(data_path, "rotmodes_test.npy"))
    v_test = np.load(os.path.join(data_path, "vertmodes_test.npy"))
    a_test = np.load(os.path.join(data_path, "alphas_test.npy"))

    log_f_train = np.log(np.maximum(f_train, 1e-7))
    mean_f = np.mean(log_f_train, axis=0).astype(np.float32)
    std_f = (np.std(log_f_train, axis=0) + 1e-8).astype(np.float32)

    f_train_scaled = ((np.log(np.maximum(f_train, 1e-7)) - mean_f) / std_f).astype(np.float32)
    f_val_scaled = ((np.log(np.maximum(f_val, 1e-7)) - mean_f) / std_f).astype(np.float32)
    f_test_scaled = ((np.log(np.maximum(f_test, 1e-7)) - mean_f) / std_f).astype(np.float32)

    return (f_train_scaled, r_train, v_train, a_train,
            f_val_scaled, r_val, v_val, a_val,
            f_test_scaled, r_test, v_test, a_test,
            mean_f, std_f)


def load_baseline_matrices(data_path):
    """Loads Mfree (12, 12), K_basis (7, 12, 12), and L_inv (12, 12)."""
    Mfree = np.load(os.path.join(data_path, "Mass_matrix.npy")).astype(np.float32)
    K_basis = np.load(os.path.join(data_path, "Ke_matrix.npy")).astype(np.float32)
    L_inv = np.load(os.path.join(data_path, "L_inv.npy")).astype(np.float32)
    return Mfree, K_basis, L_inv


@tf.function
def assemble_global_Kmatrices_bearings(z_samples, K_basis):
    """
    Fast global stiffness assembly via basis linear combination.
    z_samples: [Batch_total, n_dims]
    K_basis:   [n_dims, n_dofs, n_dofs]
    """
    return tf.einsum('BD, DKQ -> BKQ', z_samples, K_basis)


class Solve_Bridge_Eigenproblem(K.layers.Layer):
    """
    Differentiable Generalized Eigensolver Layer.
    Uses CPU multi-threaded LAPACK for the 12x12 eigendecomposition to avoid cuSOLVER's
    small-matrix GPU synchronization overhead (~10x faster for 12x12 matrices).
    """
    def __init__(self, num_dofs, n_modes, L_inv, **kwargs):
        super(Solve_Bridge_Eigenproblem, self).__init__(**kwargs)
        self.num_dofs = int(num_dofs)
        self.n_modes = int(n_modes)
        self.L_inv_np = np.array(L_inv, dtype=np.float32)
        self.L_inv = tf.constant(self.L_inv_np, dtype=tf.float32)
        self.eye_jitter = tf.eye(self.num_dofs, dtype=tf.float32)[tf.newaxis, :, :] * 1e-6

    def call(self, K_batch):
        A_batch = tf.matmul(self.L_inv, tf.matmul(K_batch, self.L_inv, transpose_b=True))
        A_batch = 0.5 * (A_batch + tf.transpose(A_batch, perm=[0, 2, 1])) + self.eye_jitter

        with tf.device('/CPU:0'):
            eigvals, eigvecs = tf.linalg.eigh(A_batch)

        vals_trunc = tf.clip_by_value(eigvals[:, :self.n_modes], 1e-6, 1e20)
        freqs_hz = tf.math.sqrt(vals_trunc) * (1.0 / (2.0 * np.pi))

        modes_full = tf.matmul(self.L_inv, eigvecs[:, :, :self.n_modes], transpose_a=True)

        vert_modes = modes_full[:, 0::2, :]  # [B, n_nodes, n_modes]
        rot_modes = modes_full[:, 1::2, :]   # [B, n_nodes, n_modes]

        abs_vert = tf.abs(vert_modes)
        max_mask = tf.one_hot(tf.argmax(abs_vert, axis=1), depth=tf.shape(vert_modes)[1], axis=1, dtype=tf.float32)
        peak_vals = tf.reduce_sum(vert_modes * max_mask, axis=1, keepdims=True)
        signs = tf.where(peak_vals >= 0.0, tf.ones_like(peak_vals), -tf.ones_like(peak_vals))

        vert_modes = vert_modes * signs
        rot_modes = rot_modes * signs

        vert_modes = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_modes = tf.transpose(rot_modes, perm=[0, 2, 1])

        vert_modes = tf.math.l2_normalize(vert_modes, axis=-1, epsilon=1e-10)
        rot_modes = tf.math.l2_normalize(rot_modes, axis=-1, epsilon=1e-10)

        return freqs_hz, rot_modes, vert_modes

    def get_config(self):
        config = super(Solve_Bridge_Eigenproblem, self).get_config()
        config.update({
            "num_dofs": self.num_dofs,
            "n_modes": self.n_modes,
            "L_inv": self.L_inv_np.tolist()
        })
        return config