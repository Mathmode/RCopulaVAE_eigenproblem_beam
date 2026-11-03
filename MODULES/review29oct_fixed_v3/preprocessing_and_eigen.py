#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v3/preprocessing_and_eigen.py

Data loading and differentiable eigensolver (as in MODULES/review29oct/preprocessing_and_eigen.py, except that
the mode shapes are read at the sensor nodes only, so the decoder mesh can be finer than the sensor grid, and
the largest-|vertical| sign flip is gone: the likelihood uses the MAC, which does not depend on the mode sign).
"""

import os
import json
import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed_v3 import tf_keras

K = tf_keras()


def load_problem_config(data_path):
    with open(os.path.join(data_path, "problem_config.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_bridge_data(data_path):
    """Loads train, val and test sets with training-only log-frequency scaling."""
    out = []
    for split in ("train", "val", "test"):
        f = np.load(os.path.join(data_path, f"freqs_{split}.npy"))
        r = np.load(os.path.join(data_path, f"rotmodes_{split}.npy"))
        v = np.load(os.path.join(data_path, f"vertmodes_{split}.npy"))
        a = np.load(os.path.join(data_path, f"alphas_{split}.npy"))
        out.append([f, r, v, a])

    log_f_train = np.log(np.maximum(out[0][0], 1e-7))
    mean_f = np.mean(log_f_train, axis=0).astype(np.float32)
    std_f = (np.std(log_f_train, axis=0) + 1e-8).astype(np.float32)
    for split in out:
        split[0] = ((np.log(np.maximum(split[0], 1e-7)) - mean_f) / std_f).astype(np.float32)

    (f_train, r_train, v_train, a_train), (f_val, r_val, v_val, a_val), (f_test, r_test, v_test, a_test) = out
    return (f_train, r_train, v_train, a_train,
            f_val, r_val, v_val, a_val,
            f_test, r_test, v_test, a_test,
            mean_f, std_f)


def load_test_split(data_path, split, mean_f, std_f):
    """Loads an extra test set (e.g. 'test_fine') with the training log-frequency scaling."""
    f = np.load(os.path.join(data_path, f"freqs_{split}.npy"))
    r = np.load(os.path.join(data_path, f"rotmodes_{split}.npy"))
    v = np.load(os.path.join(data_path, f"vertmodes_{split}.npy"))
    a = np.load(os.path.join(data_path, f"alphas_{split}.npy"))
    f = ((np.log(np.maximum(f, 1e-7)) - mean_f) / std_f).astype(np.float32)
    return f, r, v, a


def load_baseline_matrices(data_path):
    """Loads Mfree (12, 12), K_basis (7, 12, 12), and L_inv (12, 12)."""
    Mfree = np.load(os.path.join(data_path, "Mass_matrix.npy")).astype(np.float32)
    K_basis = np.load(os.path.join(data_path, "Ke_matrix.npy")).astype(np.float32)
    L_inv = np.load(os.path.join(data_path, "L_inv.npy")).astype(np.float32)
    return Mfree, K_basis, L_inv


@tf.function
def assemble_global_Kmatrices_bearings(z_samples, K_basis):
    """K(z) = sum_d z_d K_basis[d] for z_samples [Batch_total, n_dims]."""
    return tf.einsum("BD,DKQ->BKQ", z_samples, K_basis)


class Solve_Bridge_Eigenproblem(K.layers.Layer):
    """
    Differentiable generalized eigensolver. The small eigendecompositions run on the CPU (faster than
    cuSOLVER for such matrices) and in float64: with elastic bearings the condition number of L^-1 K L^-T is
    ~3e5, and float32 gave errors up to 0.6 % on the first frequency. Returns f [B, n_modes] in Hz and unit-norm
    rotational / vertical shapes [B, n_modes, n_nodes] in float32.
    """
    def __init__(self, num_dofs, n_modes, L_inv, sensor_nodes=None, **kwargs):
        super(Solve_Bridge_Eigenproblem, self).__init__(**kwargs)
        self.num_dofs = int(num_dofs)
        self.n_modes = int(n_modes)
        self.L_inv_np = np.array(L_inv, dtype=np.float64)
        self.L_inv = tf.constant(self.L_inv_np, dtype=tf.float64)
        nodes = np.arange(self.num_dofs // 2) if sensor_nodes is None else np.asarray(sensor_nodes)
        self.sensor_nodes = nodes.tolist()
        self.vert_dofs = tf.constant(2 * nodes, dtype=tf.int32)      # vertical DOFs of the sensor nodes
        self.rot_dofs = tf.constant(2 * nodes + 1, dtype=tf.int32)

    def call(self, K_batch):
        K64 = tf.cast(K_batch, tf.float64)
        A_batch = tf.matmul(self.L_inv, tf.matmul(K64, self.L_inv, transpose_b=True))
        A_batch = 0.5 * (A_batch + tf.transpose(A_batch, perm=[0, 2, 1]))

        with tf.device("/CPU:0"):
            eigvals, eigvecs = tf.linalg.eigh(A_batch)

        freqs_hz = tf.cast(tf.math.sqrt(tf.maximum(eigvals[:, :self.n_modes], 1e-6)) * (1.0 / (2.0 * np.pi)), tf.float32)
        modes_full = tf.cast(tf.matmul(self.L_inv, eigvecs[:, :, :self.n_modes], transpose_a=True), tf.float32)

        vert = tf.gather(modes_full, self.vert_dofs, axis=1)
        rot = tf.gather(modes_full, self.rot_dofs, axis=1)
        vert_modes = tf.math.l2_normalize(tf.transpose(vert, perm=[0, 2, 1]), axis=-1, epsilon=1e-12)
        rot_modes = tf.math.l2_normalize(tf.transpose(rot, perm=[0, 2, 1]), axis=-1, epsilon=1e-12)
        return freqs_hz, rot_modes, vert_modes

    def get_config(self):
        config = super(Solve_Bridge_Eigenproblem, self).get_config()
        config.update({"num_dofs": self.num_dofs, "n_modes": self.n_modes, "L_inv": self.L_inv_np.tolist(),
                       "sensor_nodes": self.sensor_nodes})
        return config
