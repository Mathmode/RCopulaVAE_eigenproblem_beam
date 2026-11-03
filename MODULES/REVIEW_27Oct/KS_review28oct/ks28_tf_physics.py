# -*- coding: utf-8 -*-
"""
ks28_tf_physics.py -- Differentiable modal decoder D(theta) for the extended case study.

The decoder only needs the constant matrices saved in decoder_basis.npz:

    K(theta) = sum_k z_k K_zone[k] + sum_j 2**eta_j K_sup[j] + K_fixed

and, with the Cholesky factor of the (known, constant) mass matrix M = C C^T, we
pre-transform the basis once:  A_. = C^-1 K_. C^-T, so that for every sample

    A(theta) = sum_k z_k A_zone[k] + sum_j s_j A_sup[j] + A_fixed      (one einsum)
    A v = lambda v  (tf.linalg.eigh)  ->  f = sqrt(lambda) / 2 pi,  phi = C^-T v

Differences with the old decoder (GC_GMm_GPU_eigen_functions.Solve_eigenproblem):
  * no element scatter: any mesh / zone layout / support model works unchanged;
  * no fixed DOFs: the beam rests on elastic bearings (Reviewer 2 c.2);
  * the mass matrix is a CONSTANT (the old layer had L_inv as a trainable Variable,
    which silently let the network change the physics);
  * float64 by default: with K ~ 1e9-1e12 the eigenvalue spread of A is ~1e8, and a
    float32 eigensolver loses most digits of the first eigenvalues;
  * only vertical DOFs at the sensors are returned (rotations are not measured);
  * no sign alignment is needed: the likelihood (MAC form) is sign-invariant.
"""

from __future__ import annotations

import numpy as np
from . import ks28_tf_setup  # noqa: F401  (forces the TensorFlow backend of Keras)
import tensorflow as tf


class ModalDecoder:
    def __init__(self, basis: dict, cfg, n_modes=None, dtype=tf.float64, eigh_device=None):
        """basis: dict loaded from decoder_basis.npz; cfg: BridgeConfig.

        eigh_device: e.g. "/CPU:0" to run the (small, batched) eigensolver on the CPU even
        when training on a GPU (often faster for 20-40 DOF matrices; benchmark with the
        smoke test). None -> default placement.
        """
        self.cfg = cfg
        self.nz = cfg.n_zones
        self.support_latent = cfg.support_latent
        self.n_modes = cfg.n_modes if n_modes is None else int(n_modes)
        self.dtype = dtype
        self.eigh_device = eigh_device

        M = np.asarray(basis["M"], dtype=np.float64)
        C = np.linalg.cholesky(M)
        Cinv = np.linalg.inv(C)
        tr = lambda K: Cinv @ K @ Cinv.T
        A_zone = np.stack([tr(K) for K in basis["K_zone"]])
        A_sup = np.stack([tr(K) for K in basis["K_sup"]])
        A_fixed = tr(np.asarray(basis["K_fixed"]))
        # Scale: work with lambda / lambda_ref (lambda_ref = first healthy eigenvalue) so the
        # numbers inside eigh are O(1) for the modes of interest.
        A_h = A_zone.sum(0) + (A_sup.sum(0) if self.support_latent else 0.0) + A_fixed
        self.lam_ref = float(np.linalg.eigvalsh(A_h)[0])
        self.A_zone = tf.constant(A_zone / self.lam_ref, dtype)
        self.A_sup = tf.constant(A_sup / self.lam_ref, dtype)
        self.A_fixed = tf.constant(A_fixed / self.lam_ref, dtype)
        self.CinvT_s = tf.constant(Cinv.T[np.asarray(basis["sensor_dofs"]), :], dtype)  # (ns, n)

    def __call__(self, theta):
        """theta (N, D) any float dtype -> f [Hz] (N, nm) float32, phi (N, nm, ns) float32."""
        th = tf.cast(theta, self.dtype)
        z = th[:, :self.nz]
        A = tf.einsum("nk,kij->nij", z, self.A_zone) + self.A_fixed
        if self.support_latent:
            s = tf.pow(tf.constant(2.0, self.dtype), th[:, self.nz:self.nz + 2])
            A = A + tf.einsum("ns,sij->nij", s, self.A_sup)
        A = 0.5 * (A + tf.linalg.matrix_transpose(A))
        if self.eigh_device is not None:
            with tf.device(self.eigh_device):
                lam, V = tf.linalg.eigh(A)
        else:
            lam, V = tf.linalg.eigh(A)
        lam = tf.maximum(lam[:, :self.n_modes], 1e-12) * self.lam_ref
        f = tf.sqrt(lam) / (2.0 * np.pi)
        phi = tf.einsum("sj,njm->nms", self.CinvT_s, V[:, :, :self.n_modes])      # (N, nm, ns)
        return tf.cast(f, tf.float32), tf.cast(phi, tf.float32)
