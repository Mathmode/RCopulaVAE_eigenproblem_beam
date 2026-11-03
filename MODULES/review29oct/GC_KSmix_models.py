#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/GC_KSmix_models.py

100% XLA-Safe Unified Physics-Informed Gaussian Copula Kumaraswamy-Mixture VAE.
  - Uses closed-form `tf.linalg.triangular_solve` for the Gaussian Copula log-density
    (avoids TFP MultivariateNormalTriL's internal `set_diag` graph ops).
  - Provides both `KL_Divergence_term` and `Joint_copula_dens_term` metric names
    for backward compatibility with `main_KS_29Oct.py` and `run_GPU.py`.
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K

tfd = tfp.distributions

from MODULES.review29oct.preprocessing_and_eigen import (
    assemble_global_Kmatrices_bearings,
    Solve_Bridge_Eigenproblem,
)
from MODULES.review29oct.GC_KSmix_architectures import (
    Fully_connected_enc_GC_KS,
    Copula_KS_pdf_layer,
)


@tf.function(jit_compile=True)
def calculate_diagonal_MAC(modes_true, modes_pred):
    """
    Computes mode-by-mode (diagonal) Modal Assurance Criterion (MAC)
    between modes_true and modes_pred of shape (Batch, n_modes, n_sensors).
    """
    eps = 1e-8
    t_norm = tf.math.l2_normalize(modes_true, axis=-1, epsilon=eps)
    p_norm = tf.math.l2_normalize(modes_pred, axis=-1, epsilon=eps)
    dot_prod = tf.reduce_sum(t_norm * p_norm, axis=-1)
    return tf.square(tf.clip_by_value(dot_prod, -1.0, 1.0))


class Inverse_Copula_KS_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_KS, num_samples, lbound, **kwargs):
        super(Inverse_Copula_KS_Model, self).__init__(**kwargs)
        self.n_dims = int(n_dims)
        self.num_KS = int(num_KS)
        self.num_samples = int(num_samples)
        self.FC_encoder = Fully_connected_enc_GC_KS(input_dim_encoder, self.n_dims, self.num_KS, lbound)

    def call(self, inputs):
        freq_data, rot_modes_data, vert_modes_data, _ = inputs

        flat_rot = tf.reshape(rot_modes_data, [-1, int(rot_modes_data.shape[1] * rot_modes_data.shape[2])])
        flat_vert = tf.reshape(vert_modes_data, [-1, int(vert_modes_data.shape[1] * vert_modes_data.shape[2])])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2

        a_ks, b_ks, weight_logits, offdiag, diag = tf.split(
            params,
            [
                self.n_dims * self.num_KS,
                self.n_dims * self.num_KS,
                self.n_dims * self.num_KS,
                n_corr,
                self.n_dims,
            ],
            axis=-1,
        )

        a_ks = tf.reshape(a_ks, (-1, self.num_KS, self.n_dims))
        b_ks = tf.reshape(b_ks, (-1, self.num_KS, self.n_dims))

        # Single softmax across the K mixture components (axis=1)
        weight_logits = tf.reshape(weight_logits, (-1, self.num_KS, self.n_dims))
        weights_prob = tf.nn.softmax(weight_logits, axis=1)

        return a_ks, b_ks, weights_prob, offdiag, diag


class My_CopulaKSVAE_withEigen(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        num_dofs,
        n_modes,
        K_basis=None,
        L_inv=None,
        n_dims=7,
        num_KS=3,
        num_samples=6,
        gamma=0.35,
        mean_f=0.0,
        std_f=1.0,
        lbound=0.45,
        prior_alpha=1.25,
        prior_beta=1.05,
        eig_device=None,
        # Optional legacy keyword arguments accepted smoothly without breaking:
        n_elements=None,
        Ke_matrices=None,
        Mfree=None,
        fixed_dofs_indices=None,
        **kwargs
    ):
        super(My_CopulaKSVAE_withEigen, self).__init__(**kwargs)
        if K_basis is None and Ke_matrices is not None:
            K_basis = Ke_matrices

        self.num_dofs = int(num_dofs)
        self.n_modes = int(n_modes)
        self.n_dims = int(n_dims)
        self.num_KS = int(num_KS)
        self.num_samples = int(num_samples)
        self.gamma = float(gamma)
        self.lbound = float(lbound)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

        self.K_basis = tf.constant(K_basis, dtype=tf.float32)
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32)
        self.std_freq = tf.constant(std_f, dtype=tf.float32)

        # Precompute static Beta prior normalizing constant
        lbeta_val = float(
            math.lgamma(self.prior_alpha)
            + math.lgamma(self.prior_beta)
            - math.lgamma(self.prior_alpha + self.prior_beta)
        )
        self.lbeta_const = tf.constant(lbeta_val, dtype=tf.float32)

        self.Encoder_model = Inverse_Copula_KS_Model(
            input_dim, self.n_dims, self.num_KS, self.num_samples, self.lbound
        )
        self.Eigen_solver = Solve_Bridge_Eigenproblem(
            self.num_dofs, self.n_modes, self.L_inv, eig_device=eig_device
        )
        self.Copula_sampling_layer = Copula_KS_pdf_layer(
            self.n_dims, self.num_KS, self.num_samples, self.lbound
        )

    def call(self, inputs):
        self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors = inputs

        self.a_ks, self.b_ks, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)

        (
            self.marginal_samples_z,
            self.copula_samples_u,
            self.LT_matrices,
        ) = self.Copula_sampling_layer(
            [self.a_ks, self.b_ks, self.weight_vals, self.offdiag_elems, self.diag_elems]
        )

        self.reshaped_z_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims))
        self.reshaped_copula_samples = tf.reshape(self.copula_samples_u, (-1, self.n_dims))

        K_global = assemble_global_Kmatrices_bearings(self.reshaped_z_samples, self.K_basis)
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(K_global)

        return self.reshaped_z_samples

    # -------------------------------------------------------------------------
    # VARIATIONAL LOSS TERMS (100% Static / XLA-Compatible)
    # -------------------------------------------------------------------------
    def Freqs_loss(self, y_true=None, y_pred=None):
        true_scaled = tf.expand_dims(self.freq_data, axis=1)  # (Batch, 1, n_modes)

        pred_hz = tf.reshape(tf.abs(self.pred_freqs), [-1, self.num_samples, self.n_modes])
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)

        sq_err = tf.square(true_scaled - pred_scaled)
        return tf.reduce_mean(tf.reduce_sum(sq_err, axis=-1))

    def MAC_modes_loss(self, y_true=None, y_pred=None):
        true_rot = tf.repeat(self.rot_modes_data, self.num_samples, axis=0)
        true_vert = tf.repeat(self.vert_modes_data, self.num_samples, axis=0)

        mac_rot = calculate_diagonal_MAC(true_rot, self.pred_rotmodes)
        mac_vert = calculate_diagonal_MAC(true_vert, self.pred_vertmodes)

        loss_modes = (1.0 - mac_rot) + (1.0 - mac_vert)
        loss_reshaped = tf.reshape(loss_modes, [-1, self.num_samples, self.n_modes])
        return tf.reduce_mean(tf.reduce_sum(loss_reshaped, axis=-1))

    def Copula_pdf_logprob(self, y_true=None, y_pred=None):
        """
        Closed-form Gaussian Copula log-density using `tf.linalg.triangular_solve`
        (completely avoids `tfd.MultivariateNormalTriL` and `set_diag` XLA issues).
          u_samples : (Batch, S, D)
          LT_matrices : (Batch, D, D)
        """
        u_clipped = tf.clip_by_value(self.copula_samples_u, 1e-5, 1.0 - 1e-5)
        sqrt2 = tf.constant(1.4142135623730951, dtype=tf.float32)
        z_norm = sqrt2 * tf.math.erfinv(2.0 * u_clipped - 1.0)  # (Batch, S, D)

        # Solve L * y = z_norm^T  =>  z_norm_T has shape (Batch, D, S)
        z_norm_T = tf.transpose(z_norm, perm=[0, 2, 1])
        y_sol = tf.linalg.triangular_solve(self.LT_matrices, z_norm_T, lower=True)  # (Batch, D, S)

        quad_joint = tf.reduce_sum(tf.square(y_sol), axis=1)      # (Batch, S)
        quad_indep = tf.reduce_sum(tf.square(z_norm), axis=-1)    # (Batch, S)

        # Log-determinant of L: sum(log(diag(L))) -> shape (Batch, 1)
        eye_d = tf.eye(self.n_dims, dtype=tf.float32)[tf.newaxis, :, :]
        diag_L = tf.reduce_sum(self.LT_matrices * eye_d, axis=-1)  # (Batch, D)
        log_det_L = tf.reduce_sum(tf.math.log(tf.maximum(diag_L, 1e-6)), axis=-1, keepdims=True)

        copula_log_dens = -log_det_L - 0.5 * (quad_joint - quad_indep)
        return copula_log_dens  # (Batch, S)

    def Marginal_KSMix_pdf_logprob(self, y_true=None, y_pred=None):
        x_unit = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        x_safe = tf.clip_by_value(x_unit, 1e-5, 1.0 - 1e-5)
        x_exp = tf.reshape(x_safe, [-1, self.num_samples, 1, self.n_dims])

        a = tf.expand_dims(self.a_ks, axis=1)         # (Batch, 1, K, D)
        b = tf.expand_dims(self.b_ks, axis=1)         # (Batch, 1, K, D)
        w = tf.expand_dims(self.weight_vals, axis=1)  # (Batch, 1, K, D)

        ln_x = tf.math.log(x_exp)
        x_a = tf.math.pow(x_exp, a)
        ln_1_minus_xa = tf.math.log(tf.clip_by_value(1.0 - x_a, 1e-7, 1.0))

        log_pdf_comp = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * ln_x + (b - 1.0) * ln_1_minus_xa
        log_w = tf.math.log(tf.clip_by_value(w, 1e-7, 1.0))

        marginal_log_dens = tf.reduce_logsumexp(log_w + log_pdf_comp, axis=2) - tf.math.log(1.0 - self.lbound)
        return tf.reduce_sum(marginal_log_dens, axis=-1)  # (Batch, S)

    def KL_Divergence_term(self, y_true=None, y_pred=None):
        copula_logq = self.Copula_pdf_logprob()
        marginal_logq = self.Marginal_KSMix_pdf_logprob()
        joint_logq = tf.clip_by_value(copula_logq + marginal_logq, -35.0, 35.0)

        x_unit = tf.clip_by_value(
            (self.marginal_samples_z - self.lbound) / (1.0 - self.lbound), 1e-5, 1.0 - 1e-5
        )
        log_p_unit = (
            (self.prior_alpha - 1.0) * tf.math.log(x_unit)
            + (self.prior_beta - 1.0) * tf.math.log(1.0 - x_unit)
            - self.lbeta_const
        )
        log_prior = tf.reduce_sum(log_p_unit - tf.math.log(1.0 - self.lbound), axis=-1)

        kl_per_sample = joint_logq - log_prior
        return tf.square(tf.constant(self.gamma, dtype=tf.float32)) * tf.reduce_mean(kl_per_sample)

    # Alias for backward compatibility with existing main scripts / metric lists
    def Joint_copula_dens_term(self, y_true=None, y_pred=None):
        return self.KL_Divergence_term(y_true, y_pred)

    def Mixture_Diversity_loss(self):
        w = self.weight_vals
        entropy = -tf.reduce_sum(w * tf.math.log(w + 1e-8), axis=1)
        return -tf.reduce_mean(entropy)

    def ELBO_Copula_loss(self, y_true, y_pred):
        l_freq = self.Freqs_loss(y_true, y_pred)
        l_mac = self.MAC_modes_loss(y_true, y_pred)
        l_kl = self.KL_Divergence_term(y_true, y_pred)
        l_div = self.Mixture_Diversity_loss()

        return 4.0 * l_freq + 12.0 * l_mac + l_kl + 0.15 * l_div