#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC_GMm_models_updated_review27oct.py
====================================
Full Updated Version of `MODULES/COPULAS/GC_GMm_models.py` for the
MS CPENG-8674 Major Revision (saved in `code/MODULES/review_27oct/`).

Supported & Verified Changes Relative to Original `GC_GMm_models.py`:
---------------------------------------------------------------------
1. [FIX 1 -- Exact Support for `num_samples >= 1` (`H in {1, 5, 10}`, Reviewer 3 Comment 7)]
   - In the original `GC_GMm_models.py` (lines 129, 147-150, 180-181), `self.freq_data`,
     `self.rot_modes_data`, `self.vert_modes_data`, `self.means`, and `self.scales` had
     batch dimension `B`, whereas `self.pred_freqs`, `self.pred_rotmodes`, `self.pred_vertmodes`,
     and `self.reshaped_alpha_samples` had dimension `B * H`.
   - When `H = 1`, `B * H == B`. When `H > 1` (e.g., `H = 5` or `10`), `Freqs_loss`,
     `MAC_modes_loss`, and `Marginal_pdf_logprob` failed with shape mismatch `[B, ...] vs [B*H, ...]`.
   - Fix: Just as in your October `GC_KSmix_models.py` (lines 163-165), we tile the target modal
     inputs and marginal parameters across `self.num_samples` (`tf.repeat(..., self.num_samples, axis=0)`),
     so `H = 1, 5, 10` all work seamlessly.

2. [FIX 2 -- Uses Fixed `Solve_eigenproblem` (`L_inv` non-trainable + exact Cholesky order)
             and Fixed `gaussian_copula_samples` (no 0.99 upper-tail truncation)]
   - Imports `assemble_global_Kmatrices`, `assemble_affine_Kfree`, and `Solve_eigenproblem` from
     `GC_GMm_GPU_eigen_functions_updated_review27oct.py` and `gaussian_copula_samples`,
     `gaussian_marginal_samples`, `build_correlation_matrices_from_cholesky` from
     `GC_GMm_functions_updated_review27oct.py`.

3. [FIX 3 -- Optional Affine Zone / Elastic Bearing Stiffness Support (`Kzone_free`, `Ksupport_free`)]
   - If `Kzone_free` is passed (for 20-, 40-, or 50-element fine meshes with 5 or 10 damage
     zones and/or elastic bearings, R1.1, R2.2-R2.4), `call()` uses `assemble_affine_Kfree`.
     Otherwise, it uses the standard element assembly `assemble_global_Kmatrices` (100% backward
     compatible with the 5-element and 10-element models).

4. [FIX 4 -- Optional Informative Prior `log p(z)` in ELBO (Reviewer 3 Comment 3)]
   - By default (`prior_beta_a=1.0, prior_beta_b=1.0`), the prior `p(z) = U[lbound, 1.0]^D`
     is uniform, so `log p(z)` is constant and the KL divergence equals `E_q[log q(z|m)]`.
   - If `prior_beta_a > 1.0` (e.g. `Beta(5, 1)` scaled to `[lbound, 1.0]`, favoring near-healthy
     states), `-log p(z)` is automatically included in `Joint_copula_dens_term` so we can
     demonstrate non-uniform Bayesian priors to Reviewer 3 without changing the training dataset.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K

tfd = tfp.distributions

from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Fully_connected_enc_GC_highdims
from .GC_GMm_GPU_eigen_functions_updated_review27oct import (
    assemble_global_Kmatrices,
    assemble_affine_Kfree,
    Solve_eigenproblem,
)
from .GC_GMm_functions_updated_review27oct import (
    build_correlation_matrices_from_cholesky,
    gaussian_copula_samples,
    gaussian_marginal_samples,
)


@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """
    Computes Modal Assurance Criterion (MAC) matrix with numerical stability.
    """
    eps = 1e-8
    modes_true_norm = modes_true / (tf.norm(modes_true, axis=2, keepdims=True) + eps)
    modes_pred_norm = modes_pred / (tf.norm(modes_pred, axis=2, keepdims=True) + eps)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return mac_matrix


class Updated_Copula_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_gaussians, num_samples, lbound, **kwargs):
        super(Updated_Copula_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.lbound = lbound

    def call(self, inputs):
        means, scales, weight_vals, offdiag_elems, diag_elems = inputs
        LT_matrices = build_correlation_matrices_from_cholesky(offdiag_elems, diag_elems, self.n_dims)
        copula_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples, self.lbound)
        marginal_samples = gaussian_marginal_samples(means, scales, copula_samples, self.lbound)
        return marginal_samples, copula_samples, LT_matrices


class Inverse_Copula_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, lbound, use_highdims_enc=False, **kwargs):
        super(Inverse_Copula_Model, self).__init__()
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        if use_highdims_enc or n_dims > 6:
            self.FC_encoder = Fully_connected_enc_GC_highdims(input_dim_encoder, n_dims, num_gaussians, lbound)
        else:
            self.FC_encoder = Fully_connected_enc_GC(input_dim_encoder, n_dims, num_gaussians, lbound)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs

        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2

        means, scales, weights, offdiag, diag = tf.split(
            params,
            [
                self.n_dims * self.num_gaussians,
                self.n_dims * self.num_gaussians,
                self.n_dims * self.num_gaussians,
                n_corr,
                self.n_dims,
            ],
            axis=-1,
        )

        means = tf.reshape(means, (-1, self.num_gaussians, self.n_dims))
        scales = tf.reshape(scales, (-1, self.num_gaussians, self.n_dims))
        weights = tf.reshape(weights, (-1, self.num_gaussians, self.n_dims))
        weights = tf.nn.softmax(weights, axis=1)

        return means, scales, weights, offdiag, diag


class My_CopulaVAE_withEigen_Updated(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        num_dofs,
        n_elements,
        n_modes,
        Ke_matrices,
        Mfree,
        L_inv,
        n_dims,
        num_gaussians,
        num_samples,
        gamma,
        mean_f,
        std_f,
        lbound,
        fixed_dofs_indices=None,
        sensor_nodes=None,
        Kzone_free=None,
        Ksupport_free=None,
        K0_free=None,
        prior_beta_a=1.0,
        prior_beta_b=1.0,
        use_highdims_enc=False,
        **kwargs,
    ):
        super(My_CopulaVAE_withEigen_Updated, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices
        self.sensor_nodes = sensor_nodes

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) if Ke_matrices is not None else None
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Kzone_free = tf.constant(Kzone_free, dtype=tf.float32) if Kzone_free is not None else None
        self.Ksupport_free = tf.constant(Ksupport_free, dtype=tf.float32) if Ksupport_free is not None else None
        self.K0_free = tf.constant(K0_free, dtype=tf.float32) if K0_free is not None else None

        self.Encoder_model = Inverse_Copula_Model(
            input_dim, n_dims, num_gaussians, num_samples, lbound, use_highdims_enc=use_highdims_enc
        )
        self.Eigen_solver = Solve_eigenproblem(
            num_dofs, n_modes, L_inv, fixed_dofs_indices=fixed_dofs_indices, sensor_nodes=sensor_nodes
        )
        self.Copula_sampling_layer = Updated_Copula_pdf_layer(n_dims, num_gaussians, num_samples, lbound)

        self.mean_freq = tf.constant(mean_f, dtype=tf.float32)
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.num_gaussians = num_gaussians
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.gamma = gamma
        self.lbound = lbound
        self.prior_beta_a = float(prior_beta_a)
        self.prior_beta_b = float(prior_beta_b)

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs

        self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)

        inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)

        # Both marginal_samples_z and copula_samples_u have shape (Batch, num_samples, n_dims)
        self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims))
        self.reshaped_copula_samples = tf.reshape(self.copula_samples_u, (-1, self.n_dims))

        lt_ext = tf.repeat(self.LT_matrices[:, tf.newaxis, :, :], self.num_samples, axis=1)
        self.reshaped_LT_matrices = tf.reshape(lt_ext, [-1, self.n_dims, self.n_dims])

        # Stiffness assembly: either affine zone/bearing matrices or standard element matrices
        if self.Kzone_free is not None:
            Kfree = assemble_affine_Kfree(
                self.reshaped_alpha_samples, self.Kzone_free, self.Ksupport_free, self.K0_free
            )
        else:
            if len(self.Ke_matrices.shape) == 2:
                Ke_matrices_dam = tf.einsum("BE, KQ -> BEKQ", self.reshaped_alpha_samples, self.Ke_matrices)
            else:
                Ke_matrices_dam = tf.einsum("BE, EKQ -> BEKQ", self.reshaped_alpha_samples, self.Ke_matrices)
            Kfree = assemble_global_Kmatrices(
                Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices
            )

        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        return self.reshaped_alpha_samples

    def Freqs_loss(self, y_true, y_pred):
        # [FIX 1]: Repeat true_scaled by self.num_samples so H > 1 works seamlessly
        true_scaled = tf.repeat(self.freq_data, repeats=self.num_samples, axis=0)
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)

        huber = tf.keras.losses.Huber(delta=1.0)
        return huber(true_scaled, pred_scaled)

    def MAC_modes_loss(self, y_true, y_pred):
        # [FIX 1]: Repeat mode shapes by self.num_samples so H > 1 works seamlessly
        true_rot = tf.repeat(self.rot_modes_data, repeats=self.num_samples, axis=0)
        true_vert = tf.repeat(self.vert_modes_data, repeats=self.num_samples, axis=0)
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes

        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches)

        return get_match_loss(true_rot, pred_rot) + get_match_loss(true_vert, pred_vert)

    def Copula_pdf_logprob(self, y_true, y_pred):
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        normal_samples = normal_dist.quantile(u_clipped)

        mvn = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.n_dims),
            scale_tril=self.reshaped_LT_matrices,
        )
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        log_prob_marginals_std = tf.reduce_sum(normal_dist.log_prob(normal_samples), axis=-1)
        return log_prob_joint_normal - log_prob_marginals_std

    def Marginal_pdf_logprob(self, y_true, y_pred):
        # [FIX 1]: Repeat marginal loc and scale across self.num_samples for H >= 1
        means_2d = tf.reshape(self.means, [-1, self.n_dims])
        scales_2d = tf.reshape(self.scales, [-1, self.n_dims])
        means_rep = tf.repeat(means_2d, repeats=self.num_samples, axis=0)
        scales_rep = tf.repeat(scales_2d, repeats=self.num_samples, axis=0)

        gaussian_marginals = tfd.TruncatedNormal(
            loc=means_rep,
            scale=scales_rep + 1e-5,
            low=self.lbound - 1e-4,
            high=1.0 + 1e-4,
        )
        log_prob_marginals = gaussian_marginals.log_prob(self.reshaped_alpha_samples)
        return tf.math.reduce_sum(log_prob_marginals, axis=-1)

    def Prior_pdf_logprob(self):
        """
        Evaluates `log p(z)` under a scaled Beta(a, b) prior on `[lbound, 1.0]^D` (R3.3).
        When `a = b = 1.0` (uniform prior), returns 0.0 (since additive constant does not affect gradients).
        """
        if self.prior_beta_a == 1.0 and self.prior_beta_b == 1.0:
            return 0.0
        span = 1.0 - self.lbound
        z_unit = tf.clip_by_value((self.reshaped_alpha_samples - self.lbound) / span, 1e-5, 1.0 - 1e-5)
        beta_dist = tfd.Beta(concentration1=self.prior_beta_a, concentration0=self.prior_beta_b)
        return tf.reduce_sum(beta_dist.log_prob(z_unit) - tf.math.log(span), axis=-1)

    def Joint_copula_dens_term(self, y_true, y_pred):
        c_term = self.Copula_pdf_logprob(y_true, y_pred)
        m_term = self.Marginal_pdf_logprob(y_true, y_pred)
        p_term = self.Prior_pdf_logprob()

        kl_integrand = c_term + m_term - p_term
        joint_logprob = tf.math.reduce_mean(kl_integrand)
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * joint_logprob

    def ELBO_Copula_loss(self, y_true, y_pred):
        loss_f = self.Freqs_loss(y_true, y_pred)
        loss_m = self.MAC_modes_loss(y_true, y_pred)
        loss_j = self.Joint_copula_dens_term(y_true, y_pred)
        return loss_f + 15.0 * loss_m + loss_j

    def get_config(self):
        config = {"num_dofs": self.num_dofs, "fixed_dofs_indices": self.fixed_dofs_indices}
        base_config = super(My_CopulaVAE_withEigen_Updated, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
