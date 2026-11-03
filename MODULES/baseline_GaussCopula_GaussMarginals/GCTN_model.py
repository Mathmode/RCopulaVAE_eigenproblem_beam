#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussCopula_GaussMarginals/GCTN_model.py

BASELINE 1 of the comparison requested in R1.4: Gaussian copula with ONE truncated Gaussian per marginal (Ref. 41 and
the submitted version of the manuscript),

    q(z | m) = c_R( F_1(z_1), ..., F_7(z_7) )  prod_d TN(z_d; mu_d, sigma_d, [lbound, 1]),

i.e. the variational family of the submitted manuscript (MODULES/COPULAS/GC_GMm_*: sigmoid location inside the
support, bounded scale, TruncatedNormal(low=lbound, high=1), Gaussian copula with a Cholesky factor). Everything else
is the revised method of MODULES/review29oct_fixed and is inherited, not copied: encoder trunk (128-256-128 ReLU),
Gaussian copula, eigen-decoder, calibrated likelihood (nll_terms), negative ELBO with gamma = 1, warm-up and training
schedule. The comparison therefore isolates the family of the marginals.

Sampling: z = F^-1(Phi(L eps)) with the closed-form truncated-normal quantile (exact reparameterisation; no rejection
sampling). Entropy: 0.5 log det R (copula, exact) + the closed-form entropy of every truncated Gaussian.
Numerics: mu in (0, 1) (as in the submitted version) gives alpha = -mu/sigma <= 0 <= beta = (1-mu)/sigma, so the
normaliser Z = Phi(beta) - Phi(alpha) is >= 0.07 for every sigma in (1e-3, 5); the quantile uses the lower-tail or the
upper-tail formula so that ndtri never receives 1 - tiny.
"""

import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed import tf_keras
from MODULES.review29oct_fixed.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.review29oct_fixed.GC_KSmix_functions import (
    build_correlation_matrices_from_cholesky, gaussian_copula_samples, copula_entropy,
)

K = tf_keras()

SQRT_HALF = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327
HALF_LOG_2PI_E = 0.5 * np.log(2.0 * np.pi * np.e)
SIG_MIN, SIG_MAX = 1e-3, 5.0                      # scale of the underlying normal in unit space u in [0, 1]
LOG_S_MID, LOG_S_HALF = 0.5 * (np.log(SIG_MIN) + np.log(SIG_MAX)), 0.5 * (np.log(SIG_MAX) - np.log(SIG_MIN))


def std_normal_cdf(x):
    return 0.5 * tf.math.erfc(-x * SQRT_HALF)


def std_normal_pdf(x):
    return INV_SQRT_2PI * tf.exp(-0.5 * tf.square(x))


def tn_normaliser(mu, sigma):
    """alpha, beta, P(X < alpha), P(X > beta) and Z = P(alpha < X < beta) of the standard normal X."""
    alpha, beta = -mu / sigma, (1.0 - mu) / sigma
    p_below, p_above = std_normal_cdf(alpha), std_normal_cdf(-beta)
    return alpha, beta, p_below, p_above, 1.0 - p_below - p_above


def tn_quantile(v, mu, sigma):
    """Quantile of TN(mu, sigma) on [0, 1] at levels v (broadcast); differentiable in v, mu and sigma."""
    _, _, p_below, p_above, Z = tn_normaliser(mu, sigma)
    p_low = p_below + v * Z                 # Phi(xi), from below
    p_up = p_above + (1.0 - v) * Z          # 1 - Phi(xi), from above
    lower = p_low < 0.5
    half = 0.5 * tf.ones_like(p_low)
    xi = tf.where(lower, tf.math.ndtri(tf.where(lower, p_low, half)), -tf.math.ndtri(tf.where(lower, half, p_up)))
    return tf.clip_by_value(mu + sigma * xi, 1e-6, 1.0 - 1e-6)


def tn_entropy(mu, sigma):
    """Entropy of TN(mu, sigma) on [0, 1]: log(sqrt(2 pi e) sigma Z) + (alpha phi(alpha) - beta phi(beta)) / (2 Z)."""
    alpha, beta, _, _, Z = tn_normaliser(mu, sigma)
    return (HALF_LOG_2PI_E + tf.math.log(sigma) + tf.math.log(Z)
            + (alpha * std_normal_pdf(alpha) - beta * std_normal_pdf(beta)) / (2.0 * Z))


def Fully_connected_enc_GCTN(input_dim, n_dims):
    """Same trunk as Fully_connected_enc_GC_KS; heads: location, scale, copula factor."""
    input1 = K.Input(shape=(input_dim,), name="InputLayer")
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    loc_raw = K.layers.Dense(n_dims, activation="linear", name="loc")(x)
    # bias 1.5: sigma ~ 0.3 at the start (broad marginals, like the KS model at initialisation)
    scale_raw = K.layers.Dense(n_dims, activation="linear", bias_initializer=K.initializers.Constant(1.5),
                               name="scale")(x)
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L = K.layers.Dense(n_correlations, activation="linear", kernel_initializer="zeros", name="off_diag")(x)
    diag_raw = K.layers.Dense(n_dims, activation="linear", name="diag")(x)

    outputs = K.layers.Concatenate(axis=1)([loc_raw, scale_raw, off_diag_L, diag_raw])
    return K.Model(inputs=input1, outputs=outputs)


class Inverse_Copula_TN_Model(K.Model):
    """Encoder: modal data -> location and scale (B, D) of the truncated Gaussians (unit space) and copula factor."""

    def __init__(self, input_dim_encoder, n_dims, obs_types="fv", **kwargs):
        super(Inverse_Copula_TN_Model, self).__init__(**kwargs)
        self.n_dims = int(n_dims)
        self.obs_types = obs_types
        self.FC_encoder = Fully_connected_enc_GCTN(input_dim_encoder, n_dims)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, _] = inputs
        parts = []
        if "f" in self.obs_types:
            parts.append(freq_data)
        if "v" in self.obs_types:
            parts.append(tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]]))
        if "r" in self.obs_types:
            parts.append(tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]]))
        params = self.FC_encoder(tf.concat(parts, axis=1))
        D = self.n_dims
        loc_raw, scale_raw, offdiag, diag_raw = tf.split(params, [D, D, D * (D - 1) // 2, D], axis=-1)
        loc = tf.sigmoid(loc_raw)                                                     # in (0, 1): inside the support
        scale = tf.exp(LOG_S_MID + LOG_S_HALF * tf.tanh(scale_raw / LOG_S_HALF))     # in (1e-3, 5)
        diag = tf.nn.softplus(diag_raw) + 1e-3
        return loc, scale, offdiag, diag


class GaussCopulaTNVAE(My_CopulaKSVAE_withEigen):
    """Gaussian copula + truncated-Gaussian marginals; likelihood, decoder and training steps inherited."""
    HEAD_ORDER = ["loc", "scale", "off_diag", "diag"]
    PRED_KEYS = ["test_loc", "test_scale", "test_offdiag_elems", "test_diag_elems"]

    def make_encoder(self, input_dim, n_dims, n_components, obs_types):
        return Inverse_Copula_TN_Model(input_dim, n_dims, obs_types)

    def sample_posterior(self, inputs):
        loc, scale, offdiag, diag = self.Encoder_model(inputs)
        LT = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)
        u = gaussian_copula_samples(LT, self.num_samples)
        x = tn_quantile(u, loc[:, None, :], scale[:, None, :])
        z = self.lbound + (1.0 - self.lbound) * x
        return z, dict(loc=loc, scale=scale, LT=LT, u=u)

    def compute_terms(self, inputs):
        z, q = self.sample_posterior(inputs)
        nll, nll_f, nll_v, nll_r = self.nll_terms(inputs, z)

        # entropy of q in z units: exact copula part + exact truncated-Gaussian marginals
        H_cop = copula_entropy(q["LT"])
        H_marg = tf.reduce_sum(tn_entropy(q["loc"], q["scale"]), axis=-1) + self.n_dims * np.log(1.0 - self.lbound)
        H = H_cop + H_marg

        nll_mean = tf.reduce_mean(nll, axis=1)
        return {
            "loss": tf.reduce_mean(self.tau * nll_mean - self.gamma2 * H),
            "neg_elbo": tf.reduce_mean(nll_mean - self.gamma2 * H),
            "NLL": tf.reduce_mean(nll_mean),
            "NLL_freq": tf.reduce_mean(nll_f),
            "NLL_vert": tf.reduce_mean(nll_v),
            "NLL_rot": tf.reduce_mean(nll_r),
            "Entropy": tf.reduce_mean(H),
            "Entropy_copula": tf.reduce_mean(H_cop),
            "Entropy_marginals": tf.reduce_mean(H_marg),
        }
