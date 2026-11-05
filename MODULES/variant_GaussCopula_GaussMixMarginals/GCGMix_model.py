#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/variant_GaussCopula_GaussMixMarginals/GCGMix_model.py

VARIANT of the proposed model: Gaussian copula with a MIXTURE of K truncated Gaussians per marginal, in the place of the
mixture of K Kumaraswamy distributions,

    q(z | m) = c_R( F_1(z_1), ..., F_7(z_7) )  prod_d  sum_k w_kd TN(z_d; mu_kd, sigma_kd, [lbound, 1]).

Everything else is inherited from MODULES/review29oct_fixed: encoder trunk (128-256-128 ReLU), Gaussian copula,
eigen-decoder, calibrated likelihood (nll_terms), negative ELBO with gamma = 1, warm-up and training. The comparison
therefore isolates the family of the marginals, and K = 1 would be the GC-TN baseline apart from the entropy estimator.

No rejection sampling and no Monte Carlo approximation of the inverse CDF are needed, because the construction is the
same as for the Kumaraswamy mixture:
  * CDF and density of every component are closed-form (erfc), each component truncated with its own normaliser Z_k;
  * the quantile of the mixture has no closed form (as for the Kumaraswamy mixture when K > 1): it is found by bisection,
    with the exact implicit gradient dx/dtheta = -(dF/dtheta)(x) / f(x), dx/du = 1 / f(x);
  * entropy: 0.5 log det R (copula, exact) + Monte Carlo of the marginals with the copula path blocked, as the KS model.
Numerics as GC-TN: mu in (0, 1) gives alpha <= 0 <= beta, so the normaliser Z_k is >= 0.07 for every sigma in (1e-3, 5).
"""

import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed import tf_keras
from MODULES.review29oct_fixed.GC_KSmix_models import My_CopulaKSVAE_withEigen
from MODULES.review29oct_fixed.GC_KSmix_functions import (
    build_correlation_matrices_from_cholesky, gaussian_copula_samples, copula_entropy,
)
from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import (
    LOG_S_HALF, LOG_S_MID, std_normal_cdf, tn_normaliser,
)

K = tf_keras()
HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def gmix_cdf(x, mu, sigma, w):
    """Mixture CDF at x (B, S, D); mu, sigma, w (B, K, D). The tail formula that keeps the digits is used per component."""
    xe = x[:, :, tf.newaxis, :]
    m, s, wt = mu[:, tf.newaxis], sigma[:, tf.newaxis], w[:, tf.newaxis]
    _, _, p_below, p_above, Z = tn_normaliser(m, s)
    xi = (xe - m) / s
    lower = (std_normal_cdf(xi) - p_below) / Z
    upper = 1.0 - (std_normal_cdf(-xi) - p_above) / Z
    return tf.reduce_sum(wt * tf.where(xi < 0.0, lower, upper), axis=2)


def gmix_logpdf(x, mu, sigma, w):
    """log of the mixture density on the unit interval at x (B, S, D)."""
    xe = x[:, :, tf.newaxis, :]
    m, s = mu[:, tf.newaxis], sigma[:, tf.newaxis]
    _, _, _, _, Z = tn_normaliser(m, s)
    log_w = tf.math.log(tf.maximum(w[:, tf.newaxis], 1e-30))
    log_pdf_k = -0.5 * tf.square((xe - m) / s) - HALF_LOG_2PI - tf.math.log(s) - tf.math.log(Z)
    return tf.reduce_logsumexp(log_w + log_pdf_k, axis=2)


def _bisection_quantile(u, mu, sigma, w, n_iter=32):
    low = tf.zeros_like(u)
    high = tf.ones_like(u)
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        is_less = gmix_cdf(mid, mu, sigma, w) < u
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)
    return tf.clip_by_value(0.5 * (low + high), 1e-6, 1.0 - 1e-6)


@tf.custom_gradient
def gmix_quantile(u, mu, sigma, w):
    """x = F^{-1}(u) of the truncated-Gaussian mixture (unit interval), with exact implicit gradients."""
    x = tf.stop_gradient(_bisection_quantile(u, mu, sigma, w))

    def grad(dx):
        with tf.GradientTape() as tape:
            tape.watch([mu, sigma, w])
            cdf = gmix_cdf(x, mu, sigma, w)
        inv_pdf = 1.0 / tf.maximum(tf.exp(gmix_logpdf(x, mu, sigma, w)), 1e-12)
        g_mu, g_sigma, g_w = tape.gradient(cdf, [mu, sigma, w], output_gradients=-dx * inv_pdf)

        def finite(g):
            return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        return finite(dx * inv_pdf), finite(g_mu), finite(g_sigma), finite(g_w)

    return x, grad


def Fully_connected_enc_GCGMix(input_dim, n_dims, n_components):
    """Same trunk as Fully_connected_enc_GC_KS; heads: locations, scales, weights, copula factor."""
    input1 = K.Input(shape=(input_dim,), name="InputLayer")
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    # component k starts around u = (k + 0.5) / K in every dimension (breaks the symmetry of the components), with
    # sigma ~ 0.3 (bias 1.5, as GC-TN): broad marginals at the start
    start = np.log((np.arange(n_components) + 0.5) / (n_components - np.arange(n_components) - 0.5))
    loc_raw = K.layers.Dense(n_components * n_dims, activation="linear", name="loc",
                             bias_initializer=K.initializers.Constant(np.repeat(start, n_dims).tolist()))(x)
    scale_raw = K.layers.Dense(n_components * n_dims, activation="linear", name="scale",
                               bias_initializer=K.initializers.Constant(1.5))(x)
    weight_logits = K.layers.Dense(n_components * n_dims, activation="linear", name="weights")(x)
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L = K.layers.Dense(n_correlations, activation="linear", kernel_initializer="zeros", name="off_diag")(x)
    diag_raw = K.layers.Dense(n_dims, activation="linear", name="diag")(x)

    outputs = K.layers.Concatenate(axis=1)([loc_raw, scale_raw, weight_logits, off_diag_L, diag_raw])
    return K.Model(inputs=input1, outputs=outputs)


class Inverse_Copula_GMix_Model(K.Model):
    """Encoder: modal data -> locations, scales and weights (B, K, D) in unit space, and the copula factor parameters."""

    def __init__(self, input_dim_encoder, n_dims, n_components, obs_types="fv", **kwargs):
        super(Inverse_Copula_GMix_Model, self).__init__(**kwargs)
        self.n_dims, self.n_components = int(n_dims), int(n_components)
        self.obs_types = obs_types
        self.FC_encoder = Fully_connected_enc_GCGMix(input_dim_encoder, n_dims, n_components)

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
        D, Kc = self.n_dims, self.n_components
        loc_raw, scale_raw, w_logits, offdiag, diag_raw = tf.split(params, [Kc * D, Kc * D, Kc * D, D * (D - 1) // 2, D], axis=-1)
        loc = tf.sigmoid(tf.reshape(loc_raw, (-1, Kc, D)))                            # in (0, 1): inside the support
        scale = tf.exp(LOG_S_MID + LOG_S_HALF * tf.tanh(tf.reshape(scale_raw, (-1, Kc, D)) / LOG_S_HALF))  # in (1e-3, 5)
        weights = tf.nn.softmax(tf.reshape(w_logits, (-1, Kc, D)), axis=1)
        diag = tf.nn.softplus(diag_raw) + 1e-3
        return loc, scale, weights, offdiag, diag


class GaussCopulaGMixVAE(My_CopulaKSVAE_withEigen):
    """Gaussian copula + truncated-Gaussian mixture marginals; likelihood, decoder and training steps inherited.
    num_KS = components per marginal, num_samples = samples of q per observation (as the KS model)."""
    HEAD_ORDER = ["loc", "scale", "weights", "off_diag", "diag"]
    PRED_KEYS = ["test_mix_loc", "test_mix_scale", "test_weight_vals", "test_offdiag_elems", "test_diag_elems"]

    def make_encoder(self, input_dim, n_dims, n_components, obs_types):
        return Inverse_Copula_GMix_Model(input_dim, n_dims, n_components, obs_types)

    def sample_posterior(self, inputs):
        loc, scale, weights, offdiag, diag = self.Encoder_model(inputs)
        LT = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)
        u = gaussian_copula_samples(LT, self.num_samples)
        x = gmix_quantile(u, loc, scale, weights)
        z = self.lbound + (1.0 - self.lbound) * x
        return z, dict(loc=loc, scale=scale, w=weights, LT=LT, u=u)

    def compute_terms(self, inputs):
        z, q = self.sample_posterior(inputs)
        nll, nll_f, nll_v, nll_r = self.nll_terms(inputs, z)

        # entropy of q: exact copula part + Monte-Carlo marginal part (copula path blocked: zero mean), as the KS model
        H_cop = copula_entropy(q["LT"])
        x_marg = gmix_quantile(tf.stop_gradient(q["u"]), q["loc"], q["scale"], q["w"])
        log_q_marg = gmix_logpdf(x_marg, q["loc"], q["scale"], q["w"]) - np.log(1.0 - self.lbound)
        H_marg = -tf.reduce_mean(tf.reduce_sum(log_q_marg, axis=-1), axis=1)
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
