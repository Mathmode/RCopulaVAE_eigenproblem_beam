#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussCopula_GaussMarginals/GCLogN_model.py

One Gaussian per marginal, as the baseline GC-TN, but in y = ln z:

    q(z | m) = c_R( F_1(z_1), ..., F_7(z_7) )  prod_d  TN(ln z_d; m_d, s_d, [ln lbound, 0]) / z_d,

i.e. a log-normal stiffness factor truncated to the prior support. Same encoder, Gaussian copula, eigen-decoder, calibrated
likelihood, negative ELBO with gamma = 1, warm-up and training as GC-TN, which is subclassed. Why ln z: the marginals of the
exact posteriors are right-skewed bells in z and nearly symmetric in ln z, so one component can follow them.

Closed forms, as GC-TN: quantile z = exp(F^-1(u)), no bisection or rejection sampling, and entropy in z units
H[z] = H[y] + E[y], with the entropy and the mean of the truncated normal.

Location: m in (ln lbound - k s, k s), k = 3, so the peak may lie outside the support and the density can keep rising up to a
bound. The best fits to the exact marginals need at most 2.6 s. The normaliser Z stays >= Phi(-3) = 0.0013 and is computed from
the tails, which keeps its digits in float32.
"""

import numpy as np
import tensorflow as tf

from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import (
    GaussCopulaTNVAE, Inverse_Copula_TN_Model, HALF_LOG_2PI_E, std_normal_cdf, std_normal_pdf,
)

EXCESS_SIGMAS = 3.0              # how far outside the support the location may lie, in units of s


def log_location(loc01, scale, lbound):
    """m in (ln lbound - k s, k s) from loc01 in (0, 1), the sigmoid output of the encoder."""
    lo = float(np.log(lbound))
    return 0.5 * lo + (-0.5 * lo + EXCESS_SIGMAS * scale) * (2.0 * loc01 - 1.0)


def logn_normaliser(m, s, lbound):
    """alpha, beta, P(X < alpha), P(X > beta) and Z = P(alpha < X < beta) of the standard normal X (from the tails)."""
    alpha, beta = (float(np.log(lbound)) - m) / s, -m / s
    p_below, p_above = std_normal_cdf(alpha), std_normal_cdf(-beta)
    Z = tf.where(alpha > 0.0, std_normal_cdf(-alpha) - p_above, std_normal_cdf(beta) - p_below)
    return alpha, beta, p_below, p_above, Z


def logn_quantile(v, m, s, lbound):
    """z = exp(F^-1(v)), the quantile of the truncated log-normal at levels v (broadcast); differentiable in v, m and s."""
    _, _, p_below, p_above, Z = logn_normaliser(m, s, lbound)
    p_low = p_below + v * Z                 # Phi(xi), from below
    p_up = p_above + (1.0 - v) * Z          # 1 - Phi(xi), from above
    lower = p_low < 0.5
    half = 0.5 * tf.ones_like(p_low)
    xi = tf.where(lower, tf.math.ndtri(tf.where(lower, p_low, half)), -tf.math.ndtri(tf.where(lower, half, p_up)))
    return tf.clip_by_value(tf.exp(m + s * xi), lbound + 1e-6, 1.0 - 1e-6)


def logn_entropy(m, s, lbound):
    """Entropy in z units: H[y] + E[y], y = ln z truncated normal."""
    alpha, beta, _, _, Z = logn_normaliser(m, s, lbound)
    phi_a, phi_b = std_normal_pdf(alpha), std_normal_pdf(beta)
    H_y = HALF_LOG_2PI_E + tf.math.log(s) + tf.math.log(Z) + (alpha * phi_a - beta * phi_b) / (2.0 * Z)
    return H_y + m + s * (phi_a - phi_b) / Z


class Inverse_Copula_LogN_Model(Inverse_Copula_TN_Model):
    """Encoder of GC-TN; the location is mapped to the range of log_location."""

    def __init__(self, input_dim_encoder, n_dims, obs_types, hidden, lbound, **kwargs):
        super().__init__(input_dim_encoder, n_dims, obs_types, hidden, **kwargs)
        self.lbound = float(lbound)

    def call(self, inputs):
        loc01, scale, offdiag, diag = super().call(inputs)
        return log_location(loc01, scale, self.lbound), scale, offdiag, diag


class GaussCopulaLogNVAE(GaussCopulaTNVAE):
    """Gaussian copula + truncated log-normal marginals; likelihood, decoder and training steps inherited."""
    PRED_KEYS = ["test_logz_loc", "test_logz_scale", "test_offdiag_elems", "test_diag_elems"]

    def make_encoder(self, input_dim, n_dims, n_components, obs_types):
        return Inverse_Copula_LogN_Model(input_dim, n_dims, obs_types, self.HIDDEN, self.lbound)

    def sample_posterior(self, inputs):
        from MODULES.review29oct_fixed.GC_KSmix_functions import build_correlation_matrices_from_cholesky, gaussian_copula_samples
        m, s, offdiag, diag = self.Encoder_model(inputs)
        LT = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)
        u = gaussian_copula_samples(LT, self.num_samples)
        z = logn_quantile(u, m[:, None, :], s[:, None, :], self.lbound)
        return z, dict(m=m, s=s, LT=LT, u=u)

    def compute_terms(self, inputs):
        from MODULES.review29oct_fixed.GC_KSmix_functions import copula_entropy
        z, q = self.sample_posterior(inputs)
        nll, nll_f, nll_v, nll_r = self.nll_terms(inputs, z)

        # entropy of q in z units: exact copula part + exact marginals
        H_cop = copula_entropy(q["LT"])
        H_marg = tf.reduce_sum(logn_entropy(q["m"], q["s"], self.lbound), axis=-1)
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
