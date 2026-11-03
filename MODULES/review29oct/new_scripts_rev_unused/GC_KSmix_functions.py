#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/GC_KSmix_functions.py

Pure TensorFlow math for a MIXTURE of K Gaussian-copula components, each with
its own single-Kumaraswamy marginals on [lbound, 1]:

    q(z | m) = sum_k pi_k * c_k(F_k1(z_1), ..., F_kD(z_D)) * prod_d f_kd(z_d)

Why a mixture of copulas instead of one copula with mixture marginals:
  * One Gaussian copula with multimodal marginals can only place modes on the
    axis-aligned grid (every combination of marginal modes). The true posteriors
    here are L-shaped / "either-or" (e.g. left bearing OR right bearing degraded),
    which a single Gaussian copula cannot represent.
  * Each component has a closed-form Kumaraswamy quantile, so samples are exact
    reparameterized functions of the Gaussian noise. No bisection solver and no
    hand-written implicit gradient (with its clipping) are needed.
  * With K = 1 the model reduces to the paper's Gaussian copula with
    (Kumaraswamy) marginals, which is a convenient ablation/baseline.
"""

import numpy as np
import tensorflow as tf

_EPS_X = 1e-6     # keeps x strictly inside (0, 1) in float32
_EPS_U = 1e-12    # keeps copula arguments away from 0 and 1


def build_correlation_cholesky(off_diag, diag, n_dims):
    """
    Lower-triangular Cholesky factor of a correlation matrix (unit row norms).
    off_diag: [..., D(D-1)/2], diag: [..., D] (> 0)  ->  L: [..., D, D]
    """
    rows, cols = np.tril_indices(n_dims, k=-1)
    basis = np.zeros((len(rows), n_dims, n_dims), dtype=np.float32)
    basis[np.arange(len(rows)), rows, cols] = 1.0
    L = tf.einsum('...o,oij->...ij', off_diag, tf.constant(basis)) + tf.linalg.diag(diag)
    return L / tf.norm(L, axis=-1, keepdims=True)


def log_survival_normal(y):
    """log(1 - Phi(y)), accurate in both tails (tf.math has no log_ndtr)."""
    y = tf.clip_by_value(y, -8.0, 8.0)
    pos = y > 0.0
    # Feed each branch only the inputs it is valid for, so the unused branch of tf.where
    # never produces inf/NaN (whose gradient would otherwise leak through as NaN).
    y_pos = tf.where(pos, y, tf.zeros_like(y))
    y_neg = tf.where(pos, tf.zeros_like(y), y)
    tail = 0.5 * tf.math.erfc(y_pos * 0.7071067811865476)      # 1 - Phi(y), accurate for y > 0
    head = 0.5 * tf.math.erfc(-y_neg * 0.7071067811865476)     # Phi(y), accurate for y < 0
    return tf.where(pos, tf.math.log(tail), tf.math.log1p(-head))


def sample_components(a, b, L, n_samples, lbound):
    """
    Reparameterized samples from EVERY mixture component.
    a, b: [B, K, D]   L: [B, K, D, D]
    Returns x (unit scale) and z (physical scale), both [B, K, S, D].
    """
    B, K, D = tf.shape(a)[0], tf.shape(a)[1], tf.shape(a)[2]
    g = tf.random.normal(tf.stack([B, K, n_samples, D]))
    y = tf.einsum('bksj,bkij->bksi', g, L)                     # y ~ N(0, L L^T)

    # Closed-form Kumaraswamy quantile, x = (1 - (1-u)^(1/b))^(1/a), with
    # log(1-u) evaluated stably in the tails.
    log_1mu = log_survival_normal(y)
    a_e, b_e = a[:, :, tf.newaxis, :], b[:, :, tf.newaxis, :]
    log_x = tf.math.log(-tf.math.expm1(log_1mu / b_e)) / a_e
    x = tf.clip_by_value(tf.exp(log_x), _EPS_X, 1.0 - _EPS_X)
    z = lbound + (1.0 - lbound) * x
    return x, z


def kumaraswamy_logpdf_logcdf_terms(x, a, b):
    """
    Per-dimension Kumaraswamy terms of every component evaluated at x.
    x: [N, D]  a, b: [N, K, D]  (N = flattened batch*samples, broadcast over K)
    Returns log f(x), u = F(x) and 1 - u, all [N, K, D].
    """
    log_x = tf.math.log(x)[:, tf.newaxis, :]
    a_log_x = a * log_x
    log_1m_xa = tf.math.log(-tf.math.expm1(a_log_x))           # log(1 - x^a), stable
    log_pdf = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * log_x + (b - 1.0) * log_1m_xa
    one_minus_u = tf.exp(b * log_1m_xa)                        # (1 - x^a)^b
    u = -tf.math.expm1(b * log_1m_xa)
    return log_pdf, u, one_minus_u


def gaussian_copula_logdensity(u, one_minus_u, L):
    """
    log c(u) = -sum log diag(L) - 0.5 * y^T (Sigma^-1 - I) y,  y = Phi^-1(u).
    u, one_minus_u: [N, K, D]   L: [N, K, D, D]  ->  [N, K]
    """
    low = u < 0.5
    # Use whichever tail is accurate in float32; the unused branch gets a harmless 0.5.
    u_c = tf.where(low, tf.clip_by_value(u, _EPS_U, 0.5), 0.5 * tf.ones_like(u))
    v_c = tf.where(low, 0.5 * tf.ones_like(u), tf.clip_by_value(one_minus_u, _EPS_U, 0.5))
    y = tf.where(low, tf.math.ndtri(u_c), -tf.math.ndtri(v_c))
    w = tf.linalg.triangular_solve(L, y[..., tf.newaxis], lower=True)[..., 0]
    quad = tf.reduce_sum(tf.square(w), -1) - tf.reduce_sum(tf.square(y), -1)
    log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), -1)
    return -log_det - 0.5 * quad


def mixture_log_density(x, log_pi, a, b, L, lbound):
    """
    log q(z) of the full K-component mixture, evaluated at z = lbound + (1-lbound) x.
    x: [N, D]  log_pi: [N, K]  a, b: [N, K, D]  L: [N, K, D, D]  ->  [N]
    """
    log_pdf, u, one_minus_u = kumaraswamy_logpdf_logcdf_terms(x, a, b)
    log_comp = (gaussian_copula_logdensity(u, one_minus_u, L)
                + tf.reduce_sum(log_pdf, -1)
                - tf.cast(tf.shape(x)[-1], x.dtype) * tf.math.log(1.0 - lbound))
    return tf.reduce_logsumexp(log_pi + log_comp, axis=-1)


def beta_prior_log_density(x, alpha, beta, lbound):
    """Independent Beta(alpha, beta) prior on the unit-scaled factors, in z-space. x: [N, D] -> [N]"""
    log_norm = tf.math.lbeta(tf.constant([alpha, beta], tf.float32))
    log_p = ((alpha - 1.0) * tf.math.log(x) + (beta - 1.0) * tf.math.log1p(-x)
             - log_norm - tf.math.log(1.0 - lbound))
    return tf.reduce_sum(log_p, -1)
