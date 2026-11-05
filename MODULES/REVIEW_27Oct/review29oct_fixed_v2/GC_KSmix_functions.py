#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v2/GC_KSmix_functions.py

Gaussian copula + Kumaraswamy-mixture marginals: sampling, densities and an exact reparameterisation.

Shapes: a, b, w are (B, K, D) (batch, mixture components, parameters), as in review29oct;
uniforms u and unit-interval samples x are (B, S, D) (S Monte-Carlo samples per observation).

Changes with respect to MODULES/review29oct/GC_KSmix_functions.py:
  * the gradient of the mixture quantile is the exact implicit gradient
        dx/dtheta = -(dF/dtheta)(x) / f(x),   dx/du = 1 / f(x),
    without the old clipping (dy clipped to norm 1 over the WHOLE batch, f(x) clipped to [5e-4, 50],
    gradients clipped to +-20), which biased the gradients, most of all for narrow components (f > 50);
  * stable log(1 - x^a) and log-densities, so narrow components near z = 1 do not produce NaNs.
"""

import numpy as np
import tensorflow as tf

SQRT_HALF = 0.7071067811865476


def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """Row-normalised lower-triangular factor L of a correlation matrix R = L L^T (unit diagonal)."""
    tril_rows, tril_cols = np.tril_indices(n_dims, k=-1)
    basis_off = np.zeros((len(tril_rows), n_dims, n_dims), dtype=np.float32)
    basis_off[np.arange(len(tril_rows)), tril_rows, tril_cols] = 1.0

    L = tf.einsum("BO,OIJ->BIJ", off_diag_elements, tf.constant(basis_off)) + tf.linalg.diag(diag_elements)
    return L / tf.norm(L, axis=-1, keepdims=True)


def gaussian_copula_samples(LT_matrices, n_samples, eps=1e-6):
    """Uniforms u (B, S, D) of the Gaussian copula with correlation L L^T."""
    batch_size = tf.shape(LT_matrices)[0]
    n_dims = tf.shape(LT_matrices)[-1]
    epsi = tf.random.normal(shape=(batch_size, n_samples, n_dims), dtype=LT_matrices.dtype)
    mvn = tf.einsum("BSD,BFD->BSF", epsi, LT_matrices)
    return tf.clip_by_value(0.5 * tf.math.erfc(-mvn * SQRT_HALF), eps, 1.0 - eps)


def _log1mexp(t):
    """log(1 - exp(t)) for t < 0, accurate for t near 0 and for t << 0."""
    return tf.where(t > -0.6931472, tf.math.log(-tf.math.expm1(t)), tf.math.log1p(-tf.math.exp(t)))


def ks_mixture_cdf(x, a_params, b_params, weight_vals):
    """Mixture CDF sum_k w_k [1 - (1 - x^a_k)^b_k] at x (B, S, D)."""
    xe = x[:, :, tf.newaxis, :]
    a, b, w = a_params[:, tf.newaxis], b_params[:, tf.newaxis], weight_vals[:, tf.newaxis]
    log_1m_xa = _log1mexp(a * tf.math.log(xe))
    return tf.reduce_sum(w * (-tf.math.expm1(b * log_1m_xa)), axis=2)


def ks_mixture_logpdf(x, a_params, b_params, weight_vals):
    """log of the mixture density on the unit interval at x (B, S, D)."""
    xe = x[:, :, tf.newaxis, :]
    a, b = a_params[:, tf.newaxis], b_params[:, tf.newaxis]
    log_w = tf.math.log(tf.maximum(weight_vals[:, tf.newaxis], 1e-30))
    log_x = tf.math.log(xe)
    log_pdf_k = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * log_x + (b - 1.0) * _log1mexp(a * log_x)
    return tf.reduce_logsumexp(log_w + log_pdf_k, axis=2)


def _bisection_quantile(u, a_params, b_params, weight_vals, n_iter=32):
    low = tf.zeros_like(u)
    high = tf.ones_like(u)
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        is_less = ks_mixture_cdf(mid, a_params, b_params, weight_vals) < u
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)
    return tf.clip_by_value(0.5 * (low + high), 1e-6, 1.0 - 1e-6)


@tf.custom_gradient
def ks_mixture_quantile(u, a_params, b_params, weight_vals):
    """x = F^{-1}(u) of the Kumaraswamy mixture (unit interval), with exact implicit gradients."""
    x = tf.stop_gradient(_bisection_quantile(u, a_params, b_params, weight_vals))

    def grad(dx):
        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])
            cdf = ks_mixture_cdf(x, a_params, b_params, weight_vals)
        inv_pdf = 1.0 / tf.maximum(tf.exp(ks_mixture_logpdf(x, a_params, b_params, weight_vals)), 1e-12)
        g_a, g_b, g_w = tape.gradient(cdf, [a_params, b_params, weight_vals], output_gradients=-dx * inv_pdf)

        def finite(g):
            return tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        return finite(dx * inv_pdf), finite(g_a), finite(g_b), finite(g_w)

    return x, grad


def copula_entropy(LT_matrices):
    """Entropy contribution of a Gaussian copula: -E[log c(U)] = 0.5 log det R = sum log diag(L) (<= 0)."""
    return tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LT_matrices)), axis=-1)


# ---- NumPy versions (post-processing) ----
def ks_mixture_pdf_np(x_unit, a, b, w):
    """Density on the unit interval; x_unit (G,), a/b/w (K,) -> (G,)."""
    x = np.clip(np.asarray(x_unit, dtype=np.float64), 1e-12, 1 - 1e-12)[:, None]
    a, b, w = (np.asarray(v, dtype=np.float64)[None] for v in (a, b, w))
    pdf = a * b * x ** (a - 1.0) * np.clip(1.0 - x ** a, 1e-300, 1.0) ** (b - 1.0)
    return np.sum(w * pdf, axis=1) / np.sum(w)


def ks_mixture_cdf_np(x_unit, a, b, w):
    x = np.clip(np.asarray(x_unit, dtype=np.float64), 0.0, 1.0)[..., None]
    a, b, w = (np.asarray(v, dtype=np.float64) for v in (a, b, w))
    return np.sum(w * (1.0 - (1.0 - x ** a) ** b), axis=-1) / np.sum(w, axis=-1)
