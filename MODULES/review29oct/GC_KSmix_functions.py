#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/GC_KSmix_functions.py

100% XLA-Compatible Functions for Remote GPU Execution:
  - Eliminates `tf.linalg.set_diag` and dynamic `tf.shape` inside @tf.function(jit_compile=True),
    fixing the `gradients/set_diag_grad/zeros` tf2xla crash.
  - Uses pure `tf.einsum` and static broadcasting for Cholesky construction, Copula sampling,
    and Bisection Kumaraswamy Mixture Quantiles.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _get_tril_basis_matrix(n_dims):
    """
    Precomputes a static numpy basis tensor of shape (n_corr, n_dims, n_dims)
    that maps the flat off-diagonal vector (Batch, n_corr) directly into a strictly
    lower-triangular matrix via `tf.einsum('bc,cij->bij', off_diag, basis)`.
    Zero dynamic shape ops, zero scatter ops, zero set_diag ops!
    """
    n_dims = int(n_dims)
    rows, cols = np.tril_indices(n_dims, k=-1)
    n_corr = len(rows)
    basis = np.zeros((n_corr, n_dims, n_dims), dtype=np.float32)
    for k in range(n_corr):
        basis[k, rows[k], cols[k]] = 1.0
    return tf.constant(basis, dtype=tf.float32)


@tf.function(jit_compile=True)
def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """
    Constructs valid lower-triangular Cholesky factors L of correlation matrices R = L @ L^T
    using ONLY static `tf.einsum` and broadcasting (no `set_diag`, no `tf.shape`).
      off_diag_elements : (Batch, n_dims*(n_dims-1)//2)
      diag_elements     : (Batch, n_dims)
      Returns           : L of shape (Batch, n_dims, n_dims) with unit row L2-norms.
    """
    n_dims_int = int(n_dims)
    off_diag_elements = tf.cast(off_diag_elements, tf.float32)
    diag_elements = tf.cast(diag_elements, tf.float32)

    # 1. Strictly lower-triangular part via static einsum basis
    tril_basis = _get_tril_basis_matrix(n_dims_int)
    L_strict = tf.einsum("bc,cij->bij", off_diag_elements, tril_basis)

    # 2. Diagonal part via static eye broadcasting (avoids tf.linalg.set_diag XLA bug!)
    eye_d = tf.eye(n_dims_int, dtype=tf.float32)[tf.newaxis, :, :]
    L_diag = tf.expand_dims(tf.maximum(diag_elements, 1e-4), axis=-1) * eye_d

    L_raw = L_strict + L_diag

    # 3. Normalize each row to unit L2 norm so diag(L @ L^T) == 1.0
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(L_raw), axis=-1, keepdims=True) + 1e-12)
    L_unit = L_raw / row_norms
    return L_unit


def gaussian_copula_samples(LT_matrices, n_dims, n_samples, lbound=0.45):
    """
    Draws reparameterized samples u in (0, 1)^D from the Gaussian Copula with Cholesky factor L.
    Kept outside XLA jit_compile so `tf.random.normal` handles dynamic last-batch sizes seamlessly.
      LT_matrices : (Batch, D, D)
      Returns     : u_samples of shape (Batch, n_samples, D)
    """
    LT_matrices = tf.cast(LT_matrices, tf.float32)
    batch_size = tf.shape(LT_matrices)[0]
    n_dims_int = int(n_dims)
    n_samples_int = int(n_samples)

    eps = tf.random.normal(
        shape=tf.stack([batch_size, n_samples_int, n_dims_int]),
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
    )
    mvn_samples = tf.einsum("bsd,bjd->bsj", eps, LT_matrices)

    # Fast standard normal CDF using tf.math.erf (100% XLA and graph friendly)
    inv_sqrt2 = tf.constant(0.7071067811865475, dtype=tf.float32)
    u_samples = 0.5 * (1.0 + tf.math.erf(mvn_samples * inv_sqrt2))
    return tf.clip_by_value(u_samples, 1e-6, 1.0 - 1e-6)


@tf.function(jit_compile=True)
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weights_prob, copula_samples, lbound):
    """
    Unconditionally stable 32-step Bisection solver for Kumaraswamy Mixtures on [0, 1].
    `weights_prob` are ALREADY normalized probabilities along axis=1 (num_KS).
    """
    u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)

    # Align parameters to (Batch, 1, Dims, num_KS)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    w_t = tf.transpose(weights_prob, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    w_t = w_t / tf.maximum(tf.reduce_sum(w_t, axis=-1, keepdims=True), 1e-8)

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(32):
        mid = 0.5 * (low + high)
        mid_expand = tf.expand_dims(mid, axis=-1)

        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-8), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-12, 1.0)
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)

        cdf_mid = tf.reduce_sum(w_t * cdf_components, axis=-1)
        is_less = cdf_mid < u

        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    return 0.5 * (low + high)


@tf.custom_gradient
def fast_kumaraswamy_quantile(a_params, b_params, weights_prob, copula_samples, lbound):
    """
    Implicit differentiation through the inverse mixture CDF.
    """
    x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weights_prob, copula_samples, lbound)

    def grad(dy):
        dy_safe = tf.clip_by_value(dy, -2.0, 2.0)

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weights_prob])

            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_t = tf.transpose(weights_prob, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_t = w_t / tf.maximum(tf.reduce_sum(w_t, axis=-1, keepdims=True), 1e-8)

            x_stop = tf.stop_gradient(tf.clip_by_value(x, 1e-5, 1.0 - 1e-5))
            x_expand = tf.expand_dims(x_stop, axis=-1)

            x_a = tf.math.pow(x_expand, a_t)
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-10, 1.0)
            cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
            cdf_x = tf.reduce_sum(w_t * cdf_comp, axis=-1)

        x_pow = tf.math.pow(x_expand, a_t - 1.0)
        one_minus_x_a_pow = tf.math.pow(one_minus_x_a, b_t - 1.0)
        pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        pdf_x = tf.reduce_sum(w_t * pdf_comp, axis=-1)

        pdf_safe = tf.clip_by_value(pdf_x, 1e-2, 100.0)
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe

        grads = tape.gradient(
            cdf_x, [a_params, b_params, weights_prob],
            output_gradients=grad_common
        )

        grad_u = tf.clip_by_value(dy_safe * (1.0 - lbound) / pdf_safe, -10.0, 10.0)

        def clean(g, limit=15.0):
            if g is None:
                return None
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), clean(grad_u), None

    z = lbound + (1.0 - lbound) * x
    return z, grad


@tf.function
def ks_mixture_marginal_samples(a_params, b_params, weights_prob, copula_samples, lbound):
    return fast_kumaraswamy_quantile(a_params, b_params, weights_prob, copula_samples, lbound)