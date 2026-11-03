#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/GC_KSmix_functions.py
Restored from original GC_KSmix_functions.py with redundant second softmax removed.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


@tf.function
def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """Batched Cholesky correlation factor construction."""
    tril_rows, tril_cols = np.tril_indices(n_dims, k=-1)
    n_off = len(tril_rows)

    basis_off = np.zeros((n_off, n_dims, n_dims), dtype=np.float32)
    for k in range(n_off):
        basis_off[k, tril_rows[k], tril_cols[k]] = 1.0
    basis_off_tf = tf.constant(basis_off, dtype=tf.float32)

    L_off = tf.einsum('BO, OIJ -> BIJ', off_diag_elements, basis_off_tf)
    L_diag = tf.linalg.diag(diag_elements)
    L = L_off + L_diag

    row_norms = tf.norm(L, axis=-1, keepdims=True)
    L = L / (row_norms + 1e-7)
    return L


@tf.function
def gaussian_copula_samples(LT_matrices, n_dims, n_samples, lbound=0.45):
    """Draws uniform [0, 1] samples from the Gaussian Copula."""
    batch_size = tf.shape(LT_matrices)[0]
    epsi_samples = tf.random.normal(shape=(batch_size, n_samples, n_dims), dtype=tf.float32)
    mvn_samples = tf.einsum('BSD, BFD -> BSF', epsi_samples, LT_matrices)
    copula_samples = 0.5 * tf.math.erfc(-mvn_samples * 0.7071067811865475)
    return tf.clip_by_value(copula_samples, 1e-5, 1.0 - 1e-5)


@tf.function
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound=0.45):
    """
    Unconditionally stable Bisection solver for Kumaraswamy Mixtures.
    FIX: weight_vals are ALREADY softmax-normalized in Inverse_Copula_KS_Model,
    so we do NOT apply a second softmax here!
    """
    u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)

    # Align parameters to (Batch, 1, Dims, Components)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    weights_t = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    weights_t = weights_t / (tf.reduce_sum(weights_t, axis=-1, keepdims=True) + 1e-8)

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(30):
        mid = (low + high) / 2.0
        mid_expand = tf.expand_dims(mid, axis=-1)

        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-20, 1.0)
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)

        cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
        is_less = cdf_mid < u

        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    return (low + high) / 2.0


@tf.custom_gradient
def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
    x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)

    def grad(dy):
        dy_safe = tf.clip_by_norm(dy, 1.0)

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])

            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_soft = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_soft = w_soft / (tf.reduce_sum(w_soft, axis=-1, keepdims=True) + 1e-8)

            x_stop = tf.stop_gradient(x)
            x_expand = tf.expand_dims(x_stop, axis=-1)

            x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-20, 1.0)
            cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
            cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)

        x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), tf.maximum(a_t - 1.0, -0.9))
        one_minus_x_a_pow = tf.math.pow(one_minus_x_a, tf.maximum(b_t - 1.0, -0.9))
        pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)

        pdf_safe = tf.clip_by_value(pdf_x, 5e-4, 50.0)
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe
        grad_u = dy_safe * (1.0 - lbound) / pdf_safe

        grads = tape.gradient(
            cdf_x, [a_params, b_params, weight_vals],
            output_gradients=grad_common
        )

        def clean(g, limit=20.0):
            if g is None:
                return None
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), clean(grad_u), None

    z = lbound + (1.0 - lbound) * x
    return z, grad


@tf.function
def ks_mixture_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)