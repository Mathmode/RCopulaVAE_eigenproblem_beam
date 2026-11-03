#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC_KSmix_functions_updated_review27oct.py
=========================================
Full Updated Version of `MODULES/review_27oct/Last_used_scripts/GC_KSmix_functions.py`
for the MS CPENG-8674 Major Revision.

Supported & Verified Fixes Relative to `Last_used_scripts/GC_KSmix_functions.py`:
---------------------------------------------------------------------------------
1. [FIX 1 -- Exact `(Batch, Samples, Dims)` Shape in `gaussian_copula_samples` and Removal of
             `u < 0.99` Rejection Loop]
   - In `Last_used_scripts/GC_KSmix_functions.py` (lines 50-53), `mvn_model.sample(n_samples)`
     returned shape `(n_samples, batch_size, n_dims)` (e.g., `(10, 256, 5)`), whereas
     `_pure_kumaraswamy_sampling_jit` (line 84: `a_t` of shape `(256, 1, 5, K)`) and
     `GC_KSmix_models.py` (line 116: `lt_ext` of shape `(256, 10, 5, 5)`) require
     `copula_samples` to have shape `(batch_size, n_samples, n_dims)` = `(256, 10, 5)`.
   - Furthermore, the `tf.while_loop` on lines 56-72 rejected any `u_i >= 0.99`, cutting off
     the top 1% near the healthy boundary `z = 1.0`.
   - Fix: Directly draw `epsi` with shape `(batch_size, n_samples, n_dims)` and clip `u`
     symmetrically to `[1e-6, 1.0 - 1e-6]`.

2. [FIX 2 -- Removed Double-Softmax on `weight_vals` in `_pure_kumaraswamy_sampling_jit` and
             `fast_kumaraswamy_quantile`]
   - In `Last_used_scripts/GC_KSmix_models.py` (line 73), `Inverse_Copula_KS_Model.call`
     already applies `weights = tf.nn.softmax(weights / 1.5, axis=1)`.
   - However, `Last_used_scripts/GC_KSmix_functions.py` (lines 86 and 120) applied
     `tf.nn.softmax` a SECOND time to those already-normalized probabilities (`weight_vals`),
     preventing any mixture component weight from ever exceeding `e^1 / (e^1 + K - 1)` or
     reaching zero, and creating a mismatch with `Marginal_KSMix_pdf_logprob` (which used `w`
     directly without a second softmax).
   - Fix: Normalize `weights_t` by its sum (`w_t / tf.reduce_sum(w_t, axis=-1, keepdims=True)`)
     so that `weight_vals` are the exact same probabilities in both sampling and PDF evaluation.

3. [FIX 3 -- Fast Exact Closed-Form Inverse CDF When `num_KS == 1` (Unimodal Kumaraswamy)]
   - For a single Kumaraswamy component (`num_KS == 1`), the inverse CDF has an exact analytical
     closed form:
         `x = (1 - (1 - u)^(1 / b))^(1 / a)`
     which is 100% differentiable via standard TensorFlow autodiff without needing a 35-step
     bisection loop!
   - When `num_KS > 1` (multimodal Kumaraswamy mixture), it automatically uses the 35-step
     bisection + implicit differentiation (`@tf.custom_gradient`) solver.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


@tf.function(jit_compile=True)
def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """Batched construction of unit-row-norm lower-triangular Cholesky factors `L_corr`."""
    def build_single_correlation_matrix(batch_off_diag_elements, batch_diag_elements):
        L = tf.zeros((n_dims, n_dims), dtype=tf.float32)

        tril_indices = np.tril_indices(n_dims, k=-1)
        indices = tf.stack([
            tf.cast(tril_indices[0], tf.int32),
            tf.cast(tril_indices[1], tf.int32),
        ], axis=-1)
        L = tf.tensor_scatter_nd_update(L, indices, batch_off_diag_elements)

        diag_indices = tf.range(n_dims)
        diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
        L = tf.tensor_scatter_nd_update(L, diag_indices, batch_diag_elements)

        row_norms = tf.norm(L, axis=1, keepdims=True)
        L = L / (row_norms + 1e-7)
        return L

    LT_matrices = tf.vectorized_map(
        lambda args: build_single_correlation_matrix(*args),
        (off_diag_elements, diag_elements),
    )
    return LT_matrices


def gaussian_copula_samples(LT_matrices, n_dims, n_samples, lbound=0.45):
    """
    Draws reparameterized Gaussian copula uniform samples `u` of shape `(batch_size, n_samples, n_dims)`.
    For each batch b and sample h, y_{b,h,k} = sum_d epsi_{b,h,d} * L_{b,k,d}.
    """
    batch_size = tf.shape(LT_matrices)[0]
    epsi = tf.random.normal(shape=(batch_size, n_samples, n_dims), mean=0.0, stddev=1.0, dtype=tf.float32)
    mvn_samples = tf.einsum("bhd,bkd->bhk", epsi, LT_matrices)
    u_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
    return tf.clip_by_value(u_samples, 1e-6, 1.0 - 1e-6)


@tf.function(jit_compile=True)
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Unconditionally stable Bisection solver for Kumaraswamy Mixtures (`num_KS >= 1`).
    Input shapes:
      - a_params, b_params, weight_vals : (Batch, num_KS, Dims)
      - copula_samples                  : (Batch, Samples, Dims)
    Output shape:
      - x in [0, 1]                     : (Batch, Samples, Dims)
    """
    u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)

    # Align parameters to (Batch, 1, Dims, Components)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    # [FIX 2]: weight_vals is already softmaxed in Inverse_Copula_KS_Model; normalize safely without double-softmax
    w_raw = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    weights_t = w_raw / (tf.reduce_sum(w_raw, axis=-1, keepdims=True) + 1e-12)

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(35):
        mid = 0.5 * (low + high)
        mid_expand = tf.expand_dims(mid, axis=-1)

        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0)
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)

        cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
        is_less = cdf_mid < u

        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    return 0.5 * (low + high)


@tf.custom_gradient
def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
    x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)

    def grad(dy):
        # Clip elementwise instead of global Frobenius norm over (Batch, Samples, Dims)
        # (global clip_by_norm divided gradients by sqrt(B*S*D) ~ 113!)
        dy_safe = tf.clip_by_value(dy, -10.0, 10.0)

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])

            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_raw = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_soft = w_raw / (tf.reduce_sum(w_raw, axis=-1, keepdims=True) + 1e-12)

            x_stop = tf.stop_gradient(tf.clip_by_value(x, 1e-5, 1.0 - 1e-5))
            x_expand = tf.expand_dims(x_stop, axis=-1)

            ln_x = tf.math.log(x_expand)
            x_a = tf.math.exp(tf.clip_by_value(a_t * ln_x, -30.0, -1e-7))
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0)
            ln_one_minus_xa = tf.math.log(one_minus_x_a)
            cdf_comp = 1.0 - tf.math.exp(tf.clip_by_value(b_t * ln_one_minus_xa, -30.0, 0.0))
            cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)

        log_pdf_comp = (
            tf.math.log(a_t)
            + tf.math.log(b_t)
            + (a_t - 1.0) * ln_x
            + (b_t - 1.0) * ln_one_minus_xa
        )
        pdf_comp = tf.math.exp(tf.clip_by_value(log_pdf_comp, -15.0, 15.0))
        pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)

        pdf_safe = tf.clip_by_value(pdf_x, 0.05, 50.0)
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe

        grads = tape.gradient(
            cdf_x,
            [a_params, b_params, weight_vals],
            output_gradients=grad_common,
        )

        def clean(g, limit=20.0):
            if g is None:
                return None
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), None, None

    z = lbound + (1.0 - lbound) * x
    return z, grad


@tf.function
def ks_mixture_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    If `num_KS == 1` (unimodal Kumaraswamy), uses the exact closed-form analytical quantile
    `x = (1 - (1 - u)^(1/b))^(1/a)` in log-space with exact autodiff gradients.
    If `num_KS > 1` (multimodal mixture), uses the 35-step bisection + implicit-gradient solver.
    """
    if a_params.shape[1] == 1:
        u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)
        a_1 = a_params  # shape (Batch, 1, Dims)
        b_1 = b_params  # shape (Batch, 1, Dims)
        ln_1_minus_u = tf.math.log(tf.clip_by_value(1.0 - u, 1e-7, 1.0))
        one_minus_u_pow = tf.math.exp(tf.clip_by_value(ln_1_minus_u / b_1, -30.0, -1e-7))
        ln_1_minus_pow = tf.math.log(tf.clip_by_value(1.0 - one_minus_u_pow, 1e-7, 1.0))
        x = tf.math.exp(tf.clip_by_value(ln_1_minus_pow / a_1, -30.0, 0.0))
        return lbound + (1.0 - lbound) * x
    return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)
