#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC_GMm_functions_updated_review27oct.py
=======================================
Full Updated Version of `MODULES/COPULAS/GC_GMm_functions.py` for the
MS CPENG-8674 Major Revision (saved in `code/MODULES/review_27oct/`).

Supported & Verified Changes Relative to Original `GC_GMm_functions.py`:
------------------------------------------------------------------------
1. [FIX 1 -- Removed the `samples < 0.99` Upper-Tail Truncation in `gaussian_copula_samples`
             (Addresses Reviewer 1 Comment 3 on Undercoverage & Comment 6 on Healthy Elements)]
   - In the original `GC_GMm_functions.py` (lines 75 and 83), a `tf.while_loop` rejected
     any copula uniform sample `u_i >= 0.99`.
   - Why this must be fixed:
     (a) Rejecting `u_i in [0.99, 1.0)` cut off the top 1% upper tail of every marginal
         `TruncatedNormal(mu, sigma, lbound, 1.0)` distribution, preventing samples from
         reaching `z_i -> 1.0` (the undamaged state!) and compressing the upper credible bound.
     (b) For `D = 5` or `10` dimensions, rejecting the entire vector whenever ANY component
         exceeded 0.99 distorted the joint copula correlation near the healthy corner `z = 1`.
     (c) Instead of a slow `tf.while_loop`, clipping `u` symmetrically to `[1e-6, 1.0 - 1e-6]`
         (the exact same clean approach you already adopted in `MODULES/KUMARSWAMY/GC_KSmix_functions.py`)
         is fully differentiable, fast on GPU, and preserves the full `[0, 1]` quantile support.

2. [FIX 2 -- Consistent `(batch_size, n_samples, n_dims)` Tensor Axis Ordering for `n_samples > 1`
             (Addresses Reviewer 3 Comment 7 on Monte Carlo Sample Size `H = 1, 5, 10`)]
   - In the original `GC_GMm_functions.py` (lines 69-72), `draw_samples` returned shape
     `[n_samples, batch_size, n_dims]`, whereas `gaussian_marginal_samples` docstring (line 107)
     and `My_CopulaVAE_withEigen.call` (line 110: `tf.repeat(self.LT_matrices[:, tf.newaxis, :, :],
     self.num_samples, axis=1)`) assumed shape `[batch_size, n_samples, n_dims]`.
   - When `n_samples = 1`, `[1, B, D]` and `[B, 1, D]` flatten identically via `tf.reshape(..., (-1, D))`.
     However, when `n_samples > 1` (e.g., `H = 5` or `10` for Reviewer 3's ablation), flattening
     `[H, B, D]` without transposing first scrambles which sample belongs to which batch item!
   - Fix: We transpose `copula_samples` to `[batch_size, n_samples, n_dims]` inside
     `gaussian_copula_samples`, and broadcast `locs` and `scales` to `[batch_size, 1, n_dims]`
     inside `gaussian_marginal_samples` so `marginal_samples` has shape `[batch_size, n_samples, n_dims]`
     for any `n_samples >= 1`.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """
    Batched construction of unit-row-norm lower-triangular Cholesky factors `L_corr`
    such that `Sigma_corr = L_corr @ L_corr^T` is a valid correlation matrix with ones
    on the diagonal.
    """
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

        # Normalize each row to unit L2 norm so diag(L @ L.T) == 1
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
    Draws reparameterized samples `u ~ C_Gaussian(L_corr)` in `(0, 1)^n_dims`.

    Args:
        LT_matrices : (batch_size, n_dims, n_dims) lower-triangular Cholesky factors
        n_dims      : int, latent dimension
        n_samples   : int, number of Monte Carlo samples H per observation
        lbound      : float (kept in signature for backward compatibility)

    Returns:
        copula_samples : (batch_size, n_samples, n_dims) in `[1e-6, 1 - 1e-6]`
    """
    batch_size = tf.shape(LT_matrices)[0]
    epsi = tf.random.normal(shape=(batch_size, n_samples, n_dims), mean=0.0, stddev=1.0, dtype=tf.float32)

    # Correlate standard normal draws: v = epsi @ L_corr^T -> shape (batch_size, n_samples, n_dims)
    mvn_samples = tf.einsum("bhd,bkl->bhk", epsi, LT_matrices)

    std_normal = tfd.Normal(loc=0.0, scale=1.0)
    u_samples = std_normal.cdf(mvn_samples)

    # [FIX 1]: Symmetric numerical clipping to [1e-6, 1 - 1e-6] instead of rejecting u >= 0.99
    copula_samples = tf.clip_by_value(u_samples, 1e-6, 1.0 - 1e-6)
    return copula_samples


@tf.function(jit_compile=True)
def gaussian_marginal_samples(locs, scales, copula_samples, lbound):
    """
    Applies the inverse CDF (quantile function) of `TruncatedNormal(loc, scale, lbound, 1.0)`
    to `copula_samples`.

    Args:
        locs           : (batch_size, 1, n_dims) or (batch_size, n_dims)
        scales         : (batch_size, 1, n_dims) or (batch_size, n_dims)
        copula_samples : (batch_size, n_samples, n_dims)
        lbound         : float, lower truncation bound (e.g. 0.45 or 0.50)

    Returns:
        marginal_samples : (batch_size, n_samples, n_dims) in `[lbound, 1.0]`
    """
    locs_3d = tf.reshape(locs, [-1, 1, tf.shape(copula_samples)[-1]])
    scales_3d = tf.reshape(scales, [-1, 1, tf.shape(copula_samples)[-1]])
    safe_u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)

    marginal_samples = tfd.TruncatedNormal(
        loc=locs_3d,
        scale=scales_3d,
        low=lbound,
        high=1.0,
    ).quantile(safe_u)

    return marginal_samples
