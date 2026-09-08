# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:04:06 2026
Enhanced Functions for Gaussian Copula and Kumaraswamy Mixture Marginals.
Includes Bisection Solver for Quantiles and Physics-Prioritized Gradients.
NaN fixes applied: Removed redundant double softmax, fixed float32 clipping.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

@tf.function(jit_compile = True)
def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """Batched version using tf.vectorized_map."""
    def build_single_correlation_matrix(batch_off_diag_elements, batch_diag_elements):
        L = tf.zeros((n_dims, n_dims), dtype=tf.float32)

        # Fill the lower triangular part with off-diagonal elements
        tril_indices = np.tril_indices(n_dims, k=-1)
        indices = tf.stack([
            tf.cast(tril_indices[0], tf.int32),
            tf.cast(tril_indices[1], tf.int32)
        ], axis=-1)
        L = tf.tensor_scatter_nd_update(L, indices, batch_off_diag_elements)

        # Fill the diagonal
        diag_indices = tf.range(n_dims)
        diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
        L = tf.tensor_scatter_nd_update(L, diag_indices, batch_diag_elements)
        
        # Safe norm equivalent to tf.norm(L) + 1e-12 but without zero-gradient issues
        row_norms = tf.sqrt(tf.reduce_sum(tf.square(L), axis=1, keepdims=True) + 1e-12)
        L = L / row_norms

        return L
    
    LT_matrices = tf.vectorized_map(
        lambda args: build_single_correlation_matrix(*args),
        (off_diag_elements, diag_elements),)
    
    return LT_matrices

@tf.function(jit_compile=True)
def gaussian_copula_samples(LT_matrices, n_dims, n_samples, lbound):
    """
    Generates samples via fast differentiable einsum (bypassing the slow while_loop).
    CRITICAL FIX: 'bik, bsk -> bsi' correctly computes L * eps (covariance L*L^T),
    preventing the -3e23 PDF evaluation crash.
    """
    batch_size = tf.shape(LT_matrices)[0]
    
    # Independent noise for every sample in the batch: [Batch, Samples, Dims]
    epsilon = tf.random.normal(shape=(batch_size, n_samples, n_dims))
    
    # Matmul mapping correctly computing L * epsilon
    mvn_samples = tf.einsum('bik, bsk -> bsi', LT_matrices, epsilon)
    
    # Transform to Uniform [0, 1] via Normal CDF
    u = tfd.Normal(0.0, 1.0).cdf(mvn_samples)
    
    # Exact safety bounds from your original while_loop implementation
    return tf.clip_by_value(u, 1e-5, 1.0 - 1e-5)

@tf.function(jit_compile=True)
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Unconditionally stable Bisection solver for Kumaraswamy Mixtures.
    Double softmax removed to fix vanishing gradients when scaling components.
    """
    u = tf.clip_by_value(copula_samples, 1e-7, 1.0 - 1e-7)
    
    # Align parameters to (Batch, 1, Dims, Components)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    
    # FIX: weights are already softmaxed in the Model. Do NOT apply tf.nn.softmax here!
    weights_t = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(35):
        mid = (low + high) / 2.0
        mid_expand = tf.expand_dims(mid, axis=-1)
        
        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        # FIX: float32 clipping adjusted from 1e-30 to 1e-7. 
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0) 
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
        signal_multiplier = 5.0
        dy_safe = tf.clip_by_norm(dy * signal_multiplier, 1.0) 

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])
            
            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            
            # FIX: Double softmax removed. Use raw model weights.
            w_t = tf.transpose(weight_vals, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            
            x_stop = tf.stop_gradient(x)
            x_expand = tf.expand_dims(x_stop, axis=-1)
            
            x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0)
            cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
            cdf_x = tf.reduce_sum(w_t * cdf_comp, axis=-1)
            
        x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), tf.maximum(a_t - 1.0, -0.9))
        one_minus_x_a_pow = tf.math.pow(one_minus_x_a, tf.maximum(b_t - 1.0, -0.9))
        pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        pdf_x = tf.reduce_sum(w_t * pdf_comp, axis=-1)
        
        pdf_safe = tf.clip_by_value(pdf_x, 1e-4, 50.0) 
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe
        
        grads = tape.gradient(
            cdf_x, [a_params, b_params, weight_vals], 
            output_gradients=grad_common
        )
        
        def clean(g, limit=20.0):
            if g is None: return None
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), None, None

    z = lbound + (1.0 - lbound) * x
    return z, grad

@tf.function
def ks_mixture_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)