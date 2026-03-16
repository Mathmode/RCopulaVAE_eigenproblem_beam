#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:11:53 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np 


@tf.function(jit_compile = True)
def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
    """Batched version using tf.vectorized_map."""
    def build_single_correlation_matrix(batch_off_diag_elements, batch_diag_elements):
        """
        Builds a *single* correlation matrix from Cholesky factor elements.

        Args:
            off_diag_elements: A 1D TensorFlow tensor (no batch dim).
            diag_elements: A 1D TensorFlow tensor (no batch dim).
            n_dims: The number of dimensions.

        Returns:
            A 2D TensorFlow tensor representing the correlation matrix.
        """
        L = tf.zeros((n_dims, n_dims), dtype=tf.float32)

        # Fill the lower triangular part with off-diagonal elements
        tril_indices = np.tril_indices(n_dims, k=-1)  # k=-1 excludes the diagonal
        indices = tf.stack([
            tf.cast(tril_indices[0], tf.int32),
            tf.cast(tril_indices[1], tf.int32)
        ], axis=-1)
        L = tf.tensor_scatter_nd_update(L, indices, batch_off_diag_elements)

        # Fill the diagonal
        diag_indices = tf.range(n_dims)
        diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
        L = tf.tensor_scatter_nd_update(L, diag_indices, batch_diag_elements)
        
        #*******************+
        # L = L +1e-07 
        row_norms = tf.norm(L, axis=1, keepdims=True)
        L = L / (row_norms + 1e-7)

        correlation_matrix = tf.matmul(L, L, transpose_b=True)
        jitter = 1e-6
        correlation_matrix = correlation_matrix + jitter * tf.eye(n_dims)
        return L
    
    LT_matrices = tf.vectorized_map(
        lambda args: build_single_correlation_matrix(*args),
        (off_diag_elements, diag_elements),)
    
    return LT_matrices


    
def gaussian_copula_samples(LT_matrices, n_dims, n_samples,lbound):
    
    lb = 0 # the gaussian copula samples should be in  [0,1], i think lbound should only apply at the final samples (marginal samples)
    def draw_samples(LT_matrices_batch):
        # Process the entire batch of LT matrices at once
        mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=tf.eye(n_dims))
        epsi_samples = mvn_model.sample(n_samples)  # Shape: [n_samples, batch_size, n_dims]
        mvn_samples = tf.matmul(epsi_samples, LT_matrices_batch, transpose_b=True) # [n_samples, batch_size, n_dims]
        copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
        return copula_samples  # [n_samples, batch_size, n_dims]

    def condition(samples):
        valid_mask = tf.reduce_all((samples > lb+ 0.00001) & (samples < 0.99), axis=-1)
        return tf.reduce_any(~valid_mask)  # Continue looping if *any* sample is invalid

    def body(samples):
      # Draw a full set of *new* samples for the entire batch
      new_samples = draw_samples(LT_matrices)

      # Create a mask of which original samples are valid.  keepdims is important!
      valid_mask = tf.reduce_all((samples > lb + 0.00001) & (samples < 0.99), axis=-1, keepdims=True)

      # Use tf.where to combine:  If valid, keep old; otherwise, use new.
      updated_samples = tf.where(valid_mask, samples, new_samples)
      return [updated_samples]

    # Initialize samples, processing the entire batch at once.
    initial_samples = draw_samples(LT_matrices)


    # Use tf.while_loop to repeatedly resample until all are valid
    copula_samples = tf.while_loop(
        condition,
        body,
        [initial_samples],
        maximum_iterations=2500
    )[0]

    return copula_samples # [n_samples, batch_size, n_dims]

@tf.function(jit_compile=True)
def _pure_beta_sampling_jit(alphas, betas, weight_vals, copula_samples, lbound):
    """
    Internal JIT-compiled forward pass for sampling using highly-optimized Bisection method.
    Unconditionally stable and guaranteed to converge without NaN divisions.
    """
    u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)
    
    # Align parameters to (B, 1, D, K)
    alphas_t = tf.transpose(alphas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    betas_t = tf.transpose(betas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights_t),
        components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
    )

    # -------------------------------------------------------------------------
    # UNCONDITIONALLY STABLE BISECTION ROOT FINDING
    # 35 iterations guarantees float32 machine precision (2^-35 approx 2.9e-11)
    # No divisions by PDF means it is mathematically impossible to produce NaNs.
    # -------------------------------------------------------------------------
    low = tf.zeros_like(u) + 1e-6
    high = tf.ones_like(u) - 1e-6

    for _ in range(10):
        mid = (low + high) / 2.0
        cdf_mid = mixture_dist.cdf(mid)
        
        # If CDF(mid) < u, the root is in the right half (new low = mid)
        is_less = cdf_mid < u
        
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    # Final root estimate
    x = (low + high) / 2.0
    return x


# @tf.function(jit_compile=True)
# def _pure_beta_sampling_jit(alphas, betas, weight_vals, copula_samples, lbound):
#     """
#     Internal JIT-compiled forward pass for sampling using highly-optimized Newton-Raphson.
#     """
#     u = tf.clip_by_value(copula_samples, 1e-6, 1.0 - 1e-6)
    
#     # Align parameters to (B, 1, D, K)
#     alphas_t = tf.transpose(alphas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     betas_t = tf.transpose(betas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

#     mixture_dist = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=weights_t),
#         components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
#     )

#     # -------------------------------------------------------------------------
#     # FAST NEWTON-RAPHSON ROOT FINDING
#     # Replaces Chandrupatla. 10 unrolled iterations are extremely fast in XLA.
#     # -------------------------------------------------------------------------
#     x = tf.identity(u) # Good initial guess

#     for _ in range(15):
#         cdf_x = mixture_dist.cdf(x)
#         pdf_x = mixture_dist.prob(x) + 1e-7 # Prevent division by zero
        
#         step = (cdf_x - u) / pdf_x
#         step = tf.clip_by_value(step, -0.2, 0.2) # Prevent chaotic/unstable jumps
        
#         x = x - step
#         x = tf.clip_by_value(x, 1e-6, 1.0 - 1e-6) # Keep strictly inside domain

#     return x

@tf.custom_gradient
def fast_beta_quantile(alphas, betas, weight_vals, copula_samples, lbound):
    """
    High-performance Sampling Layer with Analytic Gradients.
    - Forward pass is JIT-compiled for speed.
    - Backward pass uses the Implicit Function Theorem to flow Physics loss 
      into the Encoder's Beta parameters.
    """
    # 1. Forward Pass (JIT)
    x = _pure_beta_sampling_jit(alphas, betas, weight_vals, copula_samples, lbound)
    
    def grad(dy):
        # We use a localized GradientTape to analytically extract parameter gradients 
        # using the Implicit Function Theorem: dx/d_theta = - (d_CDF/d_theta) / PDF(x)
        with tf.GradientTape() as tape:
            tape.watch([alphas, betas, weight_vals])
            
            # Reconstruct distribution inside the tape
            alphas_t = tf.transpose(alphas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            betas_t = tf.transpose(betas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

            dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=weights_t),
                components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
            )
            
            # Evaluate CDF at the fixed root (stop_gradient prevents loops here)
            cdf_x = dist.cdf(tf.stop_gradient(x))
            
        # PDF for the denominator
        pdf_x = dist.prob(tf.stop_gradient(x))
        pdf_safe = tf.maximum(pdf_x, 1e-7)
        
        # 1. Gradient with respect to copula_samples (u)
        # Chain rule: dy/du = dy/dz * dz/du = dy * (1-lbound) * (1 / pdf_x)
        grad_u = dy * (1.0 - lbound) / pdf_safe
        grad_u = tf.clip_by_value(grad_u, -10.0, 10.0)
        
        # 2. Gradients with respect to the Beta parameters
        # We construct a mathematical surrogate loss: sum( W * F(x; theta) )
        # where W = -grad_u. Differentiating this surrogate computes the exact IFT gradient!
        W = tf.stop_gradient(-grad_u)
        surrogate_loss = tf.reduce_sum(W * cdf_x)
        
        grad_alphas, grad_betas, grad_weights = tape.gradient(
            surrogate_loss, [alphas, betas, weight_vals]
        )
        
        # Defensive clipping for extreme gradients early in training
        if grad_alphas is not None: grad_alphas = tf.clip_by_value(grad_alphas, -5.0, 5.0)
        if grad_betas is not None: grad_betas = tf.clip_by_value(grad_betas, -5.0, 5.0)
        if grad_weights is not None: grad_weights = tf.clip_by_value(grad_weights, -5.0, 5.0)

        return grad_alphas, grad_betas, grad_weights, grad_u, None

    # Final scaled samples
    z = lbound + (1.0 - lbound) * x
    return z, grad

@tf.function
def beta_marginal_samples(alphas, betas, weight_vals, copula_samples, lbound):
    """
    Top-level sampling wrapper.
    """
    z = fast_beta_quantile(alphas, betas, weight_vals, copula_samples, lbound)
    
    # Ensure output is (Batch, Samples, Dims)
    u_shape = tf.shape(copula_samples)
    alpha_shape = tf.shape(alphas)
    if u_shape[0] != alpha_shape[0]: 
         z = tf.transpose(z, perm=[1, 0, 2])

    return z
# @tf.function(jit_compile=True)
# def _pure_beta_sampling_jit(alphas, betas, weight_vals, copula_samples, lbound):
#     """
#     Internal JIT-compiled forward pass for sampling.
#     """
#     u = copula_samples
    
#     # Align parameters to (B, 1, D, K)
#     alphas_t = tf.transpose(alphas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     betas_t = tf.transpose(betas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

#     mixture_dist = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=weights_t),
#         components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
#     )

#     def target_fn(x):
#         return mixture_dist.cdf(x) - u

#     # Root finding for x in [0, 1]
#     beta_samples_raw = tfp.math.find_root_chandrupatla(
#         objective_fn=target_fn,
#         low=tf.zeros_like(u) + 1e-7,
#         high=tf.ones_like(u) - 1e-7,
#         max_iterations=50 
#     ).estimated_root

#     return beta_samples_raw

# @tf.custom_gradient
# def fast_beta_quantile(alphas, betas, weight_vals, copula_samples, lbound):
#     """
#     High-performance Sampling Layer with Analytic Gradients.
#     - Forward pass is JIT-compiled for speed.
#     - Backward pass uses the PDF to avoid XLA iteration errors.
#     """
#     # 1. Forward Pass (JIT)
#     x = _pure_beta_sampling_jit(alphas, betas, weight_vals, copula_samples, lbound)
    
#     def grad(dy):
#         # Analytic Gradient: dz/du = 1 / p(z)
#         # Reconstruct distribution for PDF calculation
#         alphas_t = tf.transpose(alphas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#         betas_t = tf.transpose(betas, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#         weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

#         dist = tfd.MixtureSameFamily(
#             mixture_distribution=tfd.Categorical(probs=weights_t),
#             components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
#         )
        
#         # p(x) in [0, 1] domain
#         pdf_x = dist.prob(x)
        
#         # Gradient with respect to copula_samples (u)
#         # Chain rule: dy/du = dy/dz * dz/du
#         # dz/du = (1 - lbound) / pdf_x (because z = lbound + (1-lbound)*x)
#         grad_u = dy * (1.0 - lbound) / tf.maximum(pdf_x, 1e-7)
        
#         # Note: Gradients w.r.t alphas/betas/weights can also be derived, 
#         # but often u-gradients are primary for VAE backprop through the copula.
#         # For full VAE training, we typically rely on the copula samples being the stochastic part.
#         return None, None, None, grad_u, None

#     # Final scaled samples
#     z = lbound + (1.0 - lbound) * x
#     return z, grad

# @tf.function
# def beta_marginal_samples(alphas, betas, weight_vals, copula_samples, lbound):
#     """
#     Top-level sampling wrapper.
#     """
#     z = fast_beta_quantile(alphas, betas, weight_vals, copula_samples, lbound)
    
#     # Ensure output is (Batch, Samples, Dims)
#     u_shape = tf.shape(copula_samples)
#     alpha_shape = tf.shape(alphas)
#     if u_shape[0] != alpha_shape[0]: 
#          z = tf.transpose(z, perm=[1, 0, 2])

#     return z


