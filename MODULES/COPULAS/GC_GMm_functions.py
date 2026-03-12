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


# @tf.function(jit_compile = True)
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



# @tf.function(jit_compile=True)  # Keep jit_compile=True for performance, but remove temporarily for debugging
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

    return copula_samples

@tf.function(jit_compile=True)
def gaussian_marginal_samples(locs, scales, copula_samples, lbound):
    """
    Builds marginal samples given copula samples and Gaussian marginal parameters.
    Returns marginal_samples with shape (batch_size, num_samples, n_dims).

    inputs: 
        Recall that We are here working with gaussian marginals (K = 1)
        locs with shape (batch_size, num_gaussians =1, n_dims)
        scales with shape (batch_size, num_gaussians =1, n_dims)
        copula_samples with shape (batch_size, num_samples, n_dims)
    """
    # safe_copula_samples = 1e-5 + (1.0 - 2e-5) * copula_samples
    # marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low= lbound+0.0001, high=1.000).quantile(safe_copula_samples)
    marginal_samples = tfd.TruncatedNormal(
    loc=locs, 
    scale=scales, 
    low=lbound,    # Use exactly lbound
    high=1.0).quantile(copula_samples)
    
    
    return marginal_samples



# @tf.function(jit_compile=True)
# def gaussian_copula_samples_optimized(LT_matrices, n_dims, n_samples, lbound):
#     """
#     Inverse Transform Sampling: Guarantees samples are in the [lbound, 1] 
#     range without needing a while_loop.
#     """
#     batch_size = tf.shape(LT_matrices)[0]
    
#     # Map physical lower bound to probability space [0, 1]
#     u_min = tfd.Normal(0.0, 1.0).cdf(tf.cast(lbound +0.0001, tf.float32))
#     u_max = 0.999 
    
#     # Generate random samples ONLY within the valid percentile window
#     u_samples = tf.random.uniform(
#         shape=[batch_size, n_samples, n_dims], 
#         minval=u_min, 
#         maxval=u_max, 
#         dtype=tf.float32
#     )
    
#     # Convert to standard Normal Z-space
#     z_samples = tfd.Normal(0.0, 1.0).quantile(u_samples)
    
#     # Apply correlation (Cholesky matrix)
#     copula_samples_z = tf.matmul(z_samples, LT_matrices, transpose_b=True)
    
#     # Transform back to Uniform space for the Copula representation
#     copula_samples_u = tfd.Normal(0.0, 1.0).cdf(copula_samples_z)
    
#     return tf.clip_by_value(copula_samples_u, 1e-7, 1.0 - 1e-7)

