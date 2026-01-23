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

        correlation_matrix = tf.matmul(L, L, transpose_b=True)
        jitter = 1e-6
        correlation_matrix = correlation_matrix + jitter * tf.eye(n_dims)
        return L
    
    LT_matrices = tf.vectorized_map(
        lambda args: build_single_correlation_matrix(*args),
        (off_diag_elements, diag_elements),)
    
    return LT_matrices


    

# @tf.function(jit_compile=True)  # Keep jit_compile=True for performance, but remove temporarily for debugging
def gaussian_copula_samples(LT_matrices, n_dims, n_samples):
    def draw_samples(LT_matrices_batch):
        # Process the entire batch of LT matrices at once
        mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=tf.eye(n_dims))
        epsi_samples = mvn_model.sample(n_samples)  # Shape: [n_samples, batch_size, n_dims]
        mvn_samples = tf.matmul(epsi_samples, LT_matrices_batch, transpose_b=True) # [n_samples, batch_size, n_dims]
        copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
        return copula_samples  # [n_samples, batch_size, n_dims]

    def condition(samples):
        valid_mask = tf.reduce_all((samples > 0.001) & (samples < 0.99), axis=-1)
        return tf.reduce_any(~valid_mask)  # Continue looping if *any* sample is invalid

    def body(samples):
      # Draw a full set of *new* samples for the entire batch
      new_samples = draw_samples(LT_matrices)

      # Create a mask of which original samples are valid.  keepdims is important!
      valid_mask = tf.reduce_all((samples > 0.001) & (samples < 0.99), axis=-1, keepdims=True)

      # Use tf.where to combine:  If valid, keep old; otherwise, use new.
      updated_samples = tf.where(valid_mask, samples, new_samples)
      return updated_samples

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
def gaussian_marginal_samples(locs, scales, copula_samples):
    """
    Builds marginal samples given copula samples and Gaussian marginal parameters.
    Returns marginal_samples with shape (batch_size, num_samples, n_dims).

    inputs: 
        Recall that We are here working with gaussian marginals (K = 1)
        locs with shape (batch_size, num_gaussians =1, n_dims)
        scales with shape (batch_size, num_gaussians =1, n_dims)
        copula_samples with shape (batch_size, num_samples, n_dims)
    """
    marginal_samples = tfd.TruncatedNormal(loc=locs, scale=scales, low=-0.001, high=1.001).quantile(copula_samples)
    return marginal_samples

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#$$$$$$$$$$$$$$$$$$$$$$$$for multimodal marginals. not used $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4

@tf.function(jit_compile=True)
def multimodal_marginal(x, locs, scales, weight_vals):
    """
    Defines a multimodal (mixture of Gaussians) marginal distribution.

    Args:
        x: Input values at which to evaluate the PDF, shape [n_points, 1].
        locs: Means of the Gaussian components, shape [n_gaussians].
        scales: Standard deviations of the Gaussian components, shape [n_gaussians].
        weight_vals: Mixture weights, shape [n_gaussians].  MUST sum to 1.

    Returns:
        The PDF values at the given input points x, shape [n_points].

    Connection to Overleaf function: This implements the Gaussian Mixture Model (GMM) equation
    for the marginal densities q_i(z_i | m, w).
    """
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weight_vals),
        components_distribution=tfd.TruncatedNormal(loc=locs, scale=scales, low =0.0, high=1.0)

    )
    mixture_probs = gm.prob(x)
    return mixture_probs


class Interpolation1D(tf.keras.layers.Layer):
    def __init__(self):
        super(Interpolation1D, self).__init__()

    def call(self, x, xp, fp):
        """
        Perform linear interpolation in 1D with proper shape handling and improved stability.

        Parameters:
        - x: Query points, shape [batch_size, n_points] or [n_points].
        - xp: Known domain points, shape [n_values].  Must be sorted.
        - fp: Function values at xp, shape [n_values].

        Returns:
        - Interpolated values at x, shape [batch_size, n_points] or [n_points].
        """

        # 1. Input Validation and Preprocessing
        xp = tf.sort(xp)  # Ensure xp is sorted
        tf.debugging.assert_rank(xp, 1, "xp must be a 1D tensor")  # Check rank
        tf.debugging.assert_rank(fp, 1, "fp must be a 1D tensor")
        tf.debugging.assert_equal(tf.shape(xp), tf.shape(fp), "xp and fp must have the same shape")

        # Clip x to the valid range of xp, but allow for slight extrapolation
        # by a small margin (e.g., 1% of the range)
        x_min = tf.reduce_min(xp)
        x_max = tf.reduce_max(xp)
        x_range = x_max - x_min
        margin = 0.01 * x_range  # 1% margin
        x = tf.clip_by_value(x, x_min - margin, x_max + margin)


        # Handle both 1D and 2D x cases using tf.cond
        is_rank_1 = tf.equal(tf.rank(x), 1)

        def rank_1_case():
            return tf.expand_dims(x, axis=0)  # Add batch dim if x is 1D

        def rank_2_case():
            return x  # x is already 2D or higher

        x_processed = tf.cond(is_rank_1, rank_1_case, rank_2_case)


        # 2. Find Indices (Robust to Edge Cases)

        # Compute indices for lower bound.  Use >= for correct behavior.
        indices = tf.reduce_sum(tf.cast(x_processed[:, :, None] >= xp[None, None, :], tf.int32), axis=-1) - 1

        # Clip indices to be within the valid range [0, n_values - 2].
        # This is crucial for preventing out-of-bounds errors in tf.gather.
        n_values = tf.shape(xp)[0]
        indices = tf.clip_by_value(indices, 0, n_values - 2)  # Corrected clipping


        # 3. Gather Values (Safe with Clipped Indices)
        x0 = tf.gather(xp, indices)
        x1 = tf.gather(xp, indices + 1)
        f0 = tf.gather(fp, indices)
        f1 = tf.gather(fp, indices + 1)


        # 4. Linear Interpolation (Handle Edge Cases)

        # Use tf.where to handle cases where x1 and x0 are very close (avoid division by zero).
        # If x1 == x0, the slope is effectively 0, and we just return f0.
        slope = tf.where(tf.abs(x1 - x0) > 1e-8, (f1 - f0) / (x1 - x0), 0.0)
        interpolated = f0 + slope * (x_processed - x0)

        # 6. Return Result (Correct Shape)
        return tf.squeeze(interpolated)  # Remove batch dim if input was 1D
    



    

# @tf.function(jit_compile=True)
def build_marginal_samples(locs, scales, weight_vals, copula_samples):
    """
    Builds marginal samples given copula samples and marginal parameters.

    Args:
        locs:  Shape [batch_size, n_gaussians, n_dims].
        scales: Shape [batch_size, n_gaussians, n_dims].
        weight_vals: Shape [batch_size, n_gaussians, n_dims].  
        copula_samples: Shape [batch_size, n_samples, n_dims].

    Returns:
        Marginal samples, shape [batch_size, n_samples, n_dims].

    Connection to Equations: This function performs the transformation from uniform
    copula samples (u_i) to the actual marginal samples (z_i) by using the
    inverse CDF (F_i^{-1}(u_i)).  The inverse CDF is approximated using
    the Interpolation1D layer.
    """
    interpolator = Interpolation1D()
    def process_batch(locs_batch, scales_batch, weights_batch, copula_samples_batch):
        n_dims = locs_batch.shape[1]
        n_unif_points = 200
        x = tf.linspace(0.1, 0.9, n_unif_points)   # Shape: [n_points]
        
        marg_samples = tf.TensorArray(dtype=tf.float32, size=n_dims)
    
        for i in range(n_dims):  # Use tf.range for better XLA support
            marginal_pdf_vals = multimodal_marginal(
                x, locs_batch[:, i], scales_batch[:, i], weights_batch[:, i]
            )
            marginal_cdf_vals = tf.cumsum(marginal_pdf_vals, axis=0) / tf.reduce_sum(marginal_pdf_vals)
    
            # Use custom Interpolation1D layer for inverse CDF
            marginal = interpolator(copula_samples_batch[:, i], marginal_cdf_vals, x)
            marg_samples = marg_samples.write(i, marginal)  # Replace append with write()
    
        marginal_samples = marg_samples.stack()  # Convert TensorArray to Tensor
        return marginal_samples

    marginal_samples = tf.vectorized_map(
        lambda args: process_batch(*args),
        (locs, scales, weight_vals, copula_samples),)

    return marginal_samples







# @tf.function(jit_compile = True)
# def generate_givens_rotation(n_dims, i, j, theta):
#     """
#     Create a Givens rotation matrix for dimensions `i` and `j` in `num_dimensions`.
#     """
#     rotation_matrix = tf.eye(n_dims, dtype=tf.float32)
#     cos_theta = tf.cos(theta)
#     sin_theta = tf.sin(theta)

#     # Update the matrix elements for the Givens rotation
#     rotation_matrix = tf.tensor_scatter_nd_update(
#         rotation_matrix, [[i, i], [i, j], [j, i], [j, j]],
#         [cos_theta, -sin_theta, sin_theta, cos_theta]
#     )
#     return rotation_matrix


# @tf.function(jit_compile = True)
# def copula_batch_givens_rotation(angles, n_dims):
#     """
#     Generate Givens rotation matrices for each batch, Gaussian, and angle.
#     Args: 
#         angles: Tensor of shape (batch_size, num_gaussians, n_angles), where n_angles = n_dims * (n_dims - 1) // 2.
#         n_dims: The dimensionality of the space.
#     Returns a tensor of shape (batch_size, num_gaussians, n_angles, n_dims, n_dims).
#     """
#     indices = [
#         (i, j) for i in range(n_dims) for j in range(i + 1, n_dims)
#     ]
    
#     # Build rotation matrices for each batch and Gaussian
#     def process_for_batch(gaussian_angles):
#         """
#         Process the angles for a single Gaussian (n_angles conform the Gaussian).
#         """
#         rotation_matrices = []
#         for k, (i, j) in enumerate(indices):
#             theta = gaussian_angles[k]
#             rotation_matrix = generate_givens_rotation(n_dims, i, j, theta)
#             rotation_matrices.append(rotation_matrix)
#         return tf.stack(rotation_matrices)

#     return tf.vectorized_map(process_for_batch, angles)



