# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:04:06 2026
Enhanced Functions for Gaussian Copula and Kumaraswamy Mixture Marginals.
Includes Bisection Solver for Quantiles and Physics-Prioritized Gradients.
"""


# import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np

# tfd = tfp.distributions

# @tf.function(jit_compile=True)
# def build_correlation_matrices_from_cholesky(off_diag_elements, diag_elements, n_dims):
#     def build_single_correlation_matrix(batch_off_diag_elements, batch_diag_elements):
#         L = tf.zeros((n_dims, n_dims), dtype=tf.float32)
#         tril_indices = np.tril_indices(n_dims, k=-1)
#         indices = tf.stack([tf.cast(tril_indices[0], tf.int32), tf.cast(tril_indices[1], tf.int32)], axis=-1)
#         L = tf.tensor_scatter_nd_update(L, indices, batch_off_diag_elements)
        
#         diag_indices = tf.range(n_dims)
#         diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
#         L = tf.tensor_scatter_nd_update(L, diag_indices, batch_diag_elements)
        
#         row_norms = tf.norm(L, axis=1, keepdims=True)
#         L = L / (row_norms + 1e-7)
#         return L
    
#     LT_matrices = tf.vectorized_map(
#         lambda args: build_single_correlation_matrix(*args),
#         (off_diag_elements, diag_elements))
#     return LT_matrices

# def gaussian_copula_samples(LT_matrices, n_dims, n_samples, lbound):
#     batch_size = tf.shape(LT_matrices)[0]
#     # Use a more efficient vectorized sampling approach
#     eps = tf.random.normal(shape=(n_samples, batch_size, n_dims))
#     # mvn_samples: [n_samples, batch, dims]
#     mvn_samples = tf.einsum('sbd, bdi -> sbi', eps, LT_matrices)
#     u = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples)
#     return tf.clip_by_value(u, 1e-6, 1.0 - 1e-6)

# @tf.function(jit_compile=True)
# def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
#     u = tf.clip_by_value(copula_samples, 1e-7, 1.0 - 1e-7)
#     a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#     weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

#     low = tf.zeros_like(u)
#     high = tf.ones_like(u)

#     for _ in range(35):
#         mid = (low + high) / 2.0
#         mid_expand = tf.expand_dims(mid, axis=-1)
#         x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
#         one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0) 
#         cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)
#         cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
#         is_less = cdf_mid < u
#         low = tf.where(is_less, mid, low)
#         high = tf.where(is_less, high, mid)
#     return (low + high) / 2.0

# @tf.custom_gradient
# def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
#     x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)
    
#     def grad(dy):
#         signal_multiplier = 5.0
#         dy_safe = tf.clip_by_norm(dy * signal_multiplier, 1.0) 

#         with tf.GradientTape() as tape:
#             tape.watch([a_params, b_params, weight_vals])
#             a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#             b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
#             w_soft = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]
            
#             x_stop = tf.stop_gradient(x)
#             x_expand = tf.expand_dims(x_stop, axis=-1)
#             x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
#             one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0)
#             cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
#             cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)
            
#         x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), tf.maximum(a_t - 1.0, -0.9))
#         one_minus_x_a_pow = tf.math.pow(one_minus_x_a, tf.maximum(b_t - 1.0, -0.9))
#         pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
#         pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)
        
#         pdf_safe = tf.clip_by_value(pdf_x, 1e-3, 100.0) 
#         grad_common = -dy_safe * (1.0 - lbound) / pdf_safe
        
#         grads = tape.gradient(cdf_x, [a_params, b_params, weight_vals], output_gradients=grad_common)
        
#         def clean(g, limit=10.0):
#             if g is None: return None
#             g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
#             return tf.clip_by_value(g, -limit, limit)

#         return clean(grads[0]), clean(grads[1]), clean(grads[2]), None, None

#     z = lbound + (1.0 - lbound) * x
#     return z, grad

# @tf.function
# def ks_mixture_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
#     return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

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
# @tf.function(jit_compile=True)
# def gaussian_copula_samples(LT_matrices, n_dims, n_samples):
#     """
#     Generates samples from a Gaussian Copula using Cholesky factors.
#     Output Shape: [Batch, Samples, Dims]
#     """
#     batch_size = tf.shape(LT_matrices)[0]
#     # Sample noise: [Samples, Batch, Dims]
#     epsilon = tf.random.normal(shape=(n_samples, batch_size, n_dims))
    
#     # Matmul: [Batch, Dims, Dims] @ [Batch, Dims, Samples] -> Transpose back
#     mvn_samples = tf.einsum('sbd, bdi -> bsi', epsilon, LT_matrices)
    
#     # Transform to Uniform [0, 1] via Normal CDF
#     u = tfd.Normal(0.0, 1.0).cdf(mvn_samples)
#     return tf.clip_by_value(u, 1e-6, 1.0 - 1e-6)
@tf.function(jit_compile=True)
def _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Unconditionally stable Bisection solver for Kumaraswamy Mixtures.
    """
    u = tf.clip_by_value(copula_samples, 1e-7, 1.0 - 1e-7)
    
    # Align parameters to (Batch, 1, Dims, Components)
    a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
    # Raw logits used for bisection; softmax applied internally
    weights_t = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]

    low = tf.zeros_like(u)
    high = tf.ones_like(u)

    for _ in range(35):
        mid = (low + high) / 2.0
        mid_expand = tf.expand_dims(mid, axis=-1)
        
        # Kumaraswamy CDF: 1 - (1 - x^a)^b
        x_a = tf.math.pow(tf.maximum(mid_expand, 1e-10), a_t)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0) 
        cdf_components = 1.0 - tf.math.pow(one_minus_x_a, b_t)
        
        cdf_mid = tf.reduce_sum(weights_t * cdf_components, axis=-1)
        is_less = cdf_mid < u
        
        low = tf.where(is_less, mid, low)
        high = tf.where(is_less, high, mid)

    return (low + high) / 2.0

@tf.custom_gradient
def fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound):
    """
    Quantile function with Physics-Prioritized Gradient.
    Boosts the frequency signal while shielding against PDF-driven singularities.
    """
    x = _pure_kumaraswamy_sampling_jit(a_params, b_params, weight_vals, copula_samples, lbound)
    
    def grad(dy):
        # 1. Boost physics signal and clip norm
        # The multiplier helps dy compete with the regularizer in Beta_models.py
        signal_multiplier = 5.0
        dy_safe = tf.clip_by_norm(dy * signal_multiplier, 1.0) 

        with tf.GradientTape() as tape:
            tape.watch([a_params, b_params, weight_vals])
            
            a_t = tf.transpose(a_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            b_t = tf.transpose(b_params, perm=[0, 2, 1])[:, tf.newaxis, :, :]
            w_soft = tf.nn.softmax(tf.transpose(weight_vals, perm=[0, 2, 1]), axis=-1)[:, tf.newaxis, :, :]
            
            x_stop = tf.stop_gradient(x)
            x_expand = tf.expand_dims(x_stop, axis=-1)
            
            # Surrogate CDF for differentiation
            x_a = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t)
            one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-30, 1.0)
            cdf_comp = 1.0 - tf.math.pow(one_minus_x_a, b_t)
            cdf_x = tf.reduce_sum(w_soft * cdf_comp, axis=-1)
            
        # 2. PDF with adaptive safety floor
        x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), tf.maximum(a_t - 1.0, -0.9))
        one_minus_x_a_pow = tf.math.pow(one_minus_x_a, tf.maximum(b_t - 1.0, -0.9))
        pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)
        
        # x_pow = tf.math.pow(tf.maximum(x_expand, 1e-10), a_t - 1.0)
        # one_minus_x_a_pow = tf.math.pow(one_minus_x_a, b_t - 1.0)
        # pdf_comp = a_t * b_t * x_pow * one_minus_x_a_pow
        # pdf_x = tf.reduce_sum(w_soft * pdf_comp, axis=-1)
        
        # Prevent 1/PDF from becoming too large (vanishing frequency fit)
        # or too small (exploding gradients)
        pdf_safe = tf.clip_by_value(pdf_x, 5e-4, 50.0) 
        
        # Implicit Gradient: dx/dtheta = -(dCDF/dtheta) / (dCDF/dx)
        grad_common = -dy_safe * (1.0 - lbound) / pdf_safe
        
        # Get raw gradients for parameters
        grads = tape.gradient(
            cdf_x, [a_params, b_params, weight_vals], 
            output_gradients=grad_common
        )
        
        def clean(g, limit=20.0):
            if g is None: return None
            # Replace NaNs/Infs with 0
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            # Surgical clipping
            return tf.clip_by_value(g, -limit, limit)

        return clean(grads[0]), clean(grads[1]), clean(grads[2]), None, None

    z = lbound + (1.0 - lbound) * x
    return z, grad

@tf.function
def ks_mixture_marginal_samples(a_params, b_params, weight_vals, copula_samples, lbound):
    return fast_kumaraswamy_quantile(a_params, b_params, weight_vals, copula_samples, lbound)


