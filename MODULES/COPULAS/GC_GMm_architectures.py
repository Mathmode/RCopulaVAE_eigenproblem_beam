#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:56:02 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow.keras as K 
from MODULES.COPULAS.GC_GMm_functions import gaussian_copula_samples, build_marginal_samples, gaussian_marginal_samples, build_correlation_matrices_from_cholesky
import numpy as np 
from MODULES.TRAINING.rotation_matrices_funtions import copula_batch_givens_rotation


# Inverse architecture:
def Fully_connected_enc_GC(input_dim, n_dims, num_gaussians):
    input1 = K.Input(shape =(input_dim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(100, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(input1) #Intermediate layers
    lay2 = K.layers.Dense(150, activation = 'tanh')(lay1) #Intermediate layers
    lay2 = K.layers.Dense(100, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    lay2 = K.layers.Dense(100, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    lay3 = K.layers.Dense(50, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay4')(lay2) #Intermediate layers

    # MEANS
    means = K.layers.Dense(n_dims*num_gaussians, activation = 'sigmoid',  name = 'means')(lay3)
    means = 1e-07 + 0.99 * means  # Example: Scale/shift if needed.  Good practice.

    #SIGMAS
    sigmas = K.layers.Dense(n_dims*num_gaussians, activation = 'sigmoid', name = 'stddevs')(lay3)
    sigmas = 1e-07 + 0.99 * sigmas # Scale and shift: sigmas will be in [0.01, 1.00].  ESSENTIAL for stability.
    
    ## Lmatrix Elements to build directly the lower triangular matrix rather than the correlation
    n_correlations  = n_dims*(n_dims-1)//2
    off_diag_L_elems = K.layers.Dense(n_correlations, activation='linear', name='off_diag_elements')(lay3)
    
    # off_diag_L_elems = off_diag_L_elems +1e-07 
    diag_L_elems = K.layers.Dense(n_dims, activation = 'softplus', name = 'diag_elements')(lay3)
    diag_L_elems = diag_L_elems+1e-07

    #WEIGHTS: first sofplus because the condition of Sumup to 1 must be stasify for each dimension, not for all the weights together.
    weight_vals = K.layers.Dense(n_dims*num_gaussians, activation = 'softplus', name = 'weights')(lay3) # Enforce the weights to be positive only.
    outputs = tf.concat([means, sigmas, weight_vals, off_diag_L_elems, diag_L_elems], axis = 1)
    return K.Model(inputs = input1, outputs = outputs)




class Copula_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_gaussians, num_samples, **kwargs):
        super(Copula_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples

    def call(self, inputs):
        means, scales, weight_vals, offdiag_elems, diag_elems  = inputs
        # Input validation: Check for NaNs
        tf.debugging.assert_all_finite(means, "means contains NaN or Inf")  # Added check
        tf.debugging.assert_all_finite(scales, "scales contains NaN or Inf")  # Added check
        tf.debugging.assert_all_finite(weight_vals, "weight_vals contains NaN or Inf")
        tf.debugging.assert_all_finite(offdiag_elems, "offdiag contains NaN or Inf")

        LT_matrices  = build_correlation_matrices_from_cholesky(offdiag_elems, diag_elems, self.n_dims)
        copula_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples)
        
        # marginal_sampless = build_marginal_samples(means, scales, weight_vals, copula_samples)
        # marginal_samples = tf.reshape(marginal_sampless, shape = (-1, self.num_gaussians, self.n_dims))
        # tf.print(marginal_samples)
        marginal_samples = gaussian_marginal_samples(means, scales, copula_samples)
        
        return marginal_samples, copula_samples, LT_matrices





def Fully_connected_dec(input_dim, output_dim):
     input1 = tf.keras.Input(shape =(input_dim,), name = 'InputDecoder')
     lay1 = K.layers.Dense(10, activation = 'relu', name = 'Dl1')(input1) #Intermediate layers
     lay2 = K.layers.Dense(30, activation = 'relu', name = 'Dl2')(lay1) #Intermediate layers
     lay3 = K.layers.Dense(50, activation = 'relu', name = 'Dl3')(lay2) #Intermediate layers
     lay4 = K.layers.Dense(70, activation = 'relu', name = 'Dl4')(lay3) #Intermediate layers
     lay5 = K.layers.Dense(80, activation = 'relu', name = 'Dl5')(lay4) #Intermediate layers
     output1 = K.layers.Dense(output_dim, activation = 'linear' , name = 'outputDec')(lay5)
     return K.Model(inputs = input1, outputs = output1)

































# ## Transform copula pdf into a loop to ensure samples from the desired interval. 
# # I presume that if I want to ensure that my final samples lie in the interval [0,1], the marginal PDFs must be truncated (truncated gaussian mixtures)
# # the copula samples will already satisfy the condition thanks to the CDF
# # then we need to analyze the "build marginal samples" to enforce them to be in the interval despite the interpolation points being in [-10,10]
 
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# @tf.function(jit_compile=True)
# def rev_gaussian_copula_samples(correlation_matrices, LT_matrices, n_dims, n_samples):
#     """
#     Generates samples from a batch of Gaussian copulas.
#     """
#     def process_batch(corr_matrix, LT_matrix):
#         mvn_model = tfd.MultivariateNormalTriL(loc=tf.zeros(n_dims), scale_tril=LT_matrix)
#         mvn_samples = mvn_model.sample(n_samples)

#         copula_samples = tfd.Normal(loc=0.0, scale=1.0).cdf(mvn_samples) # applying the cdf produces samples in the desired interval [0,1]
#         return copula_samples, mvn_samples, mvn_model

#     batch_results = tf.vectorized_map(
#         lambda args: process_batch(*args),
#         (correlation_matrices, LT_matrices))
#     copula_samples, mvn_samples, mvn_models = batch_results
#     return copula_samples, mvn_samples, mvn_models

# class rev_Copula_pdf_layer(tf.keras.layers.Layer):
#     def __init__(self, n_dims, num_gaussians, num_samples, **kwargs):
#         super(Copula_pdf_layer, self).__init__(**kwargs)
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples

#     def call(self, inputs):
#         means, scales, weight_vals, rho_factors = inputs
#         rho_factors = tf.cast(rho_factors, dtype = tf.float32)
#         correlation_matrices = build_correlation_matrices(rho_factors, self.n_dims)
#         LT_matrices = tf.linalg.cholesky(correlation_matrices)

#         copula_samples, mvn_samples, mvn_models = rev_gaussian_copula_samples(correlation_matrices, LT_matrices, self.n_dims, self.num_samples)
#         marginal_samples = build_marginal_samples(means, scales, weight_vals, copula_samples)
        
        
#         return marginal_samples, copula_samples, mvn_samples, mvn_models, correlation_matrices, LT_matrices








    
# from MODULES.COPULAS.GC_GMm_functions import Interpolation1D    

# @tf.function(jit_compile=True)
# def rev_multimodal_marginal(x, locs, scales, weight_vals):
#     """
#     Defines a multimodal (mixture of Gaussians) marginal distribution.
#     """
#     gm = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=weight_vals),
#         # components_distribution=tfd.Normal(loc=locs, scale=tf.maximum(scales, 1e-06))
#         components_distribution=tfd.TruncatedNormal(loc=locs, scale=tf.maximum(scales, 1e-06), low =0.0, high=1.0)

#     )
#     mixture_probs = gm.prob(x)
#     return mixture_probs

# def rev_build_marginal_samples(locs, scales, weight_vals, copula_samples):
#     """
#     Builds marginal samples given copula samples and marginal parameters.
#     """
#     interpolator = Interpolation1D()
#     def process_batch(locs_batch, scales_batch, weights_batch, copula_samples_batch):
#         n_dims = locs_batch.shape[-1]
#         n_unif_points = 1000
#         x = tf.linspace(-10.0, 10.0, n_unif_points)
#         # x = tf.linspace(1e-05,1.0, n_unif_points)  # CRITICAL: Avoid 0 and 1
#         marg_samples = tf.TensorArray(dtype=tf.float32, size=n_dims)

#         for i in range(n_dims):
#             marginal_pdf_vals = rev_multimodal_marginal(
#                 x, locs_batch[:, i], scales_batch[:, i], weights_batch[:, i]
#             )
#             marginal_cdf_vals = tf.cumsum(marginal_pdf_vals, axis=0) / tf.reduce_sum(marginal_pdf_vals)            
#             marginal = interpolator(copula_samples_batch[:, i], marginal_cdf_vals, x)
#             marg_samples = marg_samples.write(i, marginal)

#         marginal_samples = marg_samples.stack()
#         marginal_samples = tf.transpose(marginal_samples, perm=[1, 0])
        
        
        
        
#         return marginal_samples

#     marginal_samples = tf.vectorized_map(
#         lambda args: process_batch(*args),
#         (locs, scales, weight_vals, copula_samples),)
#     return marginal_samples
# class rev_sampling_layer(tf.keras.layers.Layer):
#     def __init__(self, n_dims, num_gaussians, num_samples, **kwargs):
#         super(rev_sampling_layer, self).__init__(**kwargs)
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples

#     def call(self, inputs):
#         means, scales, weight_vals, rho_factors = inputs
#         rho_factors = tf.cast(rho_factors, dtype = tf.float32)
#         correlation_matrices = build_correlation_matrices(rho_factors, self.n_dims)
#         LT_matrices = tf.linalg.cholesky(correlation_matrices)

#         copula_samples, mvn_samples, mvn_models = rev_gaussian_copula_samples(correlation_matrices, LT_matrices, self.n_dims, self.num_samples)
#         marginal_samples = build_marginal_samples(means, scales, weight_vals, copula_samples)
        
#         def condition(samples):
#             # Check if all samples in each batch are within the [0,1] interval
#             valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)
#             return tf.reduce_any(~valid_mask)

#         def body(samples):
#             # Check validity of the current samples
#             valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)

#             # Identify invalid samples
#             invalid_batches = tf.where(~valid_mask)

#             # Resample only the invalid samples
#             resampled_samples = draw_samples()
#             new_invalid_samples = tf.gather_nd(resampled_samples, invalid_batches)

#             # Update the invalid samples in the original tensor
#             samples = tf.tensor_scatter_nd_update(samples, invalid_batches, new_invalid_samples)

#             return samples
        
        
#         # Use tf.while_loop to repeatedly resample invalid samples until all are valid
#         marginal_samples = tf.while_loop(
#             condition,
#             body,
#             [marginal_samples],
#             maximum_iterations= 500  # Ensure this doesn't run forever
#         )[0]
         
        
#         return marginal_samples
        



#         copula_samples, mvn_samples, mvn_models = gaussian_copula_samples(correlation_matrices, LT_matrices, self.n_dims, self.num_samples)
#         marginal_samples = build_marginal_samples(means, scales, weight_vals, copula_samples)
#         return marginal_samples, copula_samples, mvn_samples, mvn_models, correlation_matrices, LT_matrices


# @tf.function(jit_compile=True)
# def sample_with_condition(vect_mixture, num_samples):
#     # Function to sample one batch of points
#     marginal_samples, copula_samples, mvn_samples, mvn_models, correlation_matrices, LT_matrices = rev_Copula_pdf_layer(self.n_dims, self.num_gaussians, self.num_samples)
#     def draw_samples():
#         return vect_mixture.sample(num_samples)
    
#     # Initialize samples
#     samples = draw_samples()
    
#     def condition(samples):
#         # Check if all samples in each batch are within the [0,1] interval
#         valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)
#         return tf.reduce_any(~valid_mask)

#     def body(samples):
#         # Check validity of the current samples
#         valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)

#         # Identify invalid samples
#         invalid_batches = tf.where(~valid_mask)

#         # Resample only the invalid samples
#         resampled_samples = draw_samples()
#         new_invalid_samples = tf.gather_nd(resampled_samples, invalid_batches)

#         # Update the invalid samples in the original tensor
#         samples = tf.tensor_scatter_nd_update(samples, invalid_batches, new_invalid_samples)

#         return samples

#     # Use tf.while_loop to repeatedly resample invalid samples until all are valid
#     samples = tf.while_loop(
#         condition,
#         body,
#         [samples],
#         maximum_iterations= 500  # Ensure this doesn't run forever
#     )[0]
     
    
#     return samples





#####################OLD Versions########################################################
# class Copula_pdf_layer(tf.keras.layers.Layer):
#     def __init__(self, n_dims, num_gaussians, num_samples, **kwargs):
#         super(Copula_pdf_layer, self).__init__(**kwargs)
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples

#     def call(self, inputs):
#         means, scales, weight_vals, rho_factors, phi_angles = inputs   
#         correlation_matrices = build_correlation_matrices(rho_factors, self.n_dims)[:,0,:]
#         # 
#         # basic_rot_matrix = copula_batch_givens_rotation(phi_angles, self.n_dims)
#         # rot_matrices = tf.reduce_prod(basic_rot_matrix, axis=1)  #multiplying all the individual rotation matrices
#         # RS_matrices = rot_matrices
#         # RS_transp_matrices = tf.einsum('BDK->BKD', RS_matrices) #Obtain the transpose of the transformation matrix T = RS
#         # Sigma_matrices = tf.einsum('BDK, BKL -> BDL', RS_matrices, RS_transp_matrices) # Build the Covariance matrix as C = T T^{t}
#         # Sigma_matrices += 1e-06 * tf.eye(tf.shape(Sigma_matrices)[-1], dtype=Sigma_matrices.dtype)
# # Generate copula samples and marginal samples
#         # LT_matrices = tf.linalg.cholesky(Sigma_matrices) #Apply Cholesk
#         LT_matrices = tf.linalg.cholesky(correlation_matrices)
#         eps = 1e-08
#         ### WE MUST PREVENT SAMPLES FROM THE BOUNDARY THEY ARE PRODUCING NAN VALUES. BE CAREFFUL WITH THIS
#         ## however this is not the correct way to do it. You must do the iterative process. 
#         copula_samples, mvn_samples, mvn_models = gaussian_copula_samples(correlation_matrices, LT_matrices, self.n_dims, self.num_samples)
#         marginal_samples = build_marginal_samples(means, scales, weight_vals, copula_samples)
#         return marginal_samples, copula_samples, mvn_samples, mvn_models, correlation_matrices, LT_matrices



