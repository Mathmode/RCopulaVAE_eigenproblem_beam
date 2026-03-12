#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:56:02 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow.keras as K 
from MODULES.BETA_GCOP.Beta_functions import build_correlation_matrices_from_cholesky, gaussian_copula_samples, beta_marginal_samples


# Inverse architecture:
def Fully_connected_enc_Beta(input_dim, n_dims, num_beta_mix, lbound):
    """
    Updated Encoder for Beta Mixture Marginals.
    Estimates Alpha and Beta parameters instead of Mean and Sigma.
    """
    input1 = K.Input(shape=(input_dim,), name='InputLayer')
    
    # Shared Dense Layers
    lay1 = K.layers.Dense(1024, activation='relu', kernel_initializer="he_uniform", name='lay1')(input1) 
    lay2 = K.layers.Dense(1024, activation='relu', name='lay2')(lay1)
    lay3 = K.layers.Dense(1024, activation='relu', name='lay3')(lay2)

    # BETA PARAMETERS (Alpha and Beta must be > 0)
    # We use softplus + 1.0 to ensure stability and avoid extremely sharp peaks at 0
    alphas = K.layers.Dense(n_dims * num_beta_mix, activation='softplus', name='alphas_raw')(lay3)
    alphas = alphas + 1.001 

    betas = K.layers.Dense(n_dims * num_beta_mix, activation='softplus', name='betas_raw')(lay3)
    betas = betas + 1.001

    # MIXTURE WEIGHTS
    # Softplus to be normalized later via softmax in the Model class
    weight_vals = K.layers.Dense(n_dims * num_beta_mix, activation='softplus', name='weights')(lay3)

    # COPULA CORRELATION (L-Matrix) - Remains the same
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L_elems = K.layers.Dense(n_correlations, activation='linear', name='off_diag_elements')(lay3)
    diag_L_elems = K.layers.Dense(n_dims, activation='softplus', name='diag_elements')(lay3)
    diag_L_elems = diag_L_elems + 1e-6

    outputs = tf.concat([alphas, betas, weight_vals, off_diag_L_elems, diag_L_elems], axis=1)
    return K.Model(inputs=input1, outputs=outputs)




class Copula_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_beta_mix, num_samples, lbound ,**kwargs):
        super(Copula_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_beta_mix = num_beta_mix
        self.num_samples = num_samples
        self.lbound = lbound #lower bound for truncation 
    def call(self, inputs):
        alphas, betas, weight_vals, offdiag_elems, diag_elems = inputs
        LT_matrices  = build_correlation_matrices_from_cholesky(offdiag_elems, diag_elems, self.n_dims)
        copula_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples, self.lbound)
        
       # Build samples assuming single Gaussian marginals
        marginal_samples = beta_marginal_samples(alphas, betas, weight_vals, copula_samples, self.lbound)
        
        return marginal_samples, copula_samples, LT_matrices
