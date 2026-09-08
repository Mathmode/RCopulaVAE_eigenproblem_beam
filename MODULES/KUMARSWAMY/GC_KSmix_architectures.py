# -*- coding: utf-8 -*-
"""
Encoder and Sampling Layers for Kumaraswamy Mixture Marginals.
"""

import tensorflow as tf
import tensorflow.keras as K
from MODULES.KUMARSWAMY.GC_KSmix_functions import build_correlation_matrices_from_cholesky, gaussian_copula_samples, ks_mixture_marginal_samples

def Fully_connected_enc_GC_KS(input_dim, n_dims, num_KS, lbound):
    input1 = K.Input(shape=(input_dim,), name='InputLayer')
    
    x = K.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation='relu', kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(x)

    # a, b parameters (B, D * K)
    a_params = K.layers.Dense(n_dims * num_KS, activation='softplus', name='a_ks')(x)
    b_params = K.layers.Dense(n_dims * num_KS, activation='softplus', name='b_ks')(x)
    
    a_params = tf.clip_by_value(a_params + 0.015, 0.01, 200.0) 
    b_params = tf.clip_by_value(b_params + 0.015, 0.01, 200.0) 

    # Mixture Weights
    weight_vals = K.layers.Dense(n_dims * num_KS, activation='linear', name='weights')(x)
    
    # Copula Correlation Parameters
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L = K.layers.Dense(n_correlations, activation='linear', name='off_diag')(x)
    diag_L = K.layers.Dense(n_dims, activation='softplus', name='diag')(x)
    diag_L = diag_L + 1e-6

    outputs = tf.concat([a_params, b_params, weight_vals, off_diag_L, diag_L], axis=1)
    return K.Model(inputs=input1, outputs=outputs)

class Copula_KS_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_KS, num_samples, lbound, **kwargs):
        super(Copula_KS_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_KS = num_KS
        self.num_samples = num_samples
        self.lbound = lbound

    def call(self, inputs):
        a_ks, b_ks, weights, offdiag, diag = inputs
        
        # Build Cholesky Matrices: [Batch, Dims, Dims]
        LT_matrices = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)
        
        # Gaussian Copula Samples: [Batch, Samples, Dims]
        u_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples, self.lbound)

        # Map to Mixture Space: [Batch, Samples, Dims]
        z_samples = ks_mixture_marginal_samples(a_ks, b_ks, weights, u_samples, self.lbound)
        
        return z_samples, u_samples, LT_matrices