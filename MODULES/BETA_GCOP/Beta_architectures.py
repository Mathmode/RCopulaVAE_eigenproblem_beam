#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:56:02 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow.keras as K 
from MODULES.BETA_GCOP.Beta_functions import build_correlation_matrices_from_cholesky, gaussian_copula_samples
from MODULES.BETA_GCOP.Kumarswamy_functions import kumarswamy_marginal_samples

# Inverse architecture:
def Fully_connected_enc_Beta(input_dim, n_dims, num_beta_mix, lbound):
    """
    Updated Encoder for Beta/Kumaraswamy Mixture Marginals.
    Estimates a and b parameters.
    """
    input1 = K.Input(shape=(input_dim,), name='InputLayer')
    
    # LAYER 1: Robust 'relu' activation
    lay1 = K.layers.Dense(128, activation='relu', 
                          kernel_initializer="he_uniform", bias_initializer="zeros", 
                          name='lay1',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input1) 
    
    # LAYER 2: Robust 'relu' activation
    lay2 = K.layers.Dense(128, activation='relu', 
                          kernel_initializer="he_uniform", bias_initializer="zeros",
                          name='lay2',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(lay1)
    
    # LAYER 3: Robust 'relu' activation
    lay3 = K.layers.Dense(128, activation='relu', 
                          kernel_initializer="he_uniform", bias_initializer="zeros", 
                          name='lay3',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(lay2)

    # KUMARASWAMY/BETA PARAMETERS
    # FIX: Lower bound changed from 1.05 to 0.1. 
    # This mathematically allows the distribution to form J-shapes and place 
    # high probability density exactly at the undamaged state (z = 1.0).
    alphas = K.layers.Dense(n_dims * num_beta_mix, activation='softplus', name='alphas_raw')(lay3)
    # alphas = alphas+0.1
    alphas = tf.clip_by_value(alphas + 0.1, 0.1, 100.0) 

    betas = K.layers.Dense(n_dims * num_beta_mix, activation='softplus', name='betas_raw')(lay3)
    # betas = betas+0.1
    betas = tf.clip_by_value(betas + 0.1, 0.1, 100.0)

    # MIXTURE WEIGHTS
    weight_vals = K.layers.Dense(n_dims * num_beta_mix, activation='linear', name='weights')(lay3)

    # COPULA CORRELATION (L-Matrix)
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
        
        # Build samples assuming mixture of Betas marginals
        # marginal_samples = beta_marginal_samples(alphas, betas, weight_vals, copula_samples, self.lbound)
        
        # Build samples assuming mixture of Kumarswamy marginals
        marginal_samples = kumarswamy_marginal_samples(alphas, betas, weight_vals, copula_samples, self.lbound)
         
        
        
        
        return marginal_samples, copula_samples, LT_matrices
