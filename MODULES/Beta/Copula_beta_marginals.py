#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:48:11 2025

@author: afernandez
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class GaussianCopulaBetaLayer(tf.keras.layers.Layer):
    """
    Models the posterior using a Gaussian Copula with Beta Marginals.
    
    Inputs:
        - alphas: (Batch, D) - Shape parameters for Beta
        - betas:  (Batch, D) - Shape parameters for Beta
        - L_params: (Batch, D*(D-1)/2) - Flattened Cholesky factors for correlation
        
    Returns:
        - z_samples: (Batch, D) - Sampled latent variables in [0, 1]
        - log_prob: (Batch,) - Log probability density log q(z|x)
    """
    def __init__(self, n_dims, **kwargs):
        super(GaussianCopulaBetaLayer, self).__init__(**kwargs)
        self.n_dims = n_dims

    def call(self, inputs):
        alphas, betas, L_params = inputs
        
        # 1. Validation
        tf.debugging.assert_all_finite(alphas, "Alphas NaN")
        tf.debugging.assert_all_finite(betas, "Betas NaN")
        
        # 2. Build Correlation Matrix (Sigma)
        # We need a Correlation Matrix R (Diagonals are 1)
        # We use TFP to build a Cholesky factor L such that L @ L.T = R
        # fill_triangular creates the lower triangle. 
        # We generally use CorrelationCholesky bijector or similar logic.
        
        # Simpler approach: Use TFP's CholeskyLKJ or simpler transform
        # For simplicity here, we assume L_params maps to a Correlation Cholesky
        # We use a TFP Bijector chain to ensure valid correlation matrix
        fill_triangular = tfb.FillTriangular()
        L_raw = fill_triangular(L_params)
        
        # Normalize rows to ensure diagonal is 1 (Correlation matrix property)
        # Row norm: ||row||^2 = 1
        norm = tf.norm(L_raw, axis=-1, keepdims=True)
        L = L_raw / (norm + 1e-6)
        
        # 3. Define the Base Gaussian Distribution (Multivariate Normal)
        # Mean = 0, Covariance = L @ L.T (Correlation)
        base_mvn = tfd.MultivariateNormalTriL(
            loc=tf.zeros((self.n_dims,)),
            scale_tril=L
        )
        
        # --- SAMPLING STEP ---
        
        # A. Sample Correlated Gaussians: y ~ N(0, R)
        y = base_mvn.sample() # Shape: (Batch, D)
        
        # B. Transform to Uniform: u = Phi(y)
        # Use Normal CDF
        normal_dist = tfd.Normal(loc=0., scale=1.)
        u = normal_dist.cdf(y)
        
        # Numerical stability: Clip u to avoid 0 or 1 exactly
        u = tf.clip_by_value(u, 1e-5, 1.0 - 1e-5)
        
        # C. Transform to Beta: z = BetaInverse(u)
        # Quantile function of Beta
        beta_dist = tfd.Beta(concentration1=alphas, concentration0=betas)
        z = beta_dist.quantile(u)
        
        # --- LOG PROBABILITY STEP ---
        # Log p(z) = Log Copula Density(u) + Sum Log Marginal Density(z)
        
        # 1. Log Marginal Density
        # Sum log prob of Beta across dimensions
        log_prob_marginals = tf.reduce_sum(beta_dist.log_prob(z), axis=1)
        
        # 2. Log Copula Density
        # c(u) = |R|^{-1/2} * exp( -0.5 * y^T * (R^-1 - I) * y )
        # Ideally, computed as: log_prob_MVN(y) - sum(log_prob_Normal(y))
        
        log_prob_mvn = base_mvn.log_prob(y)
        log_prob_independent_normals = tf.reduce_sum(normal_dist.log_prob(y), axis=1)
        
        log_copula_density = log_prob_mvn - log_prob_independent_normals
        
        # Total Log Probability
        log_prob_total = log_copula_density + log_prob_marginals
        
        return z, log_prob_total

# --- HELPER TO BUILD ENCODER OUTPUT ---
# You need to adjust your Encoder to output L_params
def get_copula_n_params(n_dims):
    # Number of params for lower triangular matrix excluding diagonal (since diag is fixed by correlation constraint)
    # Actually FillTriangular expects D*(D+1)/2. 
    # But for correlation, we essentially need D*(D-1)/2 free parameters if we use strict correlation structure.
    # For simplicity using FillTriangular, we output D*(D+1)/2 and normalize.
    return n_dims * (n_dims + 1) // 2