# -*- coding: utf-8 -*-
"""
Full Covariance GMM Architecture
Uses Cholesky factors (L) to build full covariance matrices and 
safe gradient firewalls mimicking the successful Copula structure.
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np

@tf.custom_gradient
def safe_physics_mapping(z_raw, lbound):
    """
    Gradient firewall: Maps to physical space with Sigmoid, but
    intercepts, cleans, and smoothly clips gradients to prevent NaNs from the Eigensolver.
    """
    safe_margin = 1e-4
    sig = tf.math.sigmoid(z_raw)
    sig_safe = tf.clip_by_value(sig, safe_margin, 1.0 - safe_margin)
    z_phys = lbound + (1.0 - lbound) * sig_safe
    
    def grad(dy):
        dy_clean = tf.where(tf.math.is_finite(dy), dy, tf.zeros_like(dy))
        dy_safe = tf.clip_by_value(dy_clean, -50.0, 50.0)
        dz_raw = dy_safe * (1.0 - lbound) * sig_safe * (1.0 - sig_safe)
        dz_raw_safe = tf.where(tf.math.is_finite(dz_raw), dz_raw, tf.zeros_like(dz_raw))
        return dz_raw_safe, None
        
    return z_phys, grad

@tf.function(jit_compile=True)
def build_scale_tril_matrices(off_diag_elements, diag_elements, n_dims):
    """Batched version to construct Lower Triangular (Cholesky) matrices."""
    def build_single(batch_off, batch_diag):
        L = tf.zeros((n_dims, n_dims), dtype=tf.float32)
        
        # Fill off-diagonal elements
        tril_indices = np.tril_indices(n_dims, k=-1)
        indices = tf.stack([
            tf.cast(tril_indices[0], tf.int32),
            tf.cast(tril_indices[1], tf.int32)
        ], axis=-1)
        L = tf.tensor_scatter_nd_update(L, indices, batch_off)

        # Fill diagonal elements
        diag_indices = tf.range(n_dims)
        indices_d = tf.stack([diag_indices, diag_indices], axis=1)
        L = tf.tensor_scatter_nd_update(L, indices_d, batch_diag)
        return L

    # Map across the flattened batch of components
    LT_matrices = tf.vectorized_map(
        lambda args: build_single(*args),
        (off_diag_elements, diag_elements)
    )
    return LT_matrices

def Fully_connected_enc_FullCov_GMM(input_dim, n_dims, num_components):
    """Maps modal data to GMM parameters (Logits, Locs, Off-Diag, Diag)."""
    input1 = K.Input(shape=(input_dim,), name='InputLayer')
    
    x = K.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation='relu', kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")(x)

    logits = K.layers.Dense(num_components, activation='linear', name='gmm_logits')(x)
    
    raw_locs = K.layers.Dense(n_dims * num_components, activation='linear', name='gmm_locs')(x)
    locs = 4.0 * tf.math.tanh(raw_locs) # Stable location mapping
    
    n_corr = n_dims * (n_dims - 1) // 2
    raw_offdiag = K.layers.Dense(n_corr * num_components, activation='linear', name='gmm_offdiag')(x)
    # Clip off-diagonals to prevent covariance explosions
    offdiag = tf.clip_by_value(raw_offdiag, -3.0, 3.0)
    
    raw_diag = K.layers.Dense(n_dims * num_components, activation='linear', name='gmm_raw_diag')(x)
    # Clip raw diagonals before softplus to prevent infinite variance
    raw_diag = tf.clip_by_value(raw_diag, -5.0, 5.0)

    outputs = tf.concat([logits, locs, offdiag, raw_diag], axis=1)
    return K.Model(inputs=input1, outputs=outputs)

class FullCov_GMM_Sampling_Layer(tf.keras.layers.Layer):
    """Differentiable sampling layer for Full Covariance Gaussian Mixtures."""
    def __init__(self, n_dims, num_components, num_samples, lbound, tau=0.1, **kwargs):
        super(FullCov_GMM_Sampling_Layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_components = num_components
        self.num_samples = num_samples
        self.lbound = lbound
        self.tau = tau 

    def call(self, inputs):
        logits, locs, offdiag, diag = inputs
        batch_size = tf.shape(logits)[0]
        
        # 1. Gumbel-Softmax for categorical mixture sampling
        logits_ext = tf.tile(tf.expand_dims(logits, axis=1), [1, self.num_samples, 1])
        U = tf.random.uniform(tf.shape(logits_ext), minval=1e-5, maxval=1.0 - 1e-5)
        gumbel_noise = -tf.math.log(-tf.math.log(U))
        
        y = tf.nn.softmax((logits_ext + gumbel_noise) / self.tau, axis=-1)
        y_expanded = tf.expand_dims(y, axis=-1) # [Batch, Samples, Components, 1]

        # 2. Build Cholesky Covariance Matrices (L)
        offdiag_flat = tf.reshape(offdiag, [-1, self.n_dims * (self.n_dims - 1) // 2])
        diag_flat = tf.reshape(diag, [-1, self.n_dims])
        
        L_flat = build_scale_tril_matrices(offdiag_flat, diag_flat, self.n_dims)
        L = tf.reshape(L_flat, [-1, self.num_components, self.n_dims, self.n_dims])
        
        # 3. Reparameterization Trick (Full Covariance)
        locs_ext = tf.tile(tf.expand_dims(locs, axis=1), [1, self.num_samples, 1, 1])
        L_ext = tf.tile(tf.expand_dims(L, axis=1), [1, self.num_samples, 1, 1, 1])
        
        epsilon = tf.random.truncated_normal(
            shape=(batch_size, self.num_samples, self.num_components, self.n_dims, 1),
            mean=0.0, stddev=1.0
        )
        
        # MatMul: L * epsilon
        noise_term = tf.squeeze(tf.matmul(L_ext, epsilon), axis=-1) 
        z_k = locs_ext + noise_term
        
        # 4. Collapse components using Gumbel-Softmax weights
        z_raw = tf.reduce_sum(y_expanded * z_k, axis=2) 
        
        # 5. Physics Mapping
        z_phys = safe_physics_mapping(z_raw, self.lbound)
        
        # We return L so the model can evaluate the exact log-probability later
        return z_phys, z_raw, L

    def get_config(self):
        config = super(FullCov_GMM_Sampling_Layer, self).get_config()
        config.update({
            "n_dims": self.n_dims,
            "num_components": self.num_components,
            "num_samples": self.num_samples,
            "lbound": self.lbound,
            "tau": self.tau
        })
        return config