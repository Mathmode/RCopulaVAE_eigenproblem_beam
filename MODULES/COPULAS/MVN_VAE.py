
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Space: Truncated Multivariate Normal (Independent/Diagonal)
Comparison model for Gaussian Copula + GMM.
Uses identical physics loss logic and architecture components where possible.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
tfd = tfp.distributions
from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem

# -------------------------------------------------------------------------
# HELPER FUNCTIONS FOR LOSS (Matching Original Logic)
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """
    Computes Modal Assurance Criterion (MAC) with improved stability.
    Matches the implementation in GC_GMm_models.py.
    """
    eps = 1e-8
    modes_true_norm = modes_true / (tf.norm(modes_true, axis=2, keepdims=True) + eps)
    modes_pred_norm = modes_pred / (tf.norm(modes_pred, axis=2, keepdims=True) + eps)
    
    # Compute MAC Matrix
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return mac_matrix

class TMVN_Encoder_Model(tf.keras.Model):
    """
    Simplified Encoder that predicts only Loc and Scale for a 
    Truncated Normal distribution.
    """
    def __init__(self, input_dim_encoder, n_dims, lbound, **kwargs):
        super(TMVN_Encoder_Model, self).__init__()
        self.n_dims = n_dims
        self.lbound = lbound
        
        # Using a similar depth to the original FC_encoder
        self.net = K.Sequential([
            K.layers.InputLayer(input_shape=(input_dim_encoder,)),
            K.layers.Dense(256, activation='relu', kernel_initializer="he_uniform"),
            K.layers.BatchNormalization(),
            K.layers.Dense(256, activation='relu', kernel_initializer="he_uniform"),
            K.layers.BatchNormalization(),
            K.layers.Dense(128, activation='relu', kernel_initializer="he_uniform"),
            K.layers.Dense(n_dims * 2) # [means, raw_scales]
        ])

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs
        
        # Preprocessing matching the original model
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.net(modal_data)
        means_raw, scales_raw = tf.split(params, 2, axis=-1)

        # Map means to [lbound, 1] similar to GC_GMm_architectures.py
        means = self.lbound + (1.0 - self.lbound) * tf.nn.sigmoid(means_raw)
        
        # Softplus ensures positive scales (standard deviations)
        scales = tf.nn.softplus(scales_raw) + 1e-5
        
        return means, scales

class My_TMVN_VAE_withEigen(tf.keras.Model):
    """
    VAE Model using Truncated Normal Latent Space.
    Maintains the same Physics-informed training loop and loss functions.
    """
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, L_inv, n_dims, beta, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_TMVN_VAE_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = TMVN_Encoder_Model(input_dim, n_dims, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.n_dims = n_dims
        self.beta = beta
        self.lbound = lbound 

    def call(self, inputs):
        # Store inputs for loss calculation during the call
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        
        # 1. ENCODE
        self.means, self.scales = self.Encoder_model(inputs)
        
        # 2. SAMPLE (Truncated Normal)
        # Using lbound as the lower threshold for the latent samples
        dist = tfd.TruncatedNormal(loc=self.means, scale=self.scales, low=self.lbound, high=1.0)
        self.reshaped_alpha_samples = dist.sample() # Single sample (equivalent to num_samples=1)

        # 3. DECODE (Physics Solver)
        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', self.reshaped_alpha_samples, self.Ke_matrices)
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', self.reshaped_alpha_samples, self.Ke_matrices)
        
        # num_samples is 1 here
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, 1, self.fixed_dofs_indices) 
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        
        return self.reshaped_alpha_samples

    def Freqs_loss(self, y_true, y_pred):
        """ Robust Frequency Loss using Huber Loss. """
        true_scaled = self.freq_data
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        return tf.keras.losses.Huber(delta=1.0)(true_scaled, pred_scaled)
    
    def MAC_modes_loss(self, y_true, y_pred):
        """ Stabilized MAC loss with best-match logic. """
        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 

        loss_rot = get_match_loss(self.rot_modes_data, self.pred_rotmodes)
        loss_vert = get_match_loss(self.vert_modes_data, self.pred_vertmodes)
        return loss_rot + loss_vert

    def LogProb_loss(self, y_true, y_pred):
        """
        Latent space log-probability (Equivalent to the NLL/KL term).
        """
        dist = tfd.TruncatedNormal(loc=self.means, scale=self.scales, low=self.lbound, high=1.0)
        # Sum logprobs across dimensions (independence assumption)
        log_prob = tf.reduce_sum(dist.log_prob(self.reshaped_alpha_samples), axis=-1)
        return tf.reduce_mean(log_prob)

    def ELBO_TMVN_loss(self, y_true, y_pred):
        """
        Combined Loss for the TMVN Baseline.
        """
        loss_f = self.Freqs_loss(y_true, y_pred)
        loss_m = self.MAC_modes_loss(y_true, y_pred)
        loss_lp = self.LogProb_loss(y_true, y_pred)
                
        # Maintaining multipliers for fair comparison (10.0 for MAC)
        total_loss = loss_f + 10.0 * loss_m + (tf.math.square(self.beta) * loss_lp)
        
        return total_loss

    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_TMVN_VAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))