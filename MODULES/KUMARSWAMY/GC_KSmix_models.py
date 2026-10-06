#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: March 2026
Unified Model for Gaussian Copula with Kumaraswamy Mixture Marginals (KSmix).
Refined to heavily reduce TFP object-instantiation overhead and optimize graph speed.
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras as K

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.KUMARSWAMY.GC_KSmix_architectures import Fully_connected_enc_GC_KS, Copula_KS_pdf_layer

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """Computes MAC with higher epsilon and clipping to prevent NaN gradients."""
    eps = 1e-8
    t_n = tf.sqrt(tf.reduce_sum(tf.square(modes_true), axis=2, keepdims=True) + eps)
    p_n = tf.sqrt(tf.reduce_sum(tf.square(modes_pred), axis=2, keepdims=True) + eps)
    
    modes_true_norm = modes_true / t_n
    modes_pred_norm = modes_pred / p_n
    
    dot_product = tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True)
    mac_matrix = tf.square(tf.clip_by_value(dot_product, -1.0, 1.0))
    return mac_matrix

# -------------------------------------------------------------------------
# INVERSE MODEL (ENCODER)
# -------------------------------------------------------------------------

class Inverse_Copula_KS_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_KS, num_samples, lbound, **kwargs):
        super(Inverse_Copula_KS_Model, self).__init__()
        self.n_dims = n_dims
        self.num_KS = num_KS
        self.num_samples = num_samples
        self.FC_encoder = Fully_connected_enc_GC_KS(input_dim_encoder, n_dims, num_KS, lbound)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, z_factors] = inputs
        
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2

        a_ks, b_ks, weights, offdiag, diag = tf.split(params, [
            self.n_dims * self.num_KS, 
            self.n_dims * self.num_KS, 
            self.n_dims * self.num_KS, 
            n_corr,                    
            self.n_dims                 
        ], axis=-1)

        a_ks = tf.reshape(a_ks, (-1, self.num_KS, self.n_dims)) + 1.1
        b_ks = tf.reshape(b_ks, (-1, self.num_KS, self.n_dims)) + 1.1
        
        weights = tf.reshape(weights, (-1, self.num_KS, self.n_dims))
        weights = tf.nn.softmax(weights / 1.5, axis=1) 
        
        return a_ks, b_ks, weights, offdiag, diag

# -------------------------------------------------------------------------
# FULL VAE MODEL WITH KUMARASWAMY MIXTURE
# -------------------------------------------------------------------------

class My_CopulaKSVAE_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, n_dims, num_KS, num_samples, gamma, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_CopulaKSVAE_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = Inverse_Copula_KS_Model(input_dim, n_dims, num_KS, num_samples, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        self.Copula_sampling_layer = Copula_KS_pdf_layer(n_dims, num_KS, num_samples, lbound)
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.num_KS = num_KS
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.gamma = gamma
        self.lbound = lbound 

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors] = inputs
        
        self.a_ks, self.b_ks, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
        inputs_to_sampling = [self.a_ks, self.b_ks, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
        self.reshaped_z_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
        
        # Optimization: Replaced tf.tile with tf.broadcast_to which is virtually 0-copy in VRAM
        lt_ext = tf.broadcast_to(
            tf.expand_dims(self.LT_matrices, axis=1), 
            [tf.shape(self.LT_matrices)[0], self.num_samples, self.n_dims, self.n_dims]
        )
        self.reshaped_LT_matrices = tf.reshape(lt_ext, [-1, self.n_dims, self.n_dims])

        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', self.reshaped_z_samples, self.Ke_matrices)
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', self.reshaped_z_samples, self.Ke_matrices)
        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        
        return self.reshaped_z_samples

    # --- LOSS COMPONENTS ---
    
    def Freqs_loss(self, y_true, y_pred):
        true_scaled = self.freq_data
        
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        huber = tf.keras.losses.Huber(delta=1.0)
        loss = huber(true_scaled, pred_scaled)
        return loss
    
    def MAC_modes_loss(self, y_true, y_pred):
        true_rot = self.rot_modes_data
        true_vert = self.vert_modes_data
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes

        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 

        loss_rot = get_match_loss(true_rot, pred_rot)
        loss_vert = get_match_loss(true_vert, pred_vert)
        return loss_rot + loss_vert
    
    def Copula_pdf_logprob(self, y_true, y_pred):
        """
        Calculates Copula Log-PDF via optimized Raw Math. 
        Replaces massive overhead from tfd.MultivariateNormalTriL
        """
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        
        # Fast Quantile (ErfInv)
        normal_samples = tf.math.sqrt(2.0) * tf.math.erfinv(2.0 * u_clipped - 1.0)
        z = tf.expand_dims(normal_samples, axis=-1)
        
        # Solve y = L^-1 z
        y = tf.linalg.triangular_solve(self.reshaped_LT_matrices, z)
        y = tf.squeeze(y, axis=-1)
        
        # Component parts for MVN log-density
        sq_norm_y = tf.reduce_sum(tf.square(y), axis=-1)
        diag_L = tf.linalg.diag_part(self.reshaped_LT_matrices)
        log_det_L = tf.reduce_sum(tf.math.log(diag_L), axis=-1)
        
        D_float = tf.cast(self.n_dims, tf.float32)
        log_prob_joint_normal = -0.5 * D_float * tf.math.log(2.0 * np.pi) - log_det_L - 0.5 * sq_norm_y
        
        # Standard normal marginals
        sq_norm_z = tf.reduce_sum(tf.square(normal_samples), axis=-1)
        log_prob_marginals_standard = -0.5 * D_float * tf.math.log(2.0 * np.pi) - 0.5 * sq_norm_z
             
        return log_prob_joint_normal - log_prob_marginals_standard
    
    def Marginal_KSMix_pdf_logprob(self, y_true, y_pred):
        """
        Calculates Kumaraswamy Mixture PDF via LogSumExp and native math.
        Averages 5-10x faster than initializing tfd.MixtureSameFamily per batch iteration.
        """
        x = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        x = tf.clip_by_value(x, 1e-6, 1.0 - 1e-6)
        
        # Function to expand KS params to matching dimensions via 0-copy broadcasting
        batch_size = tf.shape(self.a_ks)[0]
        def expand_ks_param(tensor):
            t_ext = tf.broadcast_to(
                tf.expand_dims(tensor, 1), 
                [batch_size, self.num_samples, self.num_KS, self.n_dims]
            )
            return tf.reshape(t_ext, [-1, self.num_KS, self.n_dims])
        
        a = expand_ks_param(self.a_ks)
        b = expand_ks_param(self.b_ks)
        w = expand_ks_param(self.weight_vals)
        
        x_exp = tf.expand_dims(x, axis=1) # Shape: [Batch*Samples, 1, Dims]
        
        # Raw Kumaraswamy PDF components
        x_a = tf.math.pow(x_exp, a)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0)
        
        log_pdf_comp = (tf.math.log(a) + tf.math.log(b) + 
                        (a - 1.0) * tf.math.log(x_exp) + 
                        (b - 1.0) * tf.math.log(one_minus_x_a))
                        
        # Log mixture density = LogSumExp(log(w) + log_pdf_comp)
        log_w = tf.math.log(w + 1e-8)
        log_prob_kuma_space = tf.reduce_logsumexp(log_w + log_pdf_comp, axis=1) # Reduce across KS axis
        
        adjustment = tf.math.log(1.0 - self.lbound + 1e-7)
        log_prob_marginals = tf.reduce_sum(log_prob_kuma_space - adjustment, axis=-1)
        return log_prob_marginals
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob_term = self.Marginal_KSMix_pdf_logprob(y_true, y_pred)   

        joint_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term)
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * joint_logprob
    
    def Mixture_Diversity_loss(self):
        weights = self.weight_vals
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=1) # [Batch, n_dims]
        mean_entropy = tf.reduce_mean(entropy)
        return -mean_entropy
    
    def Boundary_Penalty_loss(self):
        x = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        penalty = -tf.reduce_mean(tf.math.log(x + 1e-4) + tf.math.log(1.0 - x + 1e-4))
        return penalty
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        Loss_mix_diveristy = self.Mixture_Diversity_loss()
        Loss_boundary = self.Boundary_Penalty_loss()

        Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
        ELBO_loss = 5.0 * Loss_freqs + 10.0 * Loss_MAC + Joint_copula_nll + 1.5 * Loss_mix_diveristy + 0.1 * Loss_boundary
        
        return ELBO_loss

    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_CopulaKSVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))