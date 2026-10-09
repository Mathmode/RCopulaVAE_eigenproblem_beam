#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: March 2026
Unified Model for Gaussian Copula with Kumaraswamy Mixture Marginals (KSmix).
Refined to fully unconstrain the correlation parameters, matching the GMM behavior.
"""

# %% IMPORTS
# =============================================================================
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import tensorflow_probability as tfp
tfd = tfp.distributions

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.KUMARSWAMY.GC_KSmix_architectures import Fully_connected_enc_GC_KS, Copula_KS_pdf_layer


# %% HELPER FUNCTIONS
# =============================================================================
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


# %% INVERSE ENCODER MODEL
# =============================================================================
class Inverse_Copula_KS_Model(tf.keras.Model):
    """
    Encoder block connecting features to parameters (Weights are correctly constrained here).
    """
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

        a_ks = tf.reshape(a_ks, (-1, self.num_KS, self.n_dims)) 
        b_ks = tf.reshape(b_ks, (-1, self.num_KS, self.n_dims)) 
        
        # --- CRITICAL FIX: WEIGHTS SOFTMAX CONSTRAINT ---
        # We correctly reshape and constrain them with a Softmax across the 'K' mixture components.
        weights = tf.reshape(weights, (-1, self.num_KS, self.n_dims))
        weights = tf.nn.softmax(weights, axis=1)
        
        # WIDENED BOUNDS: Allow Kumaraswamy parameters to reach 150.0. 
        # This gives the mixture the mathematical freedom to form extremely 
        # sharp peaks and tight ridges without hitting a ceiling.
        a_ks = tf.clip_by_value(a_ks, 0.45, 150.0)
        b_ks = tf.clip_by_value(b_ks, 0.45, 150.0)
        
        offdiag = offdiag 
        diag = diag + 1e-4 
        
        return a_ks, b_ks, weights, offdiag, diag


# %% FULL VAE PHYSICS PIPELINE
# =============================================================================
class My_CopulaKSVAE_withEigen(tf.keras.Model):
    """
    Main VAE Model uniting Inverse Encoder -> Sampler -> Forward Physics Simulation
    """
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
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        
        normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        normal_samples = normal_dist.quantile(u_clipped)
        
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), 
                                         scale_tril=self.reshaped_LT_matrices)
        
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        log_prob_marginals_standard = tf.reduce_sum(normal_dist.log_prob(normal_samples), axis=-1)      
        
        return log_prob_joint_normal - log_prob_marginals_standard
    
    def Marginal_KSMix_pdf_logprob(self, y_true, y_pred):
        x = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        x = tf.clip_by_value(x, 1e-5, 1.0 - 1e-5)
        
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
        
        x_exp = tf.expand_dims(x, axis=1) 
        
        x_a = tf.math.pow(x_exp, a)
        one_minus_x_a = tf.clip_by_value(1.0 - x_a, 1e-7, 1.0)
        
        log_pdf_comp = (tf.math.log(a) + tf.math.log(b) + 
                        (a - 1.0) * tf.math.log(x_exp) + 
                        (b - 1.0) * tf.math.log(one_minus_x_a))
                        
        log_w = tf.math.log(w + 1e-8)
        log_prob_kuma_space = tf.reduce_logsumexp(log_w + log_pdf_comp, axis=1) 
        
        adjustment = tf.math.log(1.0 - self.lbound + 1e-7)
        log_prob_marginals = tf.reduce_sum(log_prob_kuma_space - adjustment, axis=-1)
        return log_prob_marginals
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob_term = self.Marginal_KSMix_pdf_logprob(y_true, y_pred)   

        joint_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term)
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * joint_logprob
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        loss_f = self.Freqs_loss(y_true, y_pred)
        loss_m = self.MAC_modes_loss(y_true, y_pred)
        loss_j = self.Joint_copula_dens_term(y_true, y_pred)
        
        # BOOSTED MAC MULTIPLIER (10.0 -> 50.0): 
        # This breaks the "fat posterior" compromise. It heavily penalizes 
        # mode shape errors, forcing the Copula's L-matrix to aggressively 
        # capture the dominant correlations to satisfy the physics engine.
        ELBO_loss = loss_f + 50.0 * loss_m + loss_j
        
        return ELBO_loss

    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_CopulaKSVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))