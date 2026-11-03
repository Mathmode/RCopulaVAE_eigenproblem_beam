#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: March 2026
Unified Model for Gaussian Copula with Kumaraswamy Mixture Marginals (KSmix).
Refined to support multi-sample Monte Carlo (num_samples > 1), capture 
multimodal posteriors via Mixture Weights, and prevent Density Collapse.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K

tfd = tfp.distributions

from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.KUMARSWAMY.GC_KSmix_architectures import Fully_connected_enc_GC_KS, Copula_KS_pdf_layer

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """Computes MAC with higher epsilon and clipping to prevent NaN gradients."""
    eps = 1e-8
    t_n = tf.norm(modes_true, axis=2, keepdims=True) + eps
    p_n = tf.norm(modes_pred, axis=2, keepdims=True) + eps
    
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

        # Base bounds (>= 1.01) are now strictly handled by the architecture layer.
        a_ks = tf.reshape(a_ks, (-1, self.num_KS, self.n_dims))
        b_ks = tf.reshape(b_ks, (-1, self.num_KS, self.n_dims))
        
        # Temperature-scaled softmax for weights to prevent sparsity too early
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
        
        # Expansion via Tile: [Batch, 1, D, D] -> [Batch, Samples, D, D]
        lt_ext = tf.tile(tf.expand_dims(self.LT_matrices, axis=1), [1, self.num_samples, 1, 1])
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
        # Broadcast target data cleanly
        true_scaled = tf.expand_dims(self.freq_data, axis=1)
        
        pred_hz = tf.reshape(tf.abs(self.pred_freqs), [-1, self.num_samples, self.n_modes])
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        error = true_scaled - pred_scaled
        is_small_error = tf.abs(error) <= 1.0
        huber_loss = tf.where(is_small_error, 0.5 * tf.square(error), 1.0 * tf.abs(error) - 0.5)
        
        return tf.reduce_mean(tf.reduce_sum(huber_loss, axis=-1))
    
    def MAC_modes_loss(self, y_true, y_pred):
        true_rot = tf.repeat(self.rot_modes_data, self.num_samples, axis=0)
        true_vert = tf.repeat(self.vert_modes_data, self.num_samples, axis=0)

        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            best_matches = tf.reduce_max(mac_mat, axis=2) 
            loss_flat = 1.0 - best_matches 
            
            loss_reshaped = tf.reshape(loss_flat, [-1, self.num_samples, self.n_modes])
            return tf.reduce_mean(tf.reduce_sum(loss_reshaped, axis=-1)) 

        loss_rot = get_match_loss(true_rot, self.pred_rotmodes)
        loss_vert = get_match_loss(true_vert, self.pred_vertmodes)
        
        return loss_rot + loss_vert
    
    def Copula_pdf_logprob(self, y_true, y_pred):
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-4, 1.0 - 1e-4)
        normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(u_clipped)
        
        LT_matrices_ext = tf.repeat(self.LT_matrices, self.num_samples, axis=0)
        
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), scale_tril=LT_matrices_ext)
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        
        log_prob_marginals_standard = tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(normal_samples), axis=-1)
             
        copula_density_flat = log_prob_joint_normal - log_prob_marginals_standard
        return tf.reshape(copula_density_flat, [-1, self.num_samples])
    
    def Marginal_KSMix_pdf_logprob(self, y_true, y_pred):
        kuma_domain_samples = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        x_safe = tf.clip_by_value(kuma_domain_samples, 1e-5, 1.0 - 1e-5)
        x_exp = tf.reshape(x_safe, [-1, self.num_samples, 1, self.n_dims])
        
        a = tf.expand_dims(self.a_ks, axis=1) 
        b = tf.expand_dims(self.b_ks, axis=1) 
        w = tf.expand_dims(self.weight_vals, axis=1) 

        ln_x = tf.math.log(x_exp)
        x_a = tf.math.pow(x_exp, a)
        ln_1_minus_xa = tf.math.log(tf.clip_by_value(1.0 - x_a, 1e-7, 1.0))
        
        log_pdf_comp = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * ln_x + (b - 1.0) * ln_1_minus_xa
        
        log_w = tf.math.log(tf.clip_by_value(w, 1e-7, 1.0))
        
        marginal_log_densities = tf.reduce_logsumexp(log_w + log_pdf_comp, axis=2)
        adjustment = tf.math.log(1.0 - self.lbound + 1e-7)
        
        log_prob_marginals = tf.reduce_sum(marginal_log_densities - adjustment, axis=-1) 
        return log_prob_marginals
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        Copula_logprob = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob = self.Marginal_KSMix_pdf_logprob(y_true, y_pred)   

        joint_logprob = Copula_logprob + Marginal_logprob
        
        # Clipping ensures NLL does not swing infinitely if something goes wrong
        joint_logprob_clipped = tf.clip_by_value(joint_logprob, -50.0, 50.0)
        nll = -tf.reduce_mean(joint_logprob_clipped)
        
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * nll
    
    def Mixture_Diversity_loss(self):
        weights = self.weight_vals
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=1)
        mean_entropy = tf.reduce_mean(entropy)
        return -mean_entropy
    
    def Boundary_Penalty_loss(self):
        x = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        x_safe = tf.clip_by_value(x, 1e-4, 1.0 - 1e-4)
        penalty = -tf.reduce_mean(tf.math.log(x_safe) + tf.math.log(1.0 - x_safe))
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