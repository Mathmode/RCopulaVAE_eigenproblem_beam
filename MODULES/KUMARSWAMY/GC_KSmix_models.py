#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: March 2026
Unified Model for Gaussian Copula with Kumaraswamy Mixture Marginals (KSmix).
Refined to prevent "Density Collapse" and prioritize Physical Reconstruction.
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

# @tf.function(jit_compile=True)
# def calculate_MAC(modes_true, modes_pred):
#     """Computes Modal Assurance Criterion (MAC) with improved stability."""
#     eps = 1e-8
#     modes_true_norm = modes_true / (tf.norm(modes_true, axis=2, keepdims=True) + eps)
#     modes_pred_norm = modes_pred / (tf.norm(modes_pred, axis=2, keepdims=True) + eps)
#     mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
#     return mac_matrix

# stabilized version for MAC calculation
@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """Computes MAC with higher epsilon and clipping to prevent NaN gradients."""
    eps = 1e-8
    # Normalize with safety
    t_n = tf.norm(modes_true, axis=2, keepdims=True) + eps
    p_n = tf.norm(modes_pred, axis=2, keepdims=True) + eps
    
    modes_true_norm = modes_true / t_n
    modes_pred_norm = modes_pred / p_n
    
    # MAC calculation
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

        a_ks = tf.reshape(a_ks, (-1, self.num_KS, self.n_dims))+1.1
        b_ks = tf.reshape(b_ks, (-1, self.num_KS, self.n_dims))+1.1
        # weights = tf.reshape(weights, (-1, self.num_KS, self.n_dims))
        # weights = tf.nn.softmax(weights, axis=1) + 1e-8
        # weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
        
        # 2. (REVISED) Temperature-scaled softmax for weights to prevent sparsity too early
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
        """
        Robust Frequency Loss using Huber Loss to handle solver instabilities.
        """
        # 1. Denormalize true data or use direct scaled comparison
        true_scaled = self.freq_data
        
        # 2. Process predicted frequencies: Abs -> Log -> Standardize
        # Clip to prevent log(0) which causes NaNs in gradients
        pred_hz = tf.abs(self.pred_freqs)
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-7))
        pred_scaled = (pred_log - self.mean_freq) / (self.std_freq + 1e-8)
        
        # 3. Use Huber Loss instead of MSE for stability
        huber = tf.keras.losses.Huber(delta=1.0)
        loss = huber(true_scaled, pred_scaled)
        
        return loss
    
    def MAC_modes_loss(self, y_true, y_pred):
        """
        Stabilized MAC loss with best-match logic.
        """
        true_rot = self.rot_modes_data
        true_vert = self.vert_modes_data
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes

        def get_match_loss(t_group, p_group):
            mac_mat = calculate_MAC(t_group, p_group)
            # Take max across predicted modes to find best match for each ground truth mode
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 

        loss_rot = get_match_loss(true_rot, pred_rot)
        loss_vert = get_match_loss(true_vert, pred_vert)
        
        return loss_rot + loss_vert
    
    
  
    def Copula_pdf_logprob(self, y_true, y_pred):
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-4, 1.0 - 1e-4)
        normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(u_clipped)
        
        # Build LT matrices for the samples
        LT_matrices_ext = tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        reshaped_LT = tf.reshape(LT_matrices_ext, [-1, self.n_dims, self.n_dims])
        
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), scale_tril=reshaped_LT)
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        
        # Standard normal log prob (denominator of copula density)
        log_prob_marginals_standard = tf.reduce_sum(tfd.Normal(0.0, 1.0).log_prob(normal_samples), axis=-1)
             
        return log_prob_joint_normal - log_prob_marginals_standard
    
   
    
    
    def Marginal_KSMix_pdf_logprob(self, y_true, y_pred):
        # Kumaraswamy is defined on [0, 1]. Map our z samples [lbound, 1] to [0, 1]
        kuma_domain_samples = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        kuma_domain_samples = tf.clip_by_value(kuma_domain_samples, 1e-5, 1.0 - 1e-5)
        
        # Expand Encoder parameters for all samples
        a_ks_ext = tf.repeat(self.a_ks[:,tf.newaxis,:,:], self.num_samples, axis = 1) 
        reshaped_alphas = tf.reshape(a_ks_ext, [-1, self.num_KS, self.n_dims]) 
        
        b_ks_ext = tf.repeat(self.b_ks[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        reshaped_betas = tf.reshape(b_ks_ext, [-1, self.num_KS, self.n_dims])  
        
        w_ks_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        reshaped_weights = tf.reshape(w_ks_ext, [-1,self.num_KS, self.n_dims])  

        # Mixture density calculation
        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.transpose(reshaped_weights, [0, 2, 1])),
            components_distribution=tfd.Kumaraswamy(concentration1=tf.transpose(reshaped_alphas, [0, 2, 1]), 
                                                   concentration0=tf.transpose(reshaped_betas, [0, 2, 1]))
        )

        log_prob_kuma_space = mixture_dist.log_prob(kuma_domain_samples)
        
        # Jacobian adjustment for the transformation z = lbound + (1-lbound)*x
        adjustment = tf.math.log(1.0 - self.lbound + 1e-7)
        log_prob_marginals = tf.math.reduce_sum(log_prob_kuma_space - adjustment, axis=-1)
        return log_prob_marginals
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob_term = self.Marginal_KSMix_pdf_logprob(y_true, y_pred)   

        joint_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term)
        
        # Use GAMMA (regularization weight) to scale NLL
        return tf.math.square(tf.cast(self.gamma, dtype=tf.float32)) * joint_logprob
    
    # REGULARIZER TO BENEFIT SOLUTIONS WITH VARIOUS MIXTURES (TO DO)
    def Mixture_Diversity_loss(self):
        """
        Regularizer that favors solutions where weights are distributed across 
        the mixture components. This uses the Shannon Entropy of the weights.
        High entropy = More diversity. Loss = -Entropy.
        """
        # weight_vals shape: [Batch, num_KS, n_dims]
        weights = self.weight_vals
        # tf.print('holaa,weights', weights[0,:])
        
        # Calculate entropy per dimension: H = - sum(p * log(p))
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=1) # [Batch, n_dims]
        
        # Average over batch and dimensions
        mean_entropy = tf.reduce_mean(entropy)
        
        # We minimize -Entropy to maximize diversity
        return -mean_entropy
    
    def Boundary_Penalty_loss(self):
        """
        Custom penalty to prevent samples from being pushed exactly to 
        lbound or 1.0, which causes the PDF collapse you observed.
        """
        x = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
        # Logarithmic barrier: increases as x -> 0 or x -> 1
        penalty = -tf.reduce_mean(tf.math.log(x + 1e-4) + tf.math.log(1.0 - x + 1e-4))
        return penalty
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        # Diversity term: Encourage the model to use all Kumaraswamy components
        Loss_mix_diveristy = self.Mixture_Diversity_loss()
        Loss_boundary = self.Boundary_Penalty_loss()


        Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
        ELBO_loss = 5.*Loss_freqs + 10.0 * Loss_MAC + Joint_copula_nll + 1.5*Loss_mix_diveristy + 0.1 * Loss_boundary
        
        return ELBO_loss
    
    
    # def Copula_pdf_logprob(self):
    #     # Standard Gaussian Copula Log-Density
    #     u = tf.clip_by_value(tf.reshape(self.reshaped_copula_samples, [-1, self.n_dims]), 1e-6, 1.0 - 1e-6)
    #     normal_dist = tfd.Normal(loc=0.0, scale=1.0)
    #     v = normal_dist.quantile(u)
        
    #     mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), scale_tril=self.reshaped_LT_matrices)
        
    #     log_c = mvn.log_prob(v) - tf.reduce_sum(normal_dist.log_prob(v), axis=-1)
    #     return log_c

    # def Marginal_pdf_logprob(self):
    #     # Kumaraswamy Mixture Marginal Log-Density
    #     # x is in [0, 1] relative to the lbound
    #     x = (tf.reshape(self.reshaped_z_samples, [-1, self.n_dims]) - self.lbound) / (1.0 - self.lbound)
    #     x = tf.clip_by_value(x, 1e-6, 1.0 - 1e-6)
        
    #     # expansion [Batch, Components, Dims] -> [Batch, Samples, Components, Dims]
    #     a_ext = tf.tile(tf.expand_dims(self.a_ks, axis=1), [1, self.num_samples, 1, 1])
    #     a = tf.reshape(a_ext, [-1, self.num_KS, self.n_dims])
        
    #     b_ext = tf.tile(tf.expand_dims(self.b_ks, axis=1), [1, self.num_samples, 1, 1])
    #     b = tf.reshape(b_ext, [-1, self.num_KS, self.n_dims])
        
    #     w_ext = tf.tile(tf.expand_dims(self.weight_vals, axis=1), [1, self.num_samples, 1, 1])
    #     w = tf.reshape(w_ext, [-1, self.num_KS, self.n_dims])

    #     x_exp = tf.expand_dims(x, axis=1) # [TotalSamples, 1, Dims]
        
    #     # Kumaraswamy PDF components
    #     ln_a = tf.math.log(tf.maximum(a, 1e-12))
    #     ln_b = tf.math.log(tf.maximum(b, 1e-12))
    #     x_a = tf.math.pow(x_exp, a)
    #     ln_x = tf.math.log(x_exp)
    #     ln_1_minus_xa = tf.math.log(tf.maximum(1.0 - x_a, 1e-12))
        
    #     log_pdf_comp = ln_a + ln_b + (a - 1.0) * ln_x + (b - 1.0) * ln_1_minus_xa
        
    #     # Mixture logic: log(sum(w * p))
    #     log_w = tf.math.log(tf.maximum(w, 1e-12))
    #     marginal_log_densities = tf.reduce_logsumexp(log_w + log_pdf_comp, axis=1)
        
    #     # Jacobian adjustment for the transformation [0, 1] -> [lbound, 1]
    #     log_abs_det_jacobian = -tf.math.log(tf.maximum(1.0 - self.lbound, 1e-12))
        
    #     return tf.reduce_sum(marginal_log_densities + log_abs_det_jacobian, axis=-1)


    # def Joint_copula_dens_term(self, y_true, y_pred):
    #     c_term = self.Copula_pdf_logprob()
    #     m_term = self.Marginal_pdf_logprob()
    #     # VAE Regularizer: -E_q[log q(z|m)]
    #     # We want to minimize -Expectation, so we return -mean(log_prob)
    #     # If log_prob is very high (positive), this loss becomes very negative (good).
    #     return -tf.reduce_mean(c_term + m_term) * tf.square(tf.cast(self.gamma, tf.float32))

    # def ELBO_Copula_loss(self, y_true, y_pred):
    #     loss_f = self.Freqs_loss(y_true, y_pred)
    #     loss_m = self.MAC_modes_loss(y_true, y_pred)
    #     loss_j = self.Joint_copula_dens_term(y_true, y_pred)
        
    #     # Following Page 10 of your paper: L = L_freq + lambda*L_MAC + gamma^2*L_PDF
    #     # Note: Your "Joint_copula_dens_term" in the log was already negative, 
    #     # which means it was acting as the NLL.
    #     return loss_f + 10.0 * loss_m + loss_j



    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_CopulaKSVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))