#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:49:18 2025

@author: afernandez
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import numpy as np

tfd = tfp.distributions
from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.BETA_GCOP.Beta_architectures import Fully_connected_enc_Beta, Copula_pdf_layer

def calculate_MAC(modes_true, modes_pred):
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return mac_matrix

class Inverse_GaussianCopula_Betamix_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_kumar_mix, num_samples, lbound, **kwargs):
        super(Inverse_GaussianCopula_Betamix_Model, self).__init__()
        self.n_dims = n_dims
        self.num_kumar_mix = num_kumar_mix
        self.num_samples = num_samples
        self.lbound = lbound
        self.FC_encoder = Fully_connected_enc_Beta(input_dim_encoder, n_dims, num_kumar_mix, lbound)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2
        split_sizes = [self.n_dims * self.num_kumar_mix, self.n_dims * self.num_kumar_mix, 
                       self.n_dims * self.num_kumar_mix, n_corr, self.n_dims]
        
        alphas, betas, weight_vals, offdiag_elems, diag_elems = tf.split(params, split_sizes, axis=-1)

        alphas = tf.reshape(alphas, (-1, self.num_kumar_mix, self.n_dims))
        betas = tf.reshape(betas, (-1, self.num_kumar_mix, self.n_dims))
        weight_vals = tf.reshape(weight_vals, (-1, self.num_kumar_mix, self.n_dims))
        weight_vals = tf.nn.softmax(weight_vals, axis=1)
        
        return alphas, betas, weight_vals, offdiag_elems, diag_elems

class My_CopulaVAE_Kumarmarg_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, regu_weight, n_dims, num_kumar_mix, num_samples, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_CopulaVAE_Kumarmarg_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices
        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = Inverse_GaussianCopula_Betamix_Model(input_dim, n_dims, num_kumar_mix, num_samples, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_kumar_mix, num_samples, lbound)
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.n_dims = n_dims
        self.num_kumar_mix = num_kumar_mix
        self.num_samples = num_samples
        self.regu_weight = regu_weight # This is the Beta factor for NLL
        self.lbound = lbound 
            
    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors] = inputs
        self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
        inputs_to_sampling = [self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
        self.reshaped_z_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
    
        # Logic for stiffness reduction (Physics)
        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', tf.cast(self.reshaped_z_samples, dtype=tf.float32), self.Ke_matrices)
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(self.reshaped_z_samples, dtype=tf.float32), self.Ke_matrices)
        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        self.pred_freqs = tf.abs(self.pred_freqs)    
        return self.reshaped_z_samples

    def Freqs_loss(self, y_true, y_pred):
        true_scaled = self.freq_data
        pred_hz = self.pred_freqs
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-6))
        pred_scaled = (pred_log - self.mean_freq) / self.std_freq
        return tf.math.reduce_mean(tf.square(true_scaled - pred_scaled))
    
    def MAC_modes_loss(self, y_true, y_pred):
        true_rot = self.rot_modes_data
        true_vert = self.vert_modes_data
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes
        def get_best_match_loss(true_group, pred_group):
            mac_mat = calculate_MAC(true_group, pred_group)
            best_matches = tf.reduce_max(mac_mat, axis=2)
            return tf.reduce_mean(1.0 - best_matches) 
        return get_best_match_loss(true_rot, pred_rot) + get_best_match_loss(true_vert, pred_vert)

    def Copula_pdf_logprob(self, y_true, y_pred):
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
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
        alphas_ext = tf.repeat(self.alphas[:,tf.newaxis,:,:], self.num_samples, axis = 1) 
        reshaped_alphas = tf.reshape(alphas_ext, [-1, self.num_kumar_mix, self.n_dims]) 
        
        betas_ext = tf.repeat(self.betas[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        reshaped_betas = tf.reshape(betas_ext, [-1, self.num_kumar_mix, self.n_dims])  
        
        weights_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        reshaped_weights = tf.reshape(weights_ext, [-1,self.num_kumar_mix, self.n_dims])  

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
        
        # Use regu_weight (Beta factor) to scale NLL
        return tf.math.square(tf.cast(self.regu_weight, dtype=tf.float32)) * joint_logprob
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        
        # NLL Term (Minimize -LogProb)
        Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
        ELBO_loss = Loss_freqs + 10.0 * Loss_MAC + Joint_copula_nll
        
        return tf.where(tf.math.is_nan(ELBO_loss), 100.0, ELBO_loss)























































# import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow.keras.layers import Layer
# import tensorflow.keras as K
# import numpy as np

# tfd = tfp.distributions
# from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
# from MODULES.BETA_GCOP.Beta_architectures import Fully_connected_enc_Beta, Copula_pdf_layer

# def calculate_MAC(modes_true, modes_pred):
#     modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
#     modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
#     mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
#     return mac_matrix

# class Inverse_GaussianCopula_Betamix_Model(tf.keras.Model):
#     def __init__(self, input_dim_encoder, n_dims, num_kumar_mix, num_samples, lbound, **kwargs):
#         super(Inverse_GaussianCopula_Betamix_Model, self).__init__()
#         self.n_dims = n_dims
#         self.num_kumar_mix = num_kumar_mix
#         self.num_samples = num_samples
#         self.lbound = lbound
#         self.FC_encoder = Fully_connected_enc_Beta(input_dim_encoder, n_dims, num_kumar_mix, lbound)

#     def call(self, inputs):
#         [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs
#         flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
#         flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
#         modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

#         params = self.FC_encoder(modal_data)
#         n_corr = self.n_dims * (self.n_dims - 1) // 2
#         split_sizes = [self.n_dims * self.num_kumar_mix, self.n_dims * self.num_kumar_mix, 
#                        self.n_dims * self.num_kumar_mix, n_corr, self.n_dims]
        
#         alphas, betas, weight_vals, offdiag_elems, diag_elems = tf.split(params, split_sizes, axis=-1)

#         alphas = tf.reshape(alphas, (-1, self.num_kumar_mix, self.n_dims))
#         betas = tf.reshape(betas, (-1, self.num_kumar_mix, self.n_dims))
#         weight_vals = tf.reshape(weight_vals, (-1, self.num_kumar_mix, self.n_dims))
#         weight_vals = tf.nn.softmax(weight_vals, axis=1)
        
#         return alphas, betas, weight_vals, offdiag_elems, diag_elems

# class My_CopulaVAE_Betamarg_withEigen(tf.keras.Model):
#     def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, regu_weight, n_dims, num_kumar_mix, num_samples, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
#         super(My_CopulaVAE_Betamarg_withEigen, self).__init__()
#         self.num_dofs = num_dofs
#         self.n_elements = n_elements
#         self.n_modes = n_modes
#         self.fixed_dofs_indices = fixed_dofs_indices
#         self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
#         self.L_inv = tf.constant(L_inv, dtype=tf.float32)

#         self.Encoder_model = Inverse_GaussianCopula_Betamix_Model(input_dim, n_dims, num_kumar_mix, num_samples, lbound)
#         self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
#         self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_kumar_mix, num_samples, lbound)
        
#         self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
#         self.std_freq = tf.constant(std_f, dtype=tf.float32)
#         self.n_dims = n_dims
#         self.num_kumar_mix = num_kumar_mix
#         self.num_samples = num_samples
#         self.regu_weight = regu_weight
#         self.lbound = lbound 
            
#     def call(self, inputs):
#         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors] = inputs
#         self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
#         self.alphas_ext = tf.repeat(self.alphas[:,tf.newaxis,:,:], self.num_samples, axis = 1) 
#         self.reshaped_alphas = tf.reshape(self.alphas_ext, [-1,self.num_kumar_mix, self.n_dims]) 
#         self.betas_ext = tf.repeat(self.betas[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_betas = tf.reshape(self.betas_ext, [-1, self.num_kumar_mix, self.n_dims])  
#         self.weights_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_weight_vals = tf.reshape(self.weights_ext, [-1,self.num_kumar_mix, self.n_dims])  

#         inputs_to_sampling = [self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems]
#         self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
#         self.reshaped_z_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
#         self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
    
#         self.LT_matrices_ext =  tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_LT_matrices = tf.reshape(self.LT_matrices_ext, [-1,self.n_dims, self.n_dims])
        
#         if len(self.Ke_matrices.shape) == 2:
#             Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', tf.cast(self.reshaped_z_samples, dtype=tf.float32), self.Ke_matrices)
#         else:
#             Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(self.reshaped_z_samples, dtype=tf.float32), self.Ke_matrices)
        
#         Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
#         self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
#         self.pred_freqs = tf.abs(self.pred_freqs)    
#         return self.reshaped_z_samples

#     def Freqs_loss(self, y_true, y_pred):
#         true_scaled = self.freq_data
#         pred_hz = self.pred_freqs
#         pred_log = tf.math.log(tf.maximum(pred_hz, 1e-6))
#         pred_scaled = (pred_log - self.mean_freq) / self.std_freq
#         return tf.math.reduce_mean(tf.square(true_scaled - pred_scaled))
    
#     def MAC_modes_loss(self, y_true, y_pred):
#         true_rot = self.rot_modes_data
#         true_vert = self.vert_modes_data
#         pred_rot = self.pred_rotmodes
#         pred_vert = self.pred_vertmodes
#         def get_best_match_loss(true_group, pred_group):
#             mac_mat = calculate_MAC(true_group, pred_group)
#             best_matches = tf.reduce_max(mac_mat, axis=2)
#             return tf.reduce_mean(1.0 - best_matches) 
#         return get_best_match_loss(true_rot, pred_rot) + get_best_match_loss(true_vert, pred_vert)

#     def Copula_pdf_logprob(self, y_true, y_pred):
#         # GUARD 1: Stricter clipping for normal quantile to avoid Inf
#         u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
#         normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(u_clipped)
        
#         # GUARD 2: Replace any remaining NaNs/Infs with zero to prevent propagation
#         normal_samples = tf.where(tf.math.is_finite(normal_samples), normal_samples, tf.zeros_like(normal_samples))
        
#         mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), scale_tril=self.reshaped_LT_matrices)
        
#         log_prob_joint_normal = mvn.log_prob(normal_samples)
#         # Log of Standard Normal PDF: -0.5*log(2pi) - 0.5*x^2
#         log_prob_marginals_standard = tf.reduce_sum(-0.5 * tf.math.log(2.0 * np.pi) - 0.5 * tf.math.square(normal_samples), axis=-1)
             
#         result = log_prob_joint_normal - log_prob_marginals_standard
#         return tf.clip_by_value(result, -50.0, 50.0) # Tight clipping for stability

#     def Marginal_KSMix_pdf_logprob(self, y_true, y_pred):
#         kuma_domain_samples = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)
#         # kuma_domain_samples = tf.clip_by_value(kuma_domain_samples, 1e-5, 1.0 - 1e-5)
#         alphas_t = tf.transpose(self.reshaped_alphas, perm=[0, 2, 1])
#         betas_t = tf.transpose(self.reshaped_betas, perm=[0, 2, 1])
#         weights_t = tf.transpose(self.reshaped_weight_vals, perm=[0, 2, 1])

#         mixture_dist = tfd.MixtureSameFamily(
#             mixture_distribution=tfd.Categorical(probs=weights_t),
#             components_distribution=tfd.Kumaraswamy(concentration1=alphas_t, concentration0=betas_t)
#         )

#         log_prob_kuma_space = mixture_dist.log_prob(kuma_domain_samples)
        
#         # GUARD 3: Neutralize NaNs and clip the marginal probability
#         log_prob_kuma_space = tf.where(tf.math.is_finite(log_prob_kuma_space), log_prob_kuma_space, tf.fill(tf.shape(log_prob_kuma_space), -20.0))
#         log_prob_kuma_space = tf.clip_by_value(log_prob_kuma_space, -20.0, 20.0)

#         adjustment = tf.math.log(1.0 - self.lbound + 1e-7)
#         log_prob_marginals = tf.math.reduce_sum(log_prob_kuma_space - adjustment, axis=-1)
#         return log_prob_marginals
    
#     def Joint_copula_dens_term(self, y_true, y_pred):
#         Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
#         Marginal_logprob_term = self.Marginal_KSMix_pdf_logprob(y_true, y_pred)   

#         # NLL is Negative log-likelihood. Sum up terms.
#         # Use mean over batch but limit the magnitude to let frequency loss dominate
#         joint_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term)
        
#         # GUARD 4: Final NaN scrub before applying regu_weight
#         joint_logprob = tf.where(tf.math.is_finite(joint_logprob), joint_logprob, 0.0)
        
#         # regu_weight is typically small to avoid over-regularization
#         return tf.math.square(tf.cast(self.regu_weight, dtype=tf.float32)) * joint_logprob
    
#     def ELBO_Copula_loss(self, y_true, y_pred):
#         Loss_freqs = self.Freqs_loss(y_true, y_pred)
#         Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
#         Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
        
#         # Note: We MINIMIZE Reconstruction + NLL. 
#         # If NLL is -LogProb, then we want to minimize -LogProb (Maximize Likelihood)
#         # Your previous code added Joint_copula_nll. If Joint_copula_nll is LogProb, 
#         # it should be SUBTRACTED to maximize likelihood. 
#         # Assuming Joint_copula_nll is LogProb (positive is good):
#         ELBO_loss = 1.0*Loss_freqs + 10.0*Loss_MAC - Joint_copula_nll
        
#         # Final safety check on the total loss
#         return tf.where(tf.math.is_nan(ELBO_loss), 100.0, ELBO_loss)
 
    
    
    
    
    
    

