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
# Assumes Inverse_Copula_Model and Copula_pdf_layer are imported/available
from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.BETA_GCOP.Beta_architectures import Fully_connected_enc_Beta, Copula_pdf_layer
        

# Import architectures if they are in a separate module
# from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Copula_pdf_layer, Fully_connected_dec
# For this file to be self-contained or runnable in context, we assume these classes are available 
# or imported by the user. The code below focuses on the Physics and VAE integration.


# -------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR LOSS
# -------------------------------------------------------------------------

# @tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """
    Computes Modal Assurance Criterion (MAC) between two sets of modes.
    Args:
        modes_true: (Batch, Modes, Nodes)
        modes_pred: (Batch, Modes, Nodes)
    Returns:
        MAC Matrix: (Batch, Modes, Modes) where [b, i, j] is MAC(True_i, Pred_j)
    """
    # Normalize along the Node axis (axis 2)
    modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
    modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    
    # Compute MAC Matrix: (Batch, Modes, Nodes) @ (Batch, Nodes, Modes) -> (Batch, Modes, Modes)
    # transpose_b=True flips the last two dims of modes_pred_norm to (Batch, Nodes, Modes)
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    
    return mac_matrix


class Inverse_GaussianCopula_Betamix_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_betas, num_samples, lbound, **kwargs):
        super(Inverse_GaussianCopula_Betamix_Model, self).__init__()
        self.n_dims = n_dims
        self.num_betas = num_betas
        self.num_samples = num_samples
        self.lbound = lbound
        self.FC_encoder = Fully_connected_enc_Beta(input_dim_encoder, n_dims, num_betas, lbound)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs
        
        # Flatten modal data
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        
        n_corr = self.n_dims * (self.n_dims - 1) // 2
        split_sizes = [
            self.n_dims * self.num_betas, # alphas
            self.n_dims * self.num_betas, # betas
            self.n_dims * self.num_betas, # weights
            n_corr,                       # off-diag L
            self.n_dims                   # diag L
        ]
        
        alphas, betas, weight_vals, offdiag_elems, diag_elems = tf.split(params, split_sizes, axis=-1)

        # Reshape to (Batch, Num_Components, N_dims)
        alphas = tf.reshape(alphas, (-1, self.num_betas, self.n_dims))
        betas = tf.reshape(betas, (-1, self.num_betas, self.n_dims))
        weight_vals = tf.reshape(weight_vals, (-1, self.num_betas, self.n_dims))
        
        # Softmax over the mixture components for each dimension
        weight_vals = tf.nn.softmax(weight_vals, axis=1)
        
        return alphas, betas, weight_vals, offdiag_elems, diag_elems

    def get_config(self):
        config = {
            'n_dims': self.n_dims,
            'num_betas': self.num_betas,
            'num_samples': self.num_samples,
            'lbound': self.lbound
        }
        base_config = super().get_config()
        return {**base_config, **config}
# -------------------------------------------------------------------------
# 4. VAE MODEL
# -------------------------------------------------------------------------

class My_CopulaVAE_Betamarg_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, regu_weight, n_dims, num_beta_mix, num_samples, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_CopulaVAE_Betamarg_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices
        # Cast Physics Matrices once
        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = Inverse_GaussianCopula_Betamix_Model(input_dim, n_dims, num_beta_mix, num_samples, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_beta_mix, num_samples, lbound)
        
        # Hyperparameters
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.n_dims = n_dims
        self.num_beta_mix = num_beta_mix
        self.num_samples = num_samples
        self.regu_weight = regu_weight
        self.lbound = lbound # lower bound for truncation according to z domain  (minimum reduction factor --> maximum admissible damage)
            
    def call(self, inputs):
        # Unpack Inputs: [Freqs, RotModes, VertModes, z_factors]
        # Data shapes expected:
        # Freqs: (Batch, Modes)
        # Modes: (Batch, Modes, Nodes)
        # z_factors: (Batch, Elements)
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.z_factors] = inputs
        
        # 1. ENCODER
        self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
        # Expand dimensions for sampling
        self.alphas_ext = tf.repeat(self.alphas[:,tf.newaxis,:,:], self.num_samples, axis = 1) #extension to include the number of samples H
        self.reshaped_alphas = tf.reshape(self.alphas_ext, [-1,self.num_beta_mix, self.n_dims]) 
        self.betas_ext = tf.repeat(self.betas[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_betas = tf.reshape(self.betas_ext, [-1, self.num_beta_mix, self.n_dims])  
        self.weights_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_weight_vals = tf.reshape(self.weights_ext, [-1,self.num_beta_mix, self.n_dims])  

        # 2. SAMPLING
        inputs_to_sampling = [self.alphas, self.betas, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling) # dimensions of the samples: [n_samples, batch_size, n_dims]
        # the marginal samples are the estimated Z (Stiffness Factors). We flatten to H*B as th
        self.reshaped_z_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
    
        self.LT_matrices_ext =  tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_LT_matrices = tf.reshape(self.LT_matrices_ext, [-1,self.n_dims, self.n_dims])
        
        # 3. DECODER: Solve Physics
        # Ke matrices damage calculation: Scale Base Ke by Alpha samples
        # Alpha: (Batch, Elem), Ke: (4, 4) -> Damaged: (Batch, Elem, 4, 4)
        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', 
                                        tf.cast(self.reshaped_z_samples, dtype=tf.float32), 
                                        tf.cast(self.Ke_matrices, dtype=tf.float32))
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', 
                                        tf.cast(self.reshaped_z_samples, dtype=tf.float32), 
                                        tf.cast(self.Ke_matrices, dtype=tf.float32))
        
        # Assemble Global Stiffness (Free DOFs only)
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
    
        # Solve Eigenvalue Problem
        # Returns: Freqs (B, M), RotModes (B, M, N), VertModes (B, M, N)
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        self.pred_freqs = tf.abs(self.pred_freqs)    
        
        # RETURN ALL DESIRED OUTPUTS
        # return [self.reshaped_alpha_samples, self.means, self.scales, self.offdiag_elems, self.diag_elems]       
        return self.reshaped_z_samples

    # --- LOSS FUNCTIONS ---

    def Freqs_loss(self, y_true, y_pred):
        true_scaled = self.freq_data    # we applied the log and scaled before entering the model! Be careful with this!
        pred_hz = self.pred_freqs
        
        # Safe log and standardization
        pred_log = tf.math.log(tf.maximum(pred_hz, 1e-6))
        pred_scaled = (pred_log - self.mean_freq) / self.std_freq
        
        loss = tf.math.reduce_mean(tf.square(true_scaled - pred_scaled))
        return loss
    
    def MAC_modes_loss(self, y_true, y_pred):
        true_rot = self.rot_modes_data
        true_vert = self.vert_modes_data
        pred_rot = self.pred_rotmodes
        pred_vert = self.pred_vertmodes

        def get_best_match_loss(true_group, pred_group):
            # Calculate MAC Matrix: (Batch, Modes, Modes)
            # Element (b, i, j) is MAC between True Mode i and Pred Mode j
            mac_mat = calculate_MAC(true_group, pred_group)
            
            # Find the best matching Predicted Mode for each True Mode
            # This handles potential mode ordering swaps if they occur
            best_matches = tf.reduce_max(mac_mat, axis=2)
            
            # We want best matches to be 1.0. Loss = 1 - MAC
            return tf.reduce_mean(1.0 - best_matches) 

        loss_rot = get_best_match_loss(true_rot, pred_rot)
        loss_vert = get_best_match_loss(true_vert, pred_vert)
        
        return loss_rot + loss_vert      

    def Copula_pdf_logprob(self,y_true, ypred):
        """
        Compute the log-likelihood of the joint PDF defined by a Gaussian copula and Gaussian mixture marginals.
        
        Args:
            locs: Tensor of shape [n_modes, n_dims], means of the Gaussian mixture components.
            scales: Tensor of shape [n_modes, n_dims], standard deviations of the Gaussian mixture components.
            weights: Tensor of shape [n_modes, n_dims], weights of the Gaussian mixture components.
            correlation_matrix: Tensor of shape [n_dims, n_dims], correlation matrix of the Gaussian copula.
            samples: Tensor of shape [n_samples, n_dims], the observed data.
    
        Returns:
            log_likelihood: Tensor of shape [], the log-likelihood of the joint PDF.
        """
        
        # # Clip copula samples slightly away from 0 and 1 to prevent Inf in quantile
        # u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-6, 1.0 - 1e-6)
        # normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        # normal_samples = normal_dist.quantile(u_clipped)
        
        # # Convert uniform samples to standard normal
        normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(self.reshaped_copula_samples)
        
        
        
        mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)
        
        ###*****
        # log_prob_joint_normal = tf.math.log(mvn.prob(normal_samples)+1e-07)
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        
        log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(normal_samples)+1e-07), axis=-1)      
        
    
        return log_prob_joint_normal - log_prob_marginals_standard
    
    def Marginal_pdf_logprob(self, y_true, y_pred):
        """
        Log-probability calculation for a mixture of Betas on [lbound, 1] in the marginals
        """
        # Convert samples from [lbound, 1] back to [0, 1] for Beta PDF
        beta_domain_samples = (self.reshaped_z_samples - self.lbound) / (1.0 - self.lbound)

        # Setup Beta mixture per dimension (originally: [B*H, num_beta_components, n_dims])
        # If we permute it, we obtain [B*H, n_dims, num_beta_components]:
        alphas_t = tf.transpose(self.reshaped_alphas, perm=[0, 2, 1])
        betas_t = tf.transpose(self.reshaped_betas, perm=[0, 2, 1])
        weights_t = tf.transpose(self.reshaped_weight_vals, perm=[0, 2, 1])

        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=weights_t),
            components_distribution=tfd.Beta(concentration1=alphas_t, concentration0=betas_t)
        )

        # Log prob in [0, 1] domain
        # mixture_dist.log_prob expects (Batch, n_dims)
        log_prob_beta_space = mixture_dist.log_prob(beta_domain_samples)

        # Change of variables adjustment: log(p_z(z)) = log(p_x(x)) - log(|dz/dx|)
        # z = lbound + (1-lbound)*x  => dz/dx = (1-lbound)
        adjustment = tf.math.log(1.0 - self.lbound)
        
        log_prob_marginals = tf.math.reduce_sum(log_prob_beta_space - adjustment, axis=-1)
        return log_prob_marginals
    
    def Joint_copula_dens_term(self,y_true, y_pred):
        Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob_term = self.Marginal_pdf_logprob(y_true,y_pred)   
        
        joint_copula_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term, axis = None)
        joint_copula_logprob = tf.math.square(tf.cast(self.regu_weight, dtype=tf.float32))* (joint_copula_logprob)

        return joint_copula_logprob
    
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        """
        Total Loss Function: Minimizes Reconstruction Error + NLL - Regularizer
        """
        
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        
        Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
                
        ELBO_loss = Loss_freqs + 10*Loss_MAC + Joint_copula_nll
        
        return ELBO_loss
    
    def get_config(self):
        config = {
            'num_dofs': self.num_dofs,
            'M_free': self.Mfree, # Note: Not used in logic but stored
            'fixed_dofs_indices': self.fixed_dofs_indices
        }
        base_config = super(My_CopulaVAE_Betamarg_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    




