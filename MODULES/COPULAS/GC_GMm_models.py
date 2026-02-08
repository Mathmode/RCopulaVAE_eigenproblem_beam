#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:49:18 2025

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow_probability as tfp
import math as m
tfb = tfp.bijectors
tfd = tfp.distributions

from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Copula_pdf_layer, Fully_connected_dec
from MODULES.COPULAS.GC_GMm_eigen_functions import Solve_eigenproblem, SolveEigenproblemStable, assemble_global_Kmatrices
# from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import Solve_eigenproblem, SolveEigenproblemStable, assemble_global_Kmatrices


@tf.function(jit_compile = True)
def calculate_MAC(modes_true, modes_pred):
    # TODO explain the function operations and translate to Keras
    modes_true_transp = tf.einsum('BCM -> BMC', modes_true)
    modes_pred_transp  = tf.einsum('BCM -> BMC', modes_pred)
        
    MAC_numer = tf.math.square(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
    MAC_denom  = tf.multiply(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_true), tf.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred))
    MAC = tf.divide(MAC_numer, MAC_denom)
    #MAC dimension is (Batch_size, N_modes)
    return MAC



class ForwardModel(K.Model):
    def __init__(self, input_dim_decoder, output_dim, num_mixtures):
        super(ForwardModel, self).__init__()
        self.num_mixtures = num_mixtures
        self.FC_decoder = Fully_connected_dec(input_dim_decoder, output_dim)

    def call(self, inputs):
        reconstructed_inputs = self.FC_decoder(inputs)
        return reconstructed_inputs
    
    def custom_loss(self, y_true, y_pred):
        loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred), axis=None)
        return loss
    
    def get_config(self):
        config = {
            'input_dim_decoder': self.FC_decoder.input_dim_decoder,
            'output_dim': self.FC_decoder.output_dim,
            'num_mixtures': self.num_mixtures
        }
        base_config = super(ForwardModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Inverse_Copula_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, **kwargs):
        super(Inverse_Copula_Model, self).__init__()
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.FC_encoder = Fully_connected_enc_GC(input_dim_encoder, n_dims, num_gaussians)

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
        self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
        self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
        self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])

        Copula_parameters = self.FC_encoder(self.modal_data)
        n_correlation_factors = self.n_dims*(self.n_dims-1)//2

        means, scales, weight_vals, offdiag_elems, diag_elems = tf.split(Copula_parameters, [self.n_dims * self.num_gaussians,
                                                        self.n_dims*self.num_gaussians,
                                                        self.n_dims*self.num_gaussians,
                                                        n_correlation_factors,
                                                        self.n_dims
                                                        ], axis=-1)
        

        # RESHAPE TO SEPARATE by num of components (Gaussians in the mixture for each variable) and num_dims (number of variables)
        means = tf.reshape(means, (-1,self.num_gaussians, self.n_dims))
        scales = tf.reshape(scales, (-1, self.num_gaussians, self.n_dims))
        weight_vals = tf.reshape(weight_vals, (-1, self.num_gaussians, self.n_dims))
        weight_vals = tf.nn.softmax(weight_vals, axis = 1) # apply the weight normalization to the weights of each dimension (contaning n_gaussians)
        
        
        return means, scales, weight_vals, offdiag_elems, diag_elems
    
    def get_config(self):
        config = {
            'n_dims': self.n_dims,
            'num_gaussians': self.num_gaussians,
            'num_samples': self.num_samples
        }
        base_config = super(Inverse_Copula_Model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   

# @tf.function(jit_compile = True)
class My_CopulaVAE_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, n_dims, num_gaussians, num_samples, beta, **kwargs): #We add the bayesian properties (gaussians, dimensions of the latent and samples, selected_features(in case you want to work with only freqs) beta, full_cov, s_lb, s_ub)
        super(My_CopulaVAE_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.Ke_matrices = Ke_matrices # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
        self.Mfree = Mfree
        self.L_inv = L_inv
        self.Encoder_model = Inverse_Copula_Model(input_dim, n_dims, num_gaussians, num_samples)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
        self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples)

        # Helper for Logit-Normal Transformation
        # We assign it to self.bijector. 
        # Note: If loading weights or restoring model, ensure this init is called.
        # self.bijector = IntervalBijector(low=0.05, high=1.0)
        self.epsi = epsi #wight factor for the regularization term in the loss 
        self.num_gaussians = num_gaussians
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.beta = beta
            
    def call(self, inputs):
        # Original shapes before flattening: 
        #freq_data size: (Batch_size, n_modes)
        # vert_modes_data size: (Batch_size, free displ. coordinates, n_modes)
        # rot_modes_data size: (Batch_size, free rot. coordinates, n_modes)
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        self.means_ext = tf.repeat(self.means[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_means = tf.reshape(self.means_ext, [-1,self.num_gaussians, self.n_dims]) # current shape: (batch_size*num_samples, num_gaussians, n_dims)
        self.scales_ext = tf.repeat(self.scales[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_scales = tf.reshape(self.scales_ext, [-1,self.num_gaussians, self.n_dims])  # current shape: (batch_size*num_samples, num_gaussians, n_dims)
        self.weights_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_weight_vals = tf.reshape(self.weights_ext, [-1,self.num_gaussians, self.n_dims])  # current shape: (batch_size*num_samples, num_gaussians, n_dims)

        # 2. SAMPLING: Generate samples in Unbounded Space 'z'
        # Copula_sampling_layer calls 'gaussian_marginal_samples' (updated to be Normal, not Truncated)
        inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
        self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
    
        self.LT_matrices_ext =  tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
        self.reshaped_LT_matrices = tf.reshape(self.LT_matrices_ext, [-1,self.n_dims, self.n_dims])
        
        # 4. DECODER: Solve Physics using Physical Alphas
        # Use vectorized assembly (GPU Optimized)
        Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(self.reshaped_alpha_samples, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements,  self.num_samples) 
    
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        self.pred_freqs = tf.abs(self.pred_freqs) 
        
        # Return physical alphas (useful for plotting/debugging)
        return self.reshaped_alpha_samples
    
    # def Freqs_loss(self, y_true, y_pred):
    #     """
    #     Robust Sorted Frequency Loss.
    #     Ignores Shape Matching logic (which causes the '80.0' error) and 
    #     strictly forces the predicted frequency spectrum to match the target spectrum.
    #     """
    #     # 1. Flatten True Frequencies into a single list per batch
    #     # We assume self.freq_data is (Batch, 10) or similar. 
    #     # If it is split (Rot/Vert), concat them first.
    #     # (Assuming self.freq_data contains all frequencies)
    #     true_freqs = self.freq_data 
        
    #     # 2. Sort True Frequencies (Small -> Large)
    #     # This ensures we compare the 1st mode to the 1st mode, etc.
    #     true_freqs_sorted = tf.sort(true_freqs, axis=1)
        
    #     # 3. Get Pred Frequencies
    #     # Eigh outputs are ALWAYS sorted, so we don't strictly need to sort again,
    #     # but it's safe to do so.
    #     pred_freqs_sorted = self.pred_freqs
        
    #     # 4. Compute Loss (Log Squared Error)
    #     # We use a tiny clamp (1e-4) just to prevent NaN if the model predicts 0.0
    #     safe_true = tf.maximum(true_freqs_sorted, 1e-4)
    #     safe_pred = tf.maximum(pred_freqs_sorted, 1e-4)
        
    #     Freqs_sq_error = tf.square(tf.math.log(safe_true) - tf.math.log(safe_pred))
        
    #     Loss_freqs = tf.math.reduce_mean(Freqs_sq_error)
        
    #     return Loss_freqs
    
    
    
    def Freqs_loss(self, y_true, y_pred):
        true_freqs = self.freq_data
        true_freqs = tf.repeat(true_freqs[:,:,tf.newaxis], self.num_samples, axis = 2)
        true_freqs = tf.reshape(tf.transpose(true_freqs, perm = [0,2,1]), [-1,y_true.shape[1]])
        pred_freqs = self.pred_freqs
        Freqs_sq_error = tf.square(tf.math.log(true_freqs) - tf.math.log(pred_freqs))
        Loss_freqs = tf.math.reduce_mean(Freqs_sq_error, axis = None)
        return Loss_freqs
    
    # def MAC_modes_loss(self, y_true, y_pred):
    #     True_rotmodes, True_vertmodes  = self.rot_modes_data, self.vert_modes_data
        
    #     true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis = 3)
    #     true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm = [0,3,1,2]), [-1,True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
    #     true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis = 3)
    #     true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm = [0,3,1,2]), [-1,True_vertmodes.shape[1], True_vertmodes.shape[2]])
        
    #     pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
        
    #     Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) 
    #     Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) 
    #     MACs = tf.concat([Rot_MACs, Vert_MACs], axis=1)
    #     neg_MACs = tf.square(1 - MACs) ## I use tfsquare because in ELBO eq it should be the discrepancy times the discrepancy. 
    #     Loss_MAC = tf.math.reduce_mean(neg_MACs, axis  = None)
        
    #     return Loss_MAC 
    

    def compute_best_match_loss(self, true_modes, pred_modes):
        """
        Helper: Calculates the 'Best Match' MAC loss.
        Instead of comparing index-to-index (0vs0, 1vs1), this finds the 
        best matching predicted mode for every true mode.
        
        Args:
            true_modes: (Batch, N_true, D)
            pred_modes: (Batch, N_pred, D)
        """
        # 1. Normalize vectors (L2 norm) to prepare for Cosine Similarity
        # axis=2 is the DOF dimension
        true_norm = tf.math.l2_normalize(true_modes, axis=2)
        pred_norm = tf.math.l2_normalize(pred_modes, axis=2)

        # 2. Compute the Gram Matrix (Cosine Similarity between ALL pairs)
        # Result shape: (Batch, N_true, N_pred)
        # Entry [b, i, j] is the similarity between True_Mode[i] and Pred_Mode[j]
        gram_matrix = tf.matmul(true_norm, pred_norm, transpose_b=True)
        
        # 3. Square it to get MAC values (Correlation^2)
        # This fixes the "Sign Flip" issue automatically (-1 becomes 1)
        mac_matrix = tf.square(gram_matrix)

        # 4. Find the Best Match
        # For every TRUE mode (row), what is the max correlation in the PREDICTED cols?
        best_matches = tf.reduce_max(mac_matrix, axis=2) # Shape: (Batch, N_true)
        
        # 5. Compute Loss
        # We want best_matches to be close to 1.0. 
        # Loss = Mean( (1 - Best_MAC)^2 )
        loss = tf.reduce_mean(tf.square(1.0 - best_matches))
        
        return loss

    def MAC_modes_loss(self, y_true, y_pred):
        """
        Computes a loss that allows modes to change order (swap slots)
        without penalizing the model.
        """
        # --- 1. Data Preparation (Same as your original) ---
        True_rotmodes, True_vertmodes = self.rot_modes_data, self.vert_modes_data
        
        # Expand and reshape True Rotational Modes
        true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis=3)
        true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm=[0,3,1,2]), 
                                   [-1, True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
        # Expand and reshape True Vertical Modes
        true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis=3)
        true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm=[0,3,1,2]), 
                                    [-1, True_vertmodes.shape[1], True_vertmodes.shape[2]])
        
        pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
        
        # --- 2. Calculate Robust Loss (The Fix) ---
        # We calculate the loss for Rot and Vert separately using the "Best Match" logic
        # This function generates a matrix of shape (Batch, N_true, N_pred)

        loss_rot = self.compute_best_match_loss(true_rotmodes, pred_rotmodes)
        loss_vert = self.compute_best_match_loss(true_vertmodes, pred_vertmodes)
        
        # Combine them (Average)
        Loss_MAC = 0.5 * (loss_rot + loss_vert)
        
        return Loss_MAC
    
    
    def Alpha_regularizer(self, y_true, y_pred):
        # Operates on Physical Alphas (y_pred)
        min_value = tf.reduce_min(y_pred, axis=-1, keepdims=False)
        Regularizer = tf.math.reduce_mean(tf.math.reduce_sum(y_pred,axis = -1)-min_value)/(y_pred.shape[-1])-1
        return self.epsi*Regularizer
    
    
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
                
        # Convert uniform samples to standard normal
        normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(self.reshaped_copula_samples)
        mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)

        log_prob_joint_normal = tf.math.log(mvn.prob(normal_samples)+1e-07)
        log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(normal_samples)+1e-07), axis=-1)      
        
    
        return log_prob_joint_normal - log_prob_marginals_standard
    
    
    
    def Marginal_pdf_logprob(self,y_true, y_pred):
        '''
        #This one is used for Gaussian marginal directly (known inverse CDF)
        Here we calculate the log probability of the samples over the marginal distributions 
        We have n_dims marginals to consider. 
        The inputs are taken from the self, and include the marginal_samples, and the marginals parameters 
        The output will be the log_probs with shape [batch_size, num_samples]
        '''
        # Taking into account that here we have only one gaussian, we can neglect the dimension of num_gaussians as it is simply 1. 
        gaussian_marginals = tfd.TruncatedNormal(loc=self.reshaped_means[:,0,:], scale=self.reshaped_scales[:,0,:], low=-0.0001, high=1.0001)
        # This is producing one logprob value for each dimension 
        log_prob_marginals = tf.math.log(gaussian_marginals.prob(self.reshaped_alpha_samples)+ 1e-07)  
        # According to the equation (see paper), log(SUM) = SUM(logs): 
        log_prob_marginal = tf.math.reduce_sum(log_prob_marginals,axis = -1)    # to sum in the axis of n_dims        
        return log_prob_marginal
    
    
    def Joint_copula_dens_term(self,y_true, y_pred):
        Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
        Marginal_logprob_term = self.Marginal_pdf_logprob(y_true,y_pred)   
        
        joint_copula_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term, axis = None)
        joint_copula_logprob = tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (joint_copula_logprob)

        return joint_copula_logprob
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        """
        Total Loss Function.
        """
        # 1. Copula NLL (Minimize)
        Joint_copula_loss = self.Joint_copula_dens_term(y_true, y_pred)
        
        # 2. Frequencies Loss (MSE or similar)
        Loss_freqs = self.Freqs_loss(y_true, y_pred)
        
        # 3. MAC Loss (Permutation Invariant)
        Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        
        # 4. Regularizer (e.g. KL Divergence)
        Regularizer = self.Alpha_regularizer(y_true, y_pred)
        
        # --- 5. Total Sum ---
        # NOTE: Ensure these magnitudes are balanced. 
        # MAC is usually [0, 1]. NLL can be large. Freqs in Hz can be large.
        # You might need weights like: 10.0 * Loss_MAC + 0.1 * Loss_freqs...
        
        ELBO_loss = Loss_MAC + Loss_freqs - Regularizer + Joint_copula_loss
        
        return ELBO_loss
    
    # def ELBO_Copula_loss(self, y_true, y_pred):
    #     # Joint_copula_loss is now NLL (positive value to minimize)
    #     Joint_copula_loss  = self.Joint_copula_dens_term(y_true, y_pred)
    #     Loss_freqs = self.Freqs_loss(y_true,y_pred)
    #     Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
    #     Regularizer = self.Alpha_regularizer(y_true, y_pred)
                
    #     ELBO_loss = Loss_MAC + Loss_freqs - Regularizer + Joint_copula_loss
    #     return ELBO_loss
    
    

    
    
    # --------------------------------------------------------------------------
    # WARPED GAUSSIAN PROBABILITY TERMS
    # --------------------------------------------------------------------------
    
    # def Marginal_pdf_logprob(self, y_true, y_pred):
    #     """
    #     Calculates log P(alpha_samples) using the Change of Variables formula.
    #     log P(alpha) = log P(z) - log |det J|
    #     Evaluates the probability of the SAMPLED points (z).
    #     """
    #     # 1. Use the Latent Z Samples generated in call()
    #     # These are unbounded samples (-inf, inf)
    #     z_samples = self.reshaped_marginal_samples_z

    #     # 2. Calculate log P(z) using Gaussian Mixture Marginals
    #     # Since 'z' space is unbounded, we use standard Normal distributions
    #     gm = tfd.MixtureSameFamily(
    #         mixture_distribution=tfd.Categorical(probs=self.reshaped_weight_vals),
    #         components_distribution=tfd.Normal(loc=self.reshaped_means, scale=self.reshaped_scales)
    #     )
        
    #     # Log Probability of z samples under the predicted Gaussian parameters
    #     log_prob_z = tf.math.log(gm.prob(z_samples) + 1e-9)
        
    #     # 3. Calculate Jacobian Correction: log |d_alpha / d_z|
    #     log_det_jac = self.bijector.log_det_jacobian(z_samples)
        
    #     # 4. Apply Change of Variables: log P(alpha) = log P(z) - log |det J|
    #     log_prob_alpha_marginal = log_prob_z - log_det_jac
        
    #     # Sum over dimensions to get total log prob per sample vector
    #     return tf.reduce_sum(log_prob_alpha_marginal, axis=-1)

    # def Copula_pdf_logprob(self, y_true, y_pred):
    #     """
    #     Calculates the Copula density term for the SAMPLES (z).
    #     """
    #     # 1. Use the Latent Z Samples
    #     z_samples = self.reshaped_marginal_samples_z

    #     # 2. Standardize Z to get inputs for the Copula
    #     # z_std = (z - mu) / sigma
    #     # Assuming Unimodal Gaussian for standardization (simplification for GMM)
    #     z_std = (z_samples - self.reshaped_means[:,0,:]) / (self.reshaped_scales[:,0,:] + 1e-9)
        
    #     # 3. Calculate Copula Log Prob
    #     mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)
        
    #     log_prob_joint_normal = tf.math.log(mvn.prob(z_std) + 1e-7)
    #     log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(z_std) + 1e-7), axis=-1)
        
    #     return log_prob_joint_normal - log_prob_marginals_standard
    
    # def Joint_copula_dens_term(self, y_true, y_pred):
    #     Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
    #     Marginal_logprob_term = self.Marginal_pdf_logprob(y_true, y_pred)
        
    #     # Total Log Likelihood of the SAMPLES
    #     total_log_prob = Copula_density_term + Marginal_logprob_term
        
    #     # We perform Negation here because the optimizer MINIMIZES loss.
    #     # We want to MAXIMIZE likelihood.
    #     # Therefore, we return Negative Log Likelihood.
    #     joint_copula_logprob = tf.math.reduce_mean(total_log_prob, axis = None)
        
    #     # Beta weighting
    #     weighted_nll = -1.0 * tf.math.square(tf.cast(self.beta, dtype=tf.float32)) * joint_copula_logprob
    #     return weighted_nll
    
    # def ELBO_Copula_loss(self, y_true, y_pred):
    #     # Joint_copula_loss is now NLL (positive value to minimize)
    #     Joint_copula_loss  = self.Joint_copula_dens_term(y_true, y_pred)
    #     Loss_freqs = self.Freqs_loss(y_true,y_pred)
    #     Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
    #     Regularizer = self.Alpha_regularizer(y_true, y_pred)
                
    #     ELBO_loss = Loss_MAC + Loss_freqs - Regularizer + Joint_copula_loss
    #     return ELBO_loss
    
    def get_config(self):
        config = {
            'num_dofs': self.num_dofs,
            'M_free': self.Mfree,
        }
        base_config = super(My_CopulaVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    #     ## CREATE SAMPLES FROM THE DISTRIBUTIONAL LEARNING MODEL USING COPULA SAMPLING LAYER
    #     # inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
    #     inputs_to_sampling = {
    #     'means': self.means,
    #     'scales': self.scales,
    #     'weight_vals': self.weight_vals,
    #     'offdiag_elems': self.offdiag_elems,
    #     'diag_elems': self.diag_elems
    # }
        
    #     self.marginal_samples, self.copula_samples, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
    #     self.reshaped_marginal_samples = tf.reshape(self.marginal_samples, (-1, self.n_dims)) #to take Shape (Batch_size, n_dims)
    #     self.reshaped_copula_samples  = tf.reshape(self.copula_samples, (-1, self.n_dims))

    #     self.LT_matrices_ext =  tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
    #     self.reshaped_LT_matrices = tf.reshape(self.LT_matrices_ext, [-1,self.n_dims, self.n_dims])
        
        
    #     reshaped_alpha_factors = self.reshaped_marginal_samples #HEre the latent space represents the alpha factors or reduction factors affecting the stiffness matrix
    #     # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
    #     Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(reshaped_alpha_factors, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
    #     # Bear in mind that B here is B*H but we keep the same notation for that first dimension. 
    #     # So now Ke_matrices_dam must have shape (B*H, Elements, K,Q)
    #     # KQ indicate the dimension of the element matrix that here is 2x2 since we are in lienar elasticity 
        
    #     # # Assemble the element matrices to build the global matrix        
    #     Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements,  self.num_samples) # the shape is (Batch_size, n_free, n_free)

    #     # Then we enter the eigensolver (forward) function with this list to produce the eigenfrequencies
    #     self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
    #     self.pred_freqs = tf.abs(self.pred_freqs) #to enforce them to be positive***
    #     #The output of this function is the output of the inverse, i.e., the estimated damage condition described by the alpha factors.
    #     #They have shape (B*H, D) 
    #     return  reshaped_alpha_factors
        
    # def Freqs_loss(self, y_true, y_pred):
    #     true_freqs = self.freq_data
    #     true_freqs = tf.repeat(true_freqs[:,:,tf.newaxis], self.num_samples, axis = 2)
    #     #Reshape to accommodate for further steps (final shape: (batch_size*N, n_features)) that will be seen as (None, n_features)
    #     true_freqs = tf.reshape(tf.transpose(true_freqs, perm = [0,2,1]), [-1,y_true.shape[1]])
        
    #     pred_freqs = self.pred_freqs


    #     Freqs_sq_error = tf.square(tf.math.log(true_freqs) - tf.math.log(pred_freqs))
    #     Loss_freqs = tf.math.reduce_mean(Freqs_sq_error, axis = None)
    #     return Loss_freqs
    

    # def MAC_modes_loss(self, y_true, y_pred):
    #     True_rotmodes, True_vertmodes  = self.rot_modes_data, self.vert_modes_data
        
    #     true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis = 3)
    #     true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm = [0,3,1,2]), [-1,True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
    #     true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis = 3)
    #     true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm = [0,3,1,2]), [-1,True_vertmodes.shape[1], True_vertmodes.shape[2]])
        
        
        
    #     pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
    #     #Calculate the MACs
    #     Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
    #     Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
    #     # MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
    #     MACs = tf.concat([Rot_MACs, Vert_MACs], axis=1)
    #     neg_MACs = 1 - MACs
    #     Loss_MAC = tf.math.reduce_mean(neg_MACs, axis  = None)
        
    #     return Loss_MAC 
    
    # def Alpha_regularizer(self, y_true, y_pred):
    #     min_value = tf.reduce_min(y_pred, axis=-1, keepdims=False)
    #     Regularizer = tf.math.reduce_mean( tf.math.reduce_sum(y_pred,axis = -1)-min_value)/(y_pred.shape[-1])-1
    #     return self.epsi*Regularizer
     
    
    # def Copula_pdf_logprob(self,y_true, ypred):
    #     """
    #     Compute the log-likelihood of the joint PDF defined by a Gaussian copula and Gaussian mixture marginals.
        
    #     Args:
    #         locs: Tensor of shape [n_modes, n_dims], means of the Gaussian mixture components.
    #         scales: Tensor of shape [n_modes, n_dims], standard deviations of the Gaussian mixture components.
    #         weights: Tensor of shape [n_modes, n_dims], weights of the Gaussian mixture components.
    #         correlation_matrix: Tensor of shape [n_dims, n_dims], correlation matrix of the Gaussian copula.
    #         samples: Tensor of shape [n_samples, n_dims], the observed data.
    
    #     Returns:
    #         log_likelihood: Tensor of shape [], the log-likelihood of the joint PDF.
    #     """
                
    #     # Convert uniform samples to standard normal
    #     normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(self.reshaped_copula_samples)
    #     mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)

    #     log_prob_joint_normal = tf.math.log(mvn.prob(normal_samples)+1e-07)
    #     log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(normal_samples)+1e-07), axis=-1)      
        
    
    #     return log_prob_joint_normal - log_prob_marginals_standard

    # def Marginal_pdf_logprob(self, y_true, y_pred):
    #     """
    #     Calculates log P(alpha_samples) using the Change of Variables formula.
    #     log P(alpha) = log P(z) - log |det J|

    #     Note: This evaluates the probability of the SAMPLED points, not y_true.
    #     """
        
    #     # 1. Use the Latent Z Samples generated in call()
    #     # These are unbounded samples (-inf, inf)
    #     z_samples = self.reshaped_marginal_samples

    #     # 2. Calculate log P(z) using Gaussian Mixture Marginals
    #     # Since 'z' space is unbounded, we use standard Normal distributions (not Truncated).
    #     gm = tfd.MixtureSameFamily(
    #         mixture_distribution=tfd.Categorical(probs=self.reshaped_weight_vals),
    #         components_distribution=tfd.Normal(loc=self.reshaped_means, scale=self.reshaped_scales)
    #     )
        
    #     # Log Probability of z samples under the predicted Gaussian parameters
    #     # This tells us how likely the samples z are under the distribution N(mu, sigma)
    #     log_prob_z = tf.math.log(gm.prob(z_samples) + 1e-9)
        
    #     # 3. Calculate Jacobian Correction: log |d_alpha / d_z|
    #     # We need this to convert the density from Z-space to Alpha-space
    #     log_det_jac = self.bijector.log_det_jacobian(z_samples)
        
    #     # 4. Apply Change of Variables
    #     # log P(alpha) = log P(z) - log |det J|
    #     log_prob_alpha_marginal = log_prob_z - log_det_jac
        
    #     # Sum over dimensions to get total log prob per sample vector
    #     return tf.reduce_sum(log_prob_alpha_marginal, axis=-1)
    
    
    # def Marginal_pdf_logprob(self,y_true, y_pred):
    #     '''
    #     #This one is used for Gaussian marginal directly (known inverse CDF)
    #     Here we calculate the log probability of the samples over the marginal distributions 
    #     We have n_dims marginals to consider. 
    #     The inputs are taken from the self, and include the marginal_samples, and the marginals parameters 
    #     The output will be the log_probs with shape [batch_size, num_samples]
    #     '''
    #     # Taking into account that here we have only one gaussian, we can neglect the dimension of num_gaussians as it is simply 1. 
    #     gaussian_marginals = tfd.TruncatedNormal(loc=self.reshaped_means[:,0,:], scale=self.reshaped_scales[:,0,:], low=-0.0001, high=1.0001)
    #     # This is producing one logprob value for each dimension 
    #     log_prob_marginals = tf.math.log(gaussian_marginals.prob(self.reshaped_marginal_samples)+ 1e-06)  
    #     # According to the equation (see paper), log(SUM) = SUM(logs): 
    #     log_prob_marginal = tf.math.reduce_sum(log_prob_marginals,axis = -1)    # to sum in the axis of n_dims        
    #     return log_prob_marginal
    

    
    
    # def Joint_copula_dens_term(self,y_true, y_pred):
    #     Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
    #     Marginal_logprob_term = self.Marginal_pdf_logprob(y_true,y_pred)   
        
    #     joint_copula_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term, axis = None)
    #     joint_copula_logprob = tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (joint_copula_logprob)
    #     return joint_copula_logprob
    
    
    
    # def ELBO_Copula_loss(self, y_true, y_pred):
    #     Joint_copula_logprob  = self.Joint_copula_dens_term(y_true, y_pred)
    #     Loss_freqs = self.Freqs_loss(y_true,y_pred)
    #     Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
    #     Regularizer = self.Alpha_regularizer(y_true, y_pred)
                
    #     ELBO_loss = Loss_MAC + Loss_freqs - Regularizer + Joint_copula_logprob
    #     return ELBO_loss
    
    
    
    # ## use this custom loss when you want to debug the second part of the code (the eigenvalue problem)
    # def custom_loss(self,y_true, y_pred):
    #     loss = tf.math.reduce_mean(tf.square(y_true-y_pred), axis = None)    
    #     return loss
    

    # def get_config(self):
    #     config = {
    #         'num_dofs': self.num_dofs,
    #         'M_free': self.Mfree,
    #         # 'eigensolver': self.Eigen_solver
    #     }
    #     base_config = super(My_CopulaVAE_withEigen, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def from_config(cls, config):
        
    #     # config['Solve_eigenproblem'] = tf.keras.utils.deserialize_keras_object(config['Solve_eigenproblem'])
    #     return cls(**config)
    



















###############################################3
# class My_Copula_VAE(tf.keras.Model):
#     def __init__(self, input_dim_encoder, input_dim_decoder, output_dim, n_dims, num_gaussians, num_samples, model_forward, selected_features, beta):
#         super(My_Copula_VAE, self).__init__()
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples
#         self.selected_features = selected_features
#         self.beta = beta
#         self.Encoder_model = Inverse_Copula_Model(input_dim_encoder, n_dims, num_gaussians, num_samples)
#         self.Decoder_model = Forward_Model(input_dim_encoder, n_dims, num_gaussians, num_samples, model_forward)
#         self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples)
#         self.selected_features = selected_features
        
#     def ColumnSliceLayer(self, inputs):
#         selected_features = tf.constant(self.selected_features
# , dtype=tf.int32)
#         outputs = tf.gather(inputs, selected_features, axis=1)
#         return outputs
    
    
#     def call(self, inputs):
#         ## EVALUATE THE INVERSE TO ESTIMATE THE DISTRIBUTIONAL MODEL (GAUSSIAN COPULA WITH GM MARGINALS) PARAMETERS
#         [u_t, r, p] = inputs
#         self.p = p
#         u = self.ColumnSliceLayer(u_t) #this function takes the selected features
#         inputs_to_inverse = [u,r]
#         self.means, self.scales, self.weight_vals, self.rhos = self.Encoder_model(inputs_to_inverse)
        
#         ## CREATE SAMPLES FROM THE DISTRIBUTIONAL LEARNING MODEL USING COPULA SAMPLING LAYER
#         inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.rhos]
#         self.marginal_samples, self.copula_samples,  self.mvn_samples, self.mvn_models, self.correlation_matrices, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
#         self.reshaped_marginal_samples = tf.reshape(self.marginal_samples, (-1, self.n_dims))
#         ## EVALUATE THE FORWARD FED WITH THE OBTAINED MARGINAL SAMPLES (WITH THE COPULA CORRELATION EFFECT)
#         inputs_to_forward = [r, self.reshaped_marginal_samples]
#         reconstructed_inputs = self.Decoder_model(inputs_to_forward)
#         return reconstructed_inputs
    
#     def Conditional_likelihood_term(self, y_true, y_pred):
#         #Termino que mide el Data misfit en la Kullback Leibler loss. Es equivalente Data_loss pero incluye un termino asociado al ruido mediante el parametro Beta 
#         #Si solo minimizamos este término, estaríamos minimizando Data_loss. Debería salir lo mismo .
#         # We now compute eq. 9 in the paper: 
#         #First select the features involved in the problem (roll, or rollpitchyaw...)
#         y_true = self.ColumnSliceLayer(y_true)
#         y_pred = self.ColumnSliceLayer(y_pred)
#         #Repeat as many times as samples drawn from the PDF 
#         Ytrue = tf.repeat(y_true[:,:,tf.newaxis], self.num_samples, axis = 2)
#         #Reshape to accommodate for further steps (final shape: (batch_size*N, n_features)) that will be seen as (None, n_features)
#         Ytrue = tf.reshape(tf.transpose(Ytrue, perm = [0,2,1]), [-1,y_true.shape[1]])
        
#         #Calculate the discrepancy or error between the predicitons and the ground truth measurements (err.shape = (None, n_features))
#         err = Ytrue - y_pred
#         # Obtain the covariance matrix Gamma as the square of Beta times the prediction
#         Gamma = tf.math.square(tf.cast(self.beta, dtype=tf.float32)* y_pred)
#         # Compute the inverse of the diagonal matrix (inverse of the elements in the diagonal)
#         inv_Gamma = 1/Gamma
#         # Calculate the Element-wise product of inv_Gamma(None,30) and the error (None, n_features), resulting in (None, n_features) as final shape
#         y1 = inv_Gamma*err
#         # Compute the dot product of the transpose of the error tiems the previous result        
#         # Since both "err" and "G * err" are vectors, we need to sum the element-wise product
#         #Compute the dot product: Since both are vectors, this is equivalent to summing the element-wise product of err and y1 = invGamma*err.
#         likelihood = tf.reduce_sum(err * y1, axis=1, keepdims=True)  # Shape: (batch_size, 1)
#         LKLHD_Loss = 0.5* tf.math.square(tf.cast(self.beta, dtype = tf.float32))*likelihood
#         # tf.print(LKLHD_Loss)
#         return LKLHD_Loss  
    
#     def Copula_pdf_logprob(self, y_true, y_pred):       
#         """
#         Compute the log-likelihood of the joint PDF defined by a Gaussian copula and Gaussian mixture marginals.
        
#         Args:
#             locs: Tensor of shape [n_modes, n_dims], means of the Gaussian mixture components.
#             scales: Tensor of shape [n_modes, n_dims], standard deviations of the Gaussian mixture components.
#             weights: Tensor of shape [n_modes, n_dims], weights of the Gaussian mixture components.
#             correlation_matrix: Tensor of shape [n_dims, n_dims], correlation matrix of the Gaussian copula.
#             samples: Tensor of shape [n_samples, n_dims], the observed data.
    
#         Returns:
#             log_likelihood: Tensor of shape [], the log-likelihood of the joint PDF.
#         """
#         def log_prob_copula(copula_samples, correlation_matrix, LT_matrix):
#             """
#             Compute the log-probability of the Gaussian copula.
        
#             Args:
#                 copula_samples: Tensor of shape [n_samples, n_dims], copula samples in [0,1].
#                 correlation_matrix: Tensor of shape [n_dims, n_dims], correlation matrix.
        
#             Returns:
#                 log_prob_copula: Tensor of shape [n_samples], log probabilities of the copula.
#             """
#             n_dims = correlation_matrix.shape[-1]
            
#             # Convert uniform samples to standard normal
#             normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(copula_samples)
            
#             # Multivariate normal distribution with given correlation
#             # mvn = tfd.MultivariateNormalFullCovariance(
#             #     loc=tf.zeros(n_dims), covariance_matrix=correlation_matrix
#             # )
#             mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(n_dims), scale_tril = LT_matrix)

            
#             log_prob_joint_normal = tf.math.log(mvn.prob(normal_samples)+1e-07)
#             log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(normal_samples)+1e-07), axis=-1)
            
#             return log_prob_joint_normal - log_prob_marginals_standard
            


#         copula_log_probs = tf.vectorized_map(
#                 lambda args: log_prob_copula(*args),
#                 (self.copula_samples, self.correlation_matrices, self.LT_matrices),)
        
        
#         return tf.reshape(copula_log_probs, (-1, 1))


            
#     def Marginal_pdf_logprob(self, y_true, y_pred):
#         '''
#         Here we calculate the log probability of the samples over the marginal distirbutions 
#         We have n_dims marginals to consider. 
#         The inputs are taken from the self, and include the marginal_samples, and the marginals parameters 
#         The output will be the log_probs with shape [batch_size, num_samples]
#         '''
#         def log_prob_marginals(samples, locs, scales, weights):
#             """
#             Compute the log-probability of the marginal distributions.
        
#             Args:
#                 samples: Tensor of shape [n_samples, n_dims], the marginal samples.
#                 locs: Tensor of shape [n_gaussians, n_dims], means for each Gaussian component.
#                 scales: Tensor of shape [n_gaussians, n_dims], standard deviations.
#                 weights: Tensor of shape [n_gaussians, n_dims], mixture weights.
        
#             Returns:
#                 log_prob_marg: Tensor of shape [n_samples, n_dims], log probabilities of the marginals.
#             """
#             n_dims = locs.shape[-1]
#             log_prob_marginals = []
        
#             for i in range(n_dims):
#                 gm = tfd.MixtureSameFamily(
#                     mixture_distribution=tfd.Categorical(
#                         probs=weights[:, i] / (tf.reduce_sum(weights[:, i]) + 1e-10)
#                     ),
#                     components_distribution=tfd.TruncatedNormal(
#                         loc=locs[:, i], 
#                         scale=tf.maximum(scales[:, i], 1e-10),  # Avoid zero or negative scales
#                         low=-1e-5,  # Ensure within the [0,1] range
#                         high=1.001
#                     )
#                 )
        
#                 # Ensure numerical stability in log computation
#                 log_prob = tf.math.log(gm.prob(samples[:, i]) +1e-10)
#                 log_prob_marginals.append(log_prob)
        
#             return tf.stack(log_prob_marginals, axis=-1)



#         #calculate the marginal log probs and sum up for n_dims so that you end up with [batch_size, num_sammples] values. 
#         marg_log_probs = tf.math.reduce_sum(tf.vectorized_map(
#                 lambda args: log_prob_marginals(*args),
#                 (self.marginal_samples, self.means, self.scales, self.weight_vals),), axis = 2)
        
#         return tf.reshape(marg_log_probs, (-1, 1))  
      
#     def Joint_copula_dens_term(self,y_true, y_pred):
#         Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
#         Marginal_logprob_term = self.Marginal_pdf_logprob(y_true,y_pred)         

#         joint_copula_logprob = tf.math.reduce_mean(Copula_density_term +Marginal_logprob_term, axis = None)
#         joint_copula_logprob = tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (joint_copula_logprob)
#         return joint_copula_logprob + 1e-010
        

        
#     def ELBO_Copula_loss(self, y_true, y_pred):
#         # The first term is the log probability of the COPULA whose parameters we approximate with the inverse
#         # The second terms refers to the likelihood, and is related to the datamisfit, with the noise relationship between the true propoerties and the measured observations: Y = F(X) + e, siendo e el ruido, X las properties de daño e Y las mediciones
#         Conditional_lklhd_term = self.Conditional_likelihood_term(y_true, y_pred)
#         joint_copula_logprob  = self.Joint_copula_dens_term(y_true, y_pred)
        
#         tf.debugging.assert_all_finite(Conditional_lklhd_term, "Conditional Likelihood contains NaN or Inf")
#         tf.debugging.assert_all_finite(joint_copula_logprob, "joint_copula_logprob contains NaN or Inf")


        
#         ELBO_loss = Conditional_lklhd_term + joint_copula_logprob
#         return ELBO_loss
    
#     def calculate_average_log_prob(self, u_test, r_test, p_test):
#         """Calculates the average log probability on test data.
    
#         Args:
#             u_test: Test data for 'u' (observed variables).
#             r_test: Test data for 'r' (operating conditions).
#             p_test: Test data for 'p' (additional parameters, if any).
    
#         Returns:
#             A tuple: (average_joint_log_prob, average_marginal_log_prob, average_copula_log_prob)
#             Each element is a scalar (float) representing the average log probability.
#         """
    
#         # 1. Prepare Input:  Crucially, we use the *test* data here.
#         inputs = [u_test, r_test, p_test]
    
#         # 2. Get Model Outputs: Call the model to compute necessary parameters.
#         self(inputs)  # This populates self.means, self.scales, etc.
    
#         # 3. Calculate Log Probabilities: Use the pre-computed parameters.
#         joint_log_prob_tensor = self.Joint_copula_dens_term(u_test, None)  # Pass u_test as y_true
#         marginal_log_prob_tensor = self.Marginal_pdf_logprob(u_test, None)
#         copula_log_prob_tensor = self.Copula_pdf_logprob(u_test, None)
    
    
#         # 4. Average over all test samples:
#         average_joint_log_prob = tf.reduce_mean(joint_log_prob_tensor)
#         average_marginal_log_prob = tf.reduce_mean(marginal_log_prob_tensor)
#         average_copula_log_prob = tf.reduce_mean(copula_log_prob_tensor)
    
    
#         return average_joint_log_prob


    
#     def get_config(self):
#         config = {
#             'n_dims': self.n_dims,
#             'num_gaussians': self.num_gaussians,
#             'num_samples': self.num_samples,
#             'model_forward': tf.keras.utils.serialize_keras_object(self.Decoder_model)
#         }
#         base_config = super(My_Copula_VAE, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
