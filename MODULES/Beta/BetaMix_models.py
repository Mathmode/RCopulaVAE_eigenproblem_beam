#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:49:18 2025

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from MODULES.Beta.BetaMix_architectures import Fully_connected_enc_Beta, Fully_connected_dec
from MODULES.Beta.BetaMix_functions import BetaMixtureSamplingLayer
from MODULES.Beta.BetaMix_eigen_functions import Solve_eigenproblem, assemble_global_Kmatrices


# @tf.function(jit_compile = True)
# def calculate_MAC(modes_true, modes_pred):
#     # TODO explain the function operations and translate to Keras
#     modes_true_transp = tf.einsum('BCM -> BMC', modes_true)
#     modes_pred_transp  = tf.einsum('BCM -> BMC', modes_pred)
        
#     MAC_numer = tf.math.square(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
#     MAC_denom  = tf.multiply(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_true), tf.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred))
#     MAC = tf.divide(MAC_numer, MAC_denom)
#     #MAC dimension is (Batch_size, N_modes)
#     return MAC


@tf.function(jit_compile = True)
def calculate_MAC(modes_true, modes_pred):
    """
    Calculates Modal Assurance Criterion with numerical stability protection.
    """
    # modes shape: (Batch, Coords, Modes)
    modes_true_transp = tf.einsum('BCM -> BMC', modes_true)
    modes_pred_transp  = tf.einsum('BCM -> BMC', modes_pred)
        
    MAC_numer = tf.math.square(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
    
    denom_true = tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_true)
    denom_pred = tf.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred)
    
    # FIX 1: Add epsilon to denominator to prevent NaN if a mode vector is zero
    MAC_denom  = tf.multiply(denom_true, denom_pred) + 1e-8
    
    MAC = tf.divide(MAC_numer, MAC_denom)
    
    # FIX 2: Clip MAC to [0, 1] to prevent sqrt(-val) or values > 1 due to float error
    MAC = tf.clip_by_value(MAC, 0.0, 1.0)
    
    return MAC

# @tf.function(jit_compile = True)
# def orient_modes_tf(x):
#     """
#     Orients vectors in a tensor such that the element with the maximum 
#     absolute magnitude is always positive.
    
#     Args:
#         x: Tensor of shape (Batch, Coords, Modes) -> e.g., (5000, 6, 5)
        
#     Returns:
#         Tensor of the same shape with oriented vectors.
#     """
#     # Ensure input is a tensor
#     x = tf.convert_to_tensor(x)
    
#     # 1. Find the index of the max absolute value along the coordinates axis (axis=1)
#     # Output shape: (Batch, Modes)
#     max_indices = tf.argmax(tf.abs(x), axis=1)
    
#     # 2. Create a one-hot mask to extract the actual values at these indices
#     # We specify axis=1 so the 'depth' (6 coords) is inserted at the correct dimension.
#     # Output shape: (Batch, Coords, Modes) matches x
#     mask = tf.one_hot(max_indices, depth=tf.shape(x)[1], axis=1, dtype=x.dtype)
    
#     # 3. Extract the peak values (the values at the max indices)
#     # Element-wise multiply + sum reduces the Coordinate axis, leaving just the peak values.
#     # Output shape: (Batch, Modes)
#     peak_values = tf.reduce_sum(x * mask, axis=1)
    
#     # 4. Determine the sign of these peak values (-1 or 1)
#     signs = tf.math.sign(peak_values)
    
#     # 5. Handle the edge case where the peak is 0 (sign is 0)
#     # If sign is 0, we default it to 1 to leave the vector unchanged.
#     signs = tf.where(tf.equal(signs, 0), tf.ones_like(signs), signs)
    
#     # 6. Reshape for broadcasting: (Batch, Modes) -> (Batch, 1, Modes)
#     # This allows us to multiply the (Batch, 6, Modes) tensor correctly.
#     signs = tf.expand_dims(signs, axis=1)
    
#     # 7. Apply the signs to the original tensor
#     return x * signs

@tf.function(jit_compile = True)
def orient_modes_tf(x):
    """
    Orients vectors in a tensor such that the element with the maximum 
    absolute magnitude is always positive.
    """
    x = tf.convert_to_tensor(x)
    max_indices = tf.argmax(tf.abs(x), axis=1)
    mask = tf.one_hot(max_indices, depth=tf.shape(x)[1], axis=1, dtype=x.dtype)
    peak_values = tf.reduce_sum(x * mask, axis=1)
    signs = tf.math.sign(peak_values)
    signs = tf.where(tf.equal(signs, 0), tf.ones_like(signs), signs)
    signs = tf.expand_dims(signs, axis=1)
    return x * signs


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


class Inverse_Beta_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_components, num_samples, **kwargs):
        super(Inverse_Beta_Model, self).__init__()
        self.n_dims = n_dims
        self.num_components = num_components
        self.num_samples = num_samples
        self.FC_encoder = Fully_connected_enc_Beta(input_dim_encoder, n_dims, num_components)

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
        self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
        self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
        self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])
        
        Betamix_parameters = self.FC_encoder(self.modal_data)
        
        # Split output
        raw_alphas, raw_betas, logits = tf.split(Betamix_parameters, [
            self.n_dims*self.num_components, 
            self.n_dims*self.num_components, 
            self.num_components], axis = -1)
        
        # FIX 6: CRITICAL! Enforce positivity constraints on Alpha and Beta parameters.
        # Neural networks output real numbers (-inf, inf). Beta needs (>0).
        # We use Softplus + epsilon.
        alphas = tf.math.softplus(raw_alphas) + 1e-4
        betas = tf.math.softplus(raw_betas) + 1e-4
        return alphas, betas, logits
    
    def get_config(self):
        config = {
            'n_dims': self.n_dims,
            'num_components': self.num_components,
            'num_samples': self.num_samples
        }
        base_config = super(Inverse_Beta_Model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   

# @tf.function(jit_compile = True)
class My_BetaVAE_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, n_dims, num_components, num_samples, regu_weight, sigma_f, sigma_phi, **kwargs): #We add the bayesian properties (gaussians, dimensions of the latent and samples, selected_features(in case you want to work with only freqs) beta, full_cov, s_lb, s_ub)
        super(My_BetaVAE_withEigen, self).__init__()
        self.n_dims = n_dims
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.num_components = num_components

        self.Ke_matrices = Ke_matrices # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
        self.Mfree = Mfree
        self.L_inv = L_inv
        self.Encoder_model = Inverse_Beta_Model(input_dim, n_dims, num_components, num_samples)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
        
        self.Beta_sampling_layer = BetaMixtureSamplingLayer(n_dims, num_components, temperature = 0.5)

        self.epsi = epsi #wight factor for the regularization term in the loss 
        self.num_samples = num_samples
        self.regu_weight = regu_weight # weight for the regularizign terms (balances the contribution of the reconstruction and the probability terms in the loss)
        self.sigma_f = sigma_f #these are the new hyperparameters that act as weights in the loss (substitute regu_weight)
        self.sigma_phi = sigma_phi
            
    def call(self, inputs):
        # Original shapes before flattening: 
        #freq_data size: (Batch_size, n_modes)
        # vert_modes_data size: (Batch_size, free displ. coordinates, n_modes)
        # rot_modes_data size: (Batch_size, free rot. coordinates, n_modes)
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
        self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
        self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
        self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])
        
        self.alphas, self.betas, self.logits,= self.Encoder_model(inputs)
        self.alphas_ext = tf.repeat(self.alphas[:,tf.newaxis,:], self.num_samples, axis = 1)
        self.reshaped_alphas = tf.reshape(self.alphas_ext, [-1,self.num_components*self.n_dims]) # current shape: (batch_size*num_samples, num_components*n_dims)
        
        self.betas_ext = tf.repeat(self.betas[:,tf.newaxis,:], self.num_samples, axis = 1)
        self.reshaped_betas = tf.reshape(self.betas_ext, [-1, self.num_components*self.n_dims]) # current shape: (batch_size*num_samples, num_components*n_dims)

        self.weights_ext = tf.repeat(self.logits[:,tf.newaxis,:], self.num_samples, axis = 1)
        self.reshaped_weights = tf.reshape(self.weights_ext, [-1,self.num_components])  # current shape: (batch_size*num_samples, num_gaussians, n_dims)
        
        ## CREATE SAMPLES FROM THE DISTRIBUTIONAL LEARNING MODEL USING BETA SAMPLING LAYER
        # inputs_to_sampling = [self.reshaped_alphas, self.reshaped_betas, self.reshaped_weights]
        # self.z_samples, self.log_prob_z = self.Beta_sampling_layer(inputs_to_sampling)
        
        self.z_samples = self.alpha_factors 
        self.log_prob_z = tf.math.reduce_mean(self.z_samples,axis = None)
        
        # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
        Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(self.z_samples, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
        # Bear in mind that B here is B*H but we keep the same notation for that first dimension. 
        # So now Ke_matrices_dam must have shape (B*H, Elements, K,Q)
        # KQ indicate the dimension of the element matrix that here is 2x2=4 since we are in lienar elasticity 
        
        # # Assemble the element matrices to build the global matrix        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements,  self.num_samples) # the shape is (Batch_size, n_free, n_free)

        # Then we enter the eigensolver (forward) function with this list to produce the eigenfrequencies
        #TODO Watch out! REVIEW: we are enforcing the predicted modes to have unit norm. Also we must review the sign. 
        self.pred_freqs, pred_rotmodes, pred_vertmodes = self.Eigen_solver(Kfree)
        Rot_norms = tf.linalg.norm(pred_rotmodes, axis=1, keepdims=True)
        self.pred_rotmodes = orient_modes_tf(pred_rotmodes / Rot_norms)
        Vert_norms = tf.linalg.norm(pred_rotmodes, axis=1, keepdims=True)
        self.pred_vertmodes = orient_modes_tf(pred_vertmodes / Vert_norms)
        
        
        self.pred_freqs = tf.abs(self.pred_freqs) #to enforce them to be positive***
        #The output of this function is the output of the inverse, i.e., the estimated damage condition described by the alpha factors.
        #They have shape (B*H, D) 
        return  self.z_samples
        
    def Freqs_loss(self, y_true, y_pred):
        true_freqs = self.freq_data
        true_freqs = tf.repeat(true_freqs[:,:,tf.newaxis], self.num_samples, axis = 2)
        #Reshape to accommodate for further steps (final shape: (batch_size*N, n_features)) that will be seen as (None, n_features)
        true_freqs = tf.reshape(tf.transpose(true_freqs, perm = [0,2,1]), [-1,y_true.shape[1]])
        
        pred_freqs = self.pred_freqs  + 1e-6
        true_freqs = true_freqs + 1e-6


        Freqs_sq_error = tf.square(tf.math.log(true_freqs) - tf.math.log(pred_freqs))
        Loss_freqs = tf.math.reduce_mean(Freqs_sq_error, axis = None)
        return Loss_freqs
    
    
    def MSE_modes_loss(self, y_true, y_pred):
        True_rotmodes, True_vertmodes  = self.rot_modes_data, self.vert_modes_data
        
        true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis = 3)
        true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm = [0,3,1,2]), [-1,True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
        true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis = 3)
        true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm = [0,3,1,2]), [-1,True_vertmodes.shape[1], True_vertmodes.shape[2]])
        
        pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
        
        #Calculate the MACs
        Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
        Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
        
        # MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
        MACs = tf.concat([Rot_MACs, Vert_MACs], axis=1)
        sqrt_MACs = tf.math.sqrt(tf.maximum(MACs, 0.0))
        MSE_mac = 2*(1-sqrt_MACs)
        Loss_MSE_MAC = tf.math.reduce_mean(MSE_mac, axis = None)
        return Loss_MSE_MAC 

    def MAC_modes_loss(self, y_true, y_pred):
        True_rotmodes, True_vertmodes  = self.rot_modes_data, self.vert_modes_data
        
        true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis = 3)
        true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm = [0,3,1,2]), [-1,True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
        true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis = 3)
        true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm = [0,3,1,2]), [-1,True_vertmodes.shape[1], True_vertmodes.shape[2]])
        
        
        
        pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
        #Calculate the MACs
        Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
        Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
        # MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
        MACs = tf.concat([Rot_MACs, Vert_MACs], axis=1)
        neg_MACs = 1 - MACs
        Loss_MAC = tf.math.reduce_mean(neg_MACs, axis  = None)
        
        return Loss_MAC 
    
    
    def Mixture_dens_term(self, y_true, y_pred):
        """
        Computes log q(z|x).
        Since Beta is naturally normalized on [0,1], C = 1 and we don't need integral approximation.
        We simply maximize the log-probability (maximize entropy).
        """
        # self.log_prob_z is computed inside the sampling layer to be efficient
        # We want to maximize log_prob, so we usually minimize -log_prob.
        # However, following your previous code structure which returned positive log prob:
        
        Mixture_density_loss = tf.reduce_mean(self.log_prob_z)

        # You return this term. In your total loss you likely subtract it 
        # (or add it if you want to penalize low entropy). 
        # Usually: Loss = Recon - regu_weight * Entropy
        # return tf.square(tf.cast(self.regu_weight, dtype=tf.float32)) * Mixture_density_loss
        return Mixture_density_loss


    
    def ELBO_Beta_loss(self, y_true, y_pred):
        Posterior_entropy  = self.Mixture_dens_term(y_true, y_pred)
        Loss_freqs = self.Freqs_loss(y_true,y_pred)
        Loss_MSE_modes = self.MSE_modes_loss(y_true, y_pred)
        # Loss_MAC = self.MAC_modes_loss(y_true,y_pred)

        
        Cfreq = 1/(2*tf.square(tf.cast(self.sigma_f, dtype = tf.float32)))
        Cphi = 1/(2*tf.square(tf.cast(self.sigma_phi,dtype = tf.float32)))
    
        
        ELBO_loss = Cphi*Loss_MSE_modes + Cfreq*Loss_freqs - Posterior_entropy 
        return ELBO_loss
    
    # def MSE_modes_loss(self, y_true, y_pred):
    #     """
    #     Computes the Mean Squared Error (Squared L2 Norm) for mode shapes.
        
    #     Assumptions:
    #     1. Inputs are unit-norm normalized.
    #     2. Inputs are sign-consistent (e.g., phase aligned).
        
    #     Args:
    #         y_true: Tensor of shape [Batch_size, n_coords, n_modes]
    #         y_pred: Tensor of shape [Batch_size, n_coords, n_modes]
            
    #     Returns:
    #         Scalar loss value (average squared error per mode).
    #     """
        
    #     # 0. Prepare teh modeshape vectors: 
    #     True_rotmodes, True_vertmodes  = self.rot_modes_data, self.vert_modes_data
        
    #     true_rotmodes = tf.repeat(True_rotmodes[:,:,:,tf.newaxis], self.num_samples, axis = 3)
    #     true_rotmodes = tf.reshape(tf.transpose(true_rotmodes, perm = [0,3,1,2]), [-1,True_rotmodes.shape[1], True_rotmodes.shape[2]])
        
    #     true_vertmodes = tf.repeat(True_vertmodes[:,:,:, tf.newaxis], self.num_samples, axis = 3)
    #     true_vertmodes = tf.reshape(tf.transpose(true_vertmodes, perm = [0,3,1,2]), [-1,True_vertmodes.shape[1], True_vertmodes.shape[2]])
    #     pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes

    #     # 1. Calculate squared difference for every element
    #     # Shape: [Batch_size, n_coords, n_modes]
    #     squared_diff_rotmodes = tf.square(true_rotmodes - pred_rotmodes)
    #     squared_diff_vertmodes = tf.square(true_vertmodes - pred_vertmodes)
    #     squared_diff = tf.concat([squared_diff_rotmodes, squared_diff_vertmodes], axis=1)

        
    #     # 2. Sum over the coordinate dimension (axis 1)
    #     # This calculates ||phi_true - phi_pred||^2 for each mode vector.
    #     # Note: We SUM (not mean) over coords because we want the vector distance.
    #     # Result Shape: [Batch_size, n_modes]
    #     per_mode_error = tf.reduce_sum(squared_diff, axis=1)
        
    #     # 3. Average over the Batch and the Number of Modes
    #     # This gives you the scalar loss to minimize.
    #     loss = tf.reduce_mean(per_mode_error)
        
    #     return loss
     
    
    # def Copula_pdf_logprob(self, y_true, ypred):
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
    
    
    
    ## use this custom loss when you want to debug the second part of the code (the eigenvalue problem)
    def custom_loss(self,y_true, y_pred):
        loss = tf.math.reduce_mean(tf.square(y_true-y_pred), axis = None)    
        return loss
    

    def get_config(self):
        config = {
            'num_dofs': self.num_dofs,
            'M_free': self.Mfree,
            # 'eigensolver': self.Eigen_solver
        }
        base_config = super(My_BetaVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        
        # config['Solve_eigenproblem'] = tf.keras.utils.deserialize_keras_object(config['Solve_eigenproblem'])
        return cls(**config)


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
