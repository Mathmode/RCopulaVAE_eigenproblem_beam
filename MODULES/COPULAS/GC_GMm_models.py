
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated: March 2026
Unified Model for 5-element and 10-element Datasets.
Maintains original loss logic with improved numerical stability.
"""


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import numpy as np

tfd = tfp.distributions
from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Fully_connected_enc_GC_highdims, Copula_pdf_layer

# -------------------------------------------------------------------------
# HELPER FUNCTIONS FOR LOSS
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def calculate_MAC(modes_true, modes_pred):
    """
    Computes Modal Assurance Criterion (MAC) with improved stability.
    """
    # Small epsilon to avoid division by zero during normalization
    eps = 1e-8
    modes_true_norm = modes_true / (tf.norm(modes_true, axis=2, keepdims=True) + eps)
    modes_pred_norm = modes_pred / (tf.norm(modes_pred, axis=2, keepdims=True) + eps)
    
    # Compute MAC Matrix
    mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    return mac_matrix

class Inverse_Copula_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, lbound, **kwargs):
        super(Inverse_Copula_Model, self).__init__()
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        # High-dims architecture is preferred for 10 elements
        self.FC_encoder = Fully_connected_enc_GC_highdims(input_dim_encoder, n_dims, num_gaussians, lbound)

    def call(self, inputs):
        [freq_data, rot_modes_data, vert_modes_data, alpha_factors] = inputs
        
        flat_rot = tf.reshape(rot_modes_data, [-1, rot_modes_data.shape[1] * rot_modes_data.shape[2]])
        flat_vert = tf.reshape(vert_modes_data, [-1, vert_modes_data.shape[1] * vert_modes_data.shape[2]])
        modal_data = K.layers.Concatenate(axis=1)([freq_data, flat_vert, flat_rot])

        params = self.FC_encoder(modal_data)
        n_corr = self.n_dims * (self.n_dims - 1) // 2

        means, scales, weights, offdiag, diag = tf.split(params, [
            self.n_dims * self.num_gaussians,
            self.n_dims * self.num_gaussians,
            self.n_dims * self.num_gaussians,
            n_corr,
            self.n_dims
        ], axis=-1)

        means = tf.reshape(means, (-1, self.num_gaussians, self.n_dims))
        scales = tf.reshape(scales, (-1, self.num_gaussians, self.n_dims))
        weights = tf.reshape(weights, (-1, self.num_gaussians, self.n_dims))
        weights = tf.nn.softmax(weights, axis=1)
        
        return means, scales, weights, offdiag, diag

class My_CopulaVAE_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, n_dims, num_gaussians, num_samples, beta, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
        super(My_CopulaVAE_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.fixed_dofs_indices = fixed_dofs_indices

        self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
        self.L_inv = tf.constant(L_inv, dtype=tf.float32)

        self.Encoder_model = Inverse_Copula_Model(input_dim, n_dims, num_gaussians, num_samples, lbound)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
        self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples, lbound)
        
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.num_gaussians = num_gaussians
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.beta = beta
        self.lbound = lbound 

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        
        self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
        # Sampling logic
        inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
        self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
        
        self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
        self.reshaped_copula_samples = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
        
        # Replicate LT matrices for samples
        lt_ext = tf.repeat(self.LT_matrices[:, tf.newaxis, :, :], self.num_samples, axis=1)
        self.reshaped_LT_matrices = tf.reshape(lt_ext, [-1, self.n_dims, self.n_dims])

        # Stiffness scaling
        if len(self.Ke_matrices.shape) == 2:
            Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', self.reshaped_alpha_samples, self.Ke_matrices)
        else:
            Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', self.reshaped_alpha_samples, self.Ke_matrices)
        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        
        return self.reshaped_alpha_samples

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
        # Stability: Clip copula samples slightly to avoid infinity in quantile function
        u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-5, 1.0 - 1e-5)
        normal_dist = tfd.Normal(loc=0.0, scale=1.0)
        normal_samples = normal_dist.quantile(u_clipped)
               
        mvn = tfd.MultivariateNormalTriL(loc=tf.zeros(self.n_dims), 
                                         scale_tril=self.reshaped_LT_matrices)
        
        log_prob_joint_normal = mvn.log_prob(normal_samples)
        log_prob_marginals_std = tf.reduce_sum(normal_dist.log_prob(normal_samples), axis=-1)      
        
        return log_prob_joint_normal - log_prob_marginals_std
    
    def Marginal_pdf_logprob(self, y_true, y_pred):
        # High stability TruncatedNormal logprob
        gaussian_marginals = tfd.TruncatedNormal(
            loc=tf.reshape(self.means, [-1, self.n_dims]), 
            scale=tf.reshape(self.scales, [-1, self.n_dims]) + 1e-5, 
            low=self.lbound - 1e-4,  
            high=1.0 + 1e-4)

        log_prob_marginals = gaussian_marginals.log_prob(self.reshaped_alpha_samples)  
        return tf.math.reduce_sum(log_prob_marginals, axis=-1)    
    
    def Joint_copula_dens_term(self, y_true, y_pred):
        c_term = self.Copula_pdf_logprob(y_true, y_pred)
        m_term = self.Marginal_pdf_logprob(y_true, y_pred)  
        
        joint_logprob = tf.math.reduce_mean(c_term + m_term)
        # Beta scaling
        return tf.math.square(tf.cast(self.beta, dtype=tf.float32)) * joint_logprob
    
    def ELBO_Copula_loss(self, y_true, y_pred):
        """
        Combined Loss. Note the multipliers for stability.
        """
        loss_f = self.Freqs_loss(y_true, y_pred)
        loss_m = self.MAC_modes_loss(y_true, y_pred)
        loss_j = self.Joint_copula_dens_term(y_true, y_pred)
                
        # Frequency loss often dominates early on; we use weights to balance.
        # For 10 elements, MAC loss is a very reliable gradient source.
        total_loss = loss_f + 15.0 * loss_m + loss_j
        
        return total_loss

    def get_config(self):
        config = {'num_dofs': self.num_dofs, 'fixed_dofs_indices': self.fixed_dofs_indices}
        base_config = super(My_CopulaVAE_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
# import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow.keras.layers import Layer
# import tensorflow.keras as K
# import numpy as np

# tfd = tfp.distributions
# # Assumes Inverse_Copula_Model and Copula_pdf_layer are imported/available
# from MODULES.COPULAS.GC_GMm_GPU_eigen_functions import assemble_global_Kmatrices, Solve_eigenproblem
# from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Fully_connected_enc_GC_highdims,  Copula_pdf_layer
        

# # Import architectures if they are in a separate module
# # from MODULES.COPULAS.GC_GMm_architectures import Fully_connected_enc_GC, Copula_pdf_layer, Fully_connected_dec
# # For this file to be self-contained or runnable in context, we assume these classes are available 
# # or imported by the user. The code below focuses on the Physics and VAE integration.


# # -------------------------------------------------------------------------
# # 3. HELPER FUNCTIONS FOR LOSS
# # -------------------------------------------------------------------------

# @tf.function(jit_compile=True)
# def calculate_MAC(modes_true, modes_pred):
#     """
#     Computes Modal Assurance Criterion (MAC) between two sets of modes.
#     Args:
#         modes_true: (Batch, Modes, Nodes)
#         modes_pred: (Batch, Modes, Nodes)
#     Returns:
#         MAC Matrix: (Batch, Modes, Modes) where [b, i, j] is MAC(True_i, Pred_j)
#     """
#     # Normalize along the Node axis (axis 2)
#     modes_true_norm = tf.math.l2_normalize(modes_true, axis=2)
#     modes_pred_norm = tf.math.l2_normalize(modes_pred, axis=2)
    
#     # Compute MAC Matrix: (Batch, Modes, Nodes) @ (Batch, Nodes, Modes) -> (Batch, Modes, Modes)
#     # transpose_b=True flips the last two dims of modes_pred_norm to (Batch, Nodes, Modes)
#     mac_matrix = tf.square(tf.matmul(modes_true_norm, modes_pred_norm, transpose_b=True))
    
#     return mac_matrix



# class Inverse_Copula_Model(tf.keras.Model):
#     def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, lbound, **kwargs):
#         super(Inverse_Copula_Model, self).__init__()
#         self.n_dims = n_dims
#         self.num_gaussians = num_gaussians
#         self.num_samples = num_samples
#         # select the architecture to use for the encoder (for higher dimensions we use Fully_connected_enc_GC_highdims)
#         # self.FC_encoder = Fully_connected_enc_GC(input_dim_encoder, n_dims, num_gaussians, lbound)
#         self.FC_encoder = Fully_connected_enc_GC_highdims(input_dim_encoder, n_dims, num_gaussians, lbound)


#     def call(self, inputs):
#         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
#         #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
#         self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
#         self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
#         self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])

#         Copula_parameters = self.FC_encoder(self.modal_data)
#         n_correlation_factors = self.n_dims*(self.n_dims-1)//2

#         means, scales, weight_vals, offdiag_elems, diag_elems = tf.split(Copula_parameters, [self.n_dims * self.num_gaussians,
#                                                         self.n_dims*self.num_gaussians,
#                                                         self.n_dims*self.num_gaussians,
#                                                         n_correlation_factors,
#                                                         self.n_dims
#                                                         ], axis=-1)
        

#         # RESHAPE TO SEPARATE by num of components (Gaussians in the mixture for each variable) and num_dims (number of variables)
#         means = tf.reshape(means, (-1,self.num_gaussians, self.n_dims))
#         scales = tf.reshape(scales, (-1, self.num_gaussians, self.n_dims))
#         weight_vals = tf.reshape(weight_vals, (-1, self.num_gaussians, self.n_dims))
#         weight_vals = tf.nn.softmax(weight_vals, axis = 1) # apply the weight normalization to the weights of each dimension (contaning n_gaussians)
        
        
#         return means, scales, weight_vals, offdiag_elems, diag_elems
    
#     def get_config(self):
#         config = {
#             'n_dims': self.n_dims,
#             'num_gaussians': self.num_gaussians,
#             'num_samples': self.num_samples
#         }
#         base_config = super(Inverse_Copula_Model, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
   
# # -------------------------------------------------------------------------
# # 4. VAE MODEL
# # -------------------------------------------------------------------------

# class My_CopulaVAE_withEigen(tf.keras.Model):
#     def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, n_dims, num_gaussians, num_samples, beta, mean_f, std_f, lbound, fixed_dofs_indices=None, **kwargs):
#         super(My_CopulaVAE_withEigen, self).__init__()
#         self.num_dofs = num_dofs
#         self.n_elements = n_elements
#         self.n_modes = n_modes
#         self.fixed_dofs_indices = fixed_dofs_indices

#         # Cast Physics Matrices once
#         self.Ke_matrices = tf.constant(Ke_matrices, dtype=tf.float32) 
#         self.L_inv = tf.constant(L_inv, dtype=tf.float32)


#         self.Encoder_model = Inverse_Copula_Model(input_dim, n_dims, num_gaussians, num_samples, lbound)
#         self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, L_inv, fixed_dofs_indices)
#         self.Copula_sampling_layer = Copula_pdf_layer(n_dims, num_gaussians, num_samples, lbound)
        
#         # Hyperparameters
#         self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
#         self.std_freq = tf.constant(std_f, dtype=tf.float32)
#         self.num_gaussians = num_gaussians
#         self.n_dims = n_dims
#         self.num_samples = num_samples
#         self.beta = beta
#         self.lbound = lbound # lower bound for truncation according to z domain  (minimum reduction factor --> maximum admissible damage)
            
#     def call(self, inputs):
#         # Unpack Inputs: [Freqs, RotModes, VertModes, Alphas]
#         # Data shapes expected:
#         # Freqs: (Batch, Modes)
#         # Modes: (Batch, Modes, Nodes)
#         # Alphas: (Batch, Elements)
#         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        
#         # 1. ENCODER
#         self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems = self.Encoder_model(inputs)
        
#         # Expand dimensions for sampling
#         self.means_ext = tf.repeat(self.means[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_means = tf.reshape(self.means_ext, [-1,self.num_gaussians, self.n_dims]) 
#         self.scales_ext = tf.repeat(self.scales[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_scales = tf.reshape(self.scales_ext, [-1,self.num_gaussians, self.n_dims])  
#         self.weights_ext = tf.repeat(self.weight_vals[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_weight_vals = tf.reshape(self.weights_ext, [-1,self.num_gaussians, self.n_dims])  

#         # 2. SAMPLING
#         inputs_to_sampling = [self.means, self.scales, self.weight_vals, self.offdiag_elems, self.diag_elems]
#         self.marginal_samples_z, self.copula_samples_u, self.LT_matrices = self.Copula_sampling_layer(inputs_to_sampling)
#         # These are the estimated Alphas (Stiffness Factors)
#         self.reshaped_alpha_samples = tf.reshape(self.marginal_samples_z, (-1, self.n_dims)) 
#         self.reshaped_copula_samples  = tf.reshape(self.copula_samples_u, (-1, self.n_dims))
    
#         self.LT_matrices_ext =  tf.repeat(self.LT_matrices[:,tf.newaxis,:,:], self.num_samples, axis = 1)
#         self.reshaped_LT_matrices = tf.reshape(self.LT_matrices_ext, [-1,self.n_dims, self.n_dims])
        
#         # 3. DECODER: Solve Physics
#         # Ke matrices damage calculation: Scale Base Ke by Alpha samples
#         # Alpha: (Batch, Elem), Ke: (4, 4) -> Damaged: (Batch, Elem, 4, 4)
#         if len(self.Ke_matrices.shape) == 2:
#             Ke_matrices_dam = tf.einsum('BE, KQ -> BEKQ', 
#                                         tf.cast(self.reshaped_alpha_samples, dtype=tf.float32), 
#                                         tf.cast(self.Ke_matrices, dtype=tf.float32))
#         else:
#             Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', 
#                                         tf.cast(self.reshaped_alpha_samples, dtype=tf.float32), 
#                                         tf.cast(self.Ke_matrices, dtype=tf.float32))
        
#         # Assemble Global Stiffness (Free DOFs only)
#         Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.num_samples, self.fixed_dofs_indices) 
    
#         # Solve Eigenvalue Problem
#         # Returns: Freqs (B, M), RotModes (B, M, N), VertModes (B, M, N)
#         self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
#         self.pred_freqs = tf.abs(self.pred_freqs)         
#         # RETURN ALL DESIRED OUTPUTS
#         # return [self.reshaped_alpha_samples, self.means, self.scales, self.offdiag_elems, self.diag_elems]       
#         return self.reshaped_alpha_samples
#         # # --- LOSS FUNCTIONS ---

#     def Freqs_loss(self, y_true, y_pred):
#         true_scaled = self.freq_data    # we applied the log and scaled before entering the model! Be careful with this!
#         pred_hz = self.pred_freqs
        
#         # Safe log and standardization
#         pred_log = tf.math.log(tf.maximum(pred_hz, 1e-6))
#         pred_scaled = (pred_log - self.mean_freq) / self.std_freq
        
#         loss = tf.math.reduce_mean(tf.square(true_scaled - pred_scaled))
#         return loss
    
#     def MAC_modes_loss(self, y_true, y_pred):
#         true_rot = self.rot_modes_data
#         true_vert = self.vert_modes_data
#         pred_rot = self.pred_rotmodes
#         pred_vert = self.pred_vertmodes

#         def get_best_match_loss(true_group, pred_group):
#             # Calculate MAC Matrix: (Batch, Modes, Modes)
#             # Element (b, i, j) is MAC between True Mode i and Pred Mode j
#             mac_mat = calculate_MAC(true_group, pred_group)
            
#             # Find the best matching Predicted Mode for each True Mode
#             # This handles potential mode ordering swaps if they occur
#             best_matches = tf.reduce_max(mac_mat, axis=2)
            
#             # We want best matches to be 1.0. Loss = 1 - MAC
#             return tf.reduce_mean(1.0 - best_matches) 

#         loss_rot = get_best_match_loss(true_rot, pred_rot)
#         loss_vert = get_best_match_loss(true_vert, pred_vert)
        
#         return loss_rot + loss_vert


#     def Copula_pdf_logprob(self,y_true, ypred):
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
        
#         # # Clip copula samples slightly away from 0 and 1 to prevent Inf in quantile
#         # u_clipped = tf.clip_by_value(self.reshaped_copula_samples, 1e-6, 1.0 - 1e-6)
#         # normal_dist = tfd.Normal(loc=0.0, scale=1.0)
#         # normal_samples = normal_dist.quantile(u_clipped)
        
#         # # Convert uniform samples to standard normal
#         normal_samples = tfd.Normal(loc=0.0, scale=1.0).quantile(self.reshaped_copula_samples)
               
#         mvn = tfd.MultivariateNormalTriL(loc = tf.zeros(self.n_dims), scale_tril = self.reshaped_LT_matrices)
        
#         ###*****
#         # log_prob_joint_normal = tf.math.log(mvn.prob(normal_samples)+1e-07)
#         log_prob_joint_normal = mvn.log_prob(normal_samples)
        
#         log_prob_marginals_standard = tf.reduce_sum(tf.math.log(tfd.Normal(0.0, 1.0).prob(normal_samples)+1e-07), axis=-1)      
        
    
#         return log_prob_joint_normal - log_prob_marginals_standard
   
    
#     def Marginal_pdf_logprob(self,y_true, y_pred):
#         '''
#         #This one is used for Gaussian marginal directly (known inverse CDF)
#         Here we calculate the log probability of the samples over the marginal distributions 
#         We have n_dims marginals to consider. 
#         The inputs are taken from the self, and include the marginal_samples, and the marginals parameters 
#         The output will be the log_probs with shape [batch_size, num_samples]
#         '''
#         # Taking into account that here we have only one gaussian, we can neglect the dimension of num_gaussians as it is simply 1. 
#         # gaussian_marginals = tfd.TruncatedNormal(loc=self.reshaped_means[:,0,:], scale=self.reshaped_scales[:,0,:], low = self.lbound - 0.0001, high=1.0001)        
#         # # This is producing one logprob value for each dimension 
#         gaussian_marginals = tfd.TruncatedNormal(
#             loc=self.reshaped_means[:,0,:], 
#             scale=self.reshaped_scales[:,0,:], 
#             low=self.lbound - 1e-4,  
#             high=1.0 + 1e-4)

#         #****
#         # log_prob_marginals = tf.math.log(gaussian_marginals.prob(self.reshaped_alpha_samples)+ 1e-07)  
#         log_prob_marginals = gaussian_marginals.log_prob(self.reshaped_alpha_samples)  

#         # # According to the equation (see paper), log(SUM) = SUM(logs): 
#         log_prob_marginal = tf.math.reduce_sum(log_prob_marginals,axis = -1)    # to sum in the axis of n_dims        
#         return log_prob_marginal
    
    
#     def Joint_copula_dens_term(self,y_true, y_pred):
#         Copula_density_term = self.Copula_pdf_logprob(y_true, y_pred)
#         Marginal_logprob_term = self.Marginal_pdf_logprob(y_true,y_pred)  
        
#         joint_copula_logprob = tf.math.reduce_mean(Copula_density_term + Marginal_logprob_term, axis = None)
#         joint_copula_logprob = tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (joint_copula_logprob)

#         return joint_copula_logprob
    
    
#     def ELBO_Copula_loss(self, y_true, y_pred):
#         """
#         Total Loss Function: Minimizes Reconstruction Error + NLL - Regularizer
#         """
        
#         Loss_freqs = self.Freqs_loss(y_true, y_pred)
#         Loss_MAC = self.MAC_modes_loss(y_true, y_pred)
        
#         Joint_copula_nll = self.Joint_copula_dens_term(y_true, y_pred)
                
#         ELBO_loss = Loss_freqs + 10.*Loss_MAC + Joint_copula_nll
        
#         return ELBO_loss
    
#     def get_config(self):
#         config = {
#             'num_dofs': self.num_dofs,
#             'M_free': self.Mfree, # Note: Not used in logic but stored
#             'fixed_dofs_indices': self.fixed_dofs_indices
#         }
#         base_config = super(My_CopulaVAE_withEigen, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    















# class ForwardModel(K.Model):
#     def __init__(self, input_dim_decoder, output_dim, num_mixtures):
#         super(ForwardModel, self).__init__()
#         self.num_mixtures = num_mixtures
#         self.FC_decoder = Fully_connected_dec(input_dim_decoder, output_dim)

#     def call(self, inputs):
#         reconstructed_inputs = self.FC_decoder(inputs)
#         return reconstructed_inputs
    
#     def custom_loss(self, y_true, y_pred):
#         loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred), axis=None)
#         return loss
    
#     def get_config(self):
#         config = {
#             'input_dim_decoder': self.FC_decoder.input_dim_decoder,
#             'output_dim': self.FC_decoder.output_dim,
#             'num_mixtures': self.num_mixtures
#         }
#         base_config = super(ForwardModel, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))