#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:19:59 2024

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras import layers
from MODULES.TRAINING.full_multivariate_arch import  Fully_connected_enc, Fully_connected_dec, Fully_connected_gmm_inverse, Mixture_pdf_layer, Solve_eigenproblem, assemble_global_Kmatrices, ClipLayer
from tensorflow.keras.callbacks import ModelCheckpoint
#%%%% ##################################################################################3

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
    
class Inverse_Bayesian_GMM_Model(K.Model):
    def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, **kwargs):
        super(Inverse_Bayesian_GMM_Model, self).__init__()
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.FC_encoder = Fully_connected_gmm_inverse(input_dim_encoder, n_dims, num_gaussians)

    def call(self, inputs):
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
        #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
        self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
        self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
        self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])

        GMMprops = self.FC_encoder(self.modal_data)
        return GMMprops
    
    def get_config(self):
        config = {
            'n_dims': self.n_dims,
            'num_gaussians': self.num_gaussians,
            'num_samples': self.num_samples
        }
        base_config = super(Inverse_Bayesian_GMM_Model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    


# @tf.function(jit_compile = True)
class My_Bayesian_InverseForward_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, batch_size, n_dims, num_gaussians, num_samples, beta, full_cov, mean_f, std_f, lbound, **kwargs): #We add the bayesian properties (gaussians, dimensions of the latent and samples, selected_features(in case you want to work with only freqs) beta, full_cov, s_lb, s_ub)
        super(My_Bayesian_InverseForward_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.Ke_matrices = Ke_matrices # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
        self.Mfree = Mfree
        self.L_inv = L_inv
        self.Encoder_model = TMVN_Encoder_Model(input_dim, n_dims, lbound)
        self.Bayesian_encoder = Inverse_Bayesian_GMM_Model(input_dim, n_dims,num_gaussians, num_samples)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
        self.gmm_sampling = Mixture_pdf_layer(n_dims, num_gaussians, num_samples, full_cov)
        # self.global_indices = tf.cast(new_generate_indices(n_elements), dtype = 'int64')
        self.num_gaussians = num_gaussians
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.beta = beta
        self.full_cov = full_cov
        self.mean_freq = tf.constant(mean_f, dtype=tf.float32) 
        self.std_freq = tf.constant(std_f, dtype=tf.float32)
        self.lbound = lbound 
            
    def call(self, inputs):
        # Original shapes before flattening: 
        #freq_data size: (Batch_size, n_modes)
        # vert_modes_data size: (Batch_size, free displ. coordinates, n_modes)
        # rot_modes_data size: (Batch_size, free rot. coordinates, n_modes)
        [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
       
        self.GMM_props = self.Bayesian_encoder(inputs) #Delivers the means, sigmas, weights and angles
        self.means, self.sigmas, self.weights_values, self.alpha_angles = tf.split(self.GMM_props, [self.n_dims * self.num_gaussians,
                                                        self.n_dims * self.num_gaussians,
                                                        self.num_gaussians,
                                                        self.num_gaussians*((self.n_dims*(self.n_dims -1))//2)], axis=-1)
        
        self.weights_values = ClipLayer(1e-06, 1.0)(self.weights_values)
        #We denote H to the number of samples to draw
        inputs_to_sampling = [self.means, self.sigmas, self.weights_values, self.alpha_angles]
        self.LT_matrices, self.mixtures, self.reshaped_samples, self.integral_value = self.gmm_sampling(inputs_to_sampling)

        ## The reshaped samples have shape (BxH, G, D = n_dims). These are samples of the latent space described by n_dims features. 
        # We adapt the LT_matrices to the same shape of reshaped_Samples (BxH, G, D, D) #D = n_dims
        self.LT_matrices = tf.expand_dims(self.LT_matrices, axis=1)  # Shape: (B, 1, G, D,D)
        # TODO review because tile is not working in tf>2.15. . 
        self.LT_matrices = tf.tile(self.LT_matrices, [1, self.num_samples, 1, 1, 1])  # Shape: (B, H, G, D, D)
        self.LT_matrices = tf.reshape(self.LT_matrices, [-1, self.num_gaussians, self.n_dims, self.n_dims]) # Shape: (BxH, G, D,D)
        
        reshaped_alpha_factors = self.reshaped_samples        
        # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
        Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(reshaped_alpha_factors, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
        # Bear in mind that B here is B*H but we keep the same notation for that first dimension. 
        # So now Ke_matrices_dam must have shape (B*H, Elements, K,Q)
        # KQ indicate the dimension of the element matrix that here is 2x2 since we are in lienar elasticity 
        
        # # Assemble the element matrices to build the global matrix        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements,  self.num_samples) # the shape is (Batch_size, n_free, n_free)

        # Then we enter the eigensolver (forward) function with this list to produce the eigenfrequencies
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        self.pred_freqs = tf.abs(self.pred_freqs) #to enforce them to be positive***
        #The output of this function is the output of the inverse, i.e., the estimated damage condition described by the alpha factors.
        #They have shape (B*H, D) 
        return  reshaped_alpha_factors
        
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
    
    def Mixture_dens_term(self, y_true, y_pred):
        # Este termino tiene que ver con la mixtura que estimamos a traves de nuestro INVERSE. 
        # Para cada input sample, se extraen las propiedades de la mixtura (output de Inverse_model)y se calcula la log prob de las muestras que estamos tomando de esta distribucion mediante la funcion sampling
        # #Este es el termino que va a contrarrestar el efecto del misfit de datos, haciendo que las mixturas tomen una forma montañosa en lugar de un delta dirac (esto ocurre si solo minimizamos con L_data)
        mixture = self.mixtures
        # Mixture_density_loss = tf.math.reduce_mean(mixture.log_prob(self.reshaped_samples))
        epsi = 0.00001
        C = tf.cast(1.0/self.integral_value, dtype=tf.float32)
        Mixture_density_loss = tf.math.reduce_mean(tf.math.log(C*mixture.prob(self.reshaped_samples)+ epsi))

        return tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (Mixture_density_loss)
        

    def total_loss_with_regularizer(self, y_true, y_pred):    
        Loss_freqs = self.Freqs_loss(y_true,y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
        Mixture_loss = self.Mixture_dens_term(y_true,y_pred)
        
        return  Loss_MAC + Loss_freqs + Mixture_loss
    

    
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
        base_config = super(My_Bayesian_InverseForward_withEigen, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        
        # config['Solve_eigenproblem'] = tf.keras.utils.deserialize_keras_object(config['Solve_eigenproblem'])
        return cls(**config)


    
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
    


class Deterministic_Inverse_Model(tf.keras.Model):
    def __init__(self, input_dim_encoder, num_mixtures, num_gaussians, num_samples_per_mixture, s_lb, s_ub, **kwargs):
        super(Deterministic_Inverse_Model, self).__init__()
        self.num_mixtures = num_mixtures
        self.num_gaussians = num_gaussians
        self.num_samples_per_mixture = num_samples_per_mixture
        self.s_lb = s_lb
        self.s_ub = s_ub
        self.FC_encoder = Fully_connected_enc(input_dim_encoder, num_mixtures, num_gaussians)

    def call(self, inputs):
        inputs_concat = layers.Concatenate(axis=-1)(inputs)
        # inputs_concat = inputs #when we do not want to use r in the training of the Inverse

        z_pred = self.FC_encoder(inputs_concat)
        return z_pred
    
    def get_config(self):
        config = {
            'num_mixtures': self.num_mixtures,
            'num_gaussians': self.num_gaussians,
            'num_samples_per_mixture': self.num_samples_per_mixture
        }
        base_config = super(Deterministic_Inverse_Model, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


####### use this model wwhen you want to use means as inputs to the forward
################ TRANSITION MODEL: FROM DETERMINISTIC TO BAYESIAN APPROACH 
class My_Deterministic_InverseForward_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, batch_size, n_dims, num_gaussians, num_samples, beta, full_cov, s_lb, s_ub, **kwargs): #We add the bayesian properties (gaussians, dimensions of the latent and samples, selected_features(in case you want to work with only freqs) beta, full_cov, s_lb, s_ub)
        super(My_Deterministic_InverseForward_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.Ke_matrices = Ke_matrices # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
        self.Mfree = Mfree
        self.L_inv = L_inv
        self.Bayesian_encoder = Inverse_Bayesian_GMM_Model(input_dim, n_dims,num_gaussians, num_samples,s_lb, s_ub)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
        self.b_size = batch_size 
        # self.global_indices = tf.cast(new_generate_indices(n_elements), dtype = 'int64')
        self.epsi = epsi #wight factor for the regularization term in the loss 
        self.num_gaussians = num_gaussians
        self.n_dims = n_dims
        self.num_samples = num_samples
        self.beta = beta
        self.full_cov = full_cov
            
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

        self.GMM_props = self.Bayesian_encoder(inputs) #Delivers the means, sigmas, weights and angles
        self.means, self.sigmas, self.weights_values, self.alpha_angles = tf.split(self.GMM_props, [self.n_dims * self.num_gaussians,
                                                        self.n_dims * self.num_gaussians,
                                                        self.num_gaussians,
                                                        self.num_gaussians*((self.n_dims*(self.n_dims -1))//2)], axis=-1)
        

        
        #We denote H to the number of samples to draw
        inputs_to_sampling = [self.means, self.sigmas, self.weights_values, self.alpha_angles]
        self.reshaped_samples, self.LT_matrices = self.gmm_sampling(inputs_to_sampling) #These are our predicted alpha factors affecting the stiffness matrix with shape (B*H, D)
        ## The reshaped samples have shape (BxH, G, D = n_dims). These are samples of the latent space described by n_dims features. 
        # We adapt the LT_matrices to the same shape of reshaped_Samples (BxH, G, D, D) #D = n_dims
        self.LT_matrices = tf.expand_dims(self.LT_matrices, axis=1)  # Shape: (B, 1, G, D,D)
        # TODO review because tile is not working in tf>2.15. . 
        self.LT_matrices = tf.tile(self.LT_matrices, [1, self.num_samples, 1, 1, 1])  # Shape: (B, H, G, D, D)
        self.LT_matrices = tf.reshape(self.LT_matrices, [-1, self.num_gaussians, self.n_dims, self.n_dims]) # Shape: (BxH, G, D,D)
        ##CLIPPING TO [0,1] to enter the FORWARD with samples belonging to the interval known during training
        reshaped_alphas = tf.sigmoid(self.reshaped_samples) #These are our predicted alpha factors with shape (B*H, D)
        #%%%%%%%%%%%%%%%%%%%%%%%% inputting the means as deterministic estimates to the Forward (decoder)
        #The first step is the Inverse NN that estimates the stiffness properties in a list of size (Batch_size, ndofs*ndofs)
        # pred_alpha_factors =  self.means 
        # # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
        # Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(pred_alpha_factors, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
        
        # # # Assemble the element matrices to build the global matrix        
        # Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.b_size) # the shape is (Batch_size, n_free, n_free)
        # # Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements, self.global_indices) # the shape is (Batch_size, n_free, n_free)
        
        reshaped_alphas = self.means #first avoid the probabillity part by directly feeding the means to the decdoer 
        # tf.print(self.means.shape)
         #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
        Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(reshaped_alphas, dtype = tf.float32), tf.cast(self.Ke_matrices, dtype = tf.float32))
        # Bear in mind that B here is B*H but we keep the same notation for that first dimension. 
        # So now Ke_matrices_dam must have shape (B*H, Elements, K,Q)
        # KQ indicate the dimension of the element matrix that here is 2x2 since we are in lienar elasticity 
        
        # # Assemble the element matrices to build the global matrix        
        Kfree = assemble_global_Kmatrices(Ke_matrices_dam, self.n_elements,  self.num_samples) # the shape is (Batch_size, n_free, n_free)

        # Then we enter the eigensolver (forward) function with this list to produce the eigenfrequencies
        self.pred_freqs, self.pred_rotmodes, self.pred_vertmodes = self.Eigen_solver(Kfree)
        self.pred_freqs = tf.abs(self.pred_freqs) #to enforce them to be positive***
        #The output of this function is the output of the inverse, i.e., the estimated damage condition described by the alpha factors.
        #They have shape (B*H, D) 
        return  reshaped_alphas
        
    def Freqs_loss(self, y_true, y_pred):
        true_freqs = self.freq_data
        pred_freqs = self.pred_freqs
        Freqs_sq_error = tf.square(tf.math.log(true_freqs) - tf.math.log(pred_freqs))
        Loss_freqs = tf.math.reduce_mean(Freqs_sq_error, axis = None)
        return Loss_freqs
    

    def MAC_modes_loss(self, y_true, y_pred):
        true_rotmodes, true_vertmodes  = self.rot_modes_data, self.vert_modes_data
        pred_rotmodes, pred_vertmodes = self.pred_rotmodes, self.pred_vertmodes
        #Calculate the MACs
        Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
        Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
        # MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
        MACs = tf.concat([Rot_MACs, Vert_MACs], axis=1)
        neg_MACs = 1 - MACs
        Loss_MAC = tf.math.reduce_mean(neg_MACs, axis  = None)
        
        return Loss_MAC 
    
    def Alpha_regularizer(self, y_true, y_pred):
        min_value = tf.reduce_min(y_pred, axis=-1, keepdims=False)
        Regularizer = tf.math.reduce_mean( tf.math.reduce_sum(y_pred,axis = -1)-min_value)/(y_pred.shape[-1])-1
        return self.epsi*Regularizer
        
    def Mixture_dens_term(self, y_true, y_pred):

        return 0.0
        
    def total_loss_with_regularizer(self, y_true, y_pred):    
        Loss_freqs = self.Freqs_loss(y_true,y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
        Regularizer = self.Alpha_regularizer(y_true, y_pred)
        
        return  Loss_MAC + Loss_freqs - Regularizer
    
    
    
    
    ############# Old loss mixture term
    # def Mixture_dens_term(self, y_true, y_pred):
    #     # Este termino tiene que ver con la mixtura que estimamos a traves de nuestro INVERSE. 
    #     # Para cada input sample, se extraen las propiedades de la mixtura (output de Inverse_model)y se calcula la log prob de las muestras que estamos tomando de esta distribucion mediante la funcion sampling
    #     #Este es el termino que va a contrarrestar el efecto del misfit de datos, haciendo que las mixturas tomen una forma montañosa en lugar de un delta dirac (esto ocurre si solo minimizamos con L_data)
    #     means, sigmas, weights = self.means, self.sigmas, self.weights_values
        
    #     #Reshaping required to accomodate for the batch size (, num_mixtures*num_gaussians) during training
    #     means = tf.reshape(means, (-1, self.num_mixtures, self.num_gaussians))
    #     sigmas = tf.reshape(sigmas, (-1, self.num_mixtures, self.num_gaussians))
    #     weights = tf.reshape(weights, (-1, self.num_gaussians))
        # weights = tf.ClipLayer(weights, 1e-6, 1.0)
    #     ## REPEATING THE MEANS TO FIT WITH THE NUMBER OF SAMPLES ,N
    #     m = tf.repeat(means[:,:,tf.newaxis], self.num_samples_per_mixture, axis = 2)
    #     s = tf.repeat(sigmas[:,:,tf.newaxis], self.num_samples_per_mixture, axis = 2)
    #     w = tf.repeat(weights[:,tf.newaxis], self.num_samples_per_mixture, axis = 1)
    #     ## THIS PERMUTATION IS TO MAKE THE MEANS, SIGMAS AND WEIGHTS HAVE THE DESIRED SHAPE (None*N, num_mixtures, num_gaussians):
    #     means = tf.reshape(tf.transpose(m, perm = [0,2,1,3]), [-1, self.num_mixtures, self.num_gaussians])
    #     sigmas = tf.reshape(tf.transpose(s, perm = [0,2,1,3]), [-1, self.num_mixtures, self.num_gaussians])
    #     weights =tf.reshape(w, [-1,self.num_gaussians]) 
        
    #     diag_components = []
    #     for i in range(self.num_gaussians):
    #         component = tfp.distributions.MultivariateNormalDiag(loc=means[:, :, i], scale_diag=sigmas[:, :, i])
    #         diag_components.append(component)
            
        
    #     full_components = []
    #     for i in range(self.num_gaussians):
    #         LT_matrix = self.LT_matrices[:,i,:,:]
    #         component2 = tfp.distributions.MultivariateNormalTriL(loc=means[:,:,i], scale_tril=LT_matrix)
    #         full_components.append(component2)
            
    #     if self.full_cov: 
    #         Components = full_components
    #     else: 
    #         Components = diag_components 
    
    #     cat = tfp.distributions.Categorical(probs=weights)
    #     mixture = tfp.distributions.Mixture(cat=cat, components= Components)

    #     # Mixture_density_loss = tf.math.reduce_mean(mixture.log_prob(self.reshaped_samples))
    #     epsi = 1.0
    #     Mixture_density_loss = tf.math.reduce_mean(tf.math.log(mixture.prob(self.reshaped_samples)+ epsi))


    #     return tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (Mixture_density_loss)