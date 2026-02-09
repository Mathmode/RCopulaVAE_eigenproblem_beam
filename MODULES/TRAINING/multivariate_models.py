#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:19:59 2024

@author: afernandez
"""

import tensorflow as tf
import tensorflow_probability as tfp

import keras as K 
import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
from MODULES.TRAINING.multivariate_arch import ClipLayer, Solve_eigenproblem, assemble_global_Kmatrices
from MODULES.TRAINING.multivariate_arch import Fully_connected_gmm_inverse, GMMSamplingLayer  ## new! because we want to make it bayesian now
from tensorflow.keras.callbacks import ModelCheckpoint  # TODO Querida.... todavía no has aprendido a guardar el modelo antes de que casque


@tf.function(jit_compile = True)
def calculate_MAC(modes_true, modes_pred):
    modes_true_transp = tf.einsum('BCM -> BMC', modes_true)
    modes_pred_transp  = tf.einsum('BCM -> BMC', modes_pred)
        
    MAC_numer = tf.math.square(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
    MAC_denom  = tf.math.multiply(tf.einsum('BMC, BCM -> BM', modes_true_transp, modes_true), tf.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred))
    eps  = 0.0001
    MAC_denom = MAC_denom+eps
    MAC = tf.math.divide(MAC_numer, MAC_denom)
    #MAC dimension is (Batch_size, N_modes)
    return MAC

        
class Inverse_Bayesian_GMM_Model(K.Model):
    def __init__(self, input_dim_encoder, n_dims, num_gaussians, num_samples, s_lb, s_ub, **kwargs):
        super(Inverse_Bayesian_GMM_Model, self).__init__()
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.s_lb = s_lb #lower bound for the damaged condition features (in this case i guess it will be [0,1])
        self.s_ub = s_ub #Upper bound for the damaged condition features
        self.FC_encoder = Fully_connected_gmm_inverse(input_dim_encoder, n_dims, num_gaussians, s_lb, s_ub)

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
    

class My_Bayesian_InverseForward_withEigen(tf.keras.Model):
    def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, batch_size, n_dims, num_gaussians, num_samples, beta, full_cov, s_lb, s_ub, **kwargs): #We add the bayesian properties (gaussians, dimensions of the latent and samples, selected_features(in case you want to work with only freqs) beta, full_cov, s_lb, s_ub)
        super(My_Bayesian_InverseForward_withEigen, self).__init__()
        self.num_dofs = num_dofs
        self.n_elements = n_elements
        self.n_modes = n_modes
        self.Ke_matrices = Ke_matrices # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
        self.Mfree = Mfree
        self.L_inv = L_inv
        self.Bayesian_encoder = Inverse_Bayesian_GMM_Model(input_dim, n_dims,num_gaussians, num_samples,s_lb, s_ub)
        self.Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
        self.gmm_sampling = GMMSamplingLayer(n_dims, num_gaussians, num_samples, full_cov)
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
        
        self.weights_values = ClipLayer(1e-06, 1.0)(self.weights_values)
        #We denote H to the number of samples to draw
        inputs_to_sampling = [self.means, self.sigmas, self.weights_values, self.alpha_angles]
        self.reshaped_samples, self.LT_matrices = self.gmm_sampling(inputs_to_sampling) #These are our predicted alpha factors affecting the stiffness matrix with shape (B*H, D)
        # self.reshaped_samples = ClipLayer(0.0, 1.0)(self.reshaped_samples)

        ## The reshaped samples have shape (BxH, G, D = n_dims). These are samples of the latent space described by n_dims features. 
        # We adapt the LT_matrices to the same shape of reshaped_Samples (BxH, G, D, D) #D = n_dims
        self.LT_matrices = tf.expand_dims(self.LT_matrices, axis=1)  # Shape: (B, 1, G, D,D)
        # TODO review because tile is not working in tf>2.15. . 
        self.LT_matrices = tf.tile(self.LT_matrices, [1, self.num_samples, 1, 1, 1])  # Shape: (B, H, G, D, D)
        self.LT_matrices = tf.reshape(self.LT_matrices, [-1, self.num_gaussians, self.n_dims, self.n_dims]) # Shape: (BxH, G, D,D)
        ##CLIPPING TO [0,1] to enter the FORWARD with samples belonging to the interval known during training
        # reshaped_alphas = tf.sigmoid(self.reshaped_samples) #These are our predicted alpha factors with shape (B*H, D)
        reshaped_alphas = self.reshaped_samples
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
        # Este termino tiene que ver con la mixtura que estimamos a traves de nuestro INVERSE. 
        # Para cada input sample, se extraen las propiedades de la mixtura (output de Inverse_model)y se calcula la log prob de las muestras que estamos tomando de esta distribucion mediante la funcion sampling
        #Este es el termino que va a contrarrestar el efecto del misfit de datos, haciendo que las mixturas tomen una forma montañosa en lugar de un delta dirac (esto ocurre si solo minimizamos con L_data)
        means, sigmas, weights = self.means, self.sigmas, self.weights_values
        # tf.print("sigmas", sigmas)
        # tf.print("*************")
        
        #Reshaping required to accomodate for the batch size (, num_mixtures*num_gaussians) during training
        means = tf.reshape(means, (-1, self.n_dims, self.num_gaussians))
        sigmas = tf.reshape(sigmas, (-1, self.n_dims, self.num_gaussians))
        weights = tf.reshape(weights, (-1, self.num_gaussians))
        weights = tf.clip_by_value(weights, 1e-6, 1.0)
        ## REPEATING THE MEANS TO FIT WITH THE NUMBER OF SAMPLES ,N
        m = tf.repeat(means[:,:,tf.newaxis], self.num_samples, axis = 2)
        s = tf.repeat(sigmas[:,:,tf.newaxis], self.num_samples, axis = 2)
        w = tf.repeat(weights[:,tf.newaxis], self.num_samples, axis = 1)
        ## THIS PERMUTATION IS TO MAKE THE MEANS, SIGMAS AND WEIGHTS HAVE THE DESIRED SHAPE (None*N, num_mixtures, num_gaussians):
        means = tf.reshape(tf.transpose(m, perm = [0,2,1,3]), [-1, self.n_dims, self.num_gaussians])
        sigmas = tf.reshape(tf.transpose(s, perm = [0,2,1,3]), [-1, self.n_dims, self.num_gaussians])
        weights =tf.reshape(w, [-1,self.num_gaussians]) 
        
        diag_components = []
        for i in range(self.num_gaussians):
            component = tfp.distributions.MultivariateNormalDiag(loc=means[:, :, i], scale_diag=sigmas[:, :, i])
            diag_components.append(component)
            
        
        full_components = []
        for i in range(self.num_gaussians):
            LT_matrix = self.LT_matrices[:,i,:,:]
            component2 = tfp.distributions.MultivariateNormalTriL(loc=means[:,:,i], scale_tril=LT_matrix)
            full_components.append(component2)
            
        if self.full_cov: 
            Components = full_components
        else: 
            Components = diag_components 
    
        cat = tfp.distributions.Categorical(probs=weights)
        mixture = tfp.distributions.Mixture(cat=cat, components= Components)

        # Mixture_density_loss = tf.math.reduce_mean(mixture.log_prob(self.reshaped_samples))
        # we add +1 to prevent log(0) = NaN and ensure minimum value of the term = log(1)  = 0 .
        # tf.print("PROBABILITY", mixture.prob(tf.sigmoid(self.reshaped_samples)))
        
        Mixture_density_loss = tf.math.reduce_mean(tf.math.log(mixture.prob(self.reshaped_samples) + 1.))
        return tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (Mixture_density_loss)
    

    def total_loss_with_regularizer(self, y_true, y_pred):    
        Loss_freqs = self.Freqs_loss(y_true,y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
        Regularizer = self.Alpha_regularizer(y_true, y_pred)
        Mixture_loss = self.Mixture_dens_term(y_true,y_pred)
        
        return  Loss_MAC + Loss_freqs - Regularizer + Mixture_loss
    

    
    
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
        self.gmm_sampling = GMMSamplingLayer(n_dims, num_gaussians, num_samples, full_cov)
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
        # Este termino tiene que ver con la mixtura que estimamos a traves de nuestro INVERSE. 
        # Para cada input sample, se extraen las propiedades de la mixtura (output de Inverse_model)y se calcula la log prob de las muestras que estamos tomando de esta distribucion mediante la funcion sampling
        #Este es el termino que va a contrarrestar el efecto del misfit de datos, haciendo que las mixturas tomen una forma montañosa en lugar de un delta dirac (esto ocurre si solo minimizamos con L_data)
        means, sigmas, weights = self.means, self.sigmas, self.weights_values
        
        #Reshaping required to accomodate for the batch size (, num_mixtures*num_gaussians) during training
        means = tf.reshape(means, (-1, self.n_dims, self.num_gaussians))
        sigmas = tf.reshape(sigmas, (-1, self.n_dims, self.num_gaussians))
        weights = tf.reshape(weights, (-1, self.num_gaussians))
        weights = tf.clip_by_value(weights, 1e-6, 1.0)
        ## REPEATING THE MEANS TO FIT WITH THE NUMBER OF SAMPLES ,N
        m = tf.repeat(means[:,:,tf.newaxis], self.num_samples, axis = 2)
        s = tf.repeat(sigmas[:,:,tf.newaxis], self.num_samples, axis = 2)
        w = tf.repeat(weights[:,tf.newaxis], self.num_samples, axis = 1)
        ## THIS PERMUTATION IS TO MAKE THE MEANS, SIGMAS AND WEIGHTS HAVE THE DESIRED SHAPE (None*N, num_mixtures, num_gaussians):
        means = tf.reshape(tf.transpose(m, perm = [0,2,1,3]), [-1, self.n_dims, self.num_gaussians])
        sigmas = tf.reshape(tf.transpose(s, perm = [0,2,1,3]), [-1, self.n_dims, self.num_gaussians])
        weights =tf.reshape(w, [-1,self.num_gaussians]) 
        
        diag_components = []
        for i in range(self.num_gaussians):
            component = tfp.distributions.MultivariateNormalDiag(loc=means[:, :, i], scale_diag=sigmas[:, :, i])
            diag_components.append(component)
            
        
        full_components = []
        for i in range(self.num_gaussians):
            LT_matrix = self.LT_matrices[:,i,:,:]
            component2 = tfp.distributions.MultivariateNormalTriL(loc=means[:,:,i], scale_tril=LT_matrix)
            full_components.append(component2)
            
        if self.full_cov: 
            Components = full_components
        else: 
            Components = diag_components 
    
        cat = tfp.distributions.Categorical(probs=weights)
        mixture = tfp.distributions.Mixture(cat=cat, components= Components)

        Mixture_density_loss = tf.math.reduce_mean(tf.math.log(mixture.prob(self.reshaped_samples) + 1.))
        return tf.math.square(tf.cast(self.beta, dtype=tf.float32))* (Mixture_density_loss)
    

    def total_loss_with_regularizer(self, y_true, y_pred):    
        Loss_freqs = self.Freqs_loss(y_true,y_pred)
        Loss_MAC = self.MAC_modes_loss(y_true,y_pred)
        Regularizer = self.Alpha_regularizer(y_true, y_pred)
        # Mixture_loss = self.Mixture_dens_term(y_true,y_pred)
        
        return  Loss_MAC + Loss_freqs - Regularizer
    

    
    
#     ## use this custom loss when you want to debug the second part of the code (the eigenvalue problem)
#     def custom_loss(self,y_true, y_pred):
#         loss = tf.math.reduce_mean(tf.square(y_true-y_pred), axis = None)    
#         return loss
    

#     def get_config(self):
#         config = {
#             'num_dofs': self.num_dofs,
#             'M_free': self.Mfree,
#             # 'eigensolver': self.Eigen_solver
#         }
#         base_config = super(My_Bayesian_InverseForward_withEigen, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config):
        
#         # config['Solve_eigenproblem'] = tf.keras.utils.deserialize_keras_object(config['Solve_eigenproblem'])
#         return cls(**config)












































#     # def total_loss(self, y_true, y_pred):
#     #     Freqs_loss = self.Freqs_loss(y_true, y_pred)
#     #     Modes_loss = self.MAC_modes_loss( y_true, y_pred)
#     #     beta_weight = 1.5
#     #     total_loss = Freqs_loss  + beta_weight* Modes_loss
#     #     return total_loss
        
               
        
        
#     # def Custom_eigen_loss(self, y_true, y_pred):
#     #     #This term computes the loss based on the eigenfrequencies (And eigenmodes) after solving the eigenproblem. 
#     #     #We employ the squared L2 norm:
#     #     Ydiff = tf.math.square(y_true - y_pred)
#     #     Eigen_loss = tf.math.reduce_mean(Ydiff, axis = None)
#     #     return Eigen_loss
#     # TODO MAKE A LOSS TO BE APPLIED FOR EACH FREQUENCY TO TRACK INDIVIDUAL CONTRIBUTIONS TO THE LOSS
    
#     # def f1_loss(self,y_true, y_pred):
#     #     Ydiff = tf.math.square(y_true[:,0] - y_pred[:,0])
#     #     f1_loss = tf.math.reduce_mean(Ydiff, axis = None)
#     #     return f1_loss
    
    
#     # def Stiffness_loss(self, y_true, y_pred): 
#     #     # In case that we also have N available data pairs in the form {K, f}, we can add this term to match the stiffness predictions with the ground truth 
#     #     #We take the error in the Kfree_list: 
#     #     Kloss =  tf.math.reduce_mean(tf.math.square(self.Ktrue - self.Kfree_values), axis = None)
#     #     return Kloss
    
    
#     # def total_loss(self, y_true, y_pred):
#     #     Eigenval_loss = self.Custom_eigen_loss(y_true,y_pred)
#     #     Kmisfit_loss = self.Stiffness_loss(y_true, y_pred)
#     #     beta_weight = 0.0000001
#     #     total_loss = Eigenval_loss  + beta_weight* Kmisfit_loss
#     #     return total_loss
         
        

    
    
    
# ## Old model when estimating stiffness matrix entries

# # class My_Inverse_withPhysics(tf.keras.Model):
# #     def __init__(self, input_dim, num_dofs, M_free, k_lb, k_ub):
# #         super(My_Inverse_withPhysics, self).__init__()
# #         self.num_dofs = num_dofs
# #         self.M_free = M_free
# #         self.k_lb = k_lb
# #         self.k_ub = k_ub
# #         self.FC_inverse = Fully_connected_inverse(input_dim, num_dofs, k_lb, k_ub)
# #         self.Eigen_solver = Solve_eigenproblem(num_dofs, M_free)
            
# #     def call(self, inputs):
# #         # We feed the network using both the Frequency data and the Kdata in the subset of Npoints 
# #         # i think that we need a Ftrue also that has different shape than Freqs 
# #         # Freqs will contain a huge dataset of frequency vectors sampled from that space (intelligently)
# #         # Ftrue and Ktrue are the N-dimensional dataset pairs
# #         [self.Ftrue, self.Ktrue] = inputs
        
# #         #The first step is the Inverse NN that estimates the stiffness properties in a list of size (Batch_size, ndofs*ndofs)
                
# #         Klist = self.FC_inverse(self.Ftrue) # This will calcuulate the prediced Ks for the larger database of frequencies swept throughout the frequencies space
# #         # TODO: IMPORTANT: how to apply different dataset to each term in the loss function 
# #         Kfree_diagvalues, Kfree_offvalues = tf.split(Klist, [self.num_dofs, self.num_dofs*(self.num_dofs-1)//2], axis = -1)
# #         Kfree_matrices  = create_K_matrices(Kfree_diagvalues, Kfree_offvalues, self.num_dofs)
# #         #We flatten the matrices to be a vector with the n_dofs*n_dofs elements:
# #         self.Kfree_values = tf.reshape(Kfree_matrices, (-1, self.num_dofs*self.num_dofs))
                                        
# #         # Then we enter the eigensolver function with this list to produce the eigenfrequencies
# #         eigenfreqs, eigenmodes, pred_input_features = self.Eigen_solver(self.Kfree_values)
# #         # pred_input_features includes both frequencies and modes
# #         return  pred_input_features


# class My_Inverse(tf.keras.Model):
#     def __init__(self, input_dim, num_dofs, n_elements, n_modes, epsi, **kwargs):
#         super(My_Inverse, self).__init__()
#         self.num_dofs = num_dofs
#         self.n_elements = n_elements
#         self.n_modes = n_modes
#         self.FC_inverse = Fully_connected_inverse(input_dim, num_dofs, n_elements)
#         self.epsi = epsi
            
#     def call(self, inputs):
#         # Original shapes before flattening: 
#         #freq_data size: (Batch_size, n_modes)
#         # vert_modes_data size: (Batch_size, free displ. coordinates, n_modes)
#         # rot_modes_data size: (Batch_size, free rot. coordinates, n_modes)
#         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
#         #We now flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes.         
#         self.flat_rot_modes = tf.reshape(self.rot_modes_data, [-1, self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]])
#         self.flat_vert_modes = tf.reshape(self.vert_modes_data, [-1, self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]])
#         self.modal_data = K.layers.Concatenate(axis=1)([self.freq_data, self.flat_vert_modes, self.flat_rot_modes])

#         #The first step is the Inverse NN that estimates the stiffness properties in a list of size (Batch_size, ndofs*ndofs)
#         pred_alpha_factors = self.FC_inverse(self.modal_data) 
#         return  pred_alpha_factors
    
#     def Alpha_regul(self, y_true, y_pred):
#         min_value = tf.reduce_min(y_pred, axis=-1, keepdims=False)
#         Regularizer = tf.math.reduce_mean( tf.math.reduce_sum(y_pred,axis = -1)-min_value)/(y_pred.shape[-1])-1
#         return self.epsi*Regularizer
    
#     def custom_loss(self,y_true, y_pred):
#         regul = self.Alpha_regul(y_true, y_pred)
#         loss = tf.math.reduce_mean(tf.math.square(y_true-y_pred), axis = None)    
#         return loss - regul

# # class My_Inverse(tf.keras.Model):
# #     def __init__(self, input_dim, num_dofs, n_elements, n_modes, Ke, Mfree, a_lb, a_ub):
# #         super(My_Inverse, self).__init__()
# #         self.num_dofs = num_dofs
# #         self.n_elements = n_elements
# #         self.n_modes = n_modes
# #         self.Ke = Ke # Known baseline element stiffness matrix (4x4 in 2d beam elements with vcal and rot bending modes)
# #         self.Mfree = Mfree
# #         self.a_lb = a_lb
# #         self.a_ub = a_ub
# #         self.FC_inverse = Fully_connected_inverse(input_dim, num_dofs, n_elements, a_lb, a_ub)
            
# #     def call(self, inputs):
# #         # Original shapes before flattening: 
# #         #freq_data size: (Batch_size, n_modes)
# #         # vert_modes_data size: (Batch_size, free displ. coordinates, n_modes)
# #         # rot_modes_data size: (Batch_size, free rot. coordinates, n_modes)
# #         [self.freq_data, self.rot_modes_data, self.vert_modes_data, self.alpha_factors] = inputs
# #         #We noew flatten the modeshapes to feed the Inverse DNN with a vector that contains all the frequencies and mode shapes. 
# #         self.flat_rot_modes = tf.reshape(self.rot_modes_data, (self.rot_modes_data.shape[0], self.rot_modes_data.shape[1]* self.rot_modes_data.shape[2]))
# #         self.flat_vert_modes = tf.reshape(self.vert_modes_data, (self.vert_modes_data.shape[0], self.vert_modes_data.shape[1]* self.vert_modes_data.shape[2]))
        
# #         self.modal_data = tf.concat([self.freq_data, self.flat_vert_modes, self.flat_rot_modes],  axis = 1)
# #         #The first step is the Inverse NN that estimates the stiffness properties in a list of size (Batch_size, ndofs*ndofs)
# #         pred_alpha_factors = self.FC_inverse(self.modal_data)  
# #         return  pred_alpha_factors
    
# #     # def calculate_MAC(self, y_true,y_pred):
# #     def custom_loss(self,y_true, y_pred):
# #         loss = tf.math.reduce_mean(tf.math.square(y_true-y_pred), axis = None)    
# #         return loss