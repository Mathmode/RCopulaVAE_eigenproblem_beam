#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:20:39 2024

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow_probability as tfp
import numpy as np 
tfd = tfp.distributions
from tensorflow.keras.layers import Layer
from MODULES.TRAINING.rotation_matrices_funtions import batch_givens_rotation, combine_rotation_matrices


# Define a custom layer that uses tf.clip_by_value
class ClipLayer(Layer):
    def __init__(self, min_value, max_value, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)


class Mixture_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_gaussians, num_samples, full_cov, **kwargs):
        super(Mixture_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_gaussians = num_gaussians
        self.num_samples = num_samples
        self.full_cov = full_cov

    def call(self, inputs):
        means, sigmas, weights, angles = inputs
        n_angles = self.n_dims*(self.n_dims-1)//2 #Number of angles needed to build the rotation matrix for any Gaussian in the mixture. 
        # Reshaping required to accommodate for the batch size (, num_mixtures*num_gaussians) during training
        means = tf.reshape(means, (-1, self.n_dims, self.num_gaussians))
        sigmas_diag = tf.reshape(sigmas, (-1, self.n_dims, self.num_gaussians))
        weights = tf.reshape(weights, (-1, self.num_gaussians))
        angles = tf.reshape(angles, (-1, self.num_gaussians, n_angles))
        
        #Obtain the individual angle rotation matrices, with shape n_dims x n_dims.  
        rotation_matrices = batch_givens_rotation(angles, self.n_dims)
        rot_matrices = tf.reduce_prod(rotation_matrices, axis=2)  #multiplying all the individual rotation matrices 
        # rot_matrices = combine_rotation_matrices(rotation_matrices)
        #shape of rot_matrices = (batch_size, num_gaussians, n_dims, n_dims)

        RS_matrices = tf.einsum('BGDK,BKG -> BGDK', rot_matrices, sigmas_diag)
        RS_transp_matrices = tf.einsum('BGDK->BGKD', RS_matrices) #Obtain the transpose of the transformation matrix T = RS
        Sigma_matrices = tf.einsum('BGDK, BGKL -> BGDL', RS_matrices, RS_transp_matrices) # Build the Covariance matrix as C = T T^{t}
        
        # tf.print('SIGMA MATRICES', Sigma_matrices)
        LT_matrices = tf.linalg.cholesky(Sigma_matrices) #Apply Cholesk
        
        
        
        @tf.function(jit_compile=True)
        def build_mixture_single_batch(batch_means, batch_sigmas, batch_LT_matrices, batch_weights, num_gaussians, num_samples, full_cov):
            
            batch_Means = tf.transpose(batch_means, perm = [1,0])
            batch_Sigmas_diag = tf.transpose(batch_sigmas, perm = [1,0])
            if full_cov:
                components = tfd.MultivariateNormalTriL(
                    loc = batch_Means,  # Shape: [batch_size, num_gaussians, event_dim]
                    scale_tril = batch_LT_matrices  # Shape: [batch_size, num_gaussians, event_dim, event_dim]
                )
            else: 
                components = tfd.MultivariateNormalDiag(
                    loc = batch_Means,  # Shape: [batch_size, num_gaussians, event_dim]
                    scale_diag = batch_Sigmas_diag  # Shape: [batch_size, num_gaussians, event_dim, event_dim]
                )
                       
            cat = tfd.Categorical(probs=batch_weights)  # Shape: [batch_size, num_gaussians]
            mixture = tfd.MixtureSameFamily(
                mixture_distribution=cat,
                components_distribution=components)            

            return mixture
        
                
        # @tf.function(jit_compile=True)
        def sample_with_condition(vect_mixture, num_samples):
            # Function to sample one batch of points
            def draw_samples():
                return vect_mixture.sample(num_samples)
            
            # Initialize samples
            samples = draw_samples()
            
            def condition(samples):
                # Check if all samples in each batch are within the [0,1] interval
                valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)
                return tf.reduce_any(~valid_mask)
        
            def body(samples):
                # Check validity of the current samples
                valid_mask = tf.reduce_all((samples >= 0.0) & (samples <= 1.0), axis=-1)
        
                # Identify invalid samples
                invalid_batches = tf.where(~valid_mask)
        
                # Resample only the invalid samples
                resampled_samples = draw_samples()
                new_invalid_samples = tf.gather_nd(resampled_samples, invalid_batches)
        
                # Update the invalid samples in the original tensor
                samples = tf.tensor_scatter_nd_update(samples, invalid_batches, new_invalid_samples)
        
                return samples
        
            # Use tf.while_loop to repeatedly resample invalid samples until all are valid
            samples = tf.while_loop(
                condition,
                body,
                [samples],
                maximum_iterations= 500  # Ensure this doesn't run forever
            )[0]
             
            return samples

        

        # Build vectorized mixtures
        vect_mixtures = tf.vectorized_map(
            lambda batch: build_mixture_single_batch(
                batch_means=batch[0],
                batch_sigmas = batch[1],
                batch_LT_matrices=batch[2],
                batch_weights=batch[3],
                num_gaussians=self.num_gaussians,
                num_samples=self.num_samples,
                full_cov = self.full_cov
            ),
            elems=(means, sigmas_diag, LT_matrices, weights)
        )
        valid_samples = sample_with_condition(vect_mixtures, self.num_samples)
        # valid_samples  = vect_mixtures.sample(self.num_samples_per_mixture)
        valid_samples = tf.transpose(valid_samples, perm = [1,0,2])
        valid_samples = tf.reshape(valid_samples, (-1, self.n_dims))
        
        num_MC_points = 20 # we need a large number of samples to numerically approximate the analytical integral we want to solve        
        
        @tf.function(jit_compile=True)
        def compute_prob_for_batch(batch_means, batch_sigmas, batch_LT_matrices, batch_weights, num_gaussians, n_dims, num_MC_points, full_cov):
            batch_uniform_samples = tf.random.uniform(
                shape=(num_MC_points, n_dims),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32
            )
             
            batch_Means = tf.transpose(batch_means, perm = [1,0])
            batch_Sigmas_diag = tf.transpose(batch_sigmas, perm = [1,0])
            if full_cov:
                components = tfd.MultivariateNormalTriL(
                    loc = batch_Means,  # Shape: [batch_size, num_gaussians, event_dim]
                    scale_tril = batch_LT_matrices  # Shape: [batch_size, num_gaussians, event_dim, event_dim]
                )
                  
            else: 
                components = tfd.MultivariateNormalDiag(
                    loc = batch_Means,  # Shape: [batch_size, num_gaussians, event_dim]
                    scale_diag = batch_Sigmas_diag  # Shape: [batch_size, num_gaussians, event_dim, event_dim]
                )
                       
            # # Define the categorical distribution
            cat = tfd.Categorical(probs=batch_weights)  # Shape: [batch_size, num_gaussians]
            mixture = tfd.MixtureSameFamily(
                mixture_distribution=cat,
                components_distribution=components)
            prob = mixture.prob(batch_uniform_samples)     
            return prob
        
        # Apply probability computation across all batch elements
        vect_probs = tf.vectorized_map(
            lambda batch: compute_prob_for_batch(
                batch_means=batch[0],
                batch_sigmas = batch[1],
                batch_LT_matrices=batch[2],
                batch_weights=batch[3],
                num_gaussians=self.num_gaussians,
                n_dims = self.n_dims,
                num_MC_points = num_MC_points, 
                full_cov = self.full_cov
            ),
            elems=(means, sigmas_diag, LT_matrices, weights)
    )
                

        integral_value = tf.reduce_mean(vect_probs, axis= None)  # Mean over num_MC_points
        
        return LT_matrices, vect_mixtures, valid_samples, integral_value

    

def Fully_connected_gmm_inverse(input_dim, num_dimensions, num_gaussians):
    n_angles = num_dimensions*(num_dimensions-1)//2
    
    input1 = K.Input(shape =(input_dim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(256, activation='relu', kernel_initializer="he_uniform")(input1) #Intermediate layers
    lay2 = K.layers.Dense(256, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay2')(lay1) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'tanh')(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", )(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(200, activation = 'tanh')(lay2) #Intermediate layers
    # lay3 = K.layers.Dense(150, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay3')(lay2) #Intermediate layers
    lay4 = K.layers.Dense(128,  activation='relu', kernel_initializer="he_uniform", name = 'lay4')(lay2) #Intermediate layers
    means = K.layers.Dense(num_dimensions*num_gaussians, activation = 'sigmoid',  name = 'means')(lay4)
    sigmas = K.layers.Dense(num_dimensions*num_gaussians, activation = 'softplus', name = 'sigmas')(lay4)
    # sigmas = ClipLayer(s_lb, s_ub)(sigmas)
    angles = K.layers.Dense(num_gaussians*n_angles, activation = 'sigmoid', name = 'angles')(lay4)
    angles = angles*2.*np.pi  #REstriction of the angle value in rads. 
    weights = K.layers.Dense(num_gaussians, activation = 'softmax', name = 'weights')(lay4) #Intermediate layers 
    
    means = tf.convert_to_tensor(means, dtype=tf.float32)
    sigmas = tf.convert_to_tensor(sigmas, dtype=tf.float32)
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    outputs = tf.concat([means,sigmas,weights, angles], axis = 1)

    return K.Model(inputs = input1, outputs = outputs)   



def Fully_connected_enc(input_dim, num_mixtures, num_gaussians):
    input1 = K.Input(shape =(input_dim,), name = 'Inputlayer')
    lay1 = K.layers.Dense(50, activation = 'relu', name = 'El1')(input1) #Intermediate layers
    lay2 = K.layers.Dense(50, activation = 'relu', name = 'El2')(lay1) #Intermediate layers
    lay3 = K.layers.Dense(50, activation = 'relu', name = 'El3')(lay2) #Intermediate layers
    lay3 = K.layers.Dense(50, activation = 'relu', name = 'El4')(lay3) #Intermediate layers
    outputs = K.layers.Dense(num_mixtures*1, activation = tf.sigmoid, name = 'Emean')(lay3)
    return K.Model(inputs = input1, outputs = outputs)


def Fully_connected_dec(input_dim, output_dim):
     input1 = tf.keras.Input(shape =(input_dim,), name = 'InputDecoder')
     lay1 = K.layers.Dense(10, activation = 'relu', name = 'Dl1')(input1) #Intermediate layers
     lay2 = K.layers.Dense(30, activation = 'relu', name = 'Dl2')(lay1) #Intermediate layers
     lay3 = K.layers.Dense(50, activation = 'relu', name = 'Dl3')(lay2) #Intermediate layers
     lay4 = K.layers.Dense(70, activation = 'relu', name = 'Dl4')(lay3) #Intermediate layers
     lay5 = K.layers.Dense(80, activation = 'relu', name = 'Dl5')(lay4) #Intermediate layers
     output1 = K.layers.Dense(output_dim, activation = 'linear' , name = 'outputDec')(lay5)
     return K.Model(inputs = input1, outputs = output1)


class Solve_eigenproblem(Layer):
    def __init__(self, num_dofs, n_modes, Mfree, L_inv, **kwargs):
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_modes = n_modes # the number of modes we want to retain
        self.Mfree = Mfree # Fixed mass matrix (known) given a tensor*
        self.L_inv = tf.cast(L_inv,dtype = tf.float32)
        # si la funcion está escrita con otras librerias (e.j ,jax) tengo que decorarla para poder ejecutar
        #Puede que ayude a agilizar un poco el entrenamiento 
        @tf.function(jit_compile = True)
        def jit_eigh(tensor):
            eigenval, eigenvec= tf.linalg.eigh(tensor)
            return eigenval, eigenvec
        self.jit_eigh = jit_eigh
        
    def call(self, inputs):
        #The input is the tensor with the Kfree matrices with shape (Batch_size, num_dofs, num_dofs)
        Kfree_matrices = tf.cast(inputs, dtype = tf.float32) 
        ## In theory, Mfree is (nfree x nfree) but does not include batch size yet.
        
        # L = tf.linalg.cholesky(tf.cast(self.Mfree, dtype = 'float32'))
        # # Calculate the inverse of L for each matrix in the batch
        # L_inv = tf.cast(tf.linalg.inv(L), dtype = tf.float32)
        
        # Compute the generalized eigenvalue problem by transforming into a standard one for each system in the batch
        A = tf.matmul(tf.transpose(self.L_inv), tf.matmul(Kfree_matrices, self.L_inv))
        eigenvalues, eigenvectors = self.jit_eigh(A)
        # Clipping to avoid negative values (do not think it is working as we want)
        eigenvalues = tf.clip_by_value(eigenvalues, clip_value_min=1e-08, clip_value_max=1e+20) #We do this since one of the GPUs whas finding negative values (probably due to intialization issues)
        #Truncate to select the desired number of modes:
        eigenvalues_trunc = eigenvalues[:,0:self.n_modes]
        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.math.sqrt(eigenvalues_trunc)
        # Convert to frequencies in Hz (ω = 2πf)
        frequencies_Hz = frequencies_rad_s / (2 * tf.constant(np.pi, dtype='float32')) # TODO tf.constant translation to KEras
        

        eigenmodes =tf.matmul(tf.transpose(self.L_inv), eigenvectors)
        eigenmodes_trunc = eigenmodes[:,:, 0:self.n_modes]
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes_trunc[:,0:-1,:]
        rot_modes = K.layers.Concatenate(axis=1)([eig_cut[:,0::2,:],eigenmodes_trunc[:,-1:,:]])
        vert_modes = eig_cut[:,1::2,:]
        
        return frequencies_Hz, rot_modes, vert_modes


def new_generate_indices(n_elements, dof_indices):
    # Generate indices for scattering the local element matrices into global matrix
    indices = []
    for i in range(n_elements):
        idx_grid = tf.meshgrid(dof_indices[i], dof_indices[i])
        element_indices = tf.stack([tf.reshape(idx_grid[0], [-1]), tf.reshape(idx_grid[1], [-1])], axis=-1)
        # Append to indices list
        indices.append(element_indices)
    # Concatenate along the elements dimension
    return K.layers.Concatenate(axis=0)(indices)

def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples):
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype = tf.float32)
    ## Since we are drawing H samples before entering the decoder, the input dimension of Ke_matrices_dam is (Batch_size* num_samples, n_elements, 2,2)
    # TODO if num_samples brings problem, i'd avoid it as we will always use H =1 .......
    n_dofs = 2 * (n_elements + 1)  # 2 DOFs per node, n_elements + 1 nodes

    # Prepare DOF indices for each element
    #TODO meter esta variable dof_indices en el self 
    dof_indices = tf.constant([[2*i, 2*i+1, 2*i+2, 2*i+3] for i in range(n_elements)], dtype=tf.int32)

    K_flattened = tf.reshape(Ke_matrices_dam, [-1, n_elements * 16]) # Ke_matrices contains n_elements matrices of shape (4x4) and this for each sample in the batch 
    # after flattening we have the entries that are nonzero in the global assembled matrix. 
    
    nonzero_indices = new_generate_indices(n_elements, dof_indices)
    K_gg = tf.map_fn(
    lambda _: tf.zeros((n_dofs, n_dofs), dtype=tf.float32),
    elems=tf.range(tf.shape(Ke_matrices_dam)[0]),  # Dynamically infer the batch size
    fn_output_signature=tf.TensorSpec((n_dofs, n_dofs), dtype=tf.float32)
)

    def fill_matrix(nonzero_vals, matrix):
        # Assign values to the matrix
        indices = nonzero_indices  # Use the fixed indices
        updates = nonzero_vals  # Use the known nonzero values in K_flattened
        matrix = tf.tensor_scatter_nd_add(matrix, indices, updates)
        return matrix

    # Batch-wise filling
    def fillK_with_batch(batch_data):
        nonzero_vals, matrix = batch_data
        return fill_matrix(nonzero_vals, matrix)

    # Use tf.map_fn to apply the filling function across the batch
    K_global = tf.map_fn(
        fillK_with_batch,
        (K_flattened, K_gg),
        fn_output_signature=tf.TensorSpec((n_dofs, n_dofs), dtype=tf.float32)
    )
    
    fixed_dofs = [0, n_dofs-2]
    free_dofs = tf.constant([i for i in range(n_dofs) if i not in fixed_dofs], dtype= 'int32')
    ## Select the desired dofs (free ones) in the axis 1 and 2, since axis 0 corresponds to batch size. 
    K_free = tf.gather(tf.gather(K_global, free_dofs, axis=1), free_dofs, axis=2) #  transfrom tf.gather to K compatible
    # The final shape of K_free is***** (batch_size*H, n_free,n_free) * review*
    
    return K_free