#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:12:43 2024

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np 
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
tfd = tfp.distributions
from MODULES.TRAINING.rotation_matrices_funtions import batch_givens_rotation, combine_rotation_matrices

# Define a custom layer that uses tf.clip_by_value
class ClipLayer(Layer):
    def __init__(self, min_value, max_value, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

class GMMSamplingLayer(Layer):
    def __init__(self, n_dims, num_gaussians, num_samples, full_cov, **kwargs):
        super(GMMSamplingLayer, self).__init__(**kwargs)
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
        
        rotation_matrices = batch_givens_rotation(angles, self.n_dims)
        # print("Rotation Matrices shape:", rotation_matrices.shape)
        rot_matrices = tf.reduce_prod(rotation_matrices, axis=2)
        # rot_matrices = combine_rotation_matrices(rotation_matrices)
        #shape of rot_matrices = (batch_size, num_gaussians, n_dims, n_dims)

        RS_matrices = tf.einsum('BGDK,BKG -> BGDK', rot_matrices, sigmas_diag)
        # tf.print('RS_matrices', RS_matrices)
        RS_transp_matrices = tf.einsum('BGDK->BGKD', RS_matrices) #Obtain the transpose of the transformation matrix T = RS
        Sigma_matrices = tf.einsum('BGDK, BGKL -> BGDL', RS_matrices, RS_transp_matrices) # Build the Covariance matrix as C = T T^{t}
        
        # tf.print('SIGMA MATRICES', Sigma_matrices)
        LT_matrices = tf.linalg.cholesky(Sigma_matrices) #Apply Cholesk
        # tf.print('LG')
        
        diag_components = []
        for i in range(self.num_gaussians):
            component = tfp.distributions.MultivariateNormalDiag(loc=means[:,:, i], scale_diag=sigmas_diag[:, :, i ])
            diag_components.append(component)

        full_components = []
        for i in range(self.num_gaussians):
            LT_matrix = LT_matrices[:,i,:,:]
            component2 = tfp.distributions.MultivariateNormalTriL(loc=means[:,:,i], scale_tril=LT_matrix)
            full_components.append(component2)
            
        if self.full_cov: 
            Components = full_components
        else: 
            Components = diag_components 
            
        cat = tfd.Categorical(probs=weights)
        q = tfd.Mixture(cat=cat, components = Components)
        samples = q.sample(self.num_samples)
        s1=tf.transpose(samples,perm = [1,0,2])
        reshaped_samples = tf.reshape(s1, (-1, self.n_dims))
        ## The final shape after reshaping is (Batch_size * H, num_dimensions)
          
        return reshaped_samples, LT_matrices
    
## We need a neural network to map the Inverse problem: 
## We estimate one stiffness reduction factor per element (alpha) instead of the stiffness matrix elements. 
# We load the original (baseline) stiffness matrix, and apply the factors to the elements
# The estimated factors will have size (Batch_size, n_elements)
# The Stiffness Matrices (same for each element) will have size (Batch_size, n_elements, n_free, n_free)
def custom_activation(x):
    # Get the minimum value in the vector to become the one in the range (0,1)
    # and the rest will be forced to 1
    min_value = tf.math.reduce_min(x, axis=-1, keepdims=True)
    # Set the minimum value between 0 and 1 and force others to 1
    outputs = tf.where(x == min_value, tf.sigmoid(min_value), tf.ones_like(x))
    return outputs



def Fully_connected_inverse(input_dim, num_dofs, n_elements):
    input1 = K.Input(shape =(input_dim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(100, activation = 'relu', name = 'lay1')(input1) #Intermediate layers
    lay2 = K.layers.Dense(300, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay2')(lay1) #Intermediate layers
    # lay2 = K.layers.Dense(200, activation = 'tanh')(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(250, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", )(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", )(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(150, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", )(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(200, activation = 'tanh')(lay2) #Intermediate layers
    # lay3 = K.layers.Dense(50, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay3')(lay2) #Intermediate layers
    lay4 = K.layers.Dense(100, activation = 'tanh', name = 'lay4')(lay2) #Intermediate layers
    alpha_factors = K.layers.Dense(n_elements, activation = 'sigmoid', name = 'reduction_factors')(lay4)
    return K.Model(inputs = input1, outputs = alpha_factors)



def Fully_connected_gmm_inverse(input_dim, num_dimensions, num_gaussians, s_lb, s_ub):
    n_angles = num_dimensions*(num_dimensions-1)//2
    
    input1 = K.Input(shape =(input_dim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(100, activation = 'relu', name = 'lay1')(input1) #Intermediate layers
    lay2 = K.layers.Dense(250, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay2')(lay1) #Intermediate layers
    lay2 = K.layers.Dense(300, activation = 'tanh')(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", )(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(200, activation = 'tanh')(lay2) #Intermediate layers
    lay3 = K.layers.Dense(150, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros", name = 'lay3')(lay2) #Intermediate layers
    lay4 = K.layers.Dense(100, activation = 'tanh', name = 'lay4')(lay3) #Intermediate layers
    means = K.layers.Dense(num_dimensions*num_gaussians, activation = 'sigmoid',  name = 'means')(lay4)
    sigmas = K.layers.Dense(num_dimensions*num_gaussians, activation = 'softplus', name = 'sigmas')(lay4)
    sigmas  =tf.clip_by_value(sigmas, s_lb, s_ub) # use this when full_cov  = False .
    angles = K.layers.Dense(num_gaussians*n_angles, activation = 'sigmoid', name = 'angles')(lay4)
    angles = angles*2.*np.pi  #REstriction of the angle value in rads. 
    weights = K.layers.Dense(num_gaussians, activation = 'softmax', name = 'weights')(lay4) #Intermediate layers 
    
    means = tf.convert_to_tensor(means, dtype=tf.float32)
    sigmas = tf.convert_to_tensor(sigmas, dtype=tf.float32)
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    outputs = tf.concat([means,sigmas,weights, angles], axis = 1)
    # outputs = K.ops.concatenate([means,sigmas, weights, phi_angle], axis =1) # Funciona con tensorflow 2.17 

    return K.Model(inputs = input1, outputs = outputs)   




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

def new_assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples):
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype = tf.float32)
    ## Since we are drawing H samples before entering the decoder, the input dimension of Ke_matrices_dam is (Batch_size* num_samples, n_elements, 2,2)
    # TODO if num_samples brings problem, i'd avoid it as we will always use H =1 .......
    n_dofs = 2 * (n_elements + 1)  # 2 DOFs per node, n_elements + 1 nodes

    # Prepare DOF indices for each element
    #TODO meter esta variable dof_indices en el self 
    dof_indices = tf.constant([[2*i, 2*i+1, 2*i+2, 2*i+3] for i in range(n_elements)], dtype=tf.int32)

    # Flatten the Ke_matrices to shape (batch_size, n_elements*16)
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

def generate_indices(n_elements, dof_indices, batch_size):
    # Generate indices for scattering the local element matrices into global matrix
    # you can genrate the element indices without dependency on batch size, with shape (48,2) for the three element case (the nonzero entry positions)
    # then you flatten the damaged matrices that yes do have the batch size, but you can reshape as: 
    # K_flattened = tf.reshape(Ke_matrices, [-1, n_elements *16]) # the 16 is always the same as it is the element properties (4gdls in the element so the matrix is 4x4). 
        
    indices = []
    for i in range(n_elements):
        idx_grid = tf.meshgrid(dof_indices[i], dof_indices[i])
        element_indices = tf.stack([tf.reshape(idx_grid[0], [-1]), tf.reshape(idx_grid[1], [-1])], axis=-1)
        # Add batch dimension to the indices
        element_indices = tf.tile(tf.expand_dims(element_indices, axis=0), [batch_size, 1, 1])
        # Append to indices list
        indices.append(element_indices)
    # Concatenate along the elements dimension
    return K.layers.Concatenate(axis=1)(indices)

def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, batch_size):
    # TODO Batch size dependency is giving problems 
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype = tf.float32)
    batch_size = tf.cast(Ke_matrices_dam.shape[0], dtype=tf.int32)

    n_dofs = 2 * (n_elements + 1)  # 2 DOFs per node, n_elements + 1 nodes
    #Initialize the complete K matrix (Batch_size, n_dofs, n_dofs), with a total of 8*8 =64 entries per sample in the batch. 
    K_global = tf.zeros((batch_size, n_dofs, n_dofs), dtype=tf.float32)

    Kg = tf.zeros((n_dofs, n_dofs), dtype=tf.float32)
    # Prepare DOF indices for each element
    #TODO meter esta variable dof_indices en el self 
    dof_indices = tf.constant([[2*i, 2*i+1, 2*i+2, 2*i+3] for i in range(n_elements)], dtype=tf.int32)

    # Get the global indices for scattering into K_global (for all batches)
    global_indices = generate_indices(n_elements, dof_indices, batch_size)

    # Now `global_indices` has shape (batch_size, n_elements*16, 2) for each DOF combination
    # Flatten the Ke_matrices to shape (batch_size, n_elements*16)
    K_flattened = tf.reshape(Ke_matrices_dam, [-1, n_elements * 16]) # Ke_matrices contains n_elements matrices of shape (4x4) and this for each sample in the batch 
    # after flattening we have the entries that are nonzero in the global assembled matrix. 
    
    # Use tensor_scatter_nd_add to assemble the global stiffness matrices
    # Add the batch dimension to the indices to make aproper assignation of the vlaues from each matrix)
    batch_indices = tf.reshape(tf.range(batch_size, dtype=tf.int32), (batch_size, 1, 1))
    batch_indices = tf.tile(batch_indices, [1, n_elements * 16, 1])
    concat_layer = K.layers.Concatenate(axis=-1)
    full_indices = concat_layer([batch_indices, global_indices])  
    # Assemble the global stiffness matrix
    K_global = tf.tensor_scatter_nd_add(K_global, full_indices, K_flattened)
    fixed_dofs = [0, n_dofs-2]
    free_dofs = tf.constant([i for i in range(n_dofs) if i not in fixed_dofs], dtype= 'int32')
    ## Select the desired dofs (free ones) in the axis 1 and 2, since axis 0 corresponds to batch size. 
    K_free = tf.gather(tf.gather(K_global, free_dofs, axis=1), free_dofs, axis=2) # TODO transfrom tf.gather to K compatible
    return K_free


class Solve_eigenproblem(Layer):
    def __init__(self, num_dofs, n_modes, Mfree, **kwargs):
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs = num_dofs
        self.n_modes = n_modes # the number of modes we want to retain
        self.Mfree = Mfree # Fixed mass matrix (known) given a tensor*
        
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
        
        L = tf.linalg.cholesky(tf.cast(self.Mfree, dtype = 'float32'))
        # Calculate the inverse of L for each matrix in the batch
        L_inv = tf.cast(tf.linalg.inv(L), dtype = tf.float32)
        # Compute the generalized eigenvalue problem by transforming into a standard one for each system in the batch
        A = tf.matmul(tf.transpose(L_inv), tf.matmul(Kfree_matrices, L_inv))

        
        eigenvalues, eigenvectors = self.jit_eigh(A)

        # eigenvalues = tf.abs(eigenvalues)# I need to apply the sqrt so I need to enforcce them to be positive :
        eigenvalues_trunc = eigenvalues[:,0:self.n_modes]
        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.sqrt(eigenvalues_trunc)
        # Convert to frequencies in Hz (ω = 2πf)
        frequencies_Hz = frequencies_rad_s / (2 * tf.constant(np.pi, dtype='float32')) # TODO tf.constant translation to KEras
        

        eigenmodes = tf.matmul(tf.transpose(L_inv), eigenvectors)
        eigenmodes_trunc = eigenmodes[:,:, 0:self.n_modes]
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes_trunc[:,0:-1,:]
        rot_modes = K.layers.Concatenate(axis=1)([eig_cut[:,0::2,:],eigenmodes_trunc[:,-1:,:]])
        vert_modes = eig_cut[:,1::2,:]
        
        return frequencies_Hz, rot_modes, vert_modes

    