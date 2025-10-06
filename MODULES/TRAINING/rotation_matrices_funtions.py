#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:10:39 2024

@author: afernandez
"""

# FUNCTIONS REQUIRED TO CALCULATE THE ROTATION MATRICES IN HIGH DIMENSIONAL SPACES. STARTING FROM GIVENS MATRIX FORM 

import tensorflow as tf
import tensorflow_probability as tfp
import keras as K 
tfd = tfp.distributions


@tf.function(jit_compile = True)
def generate_givens_rotation(n_dims, i, j, theta):
    """
    Create a Givens rotation matrix for dimensions `i` and `j` in `num_dimensions`.
    """
    rotation_matrix = tf.eye(n_dims, dtype=tf.float32)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)

    # Update the matrix elements for the Givens rotation
    rotation_matrix = tf.tensor_scatter_nd_update(
        rotation_matrix, [[i, i], [i, j], [j, i], [j, j]],
        [cos_theta, -sin_theta, sin_theta, cos_theta]
    )
    return rotation_matrix



@tf.function(jit_compile = True)
def copula_batch_givens_rotation(angles, n_dims):
    """
    Generate Givens rotation matrices for each batch, Gaussian, and angle.
    Args: 
        angles: Tensor of shape (batch_size, num_gaussians, n_angles), where n_angles = n_dims * (n_dims - 1) // 2.
        n_dims: The dimensionality of the space.
    Returns a tensor of shape (batch_size, num_gaussians, n_angles, n_dims, n_dims).
    """
    indices = [
        (i, j) for i in range(n_dims) for j in range(i + 1, n_dims)
    ]
    
    # Build rotation matrices for each batch and Gaussian
    def process_for_batch(gaussian_angles):
        """
        Process the angles for a single Gaussian (n_angles conform the Gaussian).
        """
        rotation_matrices = []
        for k, (i, j) in enumerate(indices):
            theta = gaussian_angles[k]
            rotation_matrix = generate_givens_rotation(n_dims, i, j, theta)
            rotation_matrices.append(rotation_matrix)
        return tf.stack(rotation_matrices)

    return tf.vectorized_map(process_for_batch, angles)



# Define the main function
@tf.function(jit_compile = True)
def batch_givens_rotation(angles, n_dims):
    """
    Generate Givens rotation matrices for each batch, Gaussian, and angle.
    Args: 
        angles: Tensor of shape (batch_size, num_gaussians, n_angles), where n_angles = n_dims * (n_dims - 1) // 2.
        n_dims: The dimensionality of the space.
    Returns a tensor of shape (batch_size, num_gaussians, n_angles, n_dims, n_dims).
    """
    indices = [
        (i, j) for i in range(n_dims) for j in range(i + 1, n_dims)
    ]
    
    # Build rotation matrices for each batch and Gaussian
    def process_for_gaussian(gaussian_angles):
        """
        Process the angles for a single Gaussian (n_angles conform the Gaussian).
        """
        rotation_matrices = []
        for k, (i, j) in enumerate(indices):
            theta = gaussian_angles[k]
            rotation_matrix = generate_givens_rotation(n_dims, i, j, theta)
            rotation_matrices.append(rotation_matrix)
        return tf.stack(rotation_matrices)

    def process_for_batch(batch_angles):
        """
        Process all Gaussians for a single batch (with n_gaussians).
        """
        return tf.vectorized_map(process_for_gaussian, batch_angles)

    return tf.vectorized_map(process_for_batch, angles)



# Function to compute the final rotation matrices
@tf.function(jit_compile = True)
def combine_rotation_matrices(rotation_matrices):
    """  
    Combine rotation matrices by multiplying along the n_angles dimension.
    Args: rotation_matrices: Tensor of shape (batch_size, num_gaussians, num_angles, n_dims, n_dims)
    Returns: Tensor of shape (batch_size, num_gaussians, n_dims, n_dims)
    """
    # Function to multiply matrices along num_angles
    def multiply_along_angles(matrices):
        # Fold along num_angles (axis=2) to compute the matrix product
        return tf.foldl(lambda a, b: tf.linalg.matmul(a, b), matrices)
    
    # Apply the multiplication for each batch and Gaussian
    return tf.map_fn(
        lambda gaussian_matrices: tf.map_fn(multiply_along_angles, gaussian_matrices),
        rotation_matrices
    )