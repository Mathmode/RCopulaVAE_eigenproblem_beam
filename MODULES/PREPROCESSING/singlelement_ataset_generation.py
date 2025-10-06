#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:11:14 2024

@author: afernandez
"""
###############################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ## Brief explanation :
# This script generates a dataset asuming one single element per zone and randomly affecting one of the elements.
# We specifiy the information about the beam and indicate teh interval for the alpha factors
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import tensorflow as tf
import numpy as np
import os 
from matplotlib import pyplot as plt

def solve_eigen_problem(M_free, K_free):
    L = tf.linalg.cholesky(M_free)
    L_inv = tf.linalg.inv(L)
    A = tf.matmul(tf.transpose(L_inv), tf.matmul(K_free, L_inv))
    eigenvalues, eigenvectors = tf.linalg.eigh(A)
    
    return eigenvalues, eigenvectors, L_inv
# Function to assemble the global matrices and modify one element stiffness
def generate_dataset(n_samples, n_elements, n_modes, data_file_path):
    # Constants for the beam
    E = 210e9  # Young's modulus in Pa
    I = 8.33e-6  # Second moment of area in m^4
    rho = 7850  # Density in kg/m^3
    A = 0.005  # Cross-sectional area in m^2
    L_total = 10  # Total length of the beam in m
    L_e = L_total / n_elements  # Length of each element
    # Number of degrees of freedom
    n_dofs = 2 * (n_elements + 1)

    alpha_range=(0.05, 0.95)
    frequencies_dataset = []
    eigenmodes_dataset  = []
    rot_modes_dataset = []
    vert_modes_dataset = []
    alphas_dataset = []
    K_free_dataset = []
    
    # Create stiffness and mass matrices for the beam

    Ke_base = E * I / L_e**3 * tf.constant([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ], dtype=tf.float64)
    np.save(os.path.join(data_file_path, "Ke_matrix.npy"), Ke_base)

    # Element mass matrix Me for a single beam element (4x4)
    Me = (rho * A * L_e) * tf.constant([
        [13/35 +6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 -6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
        [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
        [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
        [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
    ], dtype=tf.float64)

    for k in range(n_samples):
        # Randomly choose an element and an alpha value in the given range
        element_index = np.random.randint(0, n_elements)
        alphas = np.ones(n_elements)
        alpha = np.random.uniform(*alpha_range)

        # alpha_vals = np.random.uniform(*alpha_range,size = n_elements)
        # alphas = 1-alpha_vals
        alphas[element_index] = 1-alpha
        alphas_dataset.append(alphas)
        # alpha = 0
        
        # Create a modified stiffness matrix for the chosen element
        Ke = Ke_base.numpy()
        Ke_mod = Ke_base.numpy()
        Ke_mod = (1- alpha) * Ke_mod

        # Global stiffness and mass matrices initialization
        K_global = tf.zeros((n_dofs, n_dofs), dtype=tf.float64)
        M_global = tf.zeros((n_dofs, n_dofs), dtype=tf.float64)
        
        # Assemble the global matrices, with modification in one element's stiffness
        for i in range(n_elements):
            # #Comment for single location damage:
            # # If you do it this way, any combination of elements can suffer damage (includes multiple location damage)
            # alpha = alpha_vals[i]
            # Ke_i = (1-alpha)*Ke 
            
            dof_indices = [2*i, 2*i+1, 2*i+2, 2*i+3]  # DOFs for current element
            
            # uncomment for single location damagE:
            if i == element_index:
                Ke_i = Ke_mod  # Use modified stiffness matrix for this element
            else:
                Ke_i = Ke  # Use base stiffness matrix for other elements
            
            # Using tf.tensor_scatter_nd_add for adding element matrices into global matrices
            indices = [[i, j] for i in dof_indices for j in dof_indices]
            K_global = tf.tensor_scatter_nd_add(K_global, indices, tf.reshape(Ke_i, [-1]))
            M_global = tf.tensor_scatter_nd_add(M_global, indices, tf.reshape(Me, [-1]))

        # Apply boundary conditions
        fixed_dofs = [0, n_dofs-2]
        free_dofs = tf.constant([i for i in range(n_dofs) if i not in fixed_dofs], dtype=tf.int32)

        K_free = tf.gather(tf.gather(K_global, free_dofs, axis=0), free_dofs, axis=1)
        M_free = tf.gather(tf.gather(M_global, free_dofs, axis=0), free_dofs, axis=1)
        np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free)

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors, L_inv = solve_eigen_problem(M_free, K_free)
        
        #truncate to select the desired number of modes while keeping all the space coordinates for the modes. 
        eigenvalues_trunc = eigenvalues[0:n_modes]
        eigenvectors_trunc = eigenvectors[:,0:n_modes]

        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.sqrt(eigenvalues_trunc)
        # Convert to frequencies in Hz (ω = 2πf)
        frequencies_Hz = frequencies_rad_s / (2 * np.pi)
        # Recover the eigenvectors (mode shapes) u = L.T @ v
        eigenmodes = tf.matmul(tf.transpose(L_inv), eigenvectors_trunc)
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes[0:-1,:]
        rot_modes = np.vstack((eig_cut[0::2,:],eigenmodes[-1:,:]))
        vert_modes = eig_cut[1::2,:]
        

        frequencies_dataset.append(frequencies_Hz.numpy())
        eigenmodes_dataset.append(eigenmodes.numpy())
        K_free_dataset.append(K_free.numpy()) 
        rot_modes_dataset.append(rot_modes)
        vert_modes_dataset.append(vert_modes)
    
    return np.array(frequencies_dataset), np.array(eigenmodes_dataset), np.array(K_free_dataset), np.array(rot_modes_dataset), np.array(vert_modes_dataset), np.array(alphas_dataset)


# Generate the dataset with random stiffness reductions
n_elements = 10
n_dofs   = 2*(n_elements+1)
num_dofs = n_dofs-2 #To remove vertical displacement at both ends
N_samples = 1000 # Number of samples to generate
n_modes = 5

date = '16Nov'
folder_name  =str(date)+"SingleElement"+ str(n_elements)+"elements"+str(n_modes)+"nmodes"
data_file_path = os.path.join("Data",folder_name)

if not os.path.exists(data_file_path):
    os.makedirs(data_file_path)
    
frequencies_dataset, eigenmodes_dataset, K_free_dataset, rot_modes_dataset, vert_modes_dataset, alphas_dataset = generate_dataset(N_samples, n_elements, n_modes, data_file_path)

    
np.save(os.path.join(data_file_path, 'freqs_data_true.npy'), frequencies_dataset)
np.save(os.path.join(data_file_path, 'rotmodes_data_true.npy'), rot_modes_dataset)
np.save(os.path.join(data_file_path, 'vertmodes_data_true.npy'), vert_modes_dataset)
np.save(os.path.join(data_file_path, 'alpha_factors_true.npy'), alphas_dataset)

