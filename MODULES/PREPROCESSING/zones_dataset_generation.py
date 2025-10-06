#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:46:08 2024

@author: afernandez
"""

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


# Function to assemble the global matrices and modify one zone stiffness
def generate_dataset(n_samples, n_elements, n_modes, data_file_path, g=5):
    # Constants for the beam
    E = 210e9  # Young's modulus in Pa
    I = 8.33e-6  # Second moment of area in m^4
    rho = 7850  # Density in kg/m^3
    A = 0.005  # Cross-sectional area in m^2
    L_total = 100  # Total length of the beam in m
    L_e = L_total / n_elements  # Length of each element
    # Number of degrees of freedom
    n_dofs = 2 * (n_elements + 1)

    alpha_range = (0.25, 1.0)
    frequencies_dataset = []
    eigenmodes_dataset = []
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
        [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210. + I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420 + I/(10*A*L_e)],
        [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
        [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2), -11*L_e/210 - I/(10*A*L_e)],
        [-13*L_e/420 + I/(10*A*L_e), -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e), L_e**2/105 + 2*I/(15*A)]
    ], dtype=tf.float64)

    # Calculate the number of zones
    n_zones = n_elements // g

    for k in range(n_samples):
        # Randomly choose a zone and an alpha value in the given range
        zone_index = np.random.randint(0, n_zones)
        element_start = zone_index * g
        element_end = element_start + g
        alphas = np.ones(n_elements)
        alpha = np.random.uniform(*alpha_range)

        # Apply the same alpha reduction to all elements in the chosen zone
        alphas[element_start:element_end] = 1 - alpha
        alphas_dataset.append(alphas)

        # Create a modified stiffness matrix for the chosen zone
        Ke = Ke_base.numpy()
        Ke_mod = Ke_base.numpy()
        Ke_mod = (1 - alpha) * Ke_mod #This is for an affected element (within the zone)
        
        # Global stiffness and mass matrices initialization
        K_global = tf.zeros((n_dofs, n_dofs), dtype=tf.float64)
        M_global = tf.zeros((n_dofs, n_dofs), dtype=tf.float64)
        
        # Assemble the global matrices, with modification in the chosen zone's stiffness
        for i in range(n_elements):
            dof_indices = [2*i, 2*i+1, 2*i+2, 2*i+3]  # DOFs for current element
            
            if element_start <= i < element_end:
                Ke_i = Ke_mod  # Use modified stiffness matrix for elements in the selected zone
            else:
                Ke_i = Ke  # Use base stiffness matrix for other elements
            
            # Using tf.tensor_scatter_nd_add for adding element matrices into global matrices
            indices = [[i, j] for i in dof_indices for j in dof_indices]
            K_global = tf.tensor_scatter_nd_add(K_global, indices, tf.reshape(Ke_i, [-1]))
            M_global = tf.tensor_scatter_nd_add(M_global, indices, tf.reshape(Me, [-1]))

        # Apply boundary conditions
        fixed_dofs = [0, n_dofs - 2]
        free_dofs = tf.constant([i for i in range(n_dofs) if i not in fixed_dofs], dtype=tf.int32)

        K_free = tf.gather(tf.gather(K_global, free_dofs, axis=0), free_dofs, axis=1)
        M_free = tf.gather(tf.gather(M_global, free_dofs, axis=0), free_dofs, axis=1)
        np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free)

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors, L_inv = solve_eigen_problem(M_free, K_free)
        
        # Truncate to select the desired number of modes while keeping all the space coordinates for the modes. 
        eigenvalues_trunc = eigenvalues[0:n_modes]
        eigenvectors_trunc = eigenvectors[:, 0:n_modes]

        # Natural frequencies (ω) in rad/s
        frequencies_rad_s = tf.sqrt(eigenvalues_trunc)
        # Convert to frequencies in Hz (ω = 2πf)
        frequencies_Hz = frequencies_rad_s / (2 * np.pi)
        # Recover the eigenvectors (mode shapes) u = L.T @ v
        eigenmodes = tf.matmul(tf.transpose(L_inv), eigenvectors_trunc)
        # R|V|R|V|R|...|R|R|
        eig_cut = eigenmodes[0:-1, :]
        rot_modes = np.vstack((eig_cut[0::2, :], eigenmodes[-1:, :]))
        vert_modes = eig_cut[1::2, :]

        frequencies_dataset.append(frequencies_Hz.numpy())
        eigenmodes_dataset.append(eigenmodes.numpy())
        K_free_dataset.append(K_free.numpy()) 
        rot_modes_dataset.append(rot_modes)
        vert_modes_dataset.append(vert_modes)
    
    return np.array(frequencies_dataset), np.array(eigenmodes_dataset), np.array(K_free_dataset), np.array(rot_modes_dataset), np.array(vert_modes_dataset), np.array(alphas_dataset)


# Define parameters for the dataset generation
n_elements = 20          # Number of elements in the beam
n_modes = 10             # Number of modes to consider
N_samples = 10000        # Number of samples to generate
g = 3                    # Number of elements per zone (only one zone can be damaged at a time)
date = '16Nov'           # Date or identifier for naming the folder
folder_name = f"{date}_ZoneData_{n_elements}elements_{n_modes}modes_{g}elements_per_zone"
data_file_path = os.path.join("Data", folder_name)

# Create the directory if it does not exist
if not os.path.exists(data_file_path):
    os.makedirs(data_file_path)

# Generate the dataset using the modified function
frequencies_dataset, eigenmodes_dataset, K_free_dataset, rot_modes_dataset, vert_modes_dataset, alphas_dataset = generate_dataset(
    N_samples, 
    n_elements, 
    n_modes, 
    data_file_path, 
    g=g
)

# Save the generated datasets to files for later use
np.save(os.path.join(data_file_path, 'freqs_data_true.npy'), frequencies_dataset)
np.save(os.path.join(data_file_path, 'rotmodes_data_true.npy'), rot_modes_dataset)
np.save(os.path.join(data_file_path, 'vertmodes_data_true.npy'), vert_modes_dataset)
np.save(os.path.join(data_file_path, 'alpha_factors_true.npy'), alphas_dataset)

print("Dataset generation complete. Files saved in:", data_file_path)




