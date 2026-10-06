#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:57:06 2026

@author: anafd

Updated for Major Comment 1:
- Added Element-wise Gaussian perturbation to the Mass Matrix to simulate Modeling Error.
- Fixed random seed for reproducibility.
- Saves both perturbed (used for generation) and unperturbed (baseline for model) mass matrices.
"""

import numpy as np
import scipy.linalg
import os 

def generate_dataset(n_samples, n_elements, n_modes, data_file_path, 
                     freq_noise_level=0.05, mode_noise_level=0.05, 
                     mass_perturbation_level=0.05, zero_rotations=False):
    # --- 1. Constants & Setup ---
    # Fix the random seed for reproducible datasets
    np.random.seed(1234)
    
    E = 210e9  
    I = 8.33e-6  
    rho = 7850  
    Area = 0.005  
    L_total = 10.0  
    L_e = L_total / n_elements  
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes

    alpha_range = (0.00, 0.5) # Damage up to 50%
    
    # Storage lists
    frequencies_dataset = []
    rot_modes_dataset = []
    vert_modes_dataset = []
    alphas_dataset = []
    
    # --- 2. Element Matrices ---
    k_const = E * I / L_e**3
    Ke_base = k_const * np.array([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ], dtype=np.float64)
    
    np.save(os.path.join(data_file_path, "Ke_matrix.npy"), Ke_base)

    m_const = rho * Area * L_e
    Me_baseline = m_const * np.array([
        [13/35 +6.*I/(5.*Area*L_e**2), 11.*L_e/210.+I/(10.*Area*L_e), 9/70 -6*I/(5*Area*L_e**2), -13*L_e/420+ I/(10*Area*L_e)],
        [11.*L_e/210. + I/(10*Area*L_e), L_e**2/105 + 2*I/(15*Area), 13*L_e/420 - I/(10*Area*L_e), -1.*L_e**2/140 - I/(30*Area)],
        [9/70 - 6*I/(5*Area*L_e**2), 13*L_e/420 - I/(10*Area*L_e), 13/35 + 6*I/(5*Area*L_e**2),  -11*L_e/210 - I/(10*Area*L_e)],
        [-13*L_e/420 + I/(10*Area*L_e),  -L_e**2/140 - I/(30*Area), -11*L_e/210 - I/(10*Area*L_e),  L_e**2/105 + 2*I/(15*Area)]
    ], dtype=np.float64)

    # --- 3. Pre-Calculate Global Mass & Cholesky ---
    M_global_baseline = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    M_global_perturbed = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    
    for i in range(n_elements):
        start_idx = 2 * i
        end_idx = start_idx + 4
        
        # Assemble Baseline (Unperturbed)
        M_global_baseline[start_idx:end_idx, start_idx:end_idx] += Me_baseline
        
        # Assemble Perturbed Mass (Modeling Error applied at element level)
        perturbation_factor = 1.0 + np.random.normal(0, mass_perturbation_level)
        Me_perturbed = Me_baseline * perturbation_factor
        M_global_perturbed[start_idx:end_idx, start_idx:end_idx] += Me_perturbed

    fixed_dofs = [0, n_dofs-2] # Simply Supported
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)

    M_free_baseline = M_global_baseline[np.ix_(free_dofs, free_dofs)]
    M_free_perturbed = M_global_perturbed[np.ix_(free_dofs, free_dofs)]
    
    # Save the baseline as 'Mass_matrix.npy' so the model loads it as the "known" formulation
    np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free_baseline)
    # Save the perturbed mass to verify what actually generated the physical data
    np.save(os.path.join(data_file_path, "Mass_matrix_perturbed.npy"), M_free_perturbed)

    # Decompose the TRUE (Perturbed) mass for forward dynamics simulation
    L = np.linalg.cholesky(M_free_perturbed)
    L_inv = np.linalg.inv(L)
    L_inv_T = L_inv.T

    # --- 4. Generation Loop ---
    print(f"Generating {n_samples} noisy samples using perturbed mass (Error Level: {mass_perturbation_level*100}%)...")
    
    for k in range(n_samples):
        # Random Stiffness Reduction
        alpha_vals = np.random.uniform(*alpha_range, size=n_elements)
        alphas = 1 - alpha_vals
        alphas_dataset.append(alphas)
        
        # Assemble Global Stiffness
        K_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
        for i in range(n_elements):
            Ke_i = (1 - alpha_vals[i]) * Ke_base 
            start_idx = 2 * i
            end_idx = start_idx + 4
            K_global[start_idx:end_idx, start_idx:end_idx] += Ke_i

        K_free = K_global[np.ix_(free_dofs, free_dofs)]

        # --- Solve Eigen Problem ---
        # The structural response is dictated by the Perturbed Mass matrix (L_inv)
        A = L_inv @ K_free @ L_inv_T
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenmodes_free = L_inv_T @ eigenvectors
        
        # Truncate
        eigenvalues_trunc = np.maximum(eigenvalues[0:n_modes], 0)
        eigenmodes_free_trunc = eigenmodes_free[:, 0:n_modes]
        
        # Frequencies (True)
        frequencies_Hz = np.sqrt(eigenvalues_trunc) / (2 * np.pi)

        # --- 5. Noise Injection Step ---
        # Frequency Noise: Gaussian relative to value
        f_noise = np.random.normal(0, freq_noise_level, size=n_modes)
        frequencies_Hz_noisy = frequencies_Hz * (1 + f_noise)

        # Mode Shape Reconstruction
        full_modes = np.zeros((n_dofs, n_modes))
        full_modes[free_dofs, :] = eigenmodes_free_trunc
        
        vert_modes_sample = full_modes[0::2, :] 
        rot_modes_sample = full_modes[1::2, :]

        # Mode Shape Noise: Gaussian relative to peak amplitude of each mode
        for m in range(n_modes):
            # Noise for Vertical
            v_max = np.max(np.abs(vert_modes_sample[:, m]))
            v_noise = np.random.normal(0, mode_noise_level * v_max, size=n_nodes)
            vert_modes_sample[:, m] += v_noise

            # Noise for Rotational
            r_max = np.max(np.abs(rot_modes_sample[:, m]))
            if r_max > 1e-12: # Avoid div by zero
                r_noise = np.random.normal(0, mode_noise_level * r_max, size=n_nodes)
                rot_modes_sample[:, m] += r_noise

        # --- 6. Sign Consistency (Post-Noise) ---
        max_indices = np.argmax(np.abs(vert_modes_sample), axis=0)
        signs = np.sign(vert_modes_sample[max_indices, range(n_modes)])
        vert_modes_sample *= signs
        rot_modes_sample *= signs

        if zero_rotations:
            rot_modes_sample *= 0.0

        frequencies_dataset.append(frequencies_Hz_noisy)
        rot_modes_dataset.append(rot_modes_sample.T)
        vert_modes_dataset.append(vert_modes_sample.T)

    return (np.array(frequencies_dataset), 
            np.array(rot_modes_dataset), 
            np.array(vert_modes_dataset), 
            np.array(alphas_dataset))

if __name__ == "__main__":
    n_elements = 5
    n_modes = 5
    N_samples = 20000 
    
    # Noise and Perturbation Levels
    F_NOISE = 0.05
    M_NOISE = 0.05
    MASS_PERTURB = 0.05 # 5% Gaussian perturbation to the baseline mass matrix (modeling error)

    folder_name = f"06Oct_Data_Noisy_E{n_elements}_Lvl{int(F_NOISE*1000)}_MassPerturb{int(MASS_PERTURB*100)}"
    data_file_path = os.path.join("Data", folder_name)

    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)

    freqs, rot, vert, alphas = generate_dataset(
        N_samples, n_elements, n_modes, data_file_path,
        freq_noise_level=F_NOISE, mode_noise_level=M_NOISE,
        mass_perturbation_level=MASS_PERTURB
    )

    # Save Arrays
    np.save(os.path.join(data_file_path, 'freqs_data_true.npy'), freqs)
    np.save(os.path.join(data_file_path, 'vertmodes_data_true.npy'), vert)
    np.save(os.path.join(data_file_path, 'rotmodes_data_true.npy'), rot)
    np.save(os.path.join(data_file_path, 'alpha_factors_true.npy'), alphas)
    
    print(f"Dataset saved to {data_file_path}")
    print(f"Measurement Noise applied: F={F_NOISE*100}%, M={M_NOISE*100}%")
    print(f"Modeling Error applied (Mass Perturbation): {MASS_PERTURB*100}%")