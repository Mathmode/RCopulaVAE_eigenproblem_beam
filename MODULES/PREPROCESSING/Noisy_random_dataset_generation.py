# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:57:06 2026

@author: anafd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 20 2026
Updated to include Stochastic Noise Injection for Uncertainty-Aware Training.

IMPROVEMENTS:
1. NOISE INJECTION: Added Gaussian noise to frequencies (1%) and mode shapes (3%).
2. STOCHASTICITY: Ensures that the VAE trains on realistic, imperfect observations.
3. PHYSICAL CONSISTENCY: Maintains sign consistency even after noise addition.
"""

import numpy as np
import scipy.linalg
import os 

def generate_dataset(n_samples, n_elements, n_modes, data_file_path, 
                     freq_noise_level=0.05, mode_noise_level=0.05, zero_rotations=False):
    # --- 1. Constants & Setup ---
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
    Me = m_const * np.array([
        [13/35 +6.*I/(5.*Area*L_e**2), 11.*L_e/210.+I/(10.*Area*L_e), 9/70 -6*I/(5*Area*L_e**2), -13*L_e/420+ I/(10*Area*L_e)],
        [11.*L_e/210. + I/(10*Area*L_e), L_e**2/105 + 2*I/(15*Area), 13*L_e/420 - I/(10*Area*L_e), -1.*L_e**2/140 - I/(30*Area)],
        [9/70 - 6*I/(5*Area*L_e**2), 13*L_e/420 - I/(10*Area*L_e), 13/35 + 6*I/(5*Area*L_e**2),  -11*L_e/210 - I/(10*Area*L_e)],
        [-13*L_e/420 + I/(10*Area*L_e),  -L_e**2/140 - I/(30*Area), -11*L_e/210 - I/(10*Area*L_e),  L_e**2/105 + 2*I/(15*Area)]
    ], dtype=np.float64)

    # --- 3. Pre-Calculate Global Mass & Cholesky ---
    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    for i in range(n_elements):
        start_idx = 2 * i
        end_idx = start_idx + 4
        M_global[start_idx:end_idx, start_idx:end_idx] += Me

    fixed_dofs = [0, n_dofs-2] # Simply Supported
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)

    M_free = M_global[np.ix_(free_dofs, free_dofs)]
    np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free)

    L = np.linalg.cholesky(M_free)
    L_inv = np.linalg.inv(L)
    L_inv_T = L_inv.T

    # --- 4. Generation Loop ---
    print(f"Generating {n_samples} noisy samples...")
    
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
    N_samples = 10000 
    
    # 2.5% Frequency Noise, 5% Mode Shape Noise
    F_NOISE = 0.025
    M_NOISE = 0.025

    folder_name = f"01Mar2026_Noisy_E{n_elements}_025_level{int(F_NOISE*100)}"
    data_file_path = os.path.join("Data", folder_name)

    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)

    freqs, rot, vert, alphas = generate_dataset(
        N_samples, n_elements, n_modes, data_file_path,
        freq_noise_level=F_NOISE, mode_noise_level=M_NOISE
    )

    # Save
    np.save(os.path.join(data_file_path, 'freqs_data_noisy.npy'), freqs)
    np.save(os.path.join(data_file_path, 'vertmodes_data_noisy.npy'), vert)
    np.save(os.path.join(data_file_path, 'rotmodes_data_noisy.npy'), rot)
    np.save(os.path.join(data_file_path, 'alpha_factors_true.npy'), alphas)
    
    print(f"Dataset saved to {data_file_path}")
    print(f"Noise levels applied: F={F_NOISE*100}%, M={M_NOISE*100}%")