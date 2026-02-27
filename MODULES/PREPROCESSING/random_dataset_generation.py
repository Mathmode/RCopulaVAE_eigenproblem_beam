#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:11:14 2024
Updated on Feb 20 2026 for Stability, Physics, and Sign Consistency

CORRECTIONS APPLIED:
1. FULL RECONSTRUCTION: Maps free DOFs back to global DOFs to ensure 
   boundary conditions (zeros) are included in the dataset.
2. OPTIMIZATION: Mass matrix is constant; Cholesky decomp is done ONCE outside the loop.
3. SLICING: Explicit separation of Vertical (u) and Rotational (theta) degrees of freedom.
"""

import numpy as np
import scipy.linalg
import os 

def generate_dataset(n_samples, n_elements, n_modes, data_file_path, zero_rotations=False):
    # --- 1. Constants & Setup ---
    E = 210e9  
    I = 8.33e-6  
    rho = 7850  
    Area = 0.005  
    L_total = 10.0  
    L_e = L_total / n_elements  
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes

    alpha_range = (0.00, 0.5)
    
    # Storage lists
    frequencies_dataset = []
    rot_modes_dataset = []
    vert_modes_dataset = []
    alphas_dataset = []
    
    # --- 2. Element Matrices ---
    k_const = E * I / L_e**3
    # Standard Euler-Bernoulli Stiffness
    Ke_base = k_const * np.array([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ], dtype=np.float64)
    
    np.save(os.path.join(data_file_path, "Ke_matrix.npy"), Ke_base)

    m_const = rho * Area * L_e
    # Consistent Mass Matrix
    Me = m_const * np.array([
        [13/35 +6.*I/(5.*Area*L_e**2), 11.*L_e/210.+I/(10.*Area*L_e), 9/70 -6*I/(5*Area*L_e**2), -13*L_e/420+ I/(10*Area*L_e)],
        [11.*L_e/210. + I/(10*Area*L_e), L_e**2/105 + 2*I/(15*Area), 13*L_e/420 - I/(10*Area*L_e), -1.*L_e**2/140 - I/(30*Area)],
        [9/70 - 6*I/(5*Area*L_e**2), 13*L_e/420 - I/(10*Area*L_e), 13/35 + 6*I/(5*Area*L_e**2),  -11*L_e/210 - I/(10*Area*L_e)],
        [-13*L_e/420 + I/(10*Area*L_e),  -L_e**2/140 - I/(30*Area), -11*L_e/210 - I/(10*Area*L_e),  L_e**2/105 + 2*I/(15*Area)]
    ], dtype=np.float64)

    # --- 3. Pre-Calculate Constant Mass Matrix & Decomposition ---
    # Assemble M_global ONCE since density/geometry don't change
    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    for i in range(n_elements):
        start_idx = 2 * i
        end_idx = start_idx + 4
        M_global[start_idx:end_idx, start_idx:end_idx] += Me

    # Define Boundary Conditions (Simply Supported)
    # Node 0: u=0 (idx 0)
    # Node N: u=0 (idx n_dofs-2)
    fixed_dofs = [0, n_dofs-2]
    all_dofs = np.arange(n_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)

    # Extract Free Mass Matrix
    M_free = M_global[np.ix_(free_dofs, free_dofs)]
    np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free)

    # Perform Cholesky Decomposition ONCE (Optimization)
    print("Pre-calculating Cholesky of Mass Matrix...")
    try:
        L = np.linalg.cholesky(M_free)
    except np.linalg.LinAlgError:
        M_free += np.eye(M_free.shape[0]) * 1e-12
        L = np.linalg.cholesky(M_free)
    L_inv = np.linalg.inv(L)
    L_inv_T = L_inv.T # Pre-compute transpose

    # --- 4. Generation Loop ---
    print(f"Generating {n_samples} samples...")
    
    for k in range(n_samples):
        if k % 500 == 0: print(f"Sample {k}")
        
        # Random Stiffness Reduction (Damage)
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

        # Apply BCs
        K_free = K_global[np.ix_(free_dofs, free_dofs)]

        # --- Solve Eigen Problem ---
        # A = L^-1 * K * L^-T
        A = L_inv @ K_free @ L_inv_T
        
        # Solve
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        # Recover Physical Modes (Free DOFs only)
        # phi_free = L^-T * y
        eigenmodes_free = L_inv_T @ eigenvectors
        
        # Truncate to n_modes
        eigenvalues_trunc = eigenvalues[0:n_modes]
        eigenmodes_free_trunc = eigenmodes_free[:, 0:n_modes]
        
        # Frequencies
        eigenvalues_trunc = np.maximum(eigenvalues_trunc, 0)
        frequencies_rad_s = np.sqrt(eigenvalues_trunc)
        frequencies_Hz = frequencies_rad_s / (2 * np.pi)

        # --- 5. Full Reconstruction & Slicing ---
        # Reconstruct the FULL vector (size n_dofs) including fixed boundaries (zeros)
        # Shape: (n_dofs, n_modes)
        full_modes = np.zeros((n_dofs, n_modes))
        full_modes[free_dofs, :] = eigenmodes_free_trunc
        
        # Now we can safely slice by definition:
        # DOFs: 0=u1, 1=th1, 2=u2, 3=th2, ...
        # Vertical (u) are even indices: 0, 2, 4...
        # Rotational (th) are odd indices: 1, 3, 5...
        vert_modes_sample = full_modes[0::2, :] # Shape: (n_nodes, n_modes)
        rot_modes_sample = full_modes[1::2, :]  # Shape: (n_nodes, n_modes)

        # --- 6. Sign Consistency ---
        # Enforce positive peak direction on the VERTICAL component (most physical)
        # Find index of max absolute value in vertical modes
        max_indices = np.argmax(np.abs(vert_modes_sample), axis=0)
        
        # Get signs at those peaks
        signs = np.sign(vert_modes_sample[max_indices, range(n_modes)])
        
        # Apply signs to both components
        vert_modes_sample = vert_modes_sample * signs
        rot_modes_sample = rot_modes_sample * signs

        if zero_rotations:
            rot_modes_sample = rot_modes_sample * 0.0

        # Transpose to store as (n_modes, n_nodes) for easier reading later?
        # Standard convention in ML is often (Features, Channels) or just flattened.
        # Here we keep (n_nodes, n_modes) to match your original flow, 
        # but usually datasets expect [Sample, Mode, Node]. 
        # Let's transpose to (n_modes, n_nodes) so each row is a mode shape.
        frequencies_dataset.append(frequencies_Hz)
        rot_modes_dataset.append(rot_modes_sample.T)
        vert_modes_dataset.append(vert_modes_sample.T)

    return (np.array(frequencies_dataset), 
            np.array(rot_modes_dataset), 
            np.array(vert_modes_dataset), 
            np.array(alphas_dataset))


# Main Execution
if __name__ == "__main__":
    n_elements = 5
    n_modes = 5
    N_samples = 10000  # Reduced for quick test
    
    # Calculate nodes for verification
    n_nodes = n_elements + 1

    folder_name = f"10D_26Feb2026_MildDam05_Randomdata{n_elements}elements"
    data_file_path = os.path.join("Data", folder_name)

    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)

    freqs, rot_modes, vert_modes, alphas = generate_dataset(
        N_samples, n_elements, n_modes, data_file_path, zero_rotations=False
    )

    print("-" * 30)
    print("GENERATION SUCCESSFUL")
    print("-" * 30)
    print(f"Number of Elements: {n_elements}")
    print(f"Number of Nodes:    {n_nodes}")
    print(f"Samples Generated:  {N_samples}")
    print("-" * 30)
    print("OUTPUT SHAPES CHECK:")
    print(f"Frequencies: {freqs.shape}  (Expected: {N_samples}, {n_modes})")
    print(f"Vert Modes:  {vert_modes.shape} (Expected: {N_samples}, {n_modes}, {n_nodes})")
    print(f"Rot Modes:   {rot_modes.shape} (Expected: {N_samples}, {n_modes}, {n_nodes})")
    
    # Verification of Boundary Conditions
    # For Simply Supported, Vertical Modes at index 0 and -1 should be 0.
    left_bc_error = np.sum(np.abs(vert_modes[:, :, 0]))
    right_bc_error = np.sum(np.abs(vert_modes[:, :, -1]))
    
    print("-" * 30)
    print(f"Boundary Condition Check (Should be 0.0):")
    print(f"Left Support Cumulative Error:  {left_bc_error:.2e}")
    print(f"Right Support Cumulative Error: {right_bc_error:.2e}")
    
    # Saving
    np.save(os.path.join(data_file_path, 'freqs_data_true.npy'), freqs)
    np.save(os.path.join(data_file_path, 'rotmodes_data_true.npy'), rot_modes)
    np.save(os.path.join(data_file_path, 'vertmodes_data_true.npy'), vert_modes)
    np.save(os.path.join(data_file_path, 'alpha_factors_true.npy'), alphas)