#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgraded FE Model & Dataset Generator for JCCE Revision (MS CPENG-8674)
Addresses:
- R2.2 & R2.3: Elastic vertical bearing springs at bridge ends (support damage identification)
- R1.1: Fine-mesh vs. Coarse-mesh model-form uncertainty (sub_elements_per_zone)
- R3.1: Unmodeled mass matrix perturbations (mass_perturb_level)
- R1.5 & R1.6: Healthy, Mild Localized (1-2 elements), Distributed, and OOD damage regimes
"""

import os
import numpy as np
import scipy.linalg


def get_beam_element_matrices(E, I, rho, Area, L_e):
    """Returns standard 4x4 Euler-Bernoulli stiffness and consistent/rotary mass matrices."""
    k_const = E * I / (L_e ** 3)
    Ke = k_const * np.array([
        [12.0,      6.0*L_e,   -12.0,      6.0*L_e],
        [6.0*L_e,   4.0*L_e**2, -6.0*L_e,  2.0*L_e**2],
        [-12.0,    -6.0*L_e,    12.0,     -6.0*L_e],
        [6.0*L_e,   2.0*L_e**2, -6.0*L_e,  4.0*L_e**2]
    ], dtype=np.float64)

    m_const = rho * Area * L_e
    Me = m_const * np.array([
        [13/35 + 6*I/(5*Area*L_e**2),      11*L_e/210 + I/(10*Area*L_e),    9/70 - 6*I/(5*Area*L_e**2),      -13*L_e/420 + I/(10*Area*L_e)],
        [11*L_e/210 + I/(10*Area*L_e),     L_e**2/105 + 2*I/(15*Area),      13*L_e/420 - I/(10*Area*L_e),    -L_e**2/140 - I/(30*Area)],
        [9/70 - 6*I/(5*Area*L_e**2),       13*L_e/420 - I/(10*Area*L_e),    13/35 + 6*I/(5*Area*L_e**2),     -11*L_e/210 - I/(10*Area*L_e)],
        [-13*L_e/420 + I/(10*Area*L_e),   -L_e**2/140 - I/(30*Area),       -11*L_e/210 - I/(10*Area*L_e),     L_e**2/105 + 2*I/(15*Area)]
    ], dtype=np.float64)

    return Ke, Me


def save_baseline_decoder_matrices(n_zones, data_file_path, E=210e9, I=8.33e-6,
                                   rho=7850.0, Area=0.005, L_total=10.0,
                                   use_elastic_bearings=True, k_bearing_0=5e7):
    """
    Builds and saves the coarse baseline matrices (Mass_matrix.npy, Ke_matrix.npy)
    used directly by the Physics-Informed VAE Decoder.
    """
    L_e = L_total / n_zones
    n_nodes = n_zones + 1
    n_dofs = 2 * n_nodes

    Ke_base, Me_base = get_beam_element_matrices(E, I, rho, Area, L_e)

    # Assemble nominal global mass matrix
    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    for i in range(n_zones):
        idx = 2 * i
        M_global[idx:idx+4, idx:idx+4] += Me_base

    if use_elastic_bearings:
        # All DOFs are free! Vertical springs sit at DOF 0 and DOF (n_dofs - 2)
        fixed_dofs = []
        free_dofs = np.arange(n_dofs)
        n_dims = n_zones + 2  # n_zones beam elements + 2 elastic bearings

        # Build global-sized Ke_matrices of shape (n_dims, n_dofs, n_dofs)
        # Or element-level if you prefer, but global-sized makes spring assembly seamless
        Ke_list = []
        for i in range(n_zones):
            K_elem_G = np.zeros((n_dofs, n_dofs), dtype=np.float64)
            idx = 2 * i
            K_elem_G[idx:idx+4, idx:idx+4] = Ke_base
            Ke_list.append(K_elem_G)

        # Left bearing vertical spring (DOF 0)
        K_left_G = np.zeros((n_dofs, n_dofs), dtype=np.float64)
        K_left_G[0, 0] = k_bearing_0
        Ke_list.append(K_left_G)

        # Right bearing vertical spring (DOF n_dofs - 2)
        K_right_G = np.zeros((n_dofs, n_dofs), dtype=np.float64)
        K_right_G[n_dofs - 2, n_dofs - 2] = k_bearing_0
        Ke_list.append(K_right_G)

        Ke_export = np.stack(Ke_list, axis=0)  # Shape: (n_zones + 2, n_dofs, n_dofs)
    else:
        fixed_dofs = [0, n_dofs - 2]
        free_dofs = np.delete(np.arange(n_dofs), fixed_dofs)
        Ke_export = Ke_base  # Original 4x4 matrix

    M_free = M_global[np.ix_(free_dofs, free_dofs)]
    np.save(os.path.join(data_file_path, "Mass_matrix.npy"), M_free)
    np.save(os.path.join(data_file_path, "Ke_matrix.npy"), Ke_export)

    return M_free, Ke_export, fixed_dofs, free_dofs


def sample_damage_vectors(n_samples, n_dims, regime="mixed_training", lbound=0.50):
    """
    Samples stiffness retention factors z in [lbound, 1.0] (1.0 = healthy).
    Addresses Reviewer 1 (localized, mild, healthy, OOD) and Reviewer 3 (prior/sampling).
    """
    z_mat = np.ones((n_samples, n_dims), dtype=np.float64)

    for k in range(n_samples):
        if regime == "mixed_training":
            # Balanced mixture for amortized encoder training:
            # 10% Healthy/near-healthy, 35% 1-element localized,
            # 25% 2-element localized, 30% fully distributed
            u = np.random.rand()
            if u < 0.10:
                z_mat[k, :] = np.random.uniform(0.96, 1.00, size=n_dims)
            elif u < 0.45:
                z_mat[k, :] = np.random.uniform(0.97, 1.00, size=n_dims)
                idx = np.random.choice(n_dims, size=1, replace=False)
                z_mat[k, idx] = np.random.uniform(lbound, 0.95, size=1)
            elif u < 0.70:
                z_mat[k, :] = np.random.uniform(0.97, 1.00, size=n_dims)
                idx = np.random.choice(n_dims, size=2, replace=False)
                z_mat[k, idx] = np.random.uniform(lbound, 0.95, size=2)
            else:
                z_mat[k, :] = np.random.uniform(lbound, 1.00, size=n_dims)

        elif regime == "healthy":
            z_mat[k, :] = np.random.uniform(0.98, 1.00, size=n_dims)

        elif regime == "localized_mild":
            z_mat[k, :] = np.random.uniform(0.98, 1.00, size=n_dims)
            n_dam = np.random.choice([1, 2])
            idx = np.random.choice(n_dims, size=n_dam, replace=False)
            z_mat[k, idx] = np.random.uniform(0.80, 0.95, size=n_dam)

        elif regime == "localized_severe":
            z_mat[k, :] = np.random.uniform(0.98, 1.00, size=n_dims)
            n_dam = np.random.choice([1, 2])
            idx = np.random.choice(n_dims, size=n_dam, replace=False)
            z_mat[k, idx] = np.random.uniform(lbound, 0.80, size=n_dam)

        elif regime == "distributed":
            z_mat[k, :] = np.random.uniform(lbound, 1.00, size=n_dims)

        elif regime == "ood_severe":
            # Out-of-distribution: damage below training lower bound [0.35, 0.50]
            z_mat[k, :] = np.random.uniform(0.95, 1.00, size=n_dims)
            n_dam = np.random.choice([1, 2, n_dims])
            idx = np.random.choice(n_dims, size=n_dam, replace=False)
            z_mat[k, idx] = np.random.uniform(0.35, lbound, size=n_dam)

        else:
            raise ValueError(f"Unknown regime: {regime}")

    return z_mat


def generate_dataset(n_samples, n_zones, n_modes,
                     sub_elements_per_zone=1,
                     mass_perturb_level=0.0,
                     freq_noise_level=0.025,
                     mode_noise_level=0.05,
                     use_elastic_bearings=True,
                     k_bearing_0=5e7,
                     regime="mixed_training",
                     lbound=0.50,
                     zero_rotations=False):
    """
    Generates modal observations (frequencies, vertical modes, rotational modes)
    at the macro-sensor nodes (n_zones + 1), while optionally solving the forward
    physics on a finer mesh (n_zones * sub_elements_per_zone) with perturbed mass.
    """
    E = 210e9
    I = 8.33e-6
    rho = 7850.0
    Area = 0.005
    L_total = 10.0

    n_fine_elems = n_zones * sub_elements_per_zone
    L_e_fine = L_total / n_fine_elems
    n_fine_nodes = n_fine_elems + 1
    n_fine_dofs = 2 * n_fine_nodes

    # Indices of the macro sensor nodes in the fine mesh
    sensor_node_indices = np.arange(0, n_fine_nodes, sub_elements_per_zone)
    n_sensor_nodes = len(sensor_node_indices)

    Ke_fine_base, Me_fine_base = get_beam_element_matrices(E, I, rho, Area, L_e_fine)

    if use_elastic_bearings:
        fixed_dofs_fine = []
        n_dims = n_zones + 2
    else:
        fixed_dofs_fine = [0, n_fine_dofs - 2]
        n_dims = n_zones

    free_dofs_fine = np.delete(np.arange(n_fine_dofs), fixed_dofs_fine)

    # Sample ground-truth stiffness factors z
    alphas_dataset = sample_damage_vectors(n_samples, n_dims, regime=regime, lbound=lbound)

    frequencies_dataset = []
    rot_modes_dataset = []
    vert_modes_dataset = []

    for k in range(n_samples):
        z_vec = alphas_dataset[k]

        K_global = np.zeros((n_fine_dofs, n_fine_dofs), dtype=np.float64)
        M_global = np.zeros((n_fine_dofs, n_fine_dofs), dtype=np.float64)

        for zone_idx in range(n_zones):
            z_zone = z_vec[zone_idx]
            for sub_idx in range(sub_elements_per_zone):
                elem_idx = zone_idx * sub_elements_per_zone + sub_idx
                start_dof = 2 * elem_idx
                end_dof = start_dof + 4

                # Stiffness assembly
                K_global[start_dof:end_dof, start_dof:end_dof] += z_zone * Ke_fine_base

                # Mass assembly with optional random element-level perturbation (R3.1)
                if mass_perturb_level > 0.0:
                    m_factor = 1.0 + np.random.uniform(-mass_perturb_level, mass_perturb_level)
                else:
                    m_factor = 1.0
                M_global[start_dof:end_dof, start_dof:end_dof] += m_factor * Me_fine_base

        # Add elastic bearing springs at ends if enabled (R2.2 & R2.3)
        if use_elastic_bearings:
            z_left_bearing = z_vec[n_zones]
            z_right_bearing = z_vec[n_zones + 1]
            K_global[0, 0] += z_left_bearing * k_bearing_0
            K_global[n_fine_dofs - 2, n_fine_dofs - 2] += z_right_bearing * k_bearing_0

        K_free = K_global[np.ix_(free_dofs_fine, free_dofs_fine)]
        M_free = M_global[np.ix_(free_dofs_fine, free_dofs_fine)]

        # Solve Generalized Eigenproblem via Cholesky
        L = np.linalg.cholesky(M_free)
        L_inv = np.linalg.inv(L)
        A = L_inv @ K_free @ L_inv.T
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenmodes_free = L_inv.T @ eigenvectors

        eigenvalues_trunc = np.maximum(eigenvalues[:n_modes], 1e-8)
        eigenmodes_free_trunc = eigenmodes_free[:, :n_modes]
        frequencies_Hz = np.sqrt(eigenvalues_trunc) / (2.0 * np.pi)

        # Reconstruct full fine-mesh mode shapes and extract at macro sensor nodes
        full_modes_fine = np.zeros((n_fine_dofs, n_modes), dtype=np.float64)
        full_modes_fine[free_dofs_fine, :] = eigenmodes_free_trunc

        vert_modes_fine = full_modes_fine[0::2, :]
        rot_modes_fine = full_modes_fine[1::2, :]

        # Subsample at the coarse sensor locations
        vert_modes_sample = vert_modes_fine[sensor_node_indices, :].copy()
        rot_modes_sample = rot_modes_fine[sensor_node_indices, :].copy()

        # Re-normalize or maintain mass-normalized scale, then inject measurement noise
        f_noise = np.random.normal(0.0, freq_noise_level, size=n_modes)
        frequencies_Hz_noisy = frequencies_Hz * (1.0 + f_noise)

        for m in range(n_modes):
            v_max = np.max(np.abs(vert_modes_sample[:, m]))
            if v_max > 1e-12:
                vert_modes_sample[:, m] += np.random.normal(0.0, mode_noise_level * v_max, size=n_sensor_nodes)

            r_max = np.max(np.abs(rot_modes_sample[:, m]))
            if r_max > 1e-12:
                rot_modes_sample[:, m] += np.random.normal(0.0, mode_noise_level * r_max, size=n_sensor_nodes)

        # Sign consistency based on peak vertical displacement
        max_indices = np.argmax(np.abs(vert_modes_sample), axis=0)
        signs = np.sign(vert_modes_sample[max_indices, range(n_modes)])
        signs[signs == 0] = 1.0
        vert_modes_sample *= signs
        rot_modes_sample *= signs

        if zero_rotations:
            rot_modes_sample *= 0.0

        frequencies_dataset.append(frequencies_Hz_noisy)
        rot_modes_dataset.append(rot_modes_sample.T)
        vert_modes_dataset.append(vert_modes_sample.T)

    return (np.array(frequencies_dataset, dtype=np.float32),
            np.array(rot_modes_dataset, dtype=np.float32),
            np.array(vert_modes_dataset, dtype=np.float32),
            np.array(alphas_dataset, dtype=np.float32))