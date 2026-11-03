#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/dataset_generation.py

Realistic Bridge with Elastic Bearings Case Study & Stratified DoE Generator.
Version: v4_consistent_30oct
"""

import os
import json
import numpy as np

DATASET_VERSION = "v4_consistent_30oct"


def get_beam_element_matrices(E, I, rho, Area, L_e):
    """Returns 4x4 Euler-Bernoulli stiffness and consistent mass matrices (with rot. inertia)."""
    k_const = E * I / (L_e ** 3)
    Ke = k_const * np.array([
        [12.0,      6.0 * L_e,     -12.0,      6.0 * L_e],
        [6.0 * L_e, 4.0 * L_e**2,  -6.0 * L_e, 2.0 * L_e**2],
        [-12.0,    -6.0 * L_e,      12.0,     -6.0 * L_e],
        [6.0 * L_e, 2.0 * L_e**2,  -6.0 * L_e, 4.0 * L_e**2]
    ], dtype=np.float64)

    m_const = rho * Area * L_e
    r2 = I / Area
    Me = m_const * np.array([
        [13/35 + 6*r2/(5*L_e**2),      11*L_e/210 + r2/(10*L_e),    9/70 - 6*r2/(5*L_e**2),      -13*L_e/420 + r2/(10*L_e)],
        [11*L_e/210 + r2/(10*L_e),     L_e**2/105 + 2*r2/15,        13*L_e/420 - r2/(10*L_e),    -L_e**2/140 - r2/30],
        [9/70 - 6*r2/(5*L_e**2),       13*L_e/420 - r2/(10*L_e),    13/35 + 6*r2/(5*L_e**2),     -11*L_e/210 - r2/(10*L_e)],
        [-13*L_e/420 + r2/(10*L_e),   -L_e**2/140 - r2/30,         -11*L_e/210 - r2/(10*L_e),     L_e**2/105 + 2*r2/15]
    ], dtype=np.float64)
    return Ke, Me


def build_coarse_baseline_matrices(n_zones=5, L_total=10.0, E=210e9, I=8.33e-6,
                                   rho=7850.0, Area=0.005, kv0=1.5e8, kr0=5.0e6):
    """
    Constructs the 5-element coarse physics model with elastic bearings at both ends.
    All 12 DOFs are active (no fixed DOFs).
    """
    n_nodes = n_zones + 1
    n_dofs = 2 * n_nodes
    L_e = L_total / n_zones
    Ke_base, Me_base = get_beam_element_matrices(E, I, rho, Area, L_e)

    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    K_basis = np.zeros((n_zones + 2, n_dofs, n_dofs), dtype=np.float64)

    for i in range(n_zones):
        idx = slice(2 * i, 2 * i + 4)
        M_global[idx, idx] += Me_base
        K_basis[i, idx, idx] = Ke_base

    # Left Bearing z_6 (Node 0: vertical DOF 0, rotational DOF 1)
    K_basis[n_zones, 0, 0] = kv0
    K_basis[n_zones, 1, 1] = kr0

    # Right Bearing z_7 (Node n_zones: vertical DOF n_dofs-2, rotational DOF n_dofs-1)
    K_basis[n_zones + 1, n_dofs - 2, n_dofs - 2] = kv0
    K_basis[n_zones + 1, n_dofs - 1, n_dofs - 1] = kr0

    L_chol = np.linalg.cholesky(M_global)
    L_inv = np.linalg.inv(L_chol)

    return M_global, K_basis, L_inv


def sample_stratified_doe(n_samples, n_zones=5, lbound=0.45, seed=1234):
    """
    Generates realistic damage scenarios across 7 parameters:
    [z1, z2, z3, z4, z5 (deck), z6 (left bearing), z7 (right bearing)].
    """
    rng = np.random.default_rng(seed)
    n_dims = n_zones + 2
    z_all = np.zeros((n_samples, n_dims), dtype=np.float64)
    scenario_labels = np.zeros(n_samples, dtype=np.int32)

    probs = [0.12, 0.33, 0.18, 0.22, 0.15]
    categories = rng.choice(5, size=n_samples, p=probs)

    for k in range(n_samples):
        cat = categories[k]
        scenario_labels[k] = cat
        z = rng.uniform(0.96, 1.00, size=n_dims)

        if cat == 0:
            pass
        elif cat == 1:
            idx = rng.integers(0, n_zones)
            z[idx] = rng.uniform(lbound + 0.05, 0.88)
        elif cat == 2:
            idx = rng.integers(n_zones, n_dims)
            z[idx] = rng.uniform(lbound + 0.05, 0.86)
        elif cat == 3:
            if rng.random() < 0.55:
                idx_d = rng.integers(0, n_zones)
                idx_b = rng.integers(n_zones, n_dims)
                z[idx_d] = rng.uniform(lbound + 0.05, 0.86)
                z[idx_b] = rng.uniform(lbound + 0.05, 0.86)
            else:
                idxs = rng.choice(n_zones, size=2, replace=False)
                z[idxs[0]] = rng.uniform(lbound + 0.05, 0.86)
                z[idxs[1]] = rng.uniform(lbound + 0.05, 0.86)
        elif cat == 4:
            idxs = rng.choice(n_dims, size=3, replace=False)
            z[idxs[0]] = rng.uniform(0.50, 0.66)
            z[idxs[1]] = rng.uniform(0.65, 0.78)
            z[idxs[2]] = rng.uniform(0.75, 0.88)

        z_all[k] = np.clip(z, lbound, 1.0)

    return z_all, scenario_labels


def solve_single_bridge_state(z, n_zones, n_modes, Ke_c, Me_c, kv0, kr0, sensor_nodes, zone_mass_pert):
    n_dofs_total = 2 * (n_zones + 1)
    K_global = np.zeros((n_dofs_total, n_dofs_total), dtype=np.float64)
    M_global = np.zeros((n_dofs_total, n_dofs_total), dtype=np.float64)

    for z_idx in range(n_zones):
        idx = slice(2 * z_idx, 2 * z_idx + 4)
        K_global[idx, idx] += z[z_idx] * Ke_c
        M_global[idx, idx] += (1.0 + zone_mass_pert[z_idx]) * Me_c

    K_global[0, 0] += z[n_zones] * kv0
    K_global[1, 1] += z[n_zones] * kr0
    K_global[n_dofs_total - 2, n_dofs_total - 2] += z[n_zones + 1] * kv0
    K_global[n_dofs_total - 1, n_dofs_total - 1] += z[n_zones + 1] * kr0

    L_chol = np.linalg.cholesky(M_global)
    L_inv = np.linalg.inv(L_chol)
    A = L_inv @ K_global @ L_inv.T
    eigvals, eigvecs = np.linalg.eigh(A)

    eigvals_trunc = np.maximum(eigvals[:n_modes], 1e-8)
    modes_full = L_inv.T @ eigvecs[:, :n_modes]
    f_hz = np.sqrt(eigvals_trunc) / (2.0 * np.pi)

    vert_s = modes_full[2 * sensor_nodes, :].copy()
    rot_s = modes_full[2 * sensor_nodes + 1, :].copy()

    max_idx = np.argmax(np.abs(vert_s), axis=0)
    signs = np.sign(vert_s[max_idx, np.arange(n_modes)])
    signs[signs == 0] = 1.0
    vert_s *= signs
    rot_s *= signs

    vert_s /= (np.linalg.norm(vert_s, axis=0, keepdims=True) + 1e-12)
    rot_s /= (np.linalg.norm(rot_s, axis=0, keepdims=True) + 1e-12)
    return f_hz, vert_s.T, rot_s.T


def solve_forward_bridge_mesh(z_samples, n_zones=5, sub_elements_per_zone=1,
                              n_modes=5, L_total=10.0, E=210e9, I=8.33e-6,
                              rho=7850.0, Area=0.005, kv0=1.5e8, kr0=5.0e6,
                              mass_perturb_std=0.015, fine_stiff_jitter=0.02,
                              freq_noise_std=0.015, mode_noise_std=0.03,
                              bimodal_sym_indices=None, seed=42):
    """
    Solves the forward eigenvalue problem.
    When sub_elements_per_zone=10 (50-element fine mesh), each zone's effective stiffness
    is the exact harmonic homogenized stiffness of 10 sub-elements in series with random
    spatial jitter, plus element-wise random mass perturbations (R1.1 & R3.1).
    """
    rng = np.random.default_rng(seed)
    n_samples = z_samples.shape[0]
    L_e_c = L_total / n_zones
    Ke_c, Me_c = get_beam_element_matrices(E, I, rho, Area, L_e_c)
    sensor_nodes_c = np.arange(0, n_zones + 1, 1)
    n_sensors = len(sensor_nodes_c)

    bimodal_set = set(bimodal_sym_indices) if bimodal_sym_indices is not None else set()

    freqs_list = np.zeros((n_samples, n_modes), dtype=np.float32)
    vert_modes_list = np.zeros((n_samples, n_modes, n_sensors), dtype=np.float32)
    rot_modes_list = np.zeros((n_samples, n_modes, n_sensors), dtype=np.float32)

    for k in range(n_samples):
        z = z_samples[k].copy()

        if sub_elements_per_zone > 1:
            sub_j = rng.normal(0.0, fine_stiff_jitter, size=(n_zones, sub_elements_per_zone))
            sub_z = np.clip(z[:n_zones, None] * (1.0 + sub_j), 0.30, 1.02)
            z_eff = z.copy()
            z_eff[:n_zones] = sub_elements_per_zone / np.sum(1.0 / sub_z, axis=1)
        else:
            z_eff = z

        zone_mass_pert = rng.normal(0.0, mass_perturb_std, size=n_zones)
        zone_mass_pert -= np.mean(zone_mass_pert)

        f_hz, vert_m, rot_m = solve_single_bridge_state(
            z_eff, n_zones, n_modes, Ke_c, Me_c, kv0, kr0, sensor_nodes_c, zone_mass_pert
        )

        if k in bimodal_set:
            z_mirror = np.array([z_eff[4], z_eff[3], z_eff[2], z_eff[1], z_eff[0], z_eff[6], z_eff[5]])
            f_m2, vert_m2, rot_m2 = solve_single_bridge_state(
                z_mirror, n_zones, n_modes, Ke_c, Me_c, kv0, kr0, sensor_nodes_c, zone_mass_pert[::-1]
            )
            f_hz = 0.5 * (f_hz + f_m2)
            vert_m = 0.5 * (vert_m + vert_m2)
            rot_m = 0.5 * (rot_m + rot_m2)

        f_noisy = f_hz * (1.0 + rng.normal(0.0, freq_noise_std, size=n_modes))

        for m in range(n_modes):
            v_max = np.max(np.abs(vert_m[m, :])) + 1e-12
            vert_m[m, :] += rng.normal(0.0, mode_noise_std * v_max, size=n_sensors)

            r_max = np.max(np.abs(rot_m[m, :])) + 1e-12
            rot_m[m, :] += rng.normal(0.0, mode_noise_std * r_max, size=n_sensors)

        vert_m /= (np.linalg.norm(vert_m, axis=-1, keepdims=True) + 1e-12)
        rot_m /= (np.linalg.norm(rot_m, axis=-1, keepdims=True) + 1e-12)

        freqs_list[k] = f_noisy.astype(np.float32)
        vert_modes_list[k] = vert_m.astype(np.float32)
        rot_modes_list[k] = rot_m.astype(np.float32)

    return freqs_list, rot_modes_list, vert_modes_list


def generate_full_benchmark_datasets(data_dir="Data/BridgeBearings_7D_DoE",
                                     n_train=14000, n_val=2400, n_test=1200,
                                     n_zones=5, n_modes=5, lbound=0.45):
    os.makedirs(data_dir, exist_ok=True)
    print(f"--- Building Baseline 7-Parameter Bridge-on-Bearings Matrices in {data_dir} ---")
    M_coarse, K_basis, L_inv = build_coarse_baseline_matrices(n_zones=n_zones)

    np.save(os.path.join(data_dir, "Mass_matrix.npy"), M_coarse.astype(np.float32))
    np.save(os.path.join(data_dir, "Ke_matrix.npy"), K_basis.astype(np.float32))
    np.save(os.path.join(data_dir, "L_inv.npy"), L_inv.astype(np.float32))

    print(f"--- Sampling Stratified DoE (Train={n_train}, Val={n_val}, Test={n_test}) ---")
    z_train, labels_train = sample_stratified_doe(n_train, n_zones=n_zones, lbound=lbound, seed=101)
    z_val, labels_val = sample_stratified_doe(n_val, n_zones=n_zones, lbound=lbound, seed=202)
    z_test, labels_test = sample_stratified_doe(n_test, n_zones=n_zones, lbound=lbound, seed=303)

    # 10 Curated Benchmark Test Scenarios (Positions 0 .. 9)
    z_test[0] = np.array([0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.98])
    labels_test[0] = 0

    z_test[1] = np.array([0.56, 0.98, 0.99, 0.98, 0.99, 0.98, 0.99])
    labels_test[1] = 1

    z_test[2] = np.array([0.98, 0.99, 0.65, 0.98, 0.99, 0.99, 0.98])
    labels_test[2] = 1

    z_test[3] = np.array([0.99, 0.78, 0.98, 0.99, 0.98, 0.98, 0.99])
    labels_test[3] = 1

    z_test[4] = np.array([0.98, 0.99, 0.98, 0.99, 0.98, 0.55, 0.98])
    labels_test[4] = 2

    z_test[5] = np.array([0.60, 0.98, 0.99, 0.98, 0.98, 0.58, 0.98])
    labels_test[5] = 3

    z_test[6] = np.array([0.98, 0.54, 0.98, 0.72, 0.98, 0.98, 0.64])
    labels_test[6] = 4

    z_test[7] = np.array([0.62, 0.82, 0.55, 0.98, 0.74, 0.68, 0.98])
    labels_test[7] = 4

    z_test[8] = np.array([0.56, 0.98, 0.99, 0.98, 0.98, 0.98, 0.98])
    labels_test[8] = 1

    z_test[9] = np.array([0.98, 0.58, 0.98, 0.98, 0.98, 0.62, 0.98])
    labels_test[9] = 3

    sym_test_idx = [5, 8, 9]

    print("--- Solving 5-Element Coarse Model for Training & Validation Sets ---")
    f_train, r_train, v_train = solve_forward_bridge_mesh(
        z_train, n_zones=n_zones, sub_elements_per_zone=1, n_modes=n_modes,
        mass_perturb_std=0.015, freq_noise_std=0.015, mode_noise_std=0.030, seed=111
    )
    f_val, r_val, v_val = solve_forward_bridge_mesh(
        z_val, n_zones=n_zones, sub_elements_per_zone=1, n_modes=n_modes,
        mass_perturb_std=0.015, freq_noise_std=0.015, mode_noise_std=0.030, seed=222
    )

    print("--- Solving 50-Element Fine Discretization Model for Test Set (Model-Form Uncertainty) ---")
    f_test, r_test, v_test = solve_forward_bridge_mesh(
        z_test, n_zones=n_zones, sub_elements_per_zone=10, n_modes=n_modes,
        mass_perturb_std=0.020, fine_stiff_jitter=0.025,
        freq_noise_std=0.015, mode_noise_std=0.030,
        bimodal_sym_indices=sym_test_idx, seed=333
    )

    np.save(os.path.join(data_dir, "freqs_train.npy"), f_train)
    np.save(os.path.join(data_dir, "rotmodes_train.npy"), r_train)
    np.save(os.path.join(data_dir, "vertmodes_train.npy"), v_train)
    np.save(os.path.join(data_dir, "alphas_train.npy"), z_train.astype(np.float32))
    np.save(os.path.join(data_dir, "labels_train.npy"), labels_train)

    np.save(os.path.join(data_dir, "freqs_val.npy"), f_val)
    np.save(os.path.join(data_dir, "rotmodes_val.npy"), r_val)
    np.save(os.path.join(data_dir, "vertmodes_val.npy"), v_val)
    np.save(os.path.join(data_dir, "alphas_val.npy"), z_val.astype(np.float32))
    np.save(os.path.join(data_dir, "labels_val.npy"), labels_val)

    np.save(os.path.join(data_dir, "freqs_test.npy"), f_test)
    np.save(os.path.join(data_dir, "rotmodes_test.npy"), r_test)
    np.save(os.path.join(data_dir, "vertmodes_test.npy"), v_test)
    np.save(os.path.join(data_dir, "alphas_test.npy"), z_test.astype(np.float32))
    np.save(os.path.join(data_dir, "labels_test.npy"), labels_test)

    with open(os.path.join(data_dir, "dataset_version.json"), "w", encoding="utf-8") as f:
        json.dump({"version": DATASET_VERSION}, f)

    print(f"Dataset generation ({DATASET_VERSION}) complete! Saved to '{data_dir}'.")


if __name__ == "__main__":
    generate_full_benchmark_datasets()