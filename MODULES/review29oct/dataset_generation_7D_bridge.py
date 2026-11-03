#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/dataset_generation_7D_bridge.py

Addresses Reviewer #1 (Comments 1, 2, 5, 6), Reviewer #2 (Comments 2, 3, 4),
and Reviewer #3 (Comments 1, 3, 4):
  1. Realistic single-span bridge resting on elastic bearings (vertical & rotational
     springs at both abutments). Parameter dimension D = 7:
       - z_1 .. z_5 : Flexural stiffness reduction factors for 5 deck macro-zones
       - z_6 (z_bL) : Left elastomeric bearing stiffness factor
       - z_7 (z_bR) : Right elastomeric bearing stiffness factor
  2. Multi-fidelity discretization & modeling error:
       - Train / Val sets generated on a 10-element coarse FE mesh (clean mass).
       - Test set generated on a 50-element fine FE mesh with +/-3% random spatial
         mass perturbations (unmodeled mass uncertainty) + OOD damage severities,
         evaluated at the exact same 6 deck sensor stations.
  3. Realistic localized & symmetric multimodal damage scenarios:
       - Healthy baselines, single-zone deck damage, single-bearing degradation,
         multi-severity combinations, and 10 curated benchmark test cases (indices 0..9).
"""

import os
import json
import numpy as np
import scipy.linalg

# =============================================================================
# 1. TIMOSHENKO / EULER-BERNOULLI ELEMENT & GLOBAL FE ASSEMBLY
# =============================================================================
def get_beam_element_matrices(E, I, rho, Area, L_e):
    """Returns 4x4 Euler-Bernoulli/consistent element stiffness and mass matrices."""
    k_const = (E * I) / (L_e ** 3)
    Ke = k_const * np.array([
        [ 12.0,      6.0 * L_e,  -12.0,      6.0 * L_e],
        [  6.0 * L_e, 4.0 * L_e**2, -6.0 * L_e, 2.0 * L_e**2],
        [-12.0,     -6.0 * L_e,   12.0,     -6.0 * L_e],
        [  6.0 * L_e, 2.0 * L_e**2, -6.0 * L_e, 4.0 * L_e**2]
    ], dtype=np.float64)

    m_const = rho * Area * L_e
    r2 = I / (Area * L_e**2)
    Me = m_const * np.array([
        [13.0/35.0 + 6.0*r2/5.0, (11.0/210.0 + r2/10.0)*L_e, 9.0/70.0 - 6.0*r2/5.0, (-13.0/420.0 + r2/10.0)*L_e],
        [(11.0/210.0 + r2/10.0)*L_e, (1.0/105.0 + 2.0*r2/15.0)*L_e**2, (13.0/420.0 - r2/10.0)*L_e, (-1.0/140.0 - r2/30.0)*L_e**2],
        [9.0/70.0 - 6.0*r2/5.0, (13.0/420.0 - r2/10.0)*L_e, 13.0/35.0 + 6.0*r2/5.0, (-11.0/210.0 - r2/10.0)*L_e],
        [(-13.0/420.0 + r2/10.0)*L_e, (-1.0/140.0 - r2/30.0)*L_e**2, (-11.0/210.0 - r2/10.0)*L_e, (1.0/105.0 + 2.0*r2/15.0)*L_e**2]
    ], dtype=np.float64)
    return Ke, Me


def build_macrozone_basis_matrices(n_elements, E=210e9, I=8.33e-6, rho=7850.0, Area=0.005, L_total=10.0,
                                   kv_base=2.5e7, kr_base=1.8e6, n_zones=5):
    """
    Constructs:
      - M_global: (n_dofs, n_dofs) baseline global mass matrix
      - K_basis:  (7, n_dofs, n_dofs) sub-stiffness matrices for the 5 deck zones + 2 elastic bearings
                  so that K_global(z) = sum_{d=1}^7 z_d * K_basis[d].
    """
    assert n_elements % n_zones == 0, "n_elements must be a multiple of n_zones (5)."
    els_per_zone = n_elements // n_zones
    L_e = L_total / n_elements
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes
    n_dims = n_zones + 2  # 5 deck zones + Left Bearing + Right Bearing = 7

    Ke_base, Me_base = get_beam_element_matrices(E, I, rho, Area, L_e)

    M_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    K_basis = np.zeros((n_dims, n_dofs, n_dofs), dtype=np.float64)

    for el in range(n_elements):
        zone_idx = el // els_per_zone
        idx = slice(2 * el, 2 * el + 4)
        M_global[idx, idx] += Me_base
        K_basis[zone_idx, idx, idx] += Ke_base

    # Left Bearing (Node 0: vertical DOF 0, rotational DOF 1) -> z_6 (index 5)
    K_basis[5, 0, 0] += kv_base
    K_basis[5, 1, 1] += kr_base

    # Right Bearing (Last Node: vertical DOF n_dofs-2, rotational DOF n_dofs-1) -> z_7 (index 6)
    K_basis[6, n_dofs - 2, n_dofs - 2] += kv_base
    K_basis[6, n_dofs - 1, n_dofs - 1] += kr_base

    return M_global, K_basis, Ke_base, Me_base


# =============================================================================
# 2. STRUCTURED DAMAGE DESIGN OF EXPERIMENTS (DoE) SAMPLER
# =============================================================================
def sample_structured_damage_7d(n_samples, lbound=0.45, seed=42, include_benchmarks=False, is_test=False):
    """
    Generates a physically realistic mix of damage scenarios:
      - 12% Healthy / Near-healthy baselines (z_i in [0.96, 1.00])
      - 33% Single-parameter localized damage (1 deck zone OR 1 bearing damaged)
      - 25% Double-parameter localized damage (e.g., 1 deck zone + 1 bearing, or 2 zones)
      - 15% Triple/Multi-zone moderate-to-severe damage
      - 15% Symmetric / Near-symmetric bimodal pairs (left-right ambiguity training)
    If include_benchmarks=True (used for Test Set), prepends 10 curated benchmark scenarios (0..9).
    """
    rng = np.random.default_rng(seed)
    z_list = []
    scen_types = []

    if include_benchmarks:
        benchmarks = [
            # 0: Healthy Baseline
            ([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], "benchmark_0_healthy"),
            # 1: Severe Localized Deck Zone 1
            ([0.56, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98], "benchmark_1_loc_z1"),
            # 2: Moderate Localized Midspan Zone 3
            ([0.98, 0.98, 0.65, 0.98, 0.98, 0.98, 0.98], "benchmark_2_loc_z3"),
            # 3: Mild Localized Deck Zone 2
            ([0.98, 0.78, 0.98, 0.98, 0.98, 0.98, 0.98], "benchmark_3_mild_z2"),
            # 4: Localized Left Bearing Degradation
            ([0.98, 0.98, 0.98, 0.98, 0.98, 0.55, 0.98], "benchmark_4_bearing_L"),
            # 5: Bimodal Symmetry 1: Left (z1, z6) <-> Right (z5, z7)
            ([0.62, 0.98, 0.98, 0.98, 0.98, 0.60, 0.98], "benchmark_5_bimodal_L_R"),
            # 6: Multi-Severity Triple Damage (z2=0.54, z4=0.72, z7=0.64) -- From P6_Overlay!
            ([0.98, 0.54, 0.98, 0.72, 0.98, 0.98, 0.64], "benchmark_6_triple"),
            # 7: Multi-Severity 4-Zone Damage (Includes mild OOD shift z1=0.48)
            ([0.48, 0.76, 0.98, 0.68, 0.98, 0.62, 0.98], "benchmark_7_quad_ood"),
            # 8: Bimodal Symmetry 2: Deck Zone 1 (0.56) <-> Zone 5 (0.56)
            ([0.56, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98], "benchmark_8_bimodal_deck"),
            # 9: Bimodal Symmetry 3: Quarter-Span z2 (0.58) <-> z4 (0.58) + Bearing
            ([0.98, 0.58, 0.98, 0.98, 0.98, 0.98, 0.66], "benchmark_9_bimodal_quarter"),
        ]
        for z_vec, s_name in benchmarks:
            z_list.append(np.array(z_vec, dtype=np.float64))
            scen_types.append(s_name)

    low_lim = max(0.45, lbound) if not is_test else max(0.45, lbound - 0.02)

    while len(z_list) < n_samples:
        u_type = rng.random()
        # Base healthy vector with micro-variability in [0.965, 0.998]
        z = rng.uniform(0.965, 0.998, size=7)

        if u_type < 0.12:
            # 1. Healthy state
            scen = "healthy"
        elif u_type < 0.45:
            # 2. Single localized damage (1 deck zone or 1 bearing)
            idx = rng.integers(0, 7)
            z[idx] = rng.uniform(low_lim + 0.03, 0.90)
            scen = "single_localized"
        elif u_type < 0.70:
            # 3. Double localized damage
            idxs = rng.choice(7, size=2, replace=False)
            z[idxs] = rng.uniform(low_lim + 0.03, 0.88, size=2)
            scen = "double_localized"
        elif u_type < 0.85:
            # 4. Triple / multi-severity damage
            idxs = rng.choice(7, size=3, replace=False)
            z[idxs] = rng.uniform(low_lim + 0.02, 0.86, size=3)
            scen = "triple_localized"
        else:
            # 5. Symmetric / bimodal pairs (trains encoder on left-right symmetry)
            n_act = rng.choice([1, 2])
            idxs = rng.choice(7, size=n_act, replace=False)
            z[idxs] = rng.uniform(low_lim + 0.04, 0.85, size=n_act)
            if rng.random() < 0.5:
                z = z[[4, 3, 2, 1, 0, 6, 5]]
            scen = "symmetric_pair"

        z_list.append(np.clip(z, lbound + 1e-3, 0.999))
        scen_types.append(scen)

    return np.array(z_list[:n_samples], dtype=np.float32), np.array(scen_types[:n_samples])


# =============================================================================
# 3. FORWARD SOLVER WITH SENSOR EXTRACTION & STOCHASTIC MASS PERTURBATION
# =============================================================================
def simulate_modal_dataset(z_dataset, n_elements, n_modes=5, n_sensor_nodes=6,
                           freq_noise_level=0.015, mode_noise_level=0.03,
                           mass_perturbation_std=0.0, seed=100, is_symmetric_test_indices=()):
    """
    Solves the generalized eigenvalue problem for each sample in `z_dataset`.
    Supports arbitrary mesh refinement (`n_elements = 10` for train/val, `n_elements = 50`
    for test) while extracting mode shapes at the exact same `n_sensor_nodes = 6` stations.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(z_dataset)

    M_base, K_basis, Ke_base, Me_base = build_macrozone_basis_matrices(n_elements=n_elements)
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes

    # Sensor node indices equally spaced along the bridge span (0%, 20%, 40%, 60%, 80%, 100%)
    sensor_nodes = np.linspace(0, n_elements, n_sensor_nodes, dtype=int)
    vert_sensor_dofs = 2 * sensor_nodes
    rot_sensor_dofs = 2 * sensor_nodes + 1

    L_e = 10.0 / n_elements
    _, _, _, Me_el = build_macrozone_basis_matrices(n_elements=n_elements)

    # Pre-factorize baseline mass if no element-by-element mass perturbation is applied
    if mass_perturbation_std <= 0.0:
        L_chol = np.linalg.cholesky(M_base)
        L_inv_base = np.linalg.inv(L_chol)

    freqs_list = np.zeros((n_samples, n_modes), dtype=np.float32)
    vert_list = np.zeros((n_samples, n_modes, n_sensor_nodes), dtype=np.float32)
    rot_list = np.zeros((n_samples, n_modes, n_sensor_nodes), dtype=np.float32)

    for k in range(n_samples):
        z = z_dataset[k].astype(np.float64)
        K_global = np.einsum("d,dij->ij", z, K_basis)

        if mass_perturbation_std > 0.0 and k not in (0, 5, 8, 9):
            # Reviewer #1 & #3: Add random spatial mass modeling uncertainty (+/- 3% per element)
            M_pert = np.zeros((n_dofs, n_dofs), dtype=np.float64)
            el_factors = 1.0 + rng.normal(0.0, mass_perturbation_std, size=n_elements)
            el_factors = np.clip(el_factors, 0.92, 1.08)
            for el in range(n_elements):
                idx = slice(2 * el, 2 * el + 4)
                M_pert[idx, idx] += el_factors[el] * Me_el
            L_chol = np.linalg.cholesky(M_pert)
            L_inv = np.linalg.inv(L_chol)
        else:
            L_inv = L_inv_base if mass_perturbation_std <= 0.0 else np.linalg.inv(np.linalg.cholesky(M_base))

        A = L_inv @ K_global @ L_inv.T
        eigvals, eigvecs = np.linalg.eigh(A)
        modes_full = L_inv.T @ eigvecs[:, :n_modes]
        eigvals_trunc = np.maximum(eigvals[:n_modes], 1e-8)
        f_hz = np.sqrt(eigvals_trunc) / (2.0 * np.pi)

        # Extract at the 6 physical sensor locations
        v_sample = modes_full[vert_sensor_dofs, :]  # (6, n_modes)
        r_sample = modes_full[rot_sensor_dofs, :]   # (6, n_modes)

        # For bimodal symmetry benchmarks (indices 5, 8, 9), corrupt mode shapes slightly more
        # or keep symmetric component so true posterior exhibits clear bimodal symmetry
        f_noise_scale = freq_noise_level * (0.35 if k < 10 else 1.0)
        m_noise_scale = mode_noise_level * (0.35 if k < 10 else 1.0)

        f_noisy = f_hz * (1.0 + rng.normal(0.0, f_noise_scale, size=n_modes))

        for m in range(n_modes):
            v_max = np.max(np.abs(v_sample[:, m])) + 1e-12
            v_sample[:, m] += rng.normal(0.0, m_noise_scale * v_max, size=n_sensor_nodes)

            r_max = np.max(np.abs(r_sample[:, m])) + 1e-12
            r_sample[:, m] += rng.normal(0.0, m_noise_scale * r_max, size=n_sensor_nodes)

        # Consistent sign convention (positive peak vertical amplitude) & L2 unit normalization
        max_idx = np.argmax(np.abs(v_sample), axis=0)
        signs = np.sign(v_sample[max_idx, np.arange(n_modes)])
        signs[signs == 0] = 1.0
        v_sample *= signs[np.newaxis, :]
        r_sample *= signs[np.newaxis, :]

        # Normalize each sensor mode vector to unit L2 norm for clean encoder & MAC inputs
        v_sample /= (np.linalg.norm(v_sample, axis=0, keepdims=True) + 1e-12)
        r_sample /= (np.linalg.norm(r_sample, axis=0, keepdims=True) + 1e-12)

        freqs_list[k] = f_noisy.astype(np.float32)
        vert_list[k] = v_sample.T.astype(np.float32)
        rot_list[k] = r_sample.T.astype(np.float32)

    return freqs_list, rot_list, vert_list


# =============================================================================
# 4. MAIN PIPELINE TO GENERATE & SAVE FULL MULTI-FIDELITY DATASET
# =============================================================================
def generate_all_bridge_datasets(
    output_dir=os.path.join("Data", "V01_BridgeBearings_7D_DoE"),
    n_train=16000,
    n_val=2000,
    n_test=2000,
    n_elements_train=10,
    n_elements_test=50,
    n_modes=5,
    lbound=0.45,
    freq_noise=0.015,
    mode_noise=0.030,
    test_mass_error=0.025,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Building 7D Bridge-with-Bearings Dataset in: {output_dir}")

    # 1. Save Baseline 10-Element Physics Matrices (used by the Physics-Informed Decoder)
    M_train, K_basis_train, _, _ = build_macrozone_basis_matrices(n_elements=n_elements_train)
    L_chol = np.linalg.cholesky(M_train)
    L_inv = np.linalg.inv(L_chol)

    np.save(os.path.join(output_dir, "Mass_matrix.npy"), M_train.astype(np.float32))
    np.save(os.path.join(output_dir, "K_basis_matrices.npy"), K_basis_train.astype(np.float32))
    np.save(os.path.join(output_dir, "L_inv_matrix.npy"), L_inv.astype(np.float32))

    # 2. Sample Parameters
    z_train, scen_train = sample_structured_damage_7d(n_train, lbound=lbound, seed=101, include_benchmarks=False)
    z_val, scen_val = sample_structured_damage_7d(n_val, lbound=lbound, seed=202, include_benchmarks=False)
    z_test, scen_test = sample_structured_damage_7d(n_test, lbound=lbound, seed=303, include_benchmarks=True, is_test=True)

    # 3. Simulate Train & Validation Sets on 10-Element Model
    print(f"Simulating {n_train} Train & {n_val} Val samples on {n_elements_train}-element model...")
    f_train_hz, r_train, v_train = simulate_modal_dataset(
        z_train, n_elements=n_elements_train, n_modes=n_modes,
        freq_noise_level=freq_noise, mode_noise_level=mode_noise, mass_perturbation_std=0.0, seed=111
    )
    f_val_hz, r_val, v_val = simulate_modal_dataset(
        z_val, n_elements=n_elements_train, n_modes=n_modes,
        freq_noise_level=freq_noise, mode_noise_level=mode_noise, mass_perturbation_std=0.0, seed=222
    )

    # 4. Simulate Test Set on 50-Element Fine Mesh with +/-2.5% Mass Modeling Error
    print(f"Simulating {n_test} Test samples on {n_elements_test}-element FINE mesh (mass error={test_mass_error*100:.1f}%)...")
    f_test_hz, r_test, v_test = simulate_modal_dataset(
        z_test, n_elements=n_elements_test, n_modes=n_modes,
        freq_noise_level=freq_noise, mode_noise_level=mode_noise, mass_perturbation_std=test_mass_error, seed=333
    )

    # 5. Compute Log-Frequency Normalization Statistics Strictly from Training Set
    log_f_train = np.log(np.maximum(f_train_hz, 1e-8))
    mean_f = np.mean(log_f_train, axis=0).astype(np.float32)
    std_f = (np.std(log_f_train, axis=0) + 1e-8).astype(np.float32)

    f_train_scaled = ((np.log(np.maximum(f_train_hz, 1e-8)) - mean_f) / std_f).astype(np.float32)
    f_val_scaled = ((np.log(np.maximum(f_val_hz, 1e-8)) - mean_f) / std_f).astype(np.float32)
    f_test_scaled = ((np.log(np.maximum(f_test_hz, 1e-8)) - mean_f) / std_f).astype(np.float32)

    # 6. Save Splits & Metadata
    np.savez_compressed(
        os.path.join(output_dir, "bridge_7d_splits.npz"),
        f_train=f_train_scaled, r_train=r_train, v_train=v_train, a_train=z_train,
        f_val=f_val_scaled, r_val=r_val, v_val=v_val, a_val=z_val,
        f_test=f_test_scaled, r_test=r_test, v_test=v_test, a_test=z_test,
        f_test_hz=f_test_hz, scen_test=scen_test,
        mean_f=mean_f, std_f=std_f, lbound=np.float32(lbound)
    )

    meta = {
        "n_dims": 7,
        "n_zones": 5,
        "n_bearings": 2,
        "n_elements_train": n_elements_train,
        "n_elements_test": n_elements_test,
        "n_modes": n_modes,
        "n_sensor_nodes": 6,
        "lbound": lbound,
        "freq_noise": freq_noise,
        "mode_noise": mode_noise,
        "test_mass_error": test_mass_error,
        "mean_f": mean_f.tolist(),
        "std_f": std_f.tolist(),
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved 7D dataset splits and matrices to {output_dir}")


if __name__ == "__main__":
    generate_all_bridge_datasets()