#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:42:54 2025

@author: afernandez
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Ensure TensorFlow is using float64 for better numerical stability
tf.keras.backend.set_floatx('float64')
print(f"TensorFlow version: {tf.__version__}")


# --- 1. Beam Properties and Element Definitions ---
n_elements = 10
n_m = 5 # Number of modes to use

# Material and Geometric Properties
E = 210e9  # Young's modulus in Pa
I = 8.33e-6  # Second moment of area in m^4
rho = 7850  # Density in kg/m^3
A = 0.0005  # Cross-sectional area in m^2
L_total = 20.0 # Total length of the beam in m (use float)
L_e = L_total / n_elements  # Length of each element

# Correct number of DOFs for 2D beam elements (v and theta at each node)
n_nodes = n_elements + 1
n_dof_full = 2 * n_nodes # Total DOFs including rotations
n_dof_vertical = n_nodes # Total DOFs considering only vertical displacement
print(f"Number of elements: {n_elements}")
print(f"Number of nodes: {n_nodes}")
print(f"Length of each element (L_e): {L_e} m")
print(f"Total number of DOFs (v, theta per node): {n_dof_full}")
print(f"Total number of DOFs (v per node): {n_dof_vertical}")


# Element stiffness matrix Ke (4x4)
Ke = E * I / L_e**3 * tf.constant([
    [12, 6*L_e, -12, 6*L_e],
    [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
    [-12, -6*L_e, 12, -6*L_e],
    [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
], dtype=tf.float64)

# Element mass matrix Me (4x4) - Consistent Mass Matrix
Me = (rho * A * L_e) * tf.constant([
    [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
    [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
    [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
    [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
], dtype=tf.float64)

print(f"\nElement Stiffness Matrix Ke (shape {Ke.shape})")
print(f"Element Mass Matrix Me (shape {Me.shape})")

# --- 2. Assemble Global Matrices (Full - including rotations) ---
K_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)
M_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)
Ke_flat = tf.reshape(Ke, [-1])
Me_flat = tf.reshape(Me, [-1])

print("\nAssembling full global matrices (including rotations)...")
for e in range(n_elements):
    node_i = e
    node_j = e + 1
    idx = tf.constant([2*node_i, 2*node_i + 1, 2*node_j, 2*node_j + 1], dtype=tf.int32)
    indices_pairs = tf.stack(tf.meshgrid(idx, idx, indexing='ij'), axis=-1)
    indices_pairs = tf.reshape(indices_pairs, [-1, 2])
    K_global_full = tf.tensor_scatter_nd_add(K_global_full, indices_pairs, Ke_flat)
    M_global_full = tf.tensor_scatter_nd_add(M_global_full, indices_pairs, Me_flat)

# --- 3. Extract Submatrices for Vertical DOFs Only ---
vertical_dof_indices = tf.range(0, n_dof_full, 2, dtype=tf.int32)
print("\nIndices for vertical DOFs:", vertical_dof_indices.numpy())

K_vertical_rows = tf.gather(K_global_full, vertical_dof_indices, axis=0)
M_vertical_rows = tf.gather(M_global_full, vertical_dof_indices, axis=0)
K_vertical = tf.gather(K_vertical_rows, vertical_dof_indices, axis=1)
M_vertical = tf.gather(M_vertical_rows, vertical_dof_indices, axis=1)

M = M_vertical
K = K_vertical

# --- Solve Generalized Eigenvalue Problem ---
try:
    M_inv = tf.linalg.inv(M)
    A_eig = M_inv @ K # Renamed to avoid conflict with cross-sectional Area A
    eigenvalues_sq, eigenvectors = tf.linalg.eigh(A_eig)
except tf.errors.InvalidArgumentError:
    print("Using Cholesky decomposition approach for eigenvalue problem.")
    L_chol = tf.linalg.cholesky(M) # Renamed L to L_chol
    L_chol_inv = tf.linalg.inv(L_chol)
    A_cholesky = L_chol_inv @ K @ tf.linalg.inv(tf.transpose(L_chol))
    eigenvalues_sq, y_eig = tf.linalg.eigh(A_cholesky) # Renamed y to y_eig
    eigenvectors = tf.linalg.solve(tf.transpose(L_chol), y_eig)



# --- Select Modes and Normalize ---
omegas_sq = eigenvalues_sq
omegas = tf.sqrt(omegas_sq)

Phi = eigenvectors[:, :n_m]
selected_omegas_sq = omegas_sq[:n_m]
selected_omegas = omegas[:n_m]

M_r_diag = tf.einsum('ji,jk,kl->il', Phi, M, Phi)
norm_factors = 1.0 / tf.sqrt(tf.linalg.diag_part(M_r_diag))
Phi_normalized = Phi * norm_factors
Phi = Phi_normalized

k_r = selected_omegas_sq # Shape [n_m] (Modal stiffnesses)
m_r = tf.ones(n_m, dtype=tf.float64) # Shape [n_m] (Modal masses, 1 due to normalization)

# *** NEW: Define Modal Damping ***
zeta_val = tf.constant(0.02, dtype=tf.float64) # Example: 2% damping for all selected modes
zeta_modal = tf.fill((n_m,), zeta_val) # Damping ratio for each mode
c_r = 2.0 * zeta_modal * selected_omegas # Modal damping coefficients c_r_i = 2 * zeta_i * omega_i (since m_r_i = 1)
print(f"\nSelected natural frequencies (rad/s): {selected_omegas.numpy()}")
print(f"Modal damping ratios (zeta): {zeta_modal.numpy()}")
print(f"Modal damping coefficients (c_r): {c_r.numpy()}")

# Reshape for calculations in Newmark
k_r_col = tf.reshape(k_r, (n_m, 1)) # [n_m, 1]
c_r_col = tf.reshape(c_r, (n_m, 1)) # [n_m, 1]
m_r_col = tf.ones_like(k_r_col)      # [n_m, 1] (modal mass = 1)


# --- 4. Define Load and Time ---
force_amplitude = tf.constant(5.0, dtype=tf.float64)
forcing_frequency_hz = tf.constant(100, dtype=tf.float64)
forcing_frequency_rad = 2.0 * np.pi * forcing_frequency_hz
Omega = forcing_frequency_rad

t_max_sim = tf.constant(10.0, dtype=tf.float64)
dt = tf.constant(0.002, dtype=tf.float64) # MODIFIED: Reduced dt for better accuracy with higher frequency forcing
load_start_time = tf.constant(1.0, dtype=tf.float64)
load_end_time = tf.constant(5.0, dtype=tf.float64)

n_steps = tf.cast(tf.round(t_max_sim / dt), dtype=tf.int32)
start_time = tf.constant(0.0, dtype=tf.float64)
num_points = n_steps + 1
t = tf.linspace(start_time, t_max_sim, num_points)
actual_dt = t[1] - t[0]
print(f"\nTime vector from {t[0]:.3f}s to {t[-1]:.3f}s with {num_points} points, dt = {actual_dt:.6f}s.")


# --- Define Load Location Vector (p) ---
n_dof = n_dof_vertical
center_dof_index = n_dof // 2 # Node index 5 for 11 nodes (0 to 10)

load_vector_np = np.zeros(n_dof) # Using numpy for easy assignment
load_vector_np[center_dof_index] = 1.0
p = tf.constant(load_vector_np, shape=(n_dof, 1), dtype=tf.float64)

# --- Define Force Magnitude over Time F(t) ---
F_t_sinusoidal = force_amplitude * tf.sin(Omega * t)
mask_condition = tf.logical_and(t >= load_start_time, t <= load_end_time)
time_mask = tf.cast(mask_condition, dtype=tf.float64)
F_t = F_t_sinusoidal * time_mask
Q_t = p @ tf.expand_dims(F_t, axis=0)
Q_r_t = tf.transpose(Phi) @ Q_t

# Plot F(t) to verify
plt.figure(figsize=(10,4))
plt.plot(t.numpy(), F_t.numpy())
plt.title('Force Magnitude F(t) Applied at Center Node')
plt.xlabel('Time (s)')
plt.ylabel('Force Amplitude (N)')
plt.grid(True)
plt.show()


# --- 6. Newmark-beta Time Integration in Modal Coordinates ---
print("\n--- Starting Newmark Method with Damping ---")
beta_newmark = tf.constant(0.25, dtype=tf.float64)   # Average acceleration method
gamma_newmark = tf.constant(0.50, dtype=tf.float64) # Average acceleration method
dt_val = tf.cast(actual_dt, dtype=tf.float64)

r_initial = tf.zeros((n_m, 1), dtype=tf.float64)
rdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)

Q_r_0 = Q_r_t[:, 0:1]
# *** MODIFIED: Initial modal acceleration with damping ***
# m_r * rddot_0 + c_r * rdot_0 + k_r * r_0 = Q_r_0
# Since m_r = 1 (identity): rddot_0 = Q_r_0 - c_r * rdot_0 - k_r * r_0
rddot_initial = Q_r_0 - c_r_col * rdot_initial - k_r_col * r_initial

r_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="r_history", clear_after_read=False)
rdot_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="rdot_history", clear_after_read=False)
rddot_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="rddot_history", clear_after_read=False)

r_history = r_history.write(0, r_initial)
rdot_history = rdot_history.write(0, rdot_initial)
rddot_history = rddot_history.write(0, rddot_initial)

r_current = r_initial
rdot_current = rdot_initial
rddot_current = rddot_initial

# *** MODIFIED: Pre-calculate LHS factor for solving rddot_next, including damping ***
# LHS_factor = m_r + gamma*dt*c_r + beta*dt^2*k_r
# Since m_r_col is ones:
LHS_factor_modal = m_r_col + gamma_newmark * dt_val * c_r_col + beta_newmark * dt_val**2 * k_r_col # Shape [n_m, 1]

print(f"Starting Newmark integration loop for {n_steps} steps...")
for i in tf.range(n_steps):
    Q_r_next = Q_r_t[:, i+1:i+2]

    # Predictor terms at t_i (current step i) for state at t_{i+1}
    # r_predictor_term = r_i + dt*rdot_i + (0.5 - beta)*dt^2 * rddot_i
    r_predictor_term = r_current + dt_val * rdot_current + (0.5 - beta_newmark) * dt_val**2 * rddot_current
    # rdot_predictor_term = rdot_i + (1-gamma)*dt*rddot_i
    rdot_predictor_term = rdot_current + (1.0 - gamma_newmark) * dt_val * rddot_current

    # *** MODIFIED: Effective force, including damping effects ***
    # RHS_force = Q_r_next - c_r * rdot_predictor - k_r * r_predictor
    RHS_force_modal = Q_r_next - c_r_col * rdot_predictor_term - k_r_col * r_predictor_term # Shape [n_m, 1]

    # Solve for modal acceleration at time t_{i+1}
    rddot_next = RHS_force_modal / LHS_factor_modal # Shape [n_m, 1]

    # Correct modal velocity at time t_{i+1}
    # rdot_next = rdot_predictor + gamma*dt*rddot_next
    rdot_next = rdot_predictor_term + gamma_newmark * dt_val * rddot_next # Shape [n_m, 1]

    # Correct modal displacement at time t_{i+1}
    # r_next = r_predictor + beta*dt^2*rddot_next
    r_next = r_predictor_term + beta_newmark * dt_val**2 * rddot_next # Shape [n_m, 1]

    r_history = r_history.write(i + 1, r_next)
    rdot_history = rdot_history.write(i + 1, rdot_next)
    rddot_history = rddot_history.write(i + 1, rddot_next)

    r_current = r_next
    rdot_current = rdot_next
    rddot_current = rddot_next

    # if (i + 1) % (n_steps // 10 or 1) == 0: # Avoid division by zero if n_steps < 10
    #      print(f"  Completed step {i+1}/{n_steps}")

print("Newmark integration finished.")

r_history_stacked = r_history.stack()
rdot_history_stacked = rdot_history.stack()
rddot_history_stacked = rddot_history.stack()

r_history_final = tf.transpose(tf.squeeze(r_history_stacked, axis=-1))
rdot_history_final = tf.transpose(tf.squeeze(rdot_history_stacked, axis=-1))
rddot_history_final = tf.transpose(tf.squeeze(rddot_history_stacked, axis=-1))

print(f"Shape of final modal displacement history (r_history_final): {r_history_final.shape}")

# Physical displacement u(t) = Phi * r(t)
u_t = tf.matmul(Phi, r_history_final)
print(f"Shape of physical displacement history (u_t): {u_t.shape}")

# Physical acceleration uddot(t) = Phi * rddot(t)
uddot_t = tf.matmul(Phi, rddot_history_final)
print(f"Shape of physical acceleration history (uddot_t): {uddot_t.shape}")


# --- 7. Define Sensor Locations and Extract Responses ---
print("\n--- Defining Sensor Locations and Extracting Responses ---")
sensor_loc_fractions = tf.constant([1/6, 1/2, 2/3], dtype=tf.float64)
sensor_actual_locations = sensor_loc_fractions * L_total

# Determine the closest node indices for each sensor location
# Node k is at x = k * L_e
sensor_node_indices_float = sensor_actual_locations / L_e
sensor_node_indices = tf.cast(tf.round(sensor_node_indices_float), dtype=tf.int32)

# Ensure indices are within bounds [0, n_nodes-1]
sensor_node_indices = tf.clip_by_value(sensor_node_indices, 0, n_nodes - 1)

print(f"Sensor locations (fractions of L): {sensor_loc_fractions.numpy()}")
print(f"Sensor locations (absolute m): {sensor_actual_locations.numpy()}")
print(f"Element length L_e: {L_e:.2f} m")
print(f"Corresponding node indices for sensors: {sensor_node_indices.numpy()}")

# Verify node coordinates (optional)
# for i, node_idx in enumerate(sensor_node_indices.numpy()):
# print(f"Sensor {i+1} at {sensor_actual_locations.numpy()[i]:.2f}m is mapped to Node {node_idx} (at x={node_idx*L_e:.2f}m)")

# Extract acceleration time series for sensor locations
# uddot_t has shape [n_dof_vertical, num_points]
sensor_accelerations = tf.gather(uddot_t, sensor_node_indices, axis=0) # Shape [num_sensors, num_points]


# --- 8. Plot Results for Sensor Locations ---
print("\n--- Plotting Sensor Acceleration Responses ---")
plt.figure(figsize=(14, 8))
for i in range(sensor_node_indices.shape[0]):
    node_idx = sensor_node_indices[i]
    loc_frac = sensor_loc_fractions[i]
    actual_loc_m = sensor_actual_locations[i]
    
    # Construct label, converting fraction to string like "L/6", "L/2", "2L/3"
    if loc_frac == 1/6: label_frac_str = "L/6"
    elif loc_frac == 1/2: label_frac_str = "L/2"
    elif loc_frac == 2/3: label_frac_str = "2L/3"
    else: label_frac_str = f"{loc_frac.numpy():.2f}L"

    plt.plot(t.numpy(), sensor_accelerations[i, :].numpy(),
             label=f'Sensor at {label_frac_str} ({actual_loc_m:.2f}m, Node {node_idx.numpy()})')

plt.title(f'Beam Acceleration Response at Sensor Locations (Modal Damping $\zeta={zeta_val.numpy()}$)')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Acceleration ($m/s^2$)')
plt.legend()
plt.grid(True)
# Add vertical lines for load duration
plt.axvline(x=load_start_time.numpy(), color='gray', linestyle='--', label=f'Load Start ({load_start_time.numpy()}s)')
plt.axvline(x=load_end_time.numpy(), color='gray', linestyle='-.', label=f'Load End ({load_end_time.numpy()}s)')
# To avoid duplicate labels for axvline if called in a loop, handle legend carefully or call once.
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # Remove duplicate labels for axvlines
plt.legend(by_label.values(), by_label.keys())
plt.show()

# --- Optional: Plot original center node acceleration for comparison (if center node is not L/2) ---
# center_node_accel = uddot_t[center_dof_index, :] # center_dof_index calculated before
# plt.figure(figsize=(12, 6))
# plt.plot(t.numpy(), center_node_accel.numpy(), label=f'Center Node (Node {center_dof_index}) Acceleration')
# plt.xlabel('Time (s)')
# plt.ylabel('Vertical Acceleration ($m/s^2$)')
# plt.title(f'Acceleration at Center Node (Node {center_dof_index}) with Damping $\zeta={zeta_val.numpy()}$')
# plt.grid(True)
# plt.axvline(x=load_start_time.numpy(), color='gray', linestyle='--', label=f'Load Start')
# plt.axvline(x=load_end_time.numpy(), color='gray', linestyle='-.', label=f'Load End')
# plt.legend()
# plt.show()