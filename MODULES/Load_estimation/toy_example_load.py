#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:41:52 2025

@author: afernandez
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Ensure TensorFlow is using float64 for better numerical stability
tf.keras.backend.set_floatx('float64')
print(f"TensorFlow version: {tf.__version__}")


# Ensure TensorFlow uses float64
tf.keras.backend.set_floatx('float64')

# --- 1. Beam Properties and Element Definitions ---
n_elements = 10
n_m = 5 # Number of modes to use (not needed for assembly) # Commented out as not used here

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
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j] for element connecting node i and j
Ke = E * I / L_e**3 * tf.constant([
    [12, 6*L_e, -12, 6*L_e],
    [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
    [-12, -6*L_e, 12, -6*L_e],
    [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
], dtype=tf.float64)

# Element mass matrix Me (4x4) - Consistent Mass Matrix
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j]
# Note: The provided matrix includes terms related to Timoshenko beam theory (shear deformation).
Me = (rho * A * L_e) * tf.constant([
    [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
    [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
    [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
    [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
], dtype=tf.float64)

print(f"\nElement Stiffness Matrix Ke (shape {Ke.shape})")
print(f"Element Mass Matrix Me (shape {Me.shape})")

# --- 2. Assemble Global Matrices (Full - including rotations) ---

# Initialize full global matrices with zeros
K_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)
M_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)

# Flatten element matrices for scatter update
Ke_flat = tf.reshape(Ke, [-1]) # Shape [16]
Me_flat = tf.reshape(Me, [-1]) # Shape [16]

print("\nAssembling full global matrices (including rotations)...")
# Loop through each element to assemble
for e in range(n_elements):
    # Node numbers for the current element
    node_i = e
    node_j = e + 1

    # Global DOF indices for the element's nodes (v_i, theta_i, v_j, theta_j)
    idx = tf.constant([2*node_i, 2*node_i + 1, 2*node_j, 2*node_j + 1], dtype=tf.int32)

    # Create the pairs of global indices for the 4x4 block
    indices_pairs = tf.stack(tf.meshgrid(idx, idx, indexing='ij'), axis=-1)
    indices_pairs = tf.reshape(indices_pairs, [-1, 2]) # Shape [16, 2]

    # Add element contributions to global matrices using scatter update
    K_global_full = tf.tensor_scatter_nd_add(K_global_full, indices_pairs, Ke_flat)
    M_global_full = tf.tensor_scatter_nd_add(M_global_full, indices_pairs, Me_flat)

# --- 3. Extract Submatrices for Vertical DOFs Only ---

# Identify the indices corresponding to vertical displacements (v)
# These are the DOFs at indices 0, 2, 4, ..., 2*n_elements
vertical_dof_indices = tf.range(0, n_dof_full, 2, dtype=tf.int32)
print("\nIndices for vertical DOFs:", vertical_dof_indices.numpy())


# Use tf.gather to select rows and columns corresponding to vertical DOFs
# First gather rows
K_vertical_rows = tf.gather(K_global_full, vertical_dof_indices, axis=0)
M_vertical_rows = tf.gather(M_global_full, vertical_dof_indices, axis=0)
# Then gather columns from the result
K_vertical = tf.gather(K_vertical_rows, vertical_dof_indices, axis=1)
M_vertical = tf.gather(M_vertical_rows, vertical_dof_indices, axis=1)

# Assign to M and K as requested by the user (though perhaps better names would be K_v, M_v)
M = M_vertical
K = K_vertical


# --- 2. Solve Generalized Eigenvalue Problem (Reusing) ---
try:
    M_inv = tf.linalg.inv(M)
    A = M_inv @ K
    eigenvalues_sq, eigenvectors = tf.linalg.eigh(A)
except tf.errors.InvalidArgumentError:
     print("Using Cholesky decomposition approach for eigenvalue problem.")
     L = tf.linalg.cholesky(M)
     L_inv = tf.linalg.inv(L)
     A_cholesky = L_inv @ K @ tf.linalg.inv(tf.transpose(L))
     eigenvalues_sq, y = tf.linalg.eigh(A_cholesky)
     eigenvectors = tf.linalg.solve(tf.transpose(L), y)

# --- 3. Select Modes and Normalize ---
omegas_sq = eigenvalues_sq
omegas = tf.sqrt(omegas_sq)

Phi = eigenvectors[:, :n_m]
selected_omegas_sq = omegas_sq[:n_m]
selected_omegas = omegas[:n_m]

# Mass normalization
M_r_diag = tf.einsum('ji,jk,kl->il', Phi, M, Phi)
norm_factors = 1.0 / tf.sqrt(tf.linalg.diag_part(M_r_diag))
Phi_normalized = Phi * norm_factors
Phi = Phi_normalized # Use normalized modes

# Modal stiffnesses (omega_i^2 for normalized modes)
k_r = selected_omegas_sq # Shape [n_m]
# Modal masses (Identity for normalized modes)
m_r = tf.ones(n_m, dtype=tf.float64) # Shape [n_m]
# Modal damping (neglected)
c_r = tf.zeros(n_m, dtype=tf.float64) # Shape [n_m]

# --- 4. Define Load and Time ---
force_amplitude = tf.constant(5.0, dtype=tf.float64)
forcing_frequency_hz = tf.constant(100, dtype=tf.float64) # Frequency of the sine wave
forcing_frequency_rad = 2.0 * np.pi * forcing_frequency_hz
Omega = forcing_frequency_rad # Angular frequency

# --- Define Time Vector ---
t_max_sim = tf.constant(10.0, dtype=tf.float64) # Total simulation time
dt = tf.constant(0.2, dtype=tf.float64)  # Desired time step
# --- Define Load Application Interval ---
load_start_time = tf.constant(1.0, dtype=tf.float64) # NEW: Load starts at 1s
load_end_time = tf.constant(5.0, dtype=tf.float64) # NEW: Load ends at 5s

# Calculate number of steps (based on t_max_sim and dt -> 500 steps)
n_steps = tf.cast(tf.round(t_max_sim / dt), dtype=tf.int32)

# Create time vector t using tf.linspace (from 0 to 5s)
start_time = tf.constant(0.0, dtype=tf.float64)
num_points = n_steps + 1
t = tf.linspace(start_time, t_max_sim, num_points) # Time vector from 0 to 5s
actual_dt = t[1] - t[0]


# --- Define Load Location Vector (p) ---
n_dof = n_dof_vertical # We consider vertical modes only 
center_dof_index = n_dof//2

load_vector = np.zeros(n_dof) # Keep as is
# center_dof_index assumed defined previously (e.g., n_dof // 2)
load_vector[center_dof_index] = 1.0 # Keep as is
p = tf.constant(load_vector, shape=(n_dof, 1), dtype=tf.float64) # Keep as is


# --- Define Force Magnitude over Time F(t) ---
# Calculate the base sinusoidal force component for all time steps (0s to 5s)
F_t_sinusoidal = force_amplitude * tf.sin(Omega * t)

# *** MODIFIED MASK ***
# Create a mask that is 1.0 ONLY between load_start_time (1s) and load_end_time (5s)
mask_condition = tf.logical_and(t >= load_start_time, t <= load_end_time)
time_mask = tf.cast(mask_condition, dtype=tf.float64) # Convert boolean (True/False) to float (1.0/0.0)

# Apply the NEW mask: Force is sinusoidal only within the interval [1s, 5s], zero otherwise
F_t = F_t_sinusoidal * time_mask # Shape [num_points]

# --- Calculate Full Nodal Force Vector Q(t) ---
# Recalculate with the new F_t
Q_t = p @ tf.expand_dims(F_t, axis=0) # Shape [n_dof, num_points]
Q_r_t = tf.transpose(Phi) @ Q_t


# ---  Plot F(t) to verify ---
plt.figure()
plt.plot(t.numpy(), F_t.numpy())
plt.title('Force Magnitude F(t)')
plt.xlabel('Time (s)')
plt.ylabel('Force Amplitude')
plt.grid(True)
plt.show()



# --- 6. Newmark-beta Time Integration in Modal Coordinates ---
print("\n--- Starting Newmark Method ---")

# Newmark parameters (Average acceleration method)
beta_newmark = tf.constant(0.25, dtype=tf.float64)
gamma_newmark = tf.constant(0.500, dtype=tf.float64)

# Time step value (already calculated as dt, ensure it's float64)
dt_val = tf.cast(actual_dt, dtype=tf.float64) # Use actual_dt from linspace calculation
print(f"Using time step dt = {dt_val.numpy():.6f} s for Newmark integration")

# Initial conditions (assuming zero displacement and velocity at t=0)
r_initial = tf.zeros((n_m, 1), dtype=tf.float64)      # Initial modal displacement [n_m, 1]
rdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)   # Initial modal velocity [n_m, 1]

# Get modal stiffness (k_r = selected_omegas_sq) and ensure correct shape
k_r_col = tf.reshape(k_r, (n_m, 1)) # Modal stiffnesses [n_m, 1]

# Calculate initial modal acceleration from equation of motion at t=0
# m_r * rddot_0 + c_r * rdot_0 + k_r * r_0 = Q_r_0
# Assuming m_r=I (mass normalized), c_r=0 (no damping):
# rddot_0 + k_r * r_0 = Q_r_0
Q_r_0 = Q_r_t[:, 0:1] # Modal force at t=0, shape [n_m, 1]
rddot_initial = Q_r_0 - k_r_col * r_initial # Initial modal acceleration [n_m, 1]

# Pre-allocate history storage using TensorArray (flexible)
# num_points = n_steps + 1 (total number of time points including t=0)
r_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="r_history", clear_after_read=False)
rdot_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="rdot_history", clear_after_read=False)
rddot_history = tf.TensorArray(dtype=tf.float64, size=num_points, name="rddot_history", clear_after_read=False)

# Store initial conditions at index 0
r_history = r_history.write(0, r_initial)
rdot_history = rdot_history.write(0, rdot_initial)
rddot_history = rddot_history.write(0, rddot_initial)

# --- Newmark Time-Stepping Loop ---
# Initialize current state
r_current = r_initial
rdot_current = rdot_initial
rddot_current = rddot_initial

# Pre-calculate constant terms in Newmark update for efficiency
# Denominator term for solving rddot_next: (m_eff + gamma*dt*c_eff + beta*dt^2*k_eff)
# Here: m_eff=1, c_eff=0, k_eff=k_r_col
den = 1.0 + beta_newmark * dt_val**2 * k_r_col # Denominator, shape [n_m, 1]

print(f"Starting Newmark integration loop for {n_steps} steps...")
# Loop from time step i = 0 to n_steps-1
# Calculates state at time t[i+1] using state at t[i]
for i in tf.range(n_steps):
    # Get modal force for the *next* time step (t_{i+1})
    Q_r_next = Q_r_t[:, i+1:i+2] # Shape [n_m, 1]

    # Predictor term based on state at time t_i (current step i)
    # r_predictor = r_i + dt*rdot_i + (0.5 - beta)*dt^2 * rddot_i
    r_predictor_term = r_current + dt_val * rdot_current + (0.5 - beta_newmark) * dt_val**2 * rddot_current

    # Calculate effective force for solving acceleration
    # F_eff = F_{i+1} - c_eff*rdot_predictor - k_eff*r_predictor
    # Here: c_eff=0
    # F_eff = Q_r_next - k_r_col * r_predictor_term
    F_eff = Q_r_next - k_r_col * r_predictor_term # Shape [n_m, 1]

    # Solve for modal acceleration at time t_{i+1}
    # rddot_next = F_eff / den
    rddot_next = F_eff / den # Shape [n_m, 1]

    # Correct modal velocity at time t_{i+1}
    # rdot_next = rdot_i + (1-gamma)*dt*rddot_i + gamma*dt*rddot_next
    rdot_next = rdot_current + (1.0 - gamma_newmark) * dt_val * rddot_current + gamma_newmark * dt_val * rddot_next # Shape [n_m, 1]

    # Correct modal displacement at time t_{i+1}
    # r_next = r_predictor + beta*dt^2*rddot_next
    r_next = r_predictor_term + beta_newmark * dt_val**2 * rddot_next # Shape [n_m, 1]

    # Store results for step i+1 (at time t[i+1])
    r_history = r_history.write(i + 1, r_next)
    rdot_history = rdot_history.write(i + 1, rdot_next)
    rddot_history = rddot_history.write(i + 1, rddot_next)

    # Update current state for the next iteration (state at i becomes state at i+1)
    r_current = r_next
    rdot_current = rdot_next
    rddot_current = rddot_next

    # Optional: Print progress
    # if (i + 1) % (n_steps // 10) == 0:
    #     print(f"  Completed step {i+1}/{n_steps}")


print("Newmark integration finished.")

# Stack results from TensorArray into single tensors
# Resulting shape from stack() is [num_points, n_m, 1]
r_history_stacked = r_history.stack()
rdot_history_stacked = rdot_history.stack()
rddot_history_stacked = rddot_history.stack()

# Reshape to [n_m, num_points] for easier use
r_history_final = tf.transpose(tf.squeeze(r_history_stacked, axis=-1)) # Shape [n_m, num_points]
rdot_history_final = tf.transpose(tf.squeeze(rdot_history_stacked, axis=-1)) # Shape [n_m, num_points]
rddot_history_final = tf.transpose(tf.squeeze(rddot_history_stacked, axis=-1)) # Shape [n_m, num_points]

print(f"Shape of final modal displacement history (r_history_final): {r_history_final.shape}")

# Physical displacement u(t) = Phi * r(t) (calculated as before)
u_t = tf.matmul(Phi, r_history_final) # Shape [n_dof, num_points]
print(f"Shape of physical displacement history (u_t): {u_t.shape}")

# Physical acceleration uddot(t) = Phi * rddot(t)
# Phi shape: [n_dof, n_m]
# rddot_history_final shape: [n_m, num_points]
uddot_t = tf.matmul(Phi, rddot_history_final) # Shape [n_dof, num_points]
print(f"Shape of physical acceleration history (uddot_t): {uddot_t.shape}")


# --- 8. Plot Results (Example: Acceleration at center node) ---
print("\n--- Plotting Results ---")
# Ensure 'center_dof_index' is defined from the load application section

# Select ACCELERATION time series for the center node
center_node_accel = uddot_t[4, :] # Acceleration time series

plt.figure(figsize=(12, 6))
# Plot acceleration vs time
plt.plot(t.numpy(), center_node_accel.numpy(), label='Center Node Acceleration')
# Add LaTeX formatting for the y-axis label
plt.xlabel('Time (s)')
plt.ylabel('Vertical Acceleration ($m/s^2$)') # Updated label with units
plt.grid(True)
# Add vertical line showing end of forcing period
# plt.axvline(x=load_duration.numpy(), color='r', linestyle='--', label=f'Force Off (t={load_duration.numpy()}s)')
plt.legend()
plt.show()

# You can keep the displacement plot code commented out or remove it if not needed
# center_node_disp = u_t[center_dof_index, :] # Displacement time series
# plt.figure(figsize=(12, 6))
# plt.plot(t.numpy(), center_node_disp.numpy(), label='Center Node Displacement')
# plt.title(f'Vertical Displacement of Center Node (DOF {center_dof_index}) vs. Time (Newmark)')
# plt.xlabel('Time (s)')
# plt.ylabel('Vertical Displacement (m)')
# plt.grid(True)
# plt.axvline(x=load_duration.numpy(), color='r', linestyle='--', label=f'Force Off (t={load_duration.numpy()}s)')
# plt.legend()
# plt.show()
# # --- 7. Reconstruct Physical Displacements ---
# print("\n--- Reconstructing Physical Displacements ---")
# # Physical displacement u(t) = Phi * r(t)
# # Phi shape: [n_dof, n_m]
# # r_history_final shape: [n_m, num_points]
# u_t = tf.matmul(Phi, r_history_final) # Shape [n_dof, num_points]

# print(f"Shape of physical displacement history (u_t): {u_t.shape}")


# # --- 8. Plot Results (Example: Displacement at center node) ---
# print("\n--- Plotting Results ---")
# # Ensure 'center_dof_index' is defined from the load application section
# center_node_disp = u_t[center_dof_index, :] # Displacement time series for the center node

# plt.figure(figsize=(12, 6))
# plt.plot(t.numpy(), center_node_disp.numpy(), label='Center Node Displacement')
# plt.title(f'Vertical Displacement of Center Node (DOF {center_dof_index}) vs. Time (Newmark)')
# plt.xlabel('Time (s)')
# plt.ylabel('Vertical Displacement (m)')
# plt.grid(True)
# # Add vertical line showing end of forcing period
# plt.axvline(x=load_duration.numpy(), color='r', linestyle='--', label=f'Force Off (t={load_duration.numpy()}s)')
# plt.legend()
# plt.show()

# # Optional: Plot modal coordinate history
# plt.figure(figsize=(12, 6))
# for j in range(n_m):
#       plt.plot(t.numpy(), r_history_final[j, :].numpy(), label=f'Mode {j+1}')
# plt.title('Modal Coordinates r_j(t) vs. Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Modal Amplitude')
# plt.legend()
# plt.grid(True)
# plt.show()







# # --- 6. Solve Uncoupled Modal Equations (Newmark-beta) ---

# # Newmark parameters (Average Acceleration Method)
# gamma = tf.constant(0.5, dtype=tf.float64)
# beta = tf.constant(0.25, dtype=tf.float64)

# # Pre-calculate Newmark constants
# a0 = 1.0 / (beta * dt**2)
# a1 = 1.0 / (beta * dt)
# a2 = 1.0 / (2.0 * beta) - 1.0
# a3 = dt * (1.0 - gamma)
# a4 = dt * gamma
# a5 = dt**2 * (0.5 - beta)
# a6 = dt * (gamma / (beta * dt) - 1.0) # Corrected term for c_r contribution to q_hat
# a7 = dt * (0.5 * gamma / beta - 1.0) # Corrected term for c_r contribution to q_hat

# # Effective stiffness k_hat = k_r + (gamma/(beta*dt))*c_r + (1/(beta*dt^2))*m_r
# # Since c_r=0 and m_r=1 (normalized):
# k_hat = k_r + a0 * m_r # Shape [n_m]

# # Initial conditions (assuming zero displacement and velocity)
# eta_0 = tf.zeros(n_m, dtype=tf.float64)
# d_eta_0 = tf.zeros(n_m, dtype=tf.float64)

# # Calculate initial acceleration: m_r*dd_eta_0 + c_r*d_eta_0 + k_r*eta_0 = q_r_0
# # Since m_r=1, c_r=0, eta_0=0, d_eta_0=0 => dd_eta_0 = q_r_0
# q_r_0 = Q_r_t[:, 0] # Modal force at t=0
# dd_eta_0 = q_r_0 / m_r # Since m_r is 1, this is just q_r_0

# # Store results
# eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)
# d_eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)
# dd_eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)

# eta_history = eta_history.write(0, eta_0)
# d_eta_history = d_eta_history.write(0, d_eta_0)
# dd_eta_history = dd_eta_history.write(0, dd_eta_0)

# # Define the function for one step of tf.scan
# def newmark_step(prev_state, forces_k1):
#     """
#     Performs one step of the Newmark-beta integration.

#     Args:
#         prev_state: Tuple (eta_k, d_eta_k, dd_eta_k) from the previous step.
#         forces_k1: Modal forces q_r at step k+1.

#     Returns:
#         Tuple (eta_k1, d_eta_k1, dd_eta_k1) for the current step.
#     """
#     eta_k, d_eta_k, dd_eta_k = prev_state
#     q_r_k1 = forces_k1

#     # Calculate effective force q_hat
#     # q_hat = q_r_k1 + m_r*(a0*eta_k + a1*d_eta_k + a2*dd_eta_k) + c_r*(a1*eta_k + (gamma/beta - 1)*d_eta_k + dt/2*(gamma/beta - 2)*dd_eta_k)
#     # Since c_r=0 and m_r=1:
#     q_hat = q_r_k1 + m_r * (a0 * eta_k + a1 * d_eta_k + a2 * dd_eta_k)

#     # Solve for displacement at k+1
#     # k_hat * eta_k1 = q_hat
#     eta_k1 = q_hat / k_hat

#     # Update acceleration and velocity at k+1
#     dd_eta_k1 = a0 * (eta_k1 - eta_k) - a1 * d_eta_k - a2 * dd_eta_k
#     d_eta_k1 = d_eta_k + a3 * dd_eta_k + a4 * dd_eta_k1

#     return (eta_k1, d_eta_k1, dd_eta_k1)




# # Prepare inputs for tf.scan
# # We need forces from step 1 to n_steps
# forces_scan_input = tf.transpose(Q_r_t[:, 1:]) # Shape [n_steps, n_m]
# initial_state = (eta_0, d_eta_0, dd_eta_0)

# # Run the time stepping integration using tf.scan
# scan_output = tf.scan(
#     newmark_step,
#     forces_scan_input,
#     initializer=initial_state
# )
# # --- Debugging Shapes ---
# print("\n--- Debugging Shapes After Scan ---")
# scan_output_type = type(scan_output)
# print(f"Type of scan_output: {scan_output_type}")

# scan_output_len = -1 # Initialize length
# if isinstance(scan_output, (list, tuple)):
#     try:
#         scan_output_len = len(scan_output)
#         print(f"Length of scan_output: {scan_output_len}") # Should print 3
#     except Exception as e:
#         print(f"Could not get length of scan_output: {e}")

# # --- CORRECTED UNPACKING ---
# if isinstance(scan_output, (list, tuple)) and scan_output_len == 3:

#     eta_scan = scan_output[0]    # Shape [n_steps, n_m]
#     d_eta_scan = scan_output[1]    # Shape [n_steps, n_m]
#     dd_eta_scan = scan_output[2]   # Shape [n_steps, n_m]

#     print(f"Shape of eta_scan: {eta_scan.shape}")     # Expect [500, 3]
#     print(f"Shape of d_eta_scan: {d_eta_scan.shape}")   # Expect [500, 3]
#     print(f"Shape of dd_eta_scan: {dd_eta_scan.shape}") # Expect [500, 3]

#     # Combine initial state with scan results
#     # Prepare initial states with expanded dim
#     eta_0_expanded = tf.expand_dims(eta_0, axis=0)     # Shape [1, 3]
#     d_eta_0_expanded = tf.expand_dims(d_eta_0, axis=0)   # Shape [1, 3]
#     dd_eta_0_expanded = tf.expand_dims(dd_eta_0, axis=0) # Shape [1, 3]

#     print(f"\nShapes before concat:")
#     print(f"Shape of eta_0_expanded: {eta_0_expanded.shape}")
#     print(f"Shape of eta_scan: {eta_scan.shape}")

#     # Check ranks (should both be 2)
#     rank_eta_0 = tf.rank(eta_0_expanded)
#     rank_eta_scan = tf.rank(eta_scan)
#     print(f"Rank of eta_0_expanded: {rank_eta_0}")
#     print(f"Rank of eta_scan: {rank_eta_scan}")

#     if rank_eta_0 == rank_eta_scan: # Proceed if ranks match
#          eta_all = tf.concat([eta_0_expanded, eta_scan], axis=0) # Shape [n_steps+1, n_m]
#          d_eta_all = tf.concat([d_eta_0_expanded, d_eta_scan], axis=0)
#          dd_eta_all = tf.concat([dd_eta_0_expanded, dd_eta_scan], axis=0)

#          # Transpose results to match previous convention [n_m, n_steps+1]
#          eta_t_numeric = tf.transpose(eta_all)
#          d_eta_t_numeric = tf.transpose(d_eta_all)
#          dd_eta_t_numeric = tf.transpose(dd_eta_all)

#          # --- Transform Back to Physical Coordinates ---
#          # Displacement: x(t) = Phi * eta(t)
#          # Shape: [n_dof, n_steps+1]
#          x_t_numeric = Phi @ eta_t_numeric

#          # Acceleration: x_ddot(t) = Phi * dd_eta(t) (Use calculated modal accelerations)
#          # Shape: [n_dof, n_steps+1]
#          x_ddot_t_numeric = Phi @ dd_eta_t_numeric


#          # --- 8. Results --- (Plotting code)
#          print(f"\nNumeric Calculation (Newmark) complete. Shapes:")
#          print(f"Time t: {t.shape}")
#          print(f"Modal displacement eta(t): {eta_t_numeric.shape}")
#          print(f"Modal acceleration dd_eta(t): {dd_eta_t_numeric.shape}")
#          print(f"Physical displacement x(t): {x_t_numeric.shape}")
#          print(f"Physical acceleration x_ddot(t): {x_ddot_t_numeric.shape}")

#          # Plot displacement of the center node
#          plt.figure(figsize=(12, 7))
#          plt.plot(t.numpy(), x_t_numeric.numpy()[center_dof_index, :], label=f'Numeric Disp. DOF {center_dof_index} (Newmark)', color='blue')
#          plt.title(f'Displacement Response (First {n_m} Modes) - Newmark Integration')
#          plt.xlabel('Time (s)')
#          plt.ylabel('Displacement')
#          plt.legend()
#          plt.grid(True)
#          plt.show()

#          # Plot acceleration of the center node
#          plt.figure(figsize=(12, 7))
#          plt.plot(t.numpy(), x_ddot_t_numeric.numpy()[center_dof_index, :], label=f'Numeric Accel. DOF {center_dof_index} (Newmark)', color='orange')
#          plt.title(f'Acceleration Response (First {n_m} Modes) - Newmark Integration')
#          plt.xlabel('Time (s)')
#          plt.ylabel('Acceleration')
#          plt.legend()
#          plt.grid(True)
#          plt.show()

#     else:
#          # This should not happen now if unpacking is correct
#          raise ValueError(f"Rank mismatch persist after correcting unpacking! Rank {rank_eta_0} vs Rank {rank_eta_scan}")

# else:
#     # Error if scan_output wasn't a tuple/list or length wasn't 3
#     raise TypeError(f"tf.scan output has unexpected structure. Expected tuple/list of length 3 based on previous error, but got type {scan_output_type} with length {scan_output_len}.")

