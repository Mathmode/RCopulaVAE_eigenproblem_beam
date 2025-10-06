#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:54:53 2025

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
n_m = 5 # Number of modes to use (not needed for assembly)

# Material and Geometric Properties
E = 210e9  # Young's modulus in Pa
I = 8.33e-6  # Second moment of area in m^4
rho = 7850  # Density in kg/m^3
A = 0.005  # Cross-sectional area in m^2
L_total = 10.0 # Total length of the beam in m (use float)
L_e = L_total / n_elements  # Length of each element

# Correct number of DOFs for 2D beam elements
n_dof = 2 * (n_elements + 1)
print(f"Number of elements: {n_elements}")
print(f"Length of each element (L_e): {L_e} m")
print(f"Total number of DOFs (2 per node): {n_dof}")

# Element stiffness matrix Ke (4x4) - Renamed Ke_base to Ke
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j] for element connecting node i and j
Ke = E * I / L_e**3 * tf.constant([
    [12, 6*L_e, -12, 6*L_e],
    [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
    [-12, -6*L_e, 12, -6*L_e],
    [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
], dtype=tf.float64)

# Element mass matrix Me (4x4) - Consistent Mass Matrix including shear deformation terms
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j]
# Note: Included terms like 6*I/(5*A*L_e**2) relate to Timoshenko beams (shear deformation)
# If using pure Euler-Bernoulli, these I/(A*L...) terms might be omitted.
# Using the matrix exactly as provided by the user.
Me = (rho * A * L_e) * tf.constant([
    [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
    [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
    [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
    [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
], dtype=tf.float64)

print(f"\nElement Stiffness Matrix Ke (shape {Ke.shape})")
# print(Ke.numpy()) # Optional: print element matrix
print(f"Element Mass Matrix Me (shape {Me.shape})")
# print(Me.numpy()) # Optional: print element matrix

# --- 2. Assemble Global Matrices ---

# Initialize global matrices with zeros
K_global = tf.zeros((n_dof, n_dof), dtype=tf.float64)
M_global = tf.zeros((n_dof, n_dof), dtype=tf.float64)

# Flatten element matrices for scatter update
Ke_flat = tf.reshape(Ke, [-1]) # Shape [16]
Me_flat = tf.reshape(Me, [-1]) # Shape [16]

print("\nAssembling global matrices...")
# Loop through each element to assemble
for e in range(n_elements):
    # Node numbers for the current element
    node_i = e
    node_j = e + 1

    # Global DOF indices for the element's nodes
    # [v_i, theta_i, v_j, theta_j] -> [2*i, 2*i+1, 2*j, 2*j+1]
    idx = tf.constant([2*node_i, 2*node_i + 1, 2*node_j, 2*node_j + 1], dtype=tf.int32)

    # Create the pairs of global indices for the 4x4 block
    # E.g., for Ke[0,0], the global indices are (idx[0], idx[0])
    #       for Ke[0,1], the global indices are (idx[0], idx[1]) ...
    indices_pairs = tf.stack(tf.meshgrid(idx, idx, indexing='ij'), axis=-1)
    indices_pairs = tf.reshape(indices_pairs, [-1, 2]) # Shape [16, 2]

    # Add element contributions to global matrices using scatter update
    K_global = tf.tensor_scatter_nd_add(K_global, indices_pairs, Ke_flat)
    M_global = tf.tensor_scatter_nd_add(M_global, indices_pairs, Me_flat)



M = M_global 
K = K_global 


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

# --- 3. Select Modes and Normalize (Reusing) ---
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

print("\nSelected Frequencies (rad/s):", selected_omegas.numpy())
# Omitting verification prints for brevity

# Modal stiffnesses (omega_i^2 for normalized modes)
k_r = selected_omegas_sq # Shape [n_m]
# Modal masses (Identity for normalized modes)
m_r = tf.ones(n_m, dtype=tf.float64) # Shape [n_m]
# Modal damping (neglected)
c_r = tf.zeros(n_m, dtype=tf.float64) # Shape [n_m]


# --- 4. Define Load and Time (Reusing) ---
force_amplitude = tf.constant(2.0, dtype=tf.float64)
forcing_frequency_hz = tf.constant(10.0, dtype=tf.float64)
forcing_frequency_rad = 2.0 * np.pi * forcing_frequency_hz

load_vector = np.zeros(n_dof)
center_dof_index = n_dof // 2
load_vector[center_dof_index] = 1.0
p = tf.constant(load_vector, shape=(n_dof, 1), dtype=tf.float64) #Only different from 0 at the application positions, that here is middle deck. Its like a mask or selection matrix

# Time vector and step
# t_max is calculated from float64 tensors, so it's float64
t_max = 10.0 / forcing_frequency_hz
n_steps = 200 # Number of steps
# CORRECTION: Ensure start value is also explicitly tf.float64
start_time = tf.constant(0.0, dtype=tf.float64)
# Now both start_time and t_max are float64
t = tf.linspace(start_time, t_max, n_steps + 1) # n_steps+1 points for n_steps intervals
# dt will be calculated from t, so it will also be float64
dt = t[1] - t[0]
# Need .numpy() to print the value of a tensor
print(f"\nTime step dt: {dt.numpy():.4f} s")



# Full force vector over time Q(t) = p * F_0 * sin(Omega*t)
Omega = forcing_frequency_rad
F_t = force_amplitude * tf.sin(Omega * t) # Shape [n_steps+1]
Q_t = p @ tf.expand_dims(F_t, axis=0) # Shape [n_dof, n_steps+1]

# --- 5. Calculate Modal Forces (Reusing) ---
# Q_r(t) = Phi^T * Q(t)
# Shape: [n_m, n_steps+1]
Q_r_t = tf.transpose(Phi) @ Q_t

# --- 6. Solve Uncoupled Modal Equations (Newmark-beta) ---

# Newmark parameters (Average Acceleration Method)
gamma = tf.constant(0.5, dtype=tf.float64)
beta = tf.constant(0.25, dtype=tf.float64)

# Pre-calculate Newmark constants
a0 = 1.0 / (beta * dt**2)
a1 = 1.0 / (beta * dt)
a2 = 1.0 / (2.0 * beta) - 1.0
a3 = dt * (1.0 - gamma)
a4 = dt * gamma
a5 = dt**2 * (0.5 - beta)
a6 = dt * (gamma / (beta * dt) - 1.0) # Corrected term for c_r contribution to q_hat
a7 = dt * (0.5 * gamma / beta - 1.0) # Corrected term for c_r contribution to q_hat

# Effective stiffness k_hat = k_r + (gamma/(beta*dt))*c_r + (1/(beta*dt^2))*m_r
# Since c_r=0 and m_r=1 (normalized):
k_hat = k_r + a0 * m_r # Shape [n_m]

# Initial conditions (assuming zero displacement and velocity)
eta_0 = tf.zeros(n_m, dtype=tf.float64)
d_eta_0 = tf.zeros(n_m, dtype=tf.float64)

# Calculate initial acceleration: m_r*dd_eta_0 + c_r*d_eta_0 + k_r*eta_0 = q_r_0
# Since m_r=1, c_r=0, eta_0=0, d_eta_0=0 => dd_eta_0 = q_r_0
q_r_0 = Q_r_t[:, 0] # Modal force at t=0
dd_eta_0 = q_r_0 / m_r # Since m_r is 1, this is just q_r_0

# Store results
eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)
d_eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)
dd_eta_history = tf.TensorArray(tf.float64, size=n_steps + 1, dynamic_size=False)

eta_history = eta_history.write(0, eta_0)
d_eta_history = d_eta_history.write(0, d_eta_0)
dd_eta_history = dd_eta_history.write(0, dd_eta_0)

# Define the function for one step of tf.scan
def newmark_step(prev_state, forces_k1):
    """
    Performs one step of the Newmark-beta integration.

    Args:
        prev_state: Tuple (eta_k, d_eta_k, dd_eta_k) from the previous step.
        forces_k1: Modal forces q_r at step k+1.

    Returns:
        Tuple (eta_k1, d_eta_k1, dd_eta_k1) for the current step.
    """
    eta_k, d_eta_k, dd_eta_k = prev_state
    q_r_k1 = forces_k1

    # Calculate effective force q_hat
    # q_hat = q_r_k1 + m_r*(a0*eta_k + a1*d_eta_k + a2*dd_eta_k) + c_r*(a1*eta_k + (gamma/beta - 1)*d_eta_k + dt/2*(gamma/beta - 2)*dd_eta_k)
    # Since c_r=0 and m_r=1:
    q_hat = q_r_k1 + m_r * (a0 * eta_k + a1 * d_eta_k + a2 * dd_eta_k)

    # Solve for displacement at k+1
    # k_hat * eta_k1 = q_hat
    eta_k1 = q_hat / k_hat

    # Update acceleration and velocity at k+1
    dd_eta_k1 = a0 * (eta_k1 - eta_k) - a1 * d_eta_k - a2 * dd_eta_k
    d_eta_k1 = d_eta_k + a3 * dd_eta_k + a4 * dd_eta_k1

    return (eta_k1, d_eta_k1, dd_eta_k1)




# Prepare inputs for tf.scan
# We need forces from step 1 to n_steps
forces_scan_input = tf.transpose(Q_r_t[:, 1:]) # Shape [n_steps, n_m]
initial_state = (eta_0, d_eta_0, dd_eta_0)

# Run the time stepping integration using tf.scan
scan_output = tf.scan(
    newmark_step,
    forces_scan_input,
    initializer=initial_state
)
# --- Debugging Shapes ---
print("\n--- Debugging Shapes After Scan ---")
scan_output_type = type(scan_output)
print(f"Type of scan_output: {scan_output_type}")

scan_output_len = -1 # Initialize length
if isinstance(scan_output, (list, tuple)):
    try:
        scan_output_len = len(scan_output)
        print(f"Length of scan_output: {scan_output_len}") # Should print 3
    except Exception as e:
        print(f"Could not get length of scan_output: {e}")

# --- CORRECTED UNPACKING ---
# Based on the error, assume scan_output is (eta_hist, d_eta_hist, dd_eta_hist)
if isinstance(scan_output, (list, tuple)) and scan_output_len == 3:

    eta_scan = scan_output[0]    # Shape [n_steps, n_m]
    d_eta_scan = scan_output[1]    # Shape [n_steps, n_m]
    dd_eta_scan = scan_output[2]   # Shape [n_steps, n_m]

    print(f"\nUnpacked Shapes:")
    print(f"Shape of eta_scan: {eta_scan.shape}")     # Expect [500, 3]
    print(f"Shape of d_eta_scan: {d_eta_scan.shape}")   # Expect [500, 3]
    print(f"Shape of dd_eta_scan: {dd_eta_scan.shape}") # Expect [500, 3]

    # Combine initial state with scan results
    # Prepare initial states with expanded dim
    eta_0_expanded = tf.expand_dims(eta_0, axis=0)     # Shape [1, 3]
    d_eta_0_expanded = tf.expand_dims(d_eta_0, axis=0)   # Shape [1, 3]
    dd_eta_0_expanded = tf.expand_dims(dd_eta_0, axis=0) # Shape [1, 3]

    print(f"\nShapes before concat:")
    print(f"Shape of eta_0_expanded: {eta_0_expanded.shape}")
    print(f"Shape of eta_scan: {eta_scan.shape}")

    # Check ranks (should both be 2)
    rank_eta_0 = tf.rank(eta_0_expanded)
    rank_eta_scan = tf.rank(eta_scan)
    print(f"Rank of eta_0_expanded: {rank_eta_0}")
    print(f"Rank of eta_scan: {rank_eta_scan}")

    if rank_eta_0 == rank_eta_scan: # Proceed if ranks match
         eta_all = tf.concat([eta_0_expanded, eta_scan], axis=0) # Shape [n_steps+1, n_m]
         d_eta_all = tf.concat([d_eta_0_expanded, d_eta_scan], axis=0)
         dd_eta_all = tf.concat([dd_eta_0_expanded, dd_eta_scan], axis=0)

         # Transpose results to match previous convention [n_m, n_steps+1]
         eta_t_numeric = tf.transpose(eta_all)
         d_eta_t_numeric = tf.transpose(d_eta_all)
         dd_eta_t_numeric = tf.transpose(dd_eta_all)

         # --- Transform Back to Physical Coordinates ---
         # Displacement: x(t) = Phi * eta(t)
         # Shape: [n_dof, n_steps+1]
         x_t_numeric = Phi @ eta_t_numeric

         # Acceleration: x_ddot(t) = Phi * dd_eta(t) (Use calculated modal accelerations)
         # Shape: [n_dof, n_steps+1]
         x_ddot_t_numeric = Phi @ dd_eta_t_numeric


         # --- 8. Results --- (Plotting code)
         print(f"\nNumeric Calculation (Newmark) complete. Shapes:")
         print(f"Time t: {t.shape}")
         print(f"Modal displacement eta(t): {eta_t_numeric.shape}")
         print(f"Modal acceleration dd_eta(t): {dd_eta_t_numeric.shape}")
         print(f"Physical displacement x(t): {x_t_numeric.shape}")
         print(f"Physical acceleration x_ddot(t): {x_ddot_t_numeric.shape}")

         # Plot displacement of the center node
         plt.figure(figsize=(12, 7))
         plt.plot(t.numpy(), x_t_numeric.numpy()[center_dof_index, :], label=f'Numeric Disp. DOF {center_dof_index} (Newmark)', color='blue')
         plt.title(f'Displacement Response (First {n_m} Modes) - Newmark Integration')
         plt.xlabel('Time (s)')
         plt.ylabel('Displacement')
         plt.legend()
         plt.grid(True)
         plt.show()

         # Plot acceleration of the center node
         plt.figure(figsize=(12, 7))
         plt.plot(t.numpy(), x_ddot_t_numeric.numpy()[center_dof_index, :], label=f'Numeric Accel. DOF {center_dof_index} (Newmark)', color='orange')
         plt.title(f'Acceleration Response (First {n_m} Modes) - Newmark Integration')
         plt.xlabel('Time (s)')
         plt.ylabel('Acceleration')
         plt.legend()
         plt.grid(True)
         plt.show()

    else:
         # This should not happen now if unpacking is correct
         raise ValueError(f"Rank mismatch persist after correcting unpacking! Rank {rank_eta_0} vs Rank {rank_eta_scan}")

else:
    # Error if scan_output wasn't a tuple/list or length wasn't 3
    raise TypeError(f"tf.scan output has unexpected structure. Expected tuple/list of length 3 based on previous error, but got type {scan_output_type} with length {scan_output_len}.")




