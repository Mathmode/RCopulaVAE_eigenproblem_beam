#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:43:44 2025

@author: afernandez
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%###############################################################################################
#Graph configuration function (font sizes)
def plot_configuration():
    plt.rc('font', size = 16)          # controls default text sizes
    plt.rc('axes', titlesize = 16)     # fontsize of the axes title
    plt.rc('axes', labelsize = 16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 16)   # legend fontsize
    plt.rc('figure', titlesize= 16)   # fontsize of the figure title

# NEWMARK BETA SOLVER FUNCTION 
def Newmark_beta_solver_singleb(Qpred, Phi, m_col, c_col, k_col, t_vector, n_steps, beta_newmark = tf.constant(0.25, dtype=tf.float64), gamma_newmark = tf.constant(0.50, dtype=tf.float64)):
    ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
    The inputs are: 
        Qpred: tensor of predicted modal forces. Shape (num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
        Phi: Truncated mode shape matrix. Shape ( n_dofs, n_m) . It is a rectangular matrix after truncation. 
        m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
        t_vector: time vector according to step Deltat and final time tf. Shape (n_steps+ 1 = num_points_sim)
        beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
    
    The outputs are: 
        [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
        The three tensors in the tuple have shape (n_dof, num_points_sim)
        '''
    
    actual_dt = t_vector[1] - t_vector[0]                  # Actual time step used by linspace
    dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)   
    n_m = Phi.shape[1]    
    # Initial conditions
    q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
    qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
    Q_at_t0 = tf.transpose(Qpred[0:1, :]) # since Qpred has shape (num_points_sim, n_m)
    qddot_initial = Q_at_t0 - c_col * qdot_initial - k_col * q_initial
    
    # LHS Factor
    LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + beta_newmark * dt_val_newmark**2 * k_col
    # if tf.reduce_any(LHS_factor_modal <= 1e-12):
    #     tf.print(f"ERROR: LHS_factor_modal is too small or zero")
    
    @tf.function(jit_compile=True)
    def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step):
        # previous_state_tuple: (r_prev, rdot_prev, rddot_prev) from time t_i
        # Q_r_force_for_current_target_step: Modal force Q_r(t_{i+1}), shape [n_m, 1]

        q_current, qdot_current, qddot_current = previous_state_tuple

        # Predictor terms based on state at time t_i
        q_predictor = q_current + dt_val_newmark * qdot_current + \
                      (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
        qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current

        # Calculate effective modal force for solving acceleration at t_{i+1}
        RHS_force_modal = Q_force_for_current_target_step - \
                          c_col * qdot_predictor - k_col * q_predictor

        # Solve for modal acc^eleration at time t_{i+1}
        qddot_next = RHS_force_modal / LHS_factor_modal

        # Correct modal velocity and displacement at time t_{i+1}
        qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
        q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next

        # The function returns the new state (r_next, rdot_next, rddot_next)
        # This tuple will be passed as 'previous_state_tuple' to the next iteration and also collected by tf.scan as the output for this step.
        return (q_next, qdot_next, qddot_next)

    # Initial state for the scan (state at t=0)
    initial_scan_state = (q_initial, qdot_initial, qddot_initial)

    # Prepare the sequence of modal forces for tf.scan. (Remove the initial time t= 0 since that is the initial condition. Then we will add it). 
    elems_for_scan = tf.expand_dims(Qpred[1:, :], axis = -1) 
    # It seems that the expanded additional dimension is required for the structure management of tf.scan. 

    tf.print(f"Starting Newmark integration using tf.scan for  {n_steps} steps ...")
    # The output will be a tuple of three tensors:  (stacked_q_next, stacked_qdot_next, stacked_qddot_next)
    # Each tensor will have shape [n_steps, n_m, 1], representing states from t_1 to t_{n_steps}
    scan_results_tuple = tf.scan(
        fn=newmark_scan_step,
        elems=elems_for_scan,
        initializer=initial_scan_state,
        name="newmark_beta_solver"
    )

    # Unpack results from scan. These are histories from t_1 to t_{n_steps} (n_steps items)
    # q_scan_output = scan_results_tuple[0]    # Shape [n_steps, n_m, 1] # We do not use it in the loss
    # qdot_scan_output = scan_results_tuple[1] # Shape [n_steps, n_m, 1] # We do not use it in the loss
    qddot_scan_output = scan_results_tuple[2]# Shape [n_steps, n_m, 1]
    qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)
    tf.print("Newmark integration with tf.scan finished.")

    # Optional: Check for NaNs in the final results (as an example)
    if tf.reduce_any(tf.math.is_nan(qddot_history_full)):
        tf.print("NaN detected in rddot_history_full. Check parameters and matrices.")

    # Reshape results to [n_m, num_points_sim] for easier use (same as original code)
    # Current shape is [num_points_sim, n_m, 1]
    qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1,0])
    # Transform modal accelerations back to physical accelerations
    uddot = tf.matmul(Phi, qddot_history_final) # Shape [n_dof_vertical, num_points_sim]
    return uddot




#%%##### MAIN #################################################################################
plot_configuration()

# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations
tf.keras.backend.set_floatx('float64')
print(f"TensorFlow version: {tf.__version__}")
# For detailed NaN/Inf checks during debugging, you can uncomment the next line:
# tf.debugging.enable_check_numerics()

# --- 1. Beam Properties and Element Definitions ---
# Define the number of finite elements and the number of modes to use in the analysis
n_elements = 10  # Number of beam elements
n_m = 5    # Number of modes to retain for modal superposition

# Material and Geometric Properties of the beam
E = 210e9        # Young's modulus in Pascals (Pa)
I = 8.33e-6      # Second moment of area in m^4
rho = 7850       # Density in kg/m^3
A = 0.0005       # Cross-sectional area in m^2
L_total = 20.0   # Total length of the beam in meters (m)
L_e = L_total / n_elements  # Length of each individual element

# Degrees of Freedom (DOF) calculations
n_nodes = n_elements + 1  # Total number of nodes
n_dof_full = 2 * n_nodes    # Total DOFs considering vertical displacement (v) and rotation (theta) at each node
n_dof_vertical = n_nodes  # Total DOFs if only considering vertical displacements (v_0, v_1, ..., v_{n_nodes-1})

# Print basic model information
print(f"Number of elements: {n_elements}")
print(f"Number of nodes: {n_nodes}")
print(f"Length of each element (L_e): {L_e} m")
print(f"Total vertical DOFs: {n_dof_vertical}")

# Element Stiffness Matrix (Ke) for a 2D Euler-Bernoulli beam element
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j] for an element connecting node i and j
Ke = E * I / L_e**3 * tf.constant([
    [12,    6*L_e,  -12,    6*L_e],
    [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
    [-12,  -6*L_e,   12,   -6*L_e],
    [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
], dtype=tf.float64)

# Element Mass Matrix (Me) - Consistent Mass Matrix for a 2D beam element
# Includes terms related to Timoshenko beam theory (shear deformation and rotatory inertia)
# Corresponds to DOFs [v_i, theta_i, v_j, theta_j]
Me = (rho * A * L_e) * tf.constant([
    [13/35 + 6.*I/(5.*A*L_e**2), 11.*L_e/210.+I/(10.*A*L_e), 9/70 - 6*I/(5*A*L_e**2), -13*L_e/420+ I/(10*A*L_e)],
    [11.*L_e/210. + I/(10*A*L_e), L_e**2/105 + 2*I/(15*A), 13*L_e/420 - I/(10*A*L_e), -1.*L_e**2/140 - I/(30*A)],
    [9/70 - 6*I/(5*A*L_e**2), 13*L_e/420 - I/(10*A*L_e), 13/35 + 6*I/(5*A*L_e**2),  -11*L_e/210 - I/(10*A*L_e)],
    [-13*L_e/420 + I/(10*A*L_e),  -L_e**2/140 - I/(30*A), -11*L_e/210 - I/(10*A*L_e),  L_e**2/105 + 2*I/(15*A)]
], dtype=tf.float64)


#%%%%%%%%%%%%%%%%%%%%%%%#############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- 2. Assemble Global Matrices (Full - including rotations) ---
# Initialize global stiffness (K_global_full) and mass (M_global_full) matrices with zeros
# These matrices will have dimensions corresponding to all DOFs (vertical and rotational)
K_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)
M_global_full = tf.zeros((n_dof_full, n_dof_full), dtype=tf.float64)

# Flatten element matrices for efficient assembly using scatter update
Ke_flat = tf.reshape(Ke, [-1]) # Shape [16]
Me_flat = tf.reshape(Me, [-1]) # Shape [16]

# Loop through each element to add its contribution to the global matrices
print("\nAssembling full global K and M matrices (including rotations)...")
for e in range(n_elements):
    node_i = e        # Start node of the current element
    node_j = e + 1    # End node of the current element
    # Global DOF indices for the element [v_i, theta_i, v_j, theta_j]
    idx_map = tf.constant([2*node_i, 2*node_i + 1, 2*node_j, 2*node_j + 1], dtype=tf.int32)
    # Create pairs of global indices for the 4x4 element matrix contributions
    indices_pairs = tf.reshape(tf.stack(tf.meshgrid(idx_map, idx_map, indexing='ij'), axis=-1), [-1, 2])
    # Add element stiffness and mass to global matrices
    K_global_full = tf.tensor_scatter_nd_add(K_global_full, indices_pairs, Ke_flat)
    M_global_full = tf.tensor_scatter_nd_add(M_global_full, indices_pairs, Me_flat)


#%% ###################################################################################################################################
# --- 3. Extract Submatrices for Vertical DOFs Only ---
# This analysis focuses on vertical displacements. We extract the parts of K and M
# corresponding only to these vertical DOFs. This is a form of static condensation
# if rotational DOFs are assumed to have negligible inertial effects or are condensed out.
vertical_dof_indices_full_system = tf.range(0, n_dof_full, 2, dtype=tf.int32) # Indices for v_0, v_1, ...
K_unconstrained = tf.gather(tf.gather(K_global_full, vertical_dof_indices_full_system, axis=0), vertical_dof_indices_full_system, axis=1)
M_unconstrained = tf.gather(tf.gather(M_global_full, vertical_dof_indices_full_system, axis=0), vertical_dof_indices_full_system, axis=1)
print(f"Shape of K_unconstrained (vertical DOFs): {K_unconstrained.shape}")
print(f"Shape of M_unconstrained (vertical DOFs): {M_unconstrained.shape}")

# --- 4. Apply Boundary Conditions (e.g., Simply Supported) ---
# To get non-rigid body motion, boundary conditions must be applied.
# Here, we assume the beam is simply supported: vertical displacement at ends is zero.
# DOF 0 (at node 0) and DOF n_dof_vertical-1 (at node n_nodes-1) are constrained.
constrained_dofs = tf.constant([0, n_dof_vertical - 1], dtype=tf.int32)
all_dofs = tf.range(n_dof_vertical, dtype=tf.int32)

# Create a boolean mask to identify free (unconstrained) DOFs
mask_free_dofs = tf.ones(n_dof_vertical, dtype=tf.bool)
for dof in constrained_dofs: # Set mask to False for constrained DOFs
    mask_free_dofs = tf.tensor_scatter_nd_update(mask_free_dofs, [[dof]], [False])
free_dofs = tf.boolean_mask(all_dofs, mask_free_dofs) # Get indices of free DOFs

print(f"\nAll vertical DOFs indices: {all_dofs.numpy()}")
print(f"Constrained DOFs indices (BCs): {constrained_dofs.numpy()}")
print(f"Free DOFs indices: {free_dofs.numpy()}")

# Extract effective stiffness (K_eff) and mass (M_eff) matrices for the free DOFs
K_eff = tf.gather(tf.gather(K_unconstrained, free_dofs, axis=0), free_dofs, axis=1)
M_eff = tf.gather(tf.gather(M_unconstrained, free_dofs, axis=0), free_dofs, axis=1)
print(f"Shape of K_eff (after BCs): {K_eff.shape}")
print(f"Shape of M_eff (after BCs): {M_eff.shape}")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- 5. Solve Generalized Eigenvalue Problem on Effective (Constrained) System ---
# Solves K_eff * Phi_eff = M_eff * Phi_eff * Omega^2 to find natural frequencies (omegas) and mode shapes (Phi_eff)
# for the system with boundary conditions applied.
print("\nSolving eigenvalue problem for the constrained system...")
try:
    M_eff_inv = tf.linalg.inv(M_eff)
    A_eig_problem = M_eff_inv @ K_eff # Form A = M^-1 * K
    eigenvalues_sq_eff, eigenvectors_eff = tf.linalg.eigh(A_eig_problem) # Solve A*v = lambda*v
except Exception as e:
    print(f"Standard eigenvalue solution failed: {e}. Trying Cholesky decomposition approach.")
    L_chol = tf.linalg.cholesky(M_eff) # M_eff = L_chol * L_chol^T
    L_chol_inv = tf.linalg.inv(L_chol)
    # Transform to standard eigenvalue problem: A_cholesky * y = lambda * y
    A_cholesky = L_chol_inv @ K_eff @ tf.linalg.inv(tf.transpose(L_chol))
    eigenvalues_sq_eff, y_eig = tf.linalg.eigh(A_cholesky)
    eigenvectors_eff = tf.linalg.solve(tf.transpose(L_chol), y_eig) # Back-transform eigenvectors

# Post-process eigenvalues (frequencies)
# Ensure eigenvalues are non-negative before taking square root
omegas_sq_eff = tf.maximum(eigenvalues_sq_eff, 0.0) # omega^2 must be >= 0
omegas_eff = tf.sqrt(omegas_sq_eff)                 # Natural angular frequencies (rad/s)

print(f"Effective system omegas_sq (rad^2/s^2, first few): {omegas_sq_eff.numpy()[:min(n_m*2, len(omegas_sq_eff))]}")
print(f"Effective system omegas (rad/s, first few): {omegas_eff.numpy()[:min(n_m, len(omegas_eff))]}")
print(f"Effective system frequencies (Hz, first few): {(omegas_eff / (2 * np.pi)).numpy()[:min(n_m, len(omegas_eff))]}")

#%% ###########################################################################################################
# --- 6. Select Modes, Normalize, and Define Modal Damping ---
# Select the first n_m modes (lowest frequencies) for the analysis
Phi_eff = eigenvectors_eff[:, :n_m]          # Mode shapes for free DOFs
selected_omegas_sq = omegas_sq_eff[:n_m]   # Selected modal frequencies squared
selected_omegas = omegas_eff[:n_m]         # Selected modal frequencies

# Mass Normalization of Mode Shapes (Phi_eff)
# Scale mode shapes such that modal mass matrix becomes an identity matrix (m_r_i = 1)
M_r_diag_eff = tf.einsum('ji,jk,kl->il', Phi_eff, M_eff, Phi_eff) # Compute M_modal_eff = Phi_eff^T * M_eff * Phi_eff
norm_factors = 1.0 / tf.sqrt(tf.maximum(tf.linalg.diag_part(M_r_diag_eff), 1e-12)) # Normalization factors (avoid div by zero)
Phi_eff_normalized = Phi_eff * norm_factors # Normalized mode shapes for free DOFs

# Reconstruct full mode shapes (Phi) for all vertical DOFs (n_dof_vertical x n_m)
# Full mode shapes will have zeros at the constrained DOFs.
Phi = tf.Variable(tf.zeros((n_dof_vertical, n_m), dtype=tf.float64))
scatter_indices = tf.expand_dims(free_dofs, axis=1) # Indices where rows of Phi_eff_normalized will be placed
# tf.scatter_nd places each row of Phi_eff_normalized into Phi at the global DOF index specified by free_dofs
Phi = tf.scatter_nd(scatter_indices, Phi_eff_normalized, shape=(n_dof_vertical, n_m))

print(f"\nSelected natural frequencies (rad/s) after BCs and normalization: {selected_omegas.numpy()}")
if tf.reduce_any(selected_omegas <= 1e-6): # Check for issues
    print("WARNING: Near-zero frequencies detected among selected modes. This might indicate issues.")

# Modal stiffnesses (k_r_i = omega_i^2, since m_r_i = 1 after mass normalization)
k_q = selected_omegas_sq
# *** TYPE OF DAMPING CONSIDERED: MODAL DAMPING ***
# We assume modal damping, where a damping ratio (zeta_i) is assigned to each mode.
# This implies that the damping matrix C is such that it can be diagonalized by the
# mass-normalized mode shapes, resulting in uncoupled modal equations of motion:
#   1 * r_ddot_i + (2*zeta_i*omega_i) * r_dot_i + (omega_i^2) * r_i = f_r_i
# This is a common and practical approach when the exact form of the physical damping matrix C is unknown.

# Define modal damping ratios (zeta)
zeta_val = tf.constant(0.1, dtype=tf.float64)  # Assume 5% damping ratio for all selected modes
zeta_modal = tf.fill((n_m,), zeta_val)          # Create a tensor of damping ratios for each mode [zeta_1, ..., zeta_n_m]

# Calculate modal damping coefficients (c_r_i = 2 * zeta_i * omega_i * m_r_i)
# Since modes are mass-normalized, m_r_i = 1. So, c_r_i = 2 * zeta_i * omega_i
c_q = 2.0 * zeta_modal * selected_omegas

# Reshape modal properties into column vectors for Newmark integration calculations
k_col = tf.reshape(k_q, (n_m, 1))      # Modal stiffnesses [n_m, 1]
c_col = tf.reshape(c_q, (n_m, 1))      # Modal damping coefficients [n_m, 1]
m_col = tf.ones_like(k_col)          # Modal masses (all 1s due to normalization) [n_m, 1]

#%%#######################################################################################
# --- 7. Define Load and Time Parameters ---
# External Force Definition
force_amplitude = tf.constant(5.0, dtype=tf.float64)         # Amplitude of the sinusoidal force
forcing_frequency_hz = tf.constant(500, dtype=tf.float64)    # Frequency of the force in Hz. Be careful with Nyquist frequency for the exitation  freq. decision
forcing_frequency_rad = 2.0 * np.pi * forcing_frequency_hz # Convert frequency to rad/s
Omega_force = forcing_frequency_rad                             # Angular frequency of the force

# Time Integration Parameters
t_max_sim = tf.constant(10.0, dtype=tf.float64) # Total simulation time
dt = tf.constant(0.001, dtype=tf.float64)      # Time step for integration (chosen small enough for 100Hz forcing & system modes)
load_start_time = tf.constant(0.1, dtype=tf.float64) # Time when the load starts
load_end_time = tf.constant(7.5, dtype=tf.float64)   # Time when the load ends

# Calculate time vector
n_steps = tf.cast(tf.round(t_max_sim / dt), dtype=tf.int32) # Number of time steps
start_time_sim = tf.constant(0.0, dtype=tf.float64)
num_points_sim = n_steps + 1  # Total number of time points (including t=0)
t_vector = tf.linspace(start_time_sim, t_max_sim, num_points_sim) # Time vector
actual_dt = t_vector[1] - t_vector[0]                  # Actual time step used by linspace
print(f"\nTime vector from {t_vector[0]:.3f}s to {t_vector[-1]:.3f}s with {num_points_sim} points, actual_dt = {actual_dt:.6f}s.")

# Define Load Application Point (spatial distribution of the force)
center_dof_index = n_dof_vertical // 2 # Example: force at the vertical DOF of the center node
# Ensure load is not applied to a constrained DOF
if center_dof_index in constrained_dofs.numpy():
    print(f"WARNING: Load application at center_dof_index {center_dof_index} which is constrained! Adjusting.")
    # Basic adjustment: try next or previous if possible (simple example)
    if center_dof_index + 1 < n_dof_vertical and (center_dof_index + 1) not in constrained_dofs.numpy():
        center_dof_index = center_dof_index + 1
    elif center_dof_index - 1 >= 0 and (center_dof_index - 1) not in constrained_dofs.numpy():
        center_dof_index = center_dof_index - 1
    else: # Fallback if no simple adjustment works
        # Find first available free DOF if center adjustment fails (more robust)
        available_free_dofs = [dof for dof in free_dofs.numpy() if 0 < dof < n_dof_vertical-1] # prefer internal
        if available_free_dofs: center_dof_index = available_free_dofs[len(available_free_dofs)//2]
        else: center_dof_index = free_dofs.numpy()[0] # any free dof as last resort
        print(f"Load application point shifted to DOF index: {center_dof_index}")
print(f"Load applied at vertical DOF index: {center_dof_index}")

# Create load vector p (defines which DOF the force magnitude F(t) applies to)
load_vector_p_physical = np.zeros(n_dof_vertical)
load_vector_p_physical[center_dof_index] = 1.0 # Unit force at the specified DOF
p_force_location = tf.constant(load_vector_p_physical, shape=(n_dof_vertical, 1), dtype=tf.float64)

# Define Force Magnitude F(t) over Time
F_t_sinusoidal = force_amplitude * tf.sin(Omega_force * t_vector) # Sinusoidal part
# Apply a time window for the load
mask_load_active = tf.logical_and(t_vector >= load_start_time, t_vector <= load_end_time)
time_mask_for_load = tf.cast(mask_load_active, dtype=tf.float64)
F_t_magnitude = F_t_sinusoidal * time_mask_for_load # Force magnitude F(t)

# Calculate Full Nodal Force Vector Q(t) in physical coordinates
Q_t_physical = p_force_location @ tf.expand_dims(F_t_magnitude, axis=0) # Shape [n_dof_vertical, num_points_sim]
# Transform to Modal Force Vector Q_r(t)
Q_r_t_modal = tf.transpose(Phi) @ Q_t_physical # Shape [n_m, num_points_sim]
#%%%%###################################################################################################################### 
#8.  calling Newmark Beta function: 
Qpred = tf.transpose(Q_r_t_modal, perm = [1,0]) # just to accommodate the prediction provided by the NN with dimension (num_points_sim, n_m) 
uddot_pred = Newmark_beta_solver_singleb(Qpred, Phi, m_col, c_col, k_col, t_vector, n_steps)
uddot_t = uddot_pred

# YOu need to vectorize this function in order to accommodate batch dimension (even if it is 1) Otherwise it is doing a wrong transpose
 
#%%%%###################################################################################################################### 
# --- 9. Define Sensor Locations and Extract Responses ---
print("\n--- Defining Sensor Locations and Extracting Acceleration Responses ---")
# Define sensor locations as fractions of the total beam length
sensor_loc_fractions = tf.constant([1/6, 1/2, 2/3], dtype=tf.float64)
sensor_actual_locations_m = sensor_loc_fractions * L_total # Absolute positions in meters

# Determine the closest node indices for each sensor location
sensor_node_indices_float = sensor_actual_locations_m / L_e
sensor_node_indices = tf.cast(tf.round(sensor_node_indices_float), dtype=tf.int32)
# Ensure indices are within valid bounds [0, n_dof_vertical-1]
sensor_node_indices = tf.clip_by_value(sensor_node_indices, 0, n_dof_vertical - 1)
# Extract acceleration time series for the specified sensor locations
# uddot_pred has shape [n_dof_vertical, num_points_sim]
sensor_accelerations_t = tf.gather(uddot_pred, sensor_node_indices, axis=0) # Shape [num_sensors, num_points_sim]
#%%###############################################################################################################################################
# --- 10. Plot Results for Sensor Locations ---
if tf.reduce_any(tf.math.is_nan(sensor_accelerations_t)):
    print("\nERROR: NaN values detected in final sensor accelerations. Plotting may fail or be meaningless.")
else:
    print("\n--- Plotting Sensor Acceleration Responses ---")

plt.figure(figsize=(14, 8))
for i in range(sensor_node_indices.shape[0]): # Iterate through each sensor
    node_idx_val = sensor_node_indices[i].numpy()
    loc_frac_val = sensor_loc_fractions[i].numpy()
    actual_loc_m_val = sensor_actual_locations_m[i].numpy()
    
    # Create a descriptive label for the plot legend
    if abs(loc_frac_val - 1/6) < 1e-6: label_frac_str = "L/6"
    elif abs(loc_frac_val - 1/2) < 1e-6: label_frac_str = "L/2"
    elif abs(loc_frac_val - 2/3) < 1e-6: label_frac_str = "2L/3"
    else: label_frac_str = f"{loc_frac_val:.2f}L" # Fallback for other fractions

    plt.plot(t_vector.numpy(), sensor_accelerations_t[i, :].numpy(),
             label=f'Sensor at {label_frac_str} ({actual_loc_m_val:.2f}m, Node {node_idx_val})')

# Add plot titles and labels
plt.title(f'Beam Acceleration Response (Simply Supported, Modal Damping $\zeta={zeta_val.numpy()}$, $dt={actual_dt.numpy():.1e}$s)')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Acceleration ($m/s^2$)')
plt.legend()
plt.grid(True)
# Add vertical lines to indicate load start and end times for clarity
plt.axvline(x=load_start_time.numpy(), color='gray', linestyle='--', linewidth=1.8, label=f'Load Start/End') # Label once
plt.axvline(x=load_end_time.numpy(), color='gray', linestyle='--', linewidth=1.8)
# Ensure unique legend entries if multiple axvlines have labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # Remove duplicate labels for axvlines
plt.legend(by_label.values(), by_label.keys(), fontsize = 18)
plt.show()


