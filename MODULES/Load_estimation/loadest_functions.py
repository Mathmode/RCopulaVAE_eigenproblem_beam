#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:21:16 2025

@author: afernandez
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
plot_configuration()


#%%#######################################################################################

# --- JIT-COMPILE TRUE VERSION OF Newmark Beta Solver with XLA-compatible NaN check ---
def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps,
                        beta_newmark=tf.constant(0.25, dtype=tf.float64),
                        gamma_newmark=tf.constant(0.50, dtype=tf.float64)):
    
    ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
    #     The inputs are: 
    #         Qpred: tensor of predicted modal forces. Shape (batch_size=1, num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
    #         t_vector: time vector according to step Deltat and final time tf. Shape (batch_size =1, n_steps+ 1 = num_points_sim)
    #         Phi: Truncated mode shape matrix. Shape (n_dofs, n_m) . It is a rectangular matrix after truncation. 
    #         m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
    #         beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
        
    #     The outputs are: 
    #         [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
    #         The three tensors in the tuple have shape (batch_size, n_dof, num_points_sim)
    #         '''

    @tf.function(jit_compile=True)
    def _internal_jitted_core_solver(batch_Qpred_single, batch_t_vector_single):
        actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
        dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)
        n_m = tf.shape(batch_Qpred_single)[1]

        q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
        qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
        Q_at_t0 = tf.transpose(batch_Qpred_single[0:1, :])
        qddot_initial = (Q_at_t0 - c_col * qdot_initial - k_col * q_initial) / m_col

        LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + \
                            beta_newmark * dt_val_newmark**2 * k_col

        def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step_scan):
            q_current, qdot_current, qddot_current = previous_state_tuple
            Q_modal_ti1 = Q_force_for_current_target_step_scan
            q_predictor = q_current + dt_val_newmark * qdot_current + \
                            (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
            qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current
            RHS_force_modal = Q_modal_ti1 - (c_col * qdot_predictor + k_col * q_predictor)
            qddot_next = RHS_force_modal / LHS_factor_modal
            qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
            q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next
            return (q_next, qdot_next, qddot_next)

        initial_scan_state = (q_initial, qdot_initial, qddot_initial)
        elems_for_scan = tf.expand_dims(batch_Qpred_single[1:, :], axis=-1)

        scan_results_tuple = tf.scan(
            fn=newmark_scan_step,
            elems=elems_for_scan,
            initializer=initial_scan_state,
            name="newmark_beta_scan_loop_jit_internal"
        )

        qddot_scan_output = scan_results_tuple[2]
        qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)

        # --- MODIFICATION START ---
        # Replace tf.cond with tf.print by tf.debugging.check_numerics for XLA compatibility
        qddot_history_full = tf.debugging.check_numerics(
            qddot_history_full,
            "Error: NaN/Inf detected in qddot_history_full inside _internal_jitted_core_solver."
        )
        # --- MODIFICATION END ---

        qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1, 0])
        uddot_physical = tf.matmul(Phi, qddot_history_final)
        return uddot_physical

    uddots = tf.vectorized_map(
        lambda packed_slices: _internal_jitted_core_solver(packed_slices[0], packed_slices[1]),
        elems=(Qpred, t_vector)
    )

    uddots_final_perm = tf.transpose(uddots, perm=[0, 2, 1])
    return uddots_final_perm



## old without jit-friendly form 
# def Newmark_beta_solver(Qpred, t_vector, Phi, m_col, c_col, k_col, n_steps, beta_newmark = tf.constant(0.25, dtype=tf.float64), gamma_newmark = tf.constant(0.50, dtype=tf.float64)):
#     ''' This function solves the system of n_m uncoupled ODEs to produce the response in physical coodrinates for the time vector. 
#     The inputs are: 
#         Qpred: tensor of predicted modal forces. Shape (batch_size=1, num_points_sim, n_m), where num_points_sim = n_steps +1 = time_vector.shape[0]
#         t_vector: time vector according to step Deltat and final time tf. Shape (batch_size =1, n_steps+ 1 = num_points_sim)
#         Phi: Truncated mode shape matrix. Shape (batch_size, n_dofs, n_m) . It is a rectangular matrix after truncation. 
#         m_col, c_col, k_col: entries of the diagonal matrices of mass, damping and stiffness as column vectors. Shape (n_m, 1). 
#         beta_newmark, gamma_newmark: parameters for the newmark method. Keep them like this unless indicated by experts. 
    
#     The outputs are: 
#         [u, udot, uddot]: response of the system in modal coordinates. Particularized only for the sensor locations. 
#         The three tensors in the tuple have shape (batch_size, n_dof, num_points_sim)
#         '''
#     def batch_NewmarkBeta(batch_Qpred, batch_t_vector):
#         actual_dt = batch_t_vector[1] - batch_t_vector[0]                  # Actual time step used by linspace
#         dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)   
#         n_m = Phi.shape[1]    
#         # Initial conditions
#         q_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         qdot_initial = tf.zeros((n_m, 1), dtype=tf.float64)
#         Q_at_t0 = tf.transpose(batch_Qpred[0:1, :]) # since Qpred has shape (num_points_sim, n_m)
#         qddot_initial = Q_at_t0 - c_col * qdot_initial - k_col * q_initial
        
#         # LHS Factor
#         LHS_factor_modal = m_col + gamma_newmark * dt_val_newmark * c_col + beta_newmark * dt_val_newmark**2 * k_col

#         # @tf.function(jit_compile=True)
#         def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step):
#             # previous_state_tuple: (r_prev, rdot_prev, rddot_prev) from time t_i
#             # Q_r_force_for_current_target_step: Modal force Q_r(t_{i+1}), shape [n_m, 1]

#             q_current, qdot_current, qddot_current = previous_state_tuple

#             # Predictor terms based on state at time t_i
#             q_predictor = q_current + dt_val_newmark * qdot_current + \
#                           (0.5 - beta_newmark) * dt_val_newmark**2 * qddot_current
#             qdot_predictor = qdot_current + (1.0 - gamma_newmark) * dt_val_newmark * qddot_current

#             # Calculate effective modal force for solving acceleration at t_{i+1}
#             RHS_force_modal = Q_force_for_current_target_step - \
#                               c_col * qdot_predictor - k_col * q_predictor

#             # Solve for modal acc^eleration at time t_{i+1}
#             qddot_next = RHS_force_modal / LHS_factor_modal

#             # Correct modal velocity and displacement at time t_{i+1}
#             qdot_next = qdot_predictor + gamma_newmark * dt_val_newmark * qddot_next
#             q_next = q_predictor + beta_newmark * dt_val_newmark**2 * qddot_next

#             # The function returns the new state (r_next, rdot_next, rddot_next)
#             # This tuple will be passed as 'previous_state_tuple' to the next iteration and also collected by tf.scan as the output for this step.
#             return (q_next, qdot_next, qddot_next)

#         # Initial state for the scan (state at t=0)
#         initial_scan_state = (q_initial, qdot_initial, qddot_initial)

#         # Prepare the sequence of modal forces for tf.scan. (Remove the initial time t= 0 since that is the initial condition. Then we will add it). 
#         elems_for_scan = tf.expand_dims(batch_Qpred[1:, :], axis = -1) 
#         # It seems that the expanded additional dimension is required for the structure management of tf.scan. 

#         # The output will be a tuple of three tensors:  (stacked_q_next, stacked_qdot_next, stacked_qddot_next)
#         # Each tensor will have shape [n_steps, n_m, 1], representing states from t_1 to t_{n_steps}
#         scan_results_tuple = tf.scan(
#             fn=newmark_scan_step,
#             elems=elems_for_scan,
#             initializer=initial_scan_state,
#             name="newmark_beta_solver"
#         )

#         # Unpack results from scan. These are histories from t_1 to t_{n_steps} (n_steps items)
#         # q_scan_output = scan_results_tuple[0]    # Shape [n_steps, n_m, 1] # We do not use it in the loss
#         # qdot_scan_output = scan_results_tuple[1] # Shape [n_steps, n_m, 1] # We do not use it in the loss
#         qddot_scan_output = scan_results_tuple[2]# Shape [n_steps, n_m, 1]
#         qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)

#         # Reshape results to [n_m, num_points_sim] for easier use (same as original code)
#         # Current shape is [num_points_sim, n_m, 1]
#         qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1,0])
#         # Transform modal accelerations back to physical accelerations
#         uddot = tf.matmul(Phi, qddot_history_final) # Shape [n_dof_vertical, num_points_sim]
#         return uddot
    
#     uddots = tf.vectorized_map(
#         lambda args: batch_NewmarkBeta(*args),
#         (Qpred, t_vector),)
    
#     uddots = tf.transpose(uddots, perm = [0,2,1])
    
#     return uddots



# # no hace falta la layer

# # Define a layer that applies the function for newmark calculation 
# class Newmark_beta_solver_layer(tf.keras.layers.Layer):
#     def __init__(self, num_points_sim, n_modes, n_steps, Phi, m_col, c_col, k_col, **kwargs):
#         super(Newmark_beta_solver_layer, self).__init__(**kwargs)
#         self.num_points_sim = num_points_sim
#         self.n_modes = n_modes
#         self.n_steps = n_steps
#         self.Phi = Phi
#         self.m_col = m_col
#         self.c_col = c_col 
#         self.k_col = k_col
        
        
#     def call(self,inputs): 
#         Qpred, t_vector = inputs         
#         uddot_pred = Newmark_beta_solver(Qpred, t_vector, self.Phi, self.m_col, self.c_col, self.k_col, self.n_steps)
        
#         return uddot_pred
        



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


    