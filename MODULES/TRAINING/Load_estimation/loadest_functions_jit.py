#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 13:12:22 2025

@author: afernandez
"""
import tensorflow as tf
import numpy as np

# Small epsilon for numerical stability
TF_EPSILON = tf.constant(1e-12, dtype=tf.float64)

# _batch_NewmarkBeta_core function remains the same as in the previous good attempt
def _batch_NewmarkBeta_core(batch_Qpred_single, batch_t_vector_single,
                             Phi_matrix, m_diag_col, c_diag_col, k_diag_col,
                             n_modes_const, # Static number of modes
                             beta_nm_const, gamma_nm_const):
    actual_dt = batch_t_vector_single[1] - batch_t_vector_single[0]
    dt_val_newmark = tf.cast(actual_dt, dtype=tf.float64)

    q_initial = tf.zeros((n_modes_const, 1), dtype=tf.float64)
    qdot_initial = tf.zeros((n_modes_const, 1), dtype=tf.float64)
    Q_at_t0 = tf.transpose(batch_Qpred_single[0:1, :])
    qddot_initial = (Q_at_t0 - c_diag_col * qdot_initial - k_diag_col * q_initial) / (m_diag_col + TF_EPSILON)

    LHS_factor_modal = m_diag_col + \
                       gamma_nm_const * dt_val_newmark * c_diag_col + \
                       beta_nm_const * dt_val_newmark**2 * k_diag_col

    def newmark_scan_step(previous_state_tuple, Q_force_for_current_target_step):
        q_current, qdot_current, qddot_current = previous_state_tuple
        q_predictor = q_current + dt_val_newmark * qdot_current + \
                      (0.5 - beta_nm_const) * dt_val_newmark**2 * qddot_current
        qdot_predictor = qdot_current + (1.0 - gamma_nm_const) * dt_val_newmark * qddot_current
        RHS_force_modal = Q_force_for_current_target_step - \
                          c_diag_col * qdot_predictor - k_diag_col * q_predictor
        qddot_next = RHS_force_modal / (LHS_factor_modal + TF_EPSILON)
        qdot_next = qdot_predictor + gamma_nm_const * dt_val_newmark * qddot_next
        q_next = q_predictor + beta_nm_const * dt_val_newmark**2 * qddot_next
        return (q_next, qdot_next, qddot_next)

    initial_scan_state = (q_initial, qdot_initial, qddot_initial)
    elems_for_scan = tf.expand_dims(batch_Qpred_single[1:, :], axis=-1)

    scan_results_tuple = tf.scan(
        fn=newmark_scan_step,
        elems=elems_for_scan,
        initializer=initial_scan_state,
        name="newmark_beta_scan_core"
    )

    qddot_scan_output = scan_results_tuple[2]
    qddot_history_full = tf.concat([tf.expand_dims(qddot_initial, axis=0), qddot_scan_output], axis=0)
    qddot_history_final = tf.transpose(tf.squeeze(qddot_history_full, axis=-1), perm=[1, 0])
    uddot_single_batch = tf.matmul(Phi_matrix, qddot_history_final)
    return uddot_single_batch


@tf.function(jit_compile=True)
def Newmark_beta_solver(Qpred, t_vector, Phi_mat, m_mat_col, c_mat_col, k_mat_col, n_s_unused,
                        beta_newmark=tf.constant(0.25, dtype=tf.float64),
                        gamma_newmark=tf.constant(0.50, dtype=tf.float64)):
    n_modes = tf.shape(Phi_mat)[1]
    batch_size = tf.shape(Qpred)[0]
    
    num_points_sim_runtime = tf.shape(Qpred)[1] # symbolic tensor, int32
    n_dof = tf.shape(Phi_mat)[0]                # symbolic tensor, int32

    # --- CORRECTED PART ---
    # Construct element_shape as a 1-D int32 Tensor from symbolic dimensions
    current_element_shape = tf.stack([n_dof, num_points_sim_runtime], name="element_shape_tensor")
    # --- END CORRECTED PART ---

    output_ta = tf.TensorArray(
        dtype=tf.float64,
        size=batch_size,                             # batch_size is a symbolic tensor, this is fine for 'size'
        element_shape=current_element_shape,         # Use the 1-D tensor here
        clear_after_read=False,
        name="output_uddots_tensor_array"
    )

    i = tf.constant(0)

    def condition(i_loop, _):
        return i_loop < batch_size

    def body(i_loop, current_ta):
        q_pred_item = Qpred[i_loop]
        t_vector_item = t_vector[i_loop]

        uddot_item_result = _batch_NewmarkBeta_core(
            q_pred_item, t_vector_item,
            Phi_mat, m_mat_col, c_mat_col, k_mat_col,
            n_modes,
            beta_newmark, gamma_newmark
        )
        updated_ta = current_ta.write(i_loop, uddot_item_result)
        return i_loop + 1, updated_ta

    _, final_ta = tf.while_loop(
        condition,
        body,
        loop_vars=[i, output_ta],
        parallel_iterations=1
    )

    uddots_batched = final_ta.stack(name="stack_batched_uddots")
    uddots_final = tf.transpose(uddots_batched, perm=[0, 2, 1])
    return uddots_final


class Newmark_beta_solver_layer(tf.keras.layers.Layer):
    def __init__(self, num_points_sim, n_modes, n_steps, Phi, m_col, c_col, k_col, **kwargs):
        super(Newmark_beta_solver_layer, self).__init__(**kwargs)
        self.num_points_sim_init = num_points_sim
        self.n_modes_init = n_modes
        self.n_steps_init = n_steps

        self.Phi_const = tf.constant(Phi, dtype=tf.float64, name="Phi_const")
        self.m_col_const = tf.constant(m_col, dtype=tf.float64, name="m_col_const")
        self.c_col_const = tf.constant(c_col, dtype=tf.float64, name="c_col_const")
        self.k_col_const = tf.constant(k_col, dtype=tf.float64, name="k_col_const")
        
        self.beta_newmark_const = tf.constant(0.25, dtype=tf.float64, name="beta_newmark")
        self.gamma_newmark_const = tf.constant(0.50, dtype=tf.float64, name="gamma_newmark")

    def call(self, inputs):
        Qpred, t_vector = inputs
        uddot_pred = Newmark_beta_solver(Qpred, t_vector,
                                         self.Phi_const,
                                         self.m_col_const, self.c_col_const, self.k_col_const,
                                         self.n_steps_init, 
                                         beta_newmark=self.beta_newmark_const,
                                         gamma_newmark=self.gamma_newmark_const)
        return uddot_pred

    def get_config(self):
        config = super(Newmark_beta_solver_layer, self).get_config()
        config.update({
            'num_points_sim': self.num_points_sim_init,
            'n_modes': self.n_modes_init,
            'n_steps': self.n_steps_init,
            'Phi': self.Phi_const.numpy().tolist(),
            'm_col': self.m_col_const.numpy().tolist(),
            'c_col': self.c_col_const.numpy().tolist(),
            'k_col': self.k_col_const.numpy().tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        phi_np = np.array(config.pop('Phi'), dtype=np.float64)
        m_col_np = np.array(config.pop('m_col'), dtype=np.float64)
        c_col_np = np.array(config.pop('c_col'), dtype=np.float64)
        k_col_np = np.array(config.pop('k_col'), dtype=np.float64)
        
        return cls(Phi=phi_np, m_col=m_col_np, c_col=c_col_np, k_col=k_col_np, **config)