#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:56:10 2025

@author: afernandez
"""

import os
import tensorflow as tf
import tensorflow.keras as K 
import numpy as np 
from MODULES.Load_estimation.loadest_architectures import  Fully_connected_arch, Fully_connected_arch_singlenode
from MODULES.Load_estimation.loadest_functions import Newmark_beta_solver

# --- 0. TensorFlow Setup ---
# Ensure TensorFlow uses float64 for better numerical stability in FEM calculations
tf.keras.backend.set_floatx('float64')
K.utils.set_random_seed(1234)


class Modal_force_estimator(K.Model):
    def __init__(self, num_points_sim, n_modes, n_steps, n_dof, Phi, m_col, c_col, k_col, **kwargs):
        super(Modal_force_estimator, self).__init__()
        self.num_points_sim = num_points_sim
        self.n_modes = n_modes
        self.n_steps = n_steps
        self.n_dof = n_dof

        self.Phi = Phi # Fixed truncated modal matrix for n_modes mode shapes in the columns. 
        self.m_col = m_col # Fixed diagonal modal mass matrix with dims (n_modes, 1). Same for damping and stiffness.
        self.c_col = c_col
        self.k_col = k_col
        self.fullyconnected_loadestimator2 = Fully_connected_arch_singlenode(num_points_sim, n_modes)
        # self.fullyconnected_loadestimator = Fully_connected_arch(num_points_sim, n_modes)

        # self.LSTM_loadestimator = LSTM_Qpred_Estimator(num_points_sim, n_modes)
        
        

    def call(self, inputs):
        self.t_vector, self.u_true = inputs 
        
        nodeF = self.fullyconnected_loadestimator2(self.t_vector) #shape [batch_size, num_points_sim] as only for one node we estimate the modal force. 
        F_at_node_load_reshaped = tf.expand_dims(nodeF, axis=-1)
        #Masking vector with zero in all locations except at node = node_load, where we assume the laod is unknonwn. 
        one_hot_vector = tf.one_hot(
        indices=5, #specifies the position with the laod. 
        depth=self.n_dof,
        dtype=F_at_node_load_reshaped.dtype # Match the dtype of the force predictions
        )
        one_hot_for_broadcast = tf.reshape(one_hot_vector, (1, 1, self.n_dof))
        Fpred = F_at_node_load_reshaped * one_hot_for_broadcast # shape (batch_size, time_points, n_dofs)
        # Q = Phi_transp * F(t) 
        Phi_transp = tf.transpose(self.Phi) #shape = (n_modes, n_dof)
        ## for the einsum: m= n_modes, d = n_dof, b  =batch_size, t = num_points_sim 
        self.Qpred = tf.einsum('md,btd->btm', Phi_transp, Fpred)
        
        # # # when estimatng all the modal forces (one time-domain vector for each mode)
        # fullyQ = self.fullyconnected_loadestimator(self.t_vector)
        # self.Qpred = tf.reshape(fullyQ, [-1, self.num_points_sim, self.n_modes])

        uddot_pred_full = Newmark_beta_solver(self.Qpred,self.t_vector,self.Phi, self.m_col, self.c_col, self.k_col, self.n_steps)
        
        sensor_locs = [3,5,7] #indicates the node id where you can compare the response as there was a sensor measuring.  It is not a distance in meters but a ID for the DOF corresponding to the sensor emplacement. 
        self.uddot_true = tf.gather(self.u_true, indices = sensor_locs, axis=2)
        uddot_pred = tf.gather(uddot_pred_full, indices = sensor_locs, axis = 2) #should be now shape (batch_size, num_sim_points, num_sensors) 

        return {"acceleration_output": uddot_pred, "modal_force_output": self.Qpred}
    
    def udata_loss(self, y_true, y_pred_acc):

        response_squared_error = tf.square(self.uddot_true - y_pred_acc)
        Loss_data = tf.math.reduce_mean(response_squared_error, axis = None)
        return Loss_data
    
    # get_config and from_config methods (ensure they are complete and handle all necessary attributes)
    def get_config(self):
        config = {
            'num_points_sim': self.num_points_sim,
            'n_modes': self.n_modes,
            'n_steps': self.n_steps,
            'n_dof': self.n_dof,
            'Phi': self.Phi.numpy().tolist() if tf.is_tensor(self.Phi) else self.Phi,
            'm_col': self.m_col.numpy().tolist() if tf.is_tensor(self.m_col) else self.m_col,
            'c_col': self.c_col.numpy().tolist() if tf.is_tensor(self.c_col) else self.c_col,
            'k_col': self.k_col.numpy().tolist() if tf.is_tensor(self.k_col) else self.k_col,
            'node_load_idx': self.node_load_idx 
        }
        base_config = super(Modal_force_estimator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        for key in ['Phi', 'm_col', 'c_col', 'k_col']:
            if key in config and isinstance(config[key], list):
                config[key] = tf.convert_to_tensor(config[key], dtype=tf.float32)
        # Ensure all __init__ args are present in config or handled
        return cls(**config)
  


# # --- Example Usage ---
# system_info_path = os.path.join("MODULES", "Load_estimation", "System_info.npy")
# system_info = np.load(system_info_path, allow_pickle = True).item()
# n_modes, Phi, m_col, c_col, k_col, uddot_true, t_vector, F_true = system_info['n_modes'], system_info['Phi'], system_info['m_col'], system_info['c_col'], system_info['k_col'], system_info['uddot_true'], system_info['t_vector'], system_info['F_true']



# # num_points_sim_example = F_true.shape[1] # Example value from your Newmark_beta_solver context
# num_points_sim_example = 800
# # 1. Prepare your t_vector:
# # It should be a 1D NumPy array or TensorFlow tensor of shape (num_points_sim_example,)
# # 2. If you have a single t_vector for prediction, add a batch dimension:
# # pos0 = 850
# t_vector = tf.expand_dims(t_vector[0:num_points_sim_example], axis=0) # Shape: (1, num_points_sim_example)
# uddot_true = tf.expand_dims(tf.transpose(uddot_true[:, 0:num_points_sim_example]), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.

# # t_vector = tf.expand_dims(t_vector, axis=0) # Shape: (1, num_points_sim_example)
# # uddot_true = tf.expand_dims(tf.transpose(uddot_true), axis = 0) # This should be loaded from wherever we saved the information of the specific problem, together with Phi, and the modal diagonal matrices of mass, damping and stiffness.
# n_dofs  =  uddot_true.shape[2]
# # we expand dimensions to add the batch size = 1 

# LR = 0.001
# batch_size = 1
# n_epochs = 10000
# n_steps   = num_points_sim_example -1
# model = Modal_force_estimator(num_points_sim_example, n_modes, n_steps, Phi, m_col, c_col, k_col)
# model.compile(optimizer = K.optimizers.Adam(learning_rate = LR), loss = model.udata_loss)


# model_history = model.fit(x = [t_vector, uddot_true],
#   y = uddot_true,
#   batch_size = batch_size,
#   epochs = n_epochs,
#   shuffle = True,
#   validation_data = ([t_vector, uddot_true], uddot_true),
#   callbacks = [])



# from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration
# plot_configuration()
# folder_path = os.path.join("MODULES","Load_estimation")

# loss_histories = model_history.history
# from matplotlib import pyplot as plt
# plt.plot(loss_histories['loss'], 'red')
# plt.yscale('log')
# # plt.xscale('log')

# uddot_pred  = model.predict([t_vector, uddot_true])

# # for i in range(n_dofs):
# #     plt.figure()
# #     plt.plot(uddot_true[0,:,i], 'blue')
# #     plt.plot(uddot_pred[0,:,i], 'red')
# #     plt.xlabel('time frame [s]')
# #     plt.ylabel('Acc [m/s2]')
# #     plt.legend(['True', 'Pred'])
    
    

# i =3
# Dt = 0.001 # Your time step

# # 1. Apply a style for overall aesthetics (optional, kept as in your original)
# # plt.style.use('seaborn-v0_8-whitegrid')

# # 2. Create the figure and axes for a single plot
# fig, ax = plt.subplots(figsize=(12, 7))

# # --- Create the time vector ---
# # Determine the number of time points from your data's shape
# # Assuming the second dimension of your data arrays is time points
# num_time_points = uddot_true.shape[1]
# # Create the time vector: t = [0, Dt, 2*Dt, ..., (N-1)*Dt]
# time_vector = np.arange(num_time_points) * Dt
# # --- End of time vector creation ---

# # 3. Plot the data for the specified 'i' using the time_vector for x-axis
# # True data: solid line
# ax.plot(time_vector, uddot_true[0, :, i], color='springgreen', alpha=0.5, linestyle='-', linewidth=5.5, label=f'True')

# # Prediction data: dashed line
# ax.plot(time_vector, uddot_pred[0, :, i], color='red', linestyle='--', linewidth=2.5, label=f'Predicted')

# # 4. Improve labels and title
# ax.set_xlabel('Time (s)') # Updated x-axis label
# ax.set_ylabel('ü(t) [m/s²] at L/3') # y-label remains appropriate

# # 5. Customize the legend
# ax.legend(frameon=True, loc='best', shadow=True)

# # 6. Add a grid
# ax.grid(True, linestyle=':', alpha=0.7)

# # 7. Adjust tick parameters
# ax.tick_params(axis='both', which='major')
# plt.tight_layout()
# plt.savefig(os.path.join(folder_path, '30Hz_result_.png'),dpi = 500, bbox_inches='tight')
# plt.show()






# Q_pred = model.predict([t_vector, uddot_true])
# Phi_T = tf.transpose(Phi)
# pseudoinv_Phi = tf.linalg.pinv(Phi_T)
# F_pred = tf.einsum('dm,btd->btd', pseudoinv_Phi, Q_pred)

# i = 5
# Fp   = F_pred[0,0:num_points_sim_example,i]
# Ft = F_true[i,0:num_points_sim_example]
# fig, ax = plt.subplots(figsize=(12, 7))
# ax.plot(time_vector, Ft, color='orange', alpha=0.5, linestyle='-', linewidth=5.5, label=f'True')
# ax.plot(time_vector, Fp, color='blue', linestyle='--', linewidth=2.5, label=f'Predicted')

# # 4. Improve labels and title
# ax.set_xlabel('Time point')
# ax.set_ylabel(f' f(t) at node {i}')
# # 5. Customize the legend
# ax.legend(frameon=True, loc='best', shadow=True)
# # 6. Add a grid
# ax.grid(True, linestyle=':', alpha=0.7)
# # 7. Adjust tick parameters
# ax.tick_params(axis='both', which='major')

# plt.tight_layout()
# plt.savefig(os.path.join(folder_path, f'load_comparison_node{i}.png'),dpi = 500, bbox_inches='tight')
# plt.show()



