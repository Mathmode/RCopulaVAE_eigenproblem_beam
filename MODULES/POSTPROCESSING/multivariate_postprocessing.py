#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:08:58 2024

@author: afernandez
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import keras as K 
import tensorflow as tf
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.TRAINING.multivariate_arch import Solve_eigenproblem, assemble_global_Kmatrices
from MODULES.TRAINING.multivariate_models import Inverse_Bayesian_GMM_Model, My_Bayesian_InverseForward_withEigen
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration, plot_loss_evolution, plot_loss_terms, plot_alpha_crossplots, show_predicted_factors
# from MODULES.POSTPROCESSING.results_analysis_tools import load_inverseonly_model, calculate_MAC, Loss_function, produce_pred_modalprops

plot_configuration()

foldername = 'PRUEBA27Nov_5els_Bayesian_0.05Beta_0.05regul_InverseForward_1.0ub_5dims_6gaussians_1Samples_1e-05LR_20000epochs_128_batchs'
folder_path = os.path.join('Output',foldername)

################ plotting loss functions to compare with Diagonal covariance:
history_path = os.path.join(folder_path, 'model_history.npy')
model_history  = np.load(history_path, allow_pickle = True)
losses = model_history.item()
names = [ 'loss', 'Freqs_loss', 'MAC_modes_loss', 'Alpha_regularizer', 'Mixture_dens_term','val_loss', 'val_Freqs_loss', 'val_MAC_modes_loss', 'val_Alpha_regularizer', 'val_Mixture_dens_term' ]

        
def compare_GMM_losses(losses, folder_path):
    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(losses[names[0]], label=r'$\mathcal{L}_{total}$', color='blue', linestyle='-', linewidth=3.)
    plt.plot(losses[names[1]], label=r'$\mathcal{L}_{freqs}$', color='tomato', linestyle='-', linewidth=3.)
    plt.plot(losses[names[2]], label=r'$\mathcal{L}_{MACs}$', color='maroon', linestyle='-', linewidth=3.)
    plt.plot(tf.abs(losses[names[3]]), label=r'$\mathcal{L}_{Regularizer}$', color='black', linestyle='--', linewidth=3.)
    plt.plot(losses[names[4]], label=r'$\mathcal{L}_{Mixture}$', color='green', linestyle='-', linewidth=3.)


    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss value', fontsize=22)
    plt.xscale('log')
    plt.yscale('log')
    # Add a legend
    plt.legend(loc='lower left', fontsize=16)
    # Add a grid for better readability
    plt.grid(True, linestyle='-.', alpha=1.)
    plt.savefig(os.path.join(folder_path, 'Comparing_LGMM.png'),dpi = 500, bbox_inches='tight')
    plt.show()
    
compare_GMM_losses(losses, folder_path)




Test_results_info = np.load(os.path.join(folder_path, 'Test_results_info.npy'), allow_pickle = True).item()
test_means, test_sigmas_diag, test_weights, test_phi_angles, pred_alpha_test =  Test_results_info['test_means'], Test_results_info['test_sigmas_diag'], Test_results_info['test_weights'], Test_results_info['test_angles_phi'], Test_results_info['pred_alpha_test']

true_alphas =  alpha_factors_true_test
pred_alphas = pred_alpha_test







# foldername = '11_Oct_logfreqs_b005_512batch1e-05LR10000epochs'
foldername = '20els_23Oct_logfreqs20elements_10modes_0.05gamma_512batch1e-05LR10000epochs'
folder_path = os.path.join('Output',foldername)
    
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

data_path = os.path.join("Data", "18OctSingleData_20elements10nmodes")

# data_path = os.path.join("Data", "07OctSingleData_10elements10nmodes")

Problem_info = np.load(os.path.join(folder_path, 'Problem_info.npy'), allow_pickle = True).item()
input_dim, n_elements, n_modes, n_dofs, batch_size, n_epochs, LR = Problem_info['input_dim'],Problem_info['n_elements'], Problem_info['num_modes'], Problem_info['n_dofs'], Problem_info['batch_size'], Problem_info['epochs'], Problem_info['LRate']

Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)

# Constraints for the Alphas

# load the global Mass and element stiffness matrices:
Mfree, Ke_matrices = load_known_matrices(data_path, n_elements)
epsi = 0.05
# num_samples_per_mixture  = 1 # you can fix this to one for the diagnostis step 
model = My_Inverse_withPhysics( input_dim, n_dofs, n_elements, n_modes, Ke_matrices, Mfree, epsi, batch_size)
opt = K.optimizers.Adam(learning_rate = LR)
model.build(input_shape = ())
weights_path = os.path.join(folder_path, "model_weights.weights.h5" )
model.load_weights(weights_path)

autoencoder_alphapred = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
##Show some randomly chosen case studies
show_predicted_factors(alpha_factors_true_test, autoencoder_alphapred)

######################## lplot each loss term 
history_path = os.path.join(folder_path, 'model_history.npy')
model_history  = np.load(history_path, allow_pickle = True).item()
plot_loss_terms(model_history, folder_path)
    
## Plot crossplots from predicted factors (Take the minimum value as the estimate)
# If it spreads the stiffness reduction within more than one element, then you will observe rare /wrong estimates in the crossplot (identify them) 
plot_alpha_crossplots(np.min(alpha_factors_true_test, axis = 1), np.min(autoencoder_alphapred, axis=1), folder_path)


###  EVALUATING THE LOSS FUNCTION WITH ANY PREDICTION 

## To evaluate predictions taken from the inverse, you need to:
    # a) evaluate the inverse_only model to produce the reduction factors
    # b) Build the matrix, solve the eigenvalue problem and extract the eigenfrequencies. 
#Build and load weights in inverse_only model:
a_lb = 0.0001
a_ub = 1.0005
inverse_path = os.path.join("Output", "Inverse_18Oct_logfreqs1024batch1e-05LR10000epochs")
inverse_model = load_inverseonly_model(inverse_path, input_dim, n_dofs, n_elements, n_modes, a_lb, a_ub)
inv_alphapred = inverse_model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])

inv_pred_freqs, inv_pred_rotmodes, inv_pred_vertmodes = produce_pred_modalprops(n_dofs, n_elements, n_modes, Mfree, inv_alphapred, Ke_matrices)
ae_pred_freqs, ae_pred_rotmodes, ae_pred_vertmodes = produce_pred_modalprops(n_dofs, n_elements, n_modes, Mfree, autoencoder_alphapred, Ke_matrices)

true_freqs, true_rotmodes, true_vertmodes =  tf.cast(Freqs_true_test, tf.float32), tf.cast(Rotmodes_true_test, tf.float32), tf.cast(Vertmodes_true_test, tf.float32)
ae_pred_freqs, ae_pred_rotmodes, ae_pred_vertmodes = tf.cast(ae_pred_freqs, tf.float32), tf.cast(ae_pred_rotmodes, tf.float32), tf.cast(ae_pred_vertmodes, tf.float32)
inv_pred_freqs, inv_pred_rotmodes, inv_pred_vertmodes = tf.cast(inv_pred_freqs, tf.float32), tf.cast(inv_pred_rotmodes, tf.float32), tf.cast(inv_pred_vertmodes, tf.float32)



loss_value_onlyinverse = Loss_function(true_freqs, inv_pred_freqs, true_rotmodes, inv_pred_rotmodes, true_vertmodes, inv_pred_vertmodes)
loss_value_autoencoder = Loss_function(true_freqs, ae_pred_freqs, true_rotmodes, ae_pred_rotmodes, true_vertmodes, ae_pred_vertmodes)

print(loss_value_onlyinverse)
print(loss_value_autoencoder)

# ltest = Loss1(alpha_factors_true_test[0,:], alphapred[0,:])






# Assuming pred_alphas and true_alphas are already defined arrays
# pred_alphas  = np.min(alphapred, axis=1)
# true_alphas = np.min(alpha_factors_true_test, axis=1)

# Create a figure and axis



# def plot_loss_terms(model_history, folder_path):
#     Freqs_loss = model_history['freqs_loss']
#     MAC_loss = model_history['mac_modes_loss']
#     Regularizer_loss = model_history['alpha_regularizer']
#     losses = plt.figure()
#     plt.plot(Freqs_loss)
#     plt.plot(MAC_loss)
#     plt.plot(Regularizer_loss)
#     plt.show()


    
    
# def plot_loss_evolution(history, output_folder_path):
#     loss_plot  = plt.figure()
#     loss = history.history['loss']
#     # loss_loc = history.history['custom_loss_Location']
#     # loss_sev = history.history['custom_loss_Severity']
#     val_loss = history.history['val_loss']
#     plt.plot(val_loss,color = '#072FD1',)
#     plt.plot(loss,color = 'red')
#     # plt.plot(loss_loc,color = 'red',)
#     # plt.plot(loss_sev,color = 'green',)
#     #plt.title('model loss')
#     plt.ylabel('$\mathcal{L}_{total}$')
#     plt.xlabel('epoch')
#     xmin, xmax = plt.xlim()
#     # ymin, ymax = plt.ylim()
#     # ymin,ymax = 0.001, 0.1
#     scale_factor = 1
#     # plt.yscale('log')
#     plt.xlim(xmin *scale_factor, xmax * scale_factor)
#     # plt.ylim(ymin * scale_factor, ymax * scale_factor)
#     plt.legend(['Validation', 'Training'], loc='upper right', fontsize = 14)
#     plt.show()
#     loss_plot.savefig(os.path.join(output_folder_path, "Loss_trainval.png"),dpi = 500, bbox_inches='tight')
