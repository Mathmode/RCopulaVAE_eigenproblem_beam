#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:07:58 2024

@author: afernandez
"""

import numpy as np
import keras as K
import os 
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from MODULES.TRAINING.multivariate_arch import Solve_eigenproblem, assemble_global_Kmatrices
from MODULES.TRAINING.multivariate_models import My_Inverse_withPhysics, My_Inverse


def calculate_MAC(modes_true, modes_pred):
    modes_true_transp = K.ops.einsum('BCM -> BMC', modes_true)
    modes_pred_transp  = K.ops.einsum('BCM -> BMC', modes_pred)
        
    MAC_numer = K.ops.square(K.ops.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
    MAC_denom  = K.ops.multiply(K.ops.einsum('BMC, BCM -> BM', modes_true_transp, modes_true), K.ops.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred))
    MAC = K.ops.divide(MAC_numer, MAC_denom)
    #MAC dimension is (Batch_size, N_modes)
    return MAC

def load_inverseonly_model(inverse_path, input_dim, n_dofs, n_elements, n_modes, a_lb, a_ub):
    inverse_path = os.path.join("Output", "Inverse_18Oct_logfreqs1024batch1e-05LR10000epochs")
    inverse_model = My_Inverse(input_dim, n_dofs, n_elements, n_modes, a_lb, a_ub)
    inverse_model.build(input_shape = ())
    inverse_weights_path = os.path.join(inverse_path, "model_weights.weights.h5" )
    inverse_model.load_weights(inverse_weights_path)
    return inverse_model
    

def Loss_function(true_freqs, pred_freqs, true_rotmodes, pred_rotmodes, true_vertmodes, pred_vertmodes):
    Freqs_sq_error = K.ops.square(K.ops.log(true_freqs) - K.ops.log(pred_freqs))
    Loss_freqs = K.ops.mean(Freqs_sq_error, axis = None)
    
    Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
    Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
    MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
    neg_MACs = 1 - MACs
    Loss_MAC = K.ops.mean(neg_MACs, axis  = None)
    return Loss_freqs + Loss_MAC

def produce_pred_modalprops(n_dofs, n_elements, n_modes, Mfree, alpha_pred, Ke_matrices):
    Ke_matrices_dam = K.ops.einsum('BE, EKQ -> BEKQ', tf.cast(alpha_pred, dtype = tf.float32), tf.cast(Ke_matrices, dtype = tf.float32))
    Kfree = assemble_global_Kmatrices(Ke_matrices_dam, n_elements, Ke_matrices_dam.shape[0]) # the shape is (Batch_size, n_free, n_free)
    Eigen_solver = Solve_eigenproblem(n_dofs, n_modes, Mfree)
    pred_freqs, pred_rotmodes, pred_vertmodes = Eigen_solver(Kfree)
    return pred_freqs, pred_rotmodes, pred_vertmodes