#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:38:41 2025

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np
import os
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import random 

from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_eigen_functions import Solve_eigenproblem, assemble_global_Kmatrices
tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
K.utils.set_random_seed(1234)
dt = 'float32' ## espcificar dtype para trabajar en float32. 
K.backend.set_floatx(dt)
folder_path = os.path.join("Output", "Gaussian_Copula")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
#Load dataset saved
data_path = os.path.join("Data", "17MARRandomdata5elements5nmodes")# case with n elements 
# data_path = os.path.join("Data", "21NovSingleData_5elements5nmodes")# case with n elements 
# 
n_elements = 5
n_dofs = 2*(n_elements +1) 
num_dofs = n_dofs -2 
batch_size = 48
# nf = 70
# total_dofs = 2*(num_elements +1) = 12 for 5 elements. From there you must remove 1 dofs at the limit nodes, resulting in 8-2 = 6 as the num_dofs
Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)
# load the mass  and stiffness matrix from wherever you have calculated it
Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)


input_dim_decoder = n_elements

def calculate_MAC(modes_true, modes_pred):
    modes_true_transp = np.einsum('BCM -> BMC', modes_true)
    modes_pred_transp  = np.einsum('BCM -> BMC', modes_pred)
        
    MAC_numer = np.square(np.einsum('BMC, BCM -> BM', modes_true_transp, modes_pred))
    MAC_denom  = np.multiply(np.einsum('BMC, BCM -> BM', modes_true_transp, modes_true), np.einsum('BMC, BCM -> BM', modes_pred_transp, modes_pred))
    MAC = np.divide(MAC_numer, MAC_denom)
    #MAC dimension is (Batch_size, N_modes)
    return MAC


def select_gt_pointcloud(input_dim_decoder, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, Mfree, Ke_matrices, L_inv, n_elements, num_dofs, folder_path ):
    point_clouds  = []
    N = 20000 # number of points to form the cloud to be used as Ground Truth (GT)
    num_samples = 1
    n_modes = Freqs_true_test.shape[1] #the number of modes we want to operate with 

    ztest = []
    for j in range(N):
        z1 = random.uniform(0.05,0.95)
        z2 = random.uniform(0.05,0.95)
        z3 = random.uniform(0.05,0.95)
        z4 = random.uniform(0.05,0.95)
        z5 = random.uniform(0.05,0.95)

        z = [z1,z2,z3,z4,z5]
        ztest.append(z)
    
    ztest = np.array(ztest) #these are the randomly generated damage features (5-dimensional case)
    
    
    
    # # Apply the corresponding factor using einsum(Batch_size, n_elements, 4x4)
    Ke_matrices_dam = tf.einsum('BE, EKQ -> BEKQ', tf.cast(ztest, dtype = tf.float32), tf.cast(Ke_matrices, dtype = tf.float32))
    Kfree = assemble_global_Kmatrices(Ke_matrices_dam, n_elements,  num_samples) # the shape is (Batch_size, n_free, n_free)
    
    Eigen_solver = Solve_eigenproblem(num_dofs, n_modes, Mfree, L_inv)
    # Then we enter the eigensolver (forward) function with this list to produce the eigenfrequencies
    pred_freqs, pred_rotmodes, pred_vertmodes = Eigen_solver(Kfree)
    pred_freqs, pred_rotmodes, pred_vertmodes = np.array(pred_freqs), np.array(pred_rotmodes), np.array(pred_vertmodes)
    pred_freqs =  np.abs(pred_freqs) #to enforce them to be positive***
   
    def Freqs_loss(y_true, y_pred, num_samples):
        Freqs_sq_error = np.square(np.log(y_true) - np.log(y_pred))
        Loss_freqs = np.mean(Freqs_sq_error, axis = 1)
        return Loss_freqs
    
    
    def MAC_modes_loss(true_rotmodes, true_vertmodes, pred_rotmodes, pred_vertmodes):
        #Calculate the MACs
        Rot_MACs = calculate_MAC(true_rotmodes, pred_rotmodes) #shape: (Batch_Size, n_modes)
        Vert_MACs = calculate_MAC(true_vertmodes, pred_vertmodes) #shape: (Batch_size, n_modes)
        # MACs = K.ops.hstack((Rot_MACs, Vert_MACs))
        MACs = np.concatenate([Rot_MACs, Vert_MACs], axis=1)
        neg_MACs = 1 - MACs
        Loss_MAC = np.mean(neg_MACs, axis  = 1)
        return Loss_MAC 
    
    
    for i in range(Freqs_true_test.shape[0]):
        pos = i
        Freqs_true = Freqs_true_test[pos,:].reshape(1,Freqs_true_test.shape[1]) #Case that we want to exlpore as input
        ftrue =  np.array(tf.repeat(Freqs_true, N, axis = 0)) #now we have it 1,000 times to associate it to 1,000 different ps
        Mrot_true = Rotmodes_true_test[pos,:].reshape(1,Rotmodes_true_test.shape[1], Rotmodes_true_test.shape[2])
        mrot_true  =  np.array(tf.repeat(Mrot_true, N, axis = 0))
        Mvert_true = Vertmodes_true_test[pos,:].reshape(1, Vertmodes_true_test.shape[1], Vertmodes_true_test.shape[2])
        mvert_true  =  np.array(tf.repeat(Mvert_true, N, axis = 0))
        
        z1 = ztest[:,0]
        z2 = ztest[:,1]
        z3 = ztest[:,2]
        z4 = ztest[:,3]
        z5 = ztest[:,4]
    
        floss = Freqs_loss(ftrue, pred_freqs, num_samples)
        MAC_loss = MAC_modes_loss(mrot_true, mvert_true, pred_rotmodes, pred_vertmodes)
        total_loss = floss + MAC_loss
    
        # Get the indices of the K smallest dloss values
        top_k_indices = np.argsort(total_loss)[:N]
    
        # Select the corresponding values for each feature in the latent space: 
        z1_filtered = z1[top_k_indices]
        z2_filtered = z2[top_k_indices]
        z3_filtered = z3[top_k_indices]
        z4_filtered = z4[top_k_indices]
        z5_filtered = z5[top_k_indices]

        Loss_data_filtered = total_loss[top_k_indices]            
        
        combined_array = np.column_stack((z1_filtered, z2_filtered, z3_filtered, z4_filtered, z5_filtered, Loss_data_filtered))
        point_clouds.append(combined_array)
    
    point_clouds = np.array(point_clouds)
    # np.savetxt(os.path.join(folder_path,"Point_clouds.csv"), combined_array, delimiter=",", header="Z1,Z2,Data_misfit", comments="", fmt="%g")        
    np.save(os.path.join(folder_path,"Prueba_Test_Point_clouds_03Feb.npy"), point_clouds)        

Test_point_clouds = select_gt_pointcloud(input_dim_decoder, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test, Mfree, Ke_matrices, L_inv, n_elements, num_dofs, folder_path )



