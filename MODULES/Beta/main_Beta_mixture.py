#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:48:23 2025

@author: afernandez
"""
def main():
    import os
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras as K 
    import time  # Import the time module
    
    
    from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
    from MODULES.Beta.BetaMix_models import My_BetaVAE_withEigen
    
    from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_regularizer_loss, plot_Gaussianmarg_joint_pdf, plot_JointPDF_loss, plot_Freqs_loss, plot_multimodal_joint_pdf, compare_losses
    
    tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
    K.utils.set_random_seed(1234)
    dt = 'float32' ## espcificar dtype para trabajar en float32. 
    K.backend.set_floatx(dt)
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
    #Load dataset saved
    data_path = os.path.join("Data", "17MARRandomdata5elements5nmodes")# case with n elements 
    # data_path = os.path.join("Data", "21NovSingleData_5elements5nmodes")# case with n elements 
    # 
    n_elements = 5
    n_dofs = 2*(n_elements +1) 
    num_dofs = n_dofs -2 
    batch_size = 1024
    
    # total_dofs = 2*(num_elements +1) = 12 for 5 elements. From there you must remove 1 dofs at the limit nodes, resulting in 12-2 = 10 as the num_dofs
    Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)
    
    ## Removing some modes (rather than creating a new database with less modes): 
    # Assuming that we retain only the first ones: 
    # n_modes = 2
    # Freqs_true_train, Freqs_true_val, Freqs_true_test = Freqs_true_train[:,0:n_modes], Freqs_true_val[:,0:n_modes], Freqs_true_test[:,0:n_modes]
    # Rotmodes_true_train, Rotmodes_true_val, Rotmodes_true_test = Rotmodes_true_train[:,:,0:n_modes], Rotmodes_true_val[:,:,0:n_modes], Rotmodes_true_test[:,:,0:n_modes]
    # Vertmodes_true_train, Vertmodes_true_val, Vertmodes_true_test = Vertmodes_true_train[:,:,0:n_modes], Vertmodes_true_val[:,:,0:n_modes], Vertmodes_true_test[:,:,0:n_modes]
    
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Specifiy models and training 
    input_dim = Freqs_true_train.shape[1]+(Rotmodes_true_train.shape[1]*Rotmodes_true_train.shape[2]) + (Vertmodes_true_train.shape[1]* Vertmodes_true_train.shape[2])
    n_modes = Freqs_true_train.shape[1] #the number of modes we want to operate with 
    
    # load the mass  and stiffness matrix from wherever you have calculated it
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
    
    ## Trainign specifications, required for the folder name 
    n_epochs = 2000
    LR = 1e-04 # with exponent 05 I observe some moments of total loss increasing, which corresponds to a bad training.......
    epsi = 0.0 # This is the weight of the regularization term that favours (or not, if 0) those solutions with a single damaged element. 
    sigma_freq = 0.08
    sigma_phi = 0.08
    
    
    ## Bayesian specifications for the Gaussian Mixture approach 
    num_components = 3
    n_dims  = alpha_factors_true_train.shape[1]
    num_samples = 1
    regu_weight = 1 ##  unused. weight factor for the Gaussian Mixture term (previously named beta)
    
    run_eagerly = False # indicate True for debugging   
    
    date = "Dec10_Beta_"+str(n_elements)+"els"+str(n_modes)+"modes_test_Data17Mar"
    starting  = date + "Prueba"
    model = My_BetaVAE_withEigen(input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, n_dims, num_components, num_samples, regu_weight, sigma_freq, sigma_phi)
    
    
    
    filename = f'{starting}_{regu_weight}RegularizerW_{n_dims}dims_{num_components}components_{num_samples}Samples_{LR}LR_{n_epochs}epochs_{batch_size}_batchs'
    folder_path = os.path.join('Output',"Beta", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
        
    model.compile(optimizer = K.optimizers.Adam(learning_rate = LR), loss = model.ELBO_Beta_loss, metrics = [model.Freqs_loss, model.MSE_modes_loss, model.ELBO_Beta_loss, model.Mixture_dens_term], run_eagerly = run_eagerly)
    start_time = time.time()  # Record the starting time
    
    model_history = model.fit(x = [Freqs_true_train,Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
      y = alpha_factors_true_train,
      batch_size = batch_size,
      epochs = n_epochs,
      shuffle = True,
      validation_data = ([Freqs_true_val,Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], alpha_factors_true_val),
      callbacks = [])
    
    
    Problem_info = {
            'input_dim_enc':input_dim,
            'n_dims': n_dims, 
            'n_components':num_components, 
            'n_samples': num_samples, 
            'epochs': n_epochs,
            'LR': LR,
            'epsi': epsi,
            'regu_weight': regu_weight,
            }
    
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    model.save_weights(os.path.join(folder_path, "model_weights.weights.h5"))
    
    end_time = time.time()  # Record the ending time
    training_time = end_time - start_time  # Calculate the training time
    
    print(f"Total training time: {training_time:.2f} seconds")
    
    
    from matplotlib import pyplot as plt
    lelbo = model_history.history['ELBO_Beta_loss']
    Freq_loss = model_history.history['Freqs_loss']
    Mixture_loss = model_history.history['Mixture_dens_term']
    MSE_modes_loss = model_history.history['MSE_modes_loss']
    
    plt.plot(Mixture_loss)

        

############## This operation allows to exeute functions in the script     
if __name__ == "__main__":
    
    main()
    

