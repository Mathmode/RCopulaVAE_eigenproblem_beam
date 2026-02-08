#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:16:13 2025

@author: afernandez
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # 1. Suppress annoying TF logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # 2. Configure GPU Memory Growth
# # This block is wrapped in a try/except to handle "Restart" situations safely
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"✅ GPU Memory Growth Enabled for {len(gpus)} GPU(s)")
#     except RuntimeError as e:
#         # If we get here, it means TF is already running.
#         # We print a warning but allow the script to continue.
#         print("⚠️ NOTICE: GPU settings could not be updated (TF already initialized).")
#         print("   If you just added these lines, RESTART YOUR KERNEL to apply them.")
#         print(f"   (Error details: {e})")
        

import numpy as np
import tensorflow as tf
import tensorflow.keras as K 
from MODULES.PREPROCESSING.preprocessing import load_data, load_known_matrices
from MODULES.COPULAS.GC_GMm_models import My_CopulaVAE_withEigen
from MODULES.COPULAS.GC_postprocessing_copulas import plot_trainval_loss, plot_regularizer_loss, plot_Gaussianmarg_joint_pdf, plot_JointPDF_loss, plot_Freqs_loss, plot_multimodal_joint_pdf

def main():
    
    # import os
    # # 1. Hide the GPU from TensorFlow (This stops the ptxas crash)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # # 2. Fix the library conflict (This stops the CPU silent crash)
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

   
    # tf.config.list_physical_devices('GPU')  # TODO I do not find the analogous in K .
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
    batch_size = 256
    # total_dofs = 2*(num_elements +1) = 12 for 5 elements. From there you must remove 1 dofs at the limit nodes, resulting in 8-2 = 6 as the num_dofs
    Freqs_true_train, Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train, Freqs_true_val, Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val, Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test =  load_data(data_path, batch_size)
    
    ## Removing some modes (rather than creating a new database with less modes): 
    # Assuming that we retain only the first ones: 
    # n_modes = 2
    # Freqs_true_train, Freqs_true_val, Freqs_true_test = Freqs_true_train[:,0:n_modes], Freqs_true_val[:,0:n_modes], Freqs_true_test[:,0:n_modes]
    # Rotmodes_true_train, Rotmodes_true_val, Rotmodes_true_test = Rotmodes_true_train[:,:,0:n_modes], Rotmodes_true_val[:,:,0:n_modes], Rotmodes_true_test[:,:,0:n_modes]
    # Vertmodes_true_train, Vertmodes_true_val, Vertmodes_true_test = Vertmodes_true_train[:,:,0:n_modes], Vertmodes_true_val[:,:,0:n_modes], Vertmodes_true_test[:,:,0:n_modes]    
    # positions = [11, 17,22,25,34,45]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Specifiy models and training 
    input_dim = Freqs_true_train.shape[1]+(Rotmodes_true_train.shape[1]*Rotmodes_true_train.shape[2]) + (Vertmodes_true_train.shape[1]* Vertmodes_true_train.shape[2])
    n_modes = Freqs_true_train.shape[1] #the number of modes we want to operate with 
    
    # load the mass  and stiffness matrix from wherever you have calculated it
    Mfree, Ke_matrices, L_inv = load_known_matrices(data_path, n_elements)
    
    ## Trainign specifications, required for the folder name 
    n_epochs = 5000
    LR = 1e-04 # with exponent 05 I observe some moments of total loss increasing, which corresponds to a bad training.......
    epsi = 0.0 #Regularizer to find one single damaged element
    
    ## Bayesian specifications for the Gaussian Mixture approach 
    num_gaussians = 1
    n_dims  = alpha_factors_true_train.shape[1]
    num_samples = 1
    beta = 0.04 ## weight factor for the Gaussian Mixture term 
    
    run_eagerly = True # indicate True for debugging   
   

    model = My_CopulaVAE_withEigen(input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, n_dims, num_gaussians, num_samples, beta)
    
    filename = r"NewMACLosss_Feb07_5els5modes_test_Data17MarCopula_"+str(beta)+"Beta_"+str(num_samples)+"Samples_"+str(LR)+"LR_"+str(n_epochs)+"epochs_"+str(batch_size)+"batchs"
    
    # date = "local_Feb02_026_WarpedGaussian"+str(n_elements)+"els"+str(n_modes)+"modes_test_Data17Mar"
    # starting  = date + "Copula"
    # filename = f'{starting}_{beta}Beta_{n_dims}dims_{num_gaussians}_{num_samples}Samples_{LR}LR_{n_epochs}epochs_{batch_size}_batchs'
   
    folder_path = os.path.join('Output',"Gaussian_Copula", filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # # 1. Definir la ruta del checkpoint
    # checkpoint_path = os.path.join("Output", "checkpoints", "cp-{epoch:04d}.ckpt")
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # # 2. ¡IMPORTANTE! Crear la carpeta 'checkpoints' si no existe
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    #     print(f"✅ Carpeta de checkpoints creada en: {checkpoint_dir}")
    
    # # 3. Callback para guardar (Checkpoint)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path, 
    #     verbose=1, 
    #     save_weights_only=True,
    #     save_freq='epoch',
    #     period = 50
    # )
    
    # # 4. Callback para limpiar memoria (Garbage Collector)
    import gc
    class MemoryCleaner(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect() # Solo usamos gc.collect(), es mas seguro durante el entrenamiento
                
    # # --- BLOQUE DE RECUPERACIÓN ---
    # # Busca el último checkpoint guardado en la carpeta
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    # if latest:
    #     print(f"🔄 Cargando pesos desde: {latest}")
    #     # Cargar pesos (expect_partial evita errores si faltan variables del optimizador)
    #     model.load_weights(latest).expect_partial()
    #     print("✅ ¡Pesos cargados! Continuamos desde donde se quedó.")
    # else:
    #     print("⚠️ No se encontraron checkpoints. Empezando desde cero.")      
    
    model.compile(optimizer = K.optimizers.Adam(learning_rate = LR), loss = model.ELBO_Copula_loss, metrics = [model.Freqs_loss,model.MAC_modes_loss, model.Joint_copula_dens_term], run_eagerly = run_eagerly)

    
    model_history = model.fit(x = [Freqs_true_train,Rotmodes_true_train, Vertmodes_true_train, alpha_factors_true_train],
      y = alpha_factors_true_train,
      batch_size = batch_size,
      epochs = n_epochs,
      shuffle = True,
      validation_data = ([Freqs_true_val,Rotmodes_true_val, Vertmodes_true_val, alpha_factors_true_val], alpha_factors_true_val),
      callbacks = [MemoryCleaner()])
    
    
    Problem_info = {
            'input_dim_enc':input_dim,
            'n_dims': n_dims, 
            'n_gaussians':num_gaussians, 
            'n_samples': num_samples, 
            'epochs': n_epochs,
            'LR': LR,
            'epsi': epsi,
            'beta': beta,
            }
    
    np.save(os.path.join(folder_path, 'Problem_info.npy'), Problem_info, allow_pickle=True)
    
    np.save(os.path.join(folder_path, 'model_history.npy'), model_history.history, allow_pickle=True)
    model.save(os.path.join(folder_path, "Full_Model_Export"), save_format="tf")
    model.save_weights(os.path.join(folder_path, "model_weights.weights.h5"))
    history_path = os.path.join(folder_path, 'model_history.npy')
    np.save(os.path.join(history_path), model_history.history, allow_pickle=True)
    
    
    plot_trainval_loss(model, folder_path)
    plot_JointPDF_loss(model, folder_path)
    plot_Freqs_loss(model, folder_path)
            
    inverse_model = model.Encoder_model
    test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems  = inverse_model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
    
    pred_alphas = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
    np.save(os.path.join(folder_path, "pred_alphas.npy"), pred_alphas, allow_pickle=True)

    

    # from MODULES.COPULAS.GC_postprocessing_copulas import plot_configuration, plot_results_PDF_uncertainty
    # plot_configuration()
    # from MODULES.COPULAS.GC_GMm_functions import build_correlation_matrices_from_cholesky
    # test_LT_matrices  = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems,n_dims)
    # positions = [0, 1, 7,  9, 11, 17,25, 34,45, 100, 138, 219, 234, 343, 456, 555, 612, 690, 761]
    # n_samples = 100
    # for i in range(len(positions)):
    #         pos = positions[i]    
    #         locs = test_means[pos,:]    
    #         scales = test_scales[pos,:]
    #         weight_vals = test_weight_vals[pos,:]
    #         LT_matrix = test_LT_matrices[pos,:]
    #         plot_results_PDF_uncertainty(n_elements, locs, scales, weight_vals, LT_matrix, n_dims, n_samples, pos, alpha_factors_true_test, folder_path)

    

    # pred_alpha_test = model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])        
    # tf.print('Predictions', pred_alpha_test[0:num_samples,:]) ## Es el output del modelo, llamado self.reshaped_alpha_samples
    # tf.print('*************************************************************')
    # tf.print('True values',alpha_factors_true_test[0,:])

      
   
    
   
    # clouds_path = os.path.join("Output", "Gaussian_Copula")
    # point_clouds = np.load(os.path.join(clouds_path,"Test_Point_clouds.npy"))


    # positions = [0, 1, 11]

    




    # all_axis = [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3],[2,4], [3,4]]
    # for j in range(len(all_axis)):
    #     chosen_axis = all_axis[j]
    #     for i in range(len(positions)):
    #         pos = positions[i]    
    #         locs = test_means[pos,:]    
    #         scales = test_scales[pos,:]
    #         weight_vals = test_weight_vals[pos,:]
    #         LT_matrix = test_LT_matrices[pos,:]
    
    #         n_samples = 50
    #         plot_KDE_pdf(locs, scales, weight_vals, LT_matrix, n_dims, n_samples, pos, chosen_axis, folder_path)
    #         # point_cloud = point_clouds[pos,:]    
    #         # plot_Datamisift_KDE(point_cloud, pos, chosen_axis, clouds_path)
            
        
    
    # for k in range(len(positions)):
    #     pos = positions[k]    
    #     point_cloud = point_clouds[pos,:]    
    #     plot_Datamisift_KDE(point_cloud, pos, chosen_axis, clouds_path)
       
    



   
   ######################################################################################################## 
    # ## loading a previously trained model:
    # model_filename = "18MAR_5els5modes_1Gaussianmarg_testCopula_0.075Beta_5dims_1gaussians_2Samples_0.001LR_5000epochs_48_batchs"
    # model_path = os.path.join("Output", "Gaussian_Copula", model_filename)
    # model = My_CopulaVAE_withEigen(input_dim, num_dofs, n_elements, n_modes, Ke_matrices, Mfree, L_inv, epsi, n_dims, num_gaussians, num_samples, beta)
    # model.build(input_shape = ())
    # ae_weights_path = os.path.join(model_path, "model_weights.weights.h5" )
    # model.load_weights(ae_weights_path)
    # inverse_model = model.Encoder_model
    # test_means, test_scales, test_weight_vals, test_offdiag_elems, test_diag_elems  = inverse_model.predict([Freqs_true_test, Rotmodes_true_test, Vertmodes_true_test, alpha_factors_true_test])
    # test_LT_matrices  = build_correlation_matrices_from_cholesky(test_offdiag_elems, test_diag_elems, n_dims)
###################################################################################################################################################
    


        

############## This operation allows to exeute functions in the script     
if __name__ == "__main__":
    
    main()
    

