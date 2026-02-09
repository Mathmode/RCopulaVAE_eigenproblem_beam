#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:06:33 2024

@author: afernandez
"""
#### RESULTS and ocmparisong with standard autoencoder predictions/estimations

import os
import tensorflow as tf
from tensorflow.keras import optimizers, models, utils
from MODULES.POSTPROCESSING.postprocessing_tools import custom_plot_loss

def training_forward(nepochs, LR, forwardmodel, num_mixtures, u_train,u_val, r_train, r_val, p_train, p_val, date):
    filename = f'{date}_2Prop_Forward'
    folder_path = os.path.join('Output', filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    input_dim_decoder = p_train.shape[1] + r_train.shape[1]
    output_dim = u_train.shape[1]
    model_forward = forwardmodel(input_dim_decoder, output_dim,num_mixtures)
    model_forward.compile(optimizer = optimizers.Adam(learning_rate = LR), loss=model_forward.custom_loss, metrics=['accuracy'])
    forward_history = model_forward.fit(x = tf.concat([p_train, r_train], 1),
    y = u_train,
    batch_size = 512,
    epochs = nepochs,
    shuffle = True,
    validation_data = (tf.concat([p_val, r_val], 1), u_val))
    tf.saved_model.save(model_forward, os.path.join(folder_path,filename))
    custom_plot_loss(forward_history, folder_path, 'loss_forward')    
    return model_forward


# def run_training(train_forward, n_epochs, LR,
#                  u_train, u_validation, u_test,
#                  r_train, r_validation, r_test,
#                  P_train, P_validation, P_test,
#                  arch_fun_f,
#                  num_mixtures, num_gaussians, num_samples_per_mixture, folder_path):

#     tf.random.set_seed(101)
#     if train_forward:
#         input_dim_decoder = num_mixtures + r_train.shape[1]
#         output_dim = u_train.shape[1]
#         model_forward = ForwardModel(input_dim_decoder, output_dim, num_mixtures, num_samples_per_mixture)
        
#         opt = optimizers.Adam(learning_rate = LR)
#         model_forward.compile(loss = 'mse', optimizer = opt)    
#         forward_history = model_forward.fit(x = tf.concat([P_train, r_train], 1),
#           y = u_train,
#           batch_size = 1024,
#           epochs = n_epochs,
#           shuffle = True,
#           validation_data = (tf.concat([P_validation, r_validation], 1), u_validation))
       
#         arch_name = 'Fully_connected_dec'
#         tf.saved_model.save(model_forward, 'model_forward_1mix')
#         custom_plot_loss(forward_history)

#         return model_forward, [], [], []

#     else:
#         input_dim_decoder = num_mixtures + r_train.shape[1]
#         output_dim = u_train.shape[1]
#         input_dim_encoder = u_train.shape[1] + r_train.shape[1]

#         model_forward = tf.saved_model.load('model_forward_1mix')
#         model_forward.trainable = False

#         model = Prueba_Autoencoder(input_dim_encoder, input_dim_decoder, output_dim, num_mixtures, num_gaussians, num_samples_per_mixture, model_forward)
#         # model = GMM_Autoencoder(input_dim_encoder, input_dim_decoder, output_dim, num_mixtures, num_gaussians, num_samples_per_mixture, model_forward)
        
#         opt = optimizers.Adam(learning_rate = LR)
#         model.compile(loss = model.averaged_loss, optimizer = opt)    
#         model_history = model.fit(x = [u_train, r_train, P_train],
#           y = u_train,
#           batch_size = 512,
#           epochs = n_epochs,
#           shuffle = True,
#           validation_data = ([u_validation, r_validation, P_validation], u_validation))
#         model.save_weights(os.path.join(folder_path,"Model_Autoencoder_weights.h5"))
#         custom_plot_loss(model_history)
#         # model_inverse = model.Encoder_model
#         # make_predictions(model, model_inverse, u_test, r_test, P_test, num_gaussians)
#         return model

