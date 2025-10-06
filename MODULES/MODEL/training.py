# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:46:07 2020

@author: 110137
"""

import tensorflow as tf
from MODULES.MODEL.submodels_creation import submodels, compact_models
from MODULES.MODEL.model_creation import PCA_model_creation, ResNet_model_creation, Compact_model_creation, ResNet_Severity_model_creation,  ResNet_Location_model_creation
from MODULES.POSTPROCESSING.postprocessing import plot_loss_evolution
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint

###########################################################################################
 
class training_info_initializer():
    #definicion e inicialización de las variables que van dentro 
    def __init__(self, n_epoch = 20, batch_size = 512, LR = 1e-03, metrics = "accuracy", loss = "mean_squared_error", shuffle = True,):        
        self.n_epoch = n_epoch
        self.Batch_size = batch_size
        self.LR = LR
        self.Metrics = metrics
        self.Loss = loss
        self.Shuffle = shuffle
        if n_epoch == None or batch_size == None or LR == None or loss == None:
            print("************************************************************************")
            print("Please initialize the info for TRAINING!")
            print("************************************************************************")
            quit()

def training_compact_model(arch_info, train_info, Xtrain, Ytrain_std, Xval, Yval_std, f_name):
    # Create the submodels based on the previos architectures. We define the encoder and decoder submodels
    input_layer,compact_model = compact_models(arch_info)
    #We create the final model of the AUTOENCODER
    #list  para guardar los modelos. Inicializamos
    filepath = os.path.join('Output','best_model' + str(f_name) +'.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    Compact_Model =Compact_model_creation(input_layer, compact_model)
    # Residual_autoencoder =ResNet_Location_model_creation(input_layer, nonlinear_encoder, nonlinear_decoder)

    # Residual_autoencoder =ResNet_model_creation(input_layer,linear_encoder,linear_decoder, nonlinear_encoder, nonlinear_decoder)
    Compact_Model.compile(metrics=[train_info.Metrics], loss= train_info.Loss, optimizer = tf.keras.optimizers.Adam(learning_rate = train_info.LR)) #Define the optimizer (SGD, RMSprop, Adam, adaline*) 
    Compact_Model.summary()
    Model_history = Compact_Model.fit(Xtrain, Ytrain_std, epochs = train_info.n_epoch, batch_size = train_info.Batch_size, callbacks = checkpoint,  shuffle = train_info.Shuffle, validation_data = (Xval, Yval_std))
    plot_loss_evolution(Model_history,f_name)
    models = Compact_Model
    np.save(os.path.join('Output','History_'+str(f_name)+'.npy'), Model_history.history)
    return models, Model_history

def building_AE_models(Residual, PCA, arch_info, train_info, filename):
    # Create the submodels based on the previos architectures. We define the encoder and decoder submodels
    input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder = submodels(arch_info)
    model_parts  = [input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder]
    models = [None, None]
    #We create the final model of the AUTOENCODER
    #list  para guardar los modelos. Inicializamos
    # models = [None,None]
    # histories = [None,None]
    # filepath = os.path.join('Output','best_model'+str(filename)+'.hdf5')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    Residual_autoencoder, Residual_encoder = ResNet_model_creation(input_layer,linear_encoder,linear_decoder, nonlinear_encoder, nonlinear_decoder)
    Residual_autoencoder.compile(metrics=[train_info.Metrics], loss= train_info.Loss, optimizer = tf.keras.optimizers.Adamax(learning_rate = train_info.LR)) #Define the optimizer (SGD, RMSprop, Adam, adaline*) 
    Residual_autoencoder.summary()
    models[0] = Residual_autoencoder
    models[1] = Residual_encoder
    return models,model_parts



def training_autoencoder(model, train_info,Xtrain_std,Xval_std,filename):
    # filepath = os.path.join('Output','best_model'+str(filename)+'.hdf5')
    filepath = os.path.join('Output','best_model'+str(filename))
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(Xtrain_std, Xtrain_std, epochs = train_info.n_epoch, batch_size = train_info.Batch_size, callbacks = checkpoint, shuffle = train_info.Shuffle, validation_data = (Xval_std, Xval_std))
    plot_loss_evolution(history,filename)
    return history



def loading_model(arch_info, train_info):
    # Create the submodels based on the previos architectures. We define the encoder and decoder submodels
    input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder = submodels(arch_info)
    #We create the final model of the AUTOENCODER
    #list  para guardar los modelos. Inicializamos
    # filepath = os.path.join('Output','best_model' + str(f_name) +'.hdf5')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    Residual_autoencoder =ResNet_Severity_model_creation(input_layer, nonlinear_encoder, nonlinear_decoder)
    # Residual_autoencoder =ResNet_Location_model_creation(input_layer, nonlinear_encoder, nonlinear_decoder)

    # Residual_autoencoder =ResNet_model_creation(input_layer,linear_encoder,linear_decoder, nonlinear_encoder, nonlinear_decoder)
    Residual_autoencoder.compile(metrics=[train_info.Metrics], loss= train_info.Loss, optimizer = tf.keras.optimizers.Adam(learning_rate = train_info.LR)) #Define the optimizer (SGD, RMSprop, Adam, adaline*) 
    Residual_autoencoder.summary()
    models = Residual_autoencoder
    
    return models

def training_model(models, train_info, Xtrain, Ytrain_std, Xval, Yval_std, f_name):
    #list  para guardar los modelos. Inicializamos
    filepath = os.path.join('Output','best_model' + str(f_name)+'.hdf5')
    # filepath = os.path.join('Output','best_model' + str(f_name))

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    models.save_weights(filepath)
    Residual_autoencoder  = models
    Model_history = Residual_autoencoder.fit(Xtrain, Ytrain_std, epochs = train_info.n_epoch, batch_size = train_info.Batch_size, callbacks = checkpoint,  shuffle = train_info.Shuffle, validation_data = (Xval, Yval_std))
    plot_loss_evolution(Model_history,f_name)
    np.save(os.path.join('Output','History_'+str(f_name)+'.npy'), Model_history.history)
    
    return Residual_autoencoder, Model_history







# # MOVER A OTRO SCRIPT!!
# #tf.math para square operation
# def my_MSE(y_actual,y_pred):
#     loss = tf.math.reduce_mean(tf.math.square(y_actual - y_pred), axis= None)
#     tf.print(loss)
#     return loss

# def custom_loss_Location(y_actual,y_pred):
#     aux_var = tf.math.square(y_actual - y_pred)
#     loss_zone1 = tf.math.reduce_mean(aux_var[:,0], axis= None)
#     return loss_zone1/y_pred.shape[1]

# def custom_loss_Severity(y_actual,y_pred):
#     aux_var = tf.math.square(y_actual - y_pred)
#     loss_zone2 = tf.math.reduce_mean(aux_var[:,1], axis= None)
#     return loss_zone2/y_pred.shape[1]


# def custom_loss(y_actual,y_pred):
#     aux_var = tf.math.square(y_actual - y_pred)
#     # tf.print(aux_var.shape)
#     loss_Location = tf.math.reduce_mean(aux_var[:,0], axis= None)
#     loss_Severity = tf.math.reduce_mean(aux_var[:,1], axis= None)
#     loss = (loss_Location + loss_Severity)/y_pred.shape[1]
#     return loss


# def custom_loss(y_actual,y_pred):
#     aux_var = tf.math.square(y_actual - y_pred)
#     # tf.print(aux_var.shape)
#     loss_Location = tf.math.reduce_mean(aux_var[:,0], axis= None)
#     loss_Severity = tf.math.reduce_mean(aux_var[:,1], axis= None)
#     loss = (loss_Location + loss_Severity)/y_pred.shape[1]
#     for i in range (aux_var.shape[0]):
#         if y_actual[i,1]<0.1:
#             loss = loss_Severity/y_pred.shape[1]
#         else:
#             loss =  (loss_Location + loss_Severity)/y_pred.shape[1]
#     return loss
