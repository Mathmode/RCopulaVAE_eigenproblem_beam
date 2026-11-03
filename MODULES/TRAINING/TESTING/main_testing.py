# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
""
help(fft)
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from MODULES.MODEL.submodels_creation import architecture_info_initializer
from MODULES.MODEL.training import training_info_initializer, loading_model
from MODULES.MODEL.Custom_losses import my_MSE
from MODULES.PREPROCESSING.preprocessing import preprocessing_interface, correct_mode_sign
from MODULES.TESTING.testing_tools import testing_synthetic_severity, testing_synthetic_location
from MODULES.POSTPROCESSING.postprocessing_tools import plot_healthy_measurements

tf.random.set_seed(7)
#INITIAL CONSIDERATIONS
filename = 'SeverityPorto22Agos'
#Load data and preprocessing
bridge_loc = "PORTO"
# bridge_loc = "Z24"

Xtrain, Xval, Xtest, Ytrain_std, Yval_std, Ytest_std, std_model, Data_real, Test_Experimental = preprocessing_interface(bridge_loc)
#Define the internal architecture of the submodels: encoder and decoder. Both architectures have an input and an output and are interconected
Input_dim = Xtrain.shape[1] ; #Input dimension (number of variables in the dataset)
# Output_dim = Ytrain_std.shape[1] #output dimension for the ENCODER --> This is the latent feature dimension (in this case, we want a single-value feature)
Output_dim = 1 #When using Severity only (or Location)
###########################################################################################3
#Initialize architecture properties and select the architectures FROM the DICTIONARY

architecture_info_Sev = architecture_info_initializer(
    PCA_enc = "arch_PCA_encoder",
    PCA_dec = "arch_PCA_decoder",
    Res_enc = "arch_Residual_encoder_Severity",
    Res_dec = "arch_Residual_decoder_Severity",
    Compact_Z24 = "arch_compact_Z24",
    input_dim = Input_dim,
    enc_dim =150,
    output_dim = Output_dim)


#BUILD AND TRAIN THE MODELS
#Initiailize the training properties
training_info_Sev = training_info_initializer(
                n_epoch =2000 ,
                batch_size =8196,
                LR = 1e-04,
                metrics = ["mse"], 
                loss = my_MSE, 
                shuffle = True)



# ###############################################################################################################################
# ##################################ESTIMATE LOCATION##############################################################
# #################################################################################################################################
f_name = 'LocationPorto15Jul'
#Load data and preprocessing
bridge_loc = "PORTO"
# bridge_loc = "Z24"

Xtrain, Xval, Xtest, Ytrain_std, Yval_std, Ytest_std, std_model, Data_real, Test_Experimental = preprocessing_interface(bridge_loc)

Input_dim = Xtrain.shape[1] ; #Input dimension (number of variables in the dataset)
Output_dim = 1

architecture_info_Loc = architecture_info_initializer(
    PCA_enc = "arch_PCA_encoder",
    PCA_dec = "arch_PCA_decoder",
    Res_enc = "arch_Residual_encoder_Location",
    Res_dec = "arch_Residual_decoder_Location",
    Compact_Z24 = "arch_compact_Z24",
    input_dim = Input_dim,
    enc_dim = 36,
    output_dim = Output_dim)

training_info_Loc = training_info_initializer(
                n_epoch = 2000, 
                batch_size =  8196, 
                LR = 1e-04,
                metrics = ["mse"], 
                loss = my_MSE, 
                shuffle = True)

###########################DAMIAGE IDENTIFICATION TESTINGSSS###############################################
# # TESTING #For Infante bridge make prediction of location once we know severity exists. 
#LOAD THE TRAINED MODELS(NOT RETRAIN!!)
path_Sev_model= os.path.join("Output","best_modelSingleCluster_DB_Severity_23Ago022_F.hdf5") #The path from which you want to inherit the weights
Model_Severity= loading_model(architecture_info_Sev, training_info_Sev)
Model_Severity.load_weights(path_Sev_model)

Model_Severity.get_config(path_Sev_model)

loss = my_MSE
tf.keras.models.load_model(path_Sev_model, custom_objects = { 'my_MSE': my_MSE})
tf.keras.models.load_model(path_Sev_model)

path_Loc_model= os.path.join("Output","best_modelClustered_DB_Location_23Ago022_F.hdf5") #The path from which you want to inherit the weights
Model_Location = loading_model(architecture_info_Loc, training_info_Loc)
Model_Location.load_weights(path_Loc_model)





#Load and merge all the acceleration signals in one single array with dimensions: [Number of samples, length of the signal, number of sensors/channels]
#Evaluation of Test_Experimental (contains 20% of the experimental data)
Healthy_test_preds = Model_Severity.predict(Test_Experimental)
Healthy_test_preds = np.transpose(np.vstack((Healthy_test_preds.reshape(Healthy_test_preds.shape[0],),Healthy_test_preds.reshape(Healthy_test_preds.shape[0],))))

fig_name = "Original_DB_20Jul_Healthytest_location"
plot_healthy_measurements(std_model.inverse_transform(Healthy_test_preds)[:,1], fig_name)


######################## NEW TESTING 04 AGOSTO  ##############################
#SEVERITY

path1 = os.path.join('Data', 'Testing_damage_scenarios02Agos22', 'Testing_data_03Agos022.npy' )
path2 = os.path.join('Data', 'Testing_data_23Agos022.npy' )
test1 = np.load(path1)
test2 = np.load(path2)
new_test = np.vstack((test2[0:800,:], test1[800:,:]))
Data_syn_test = correct_mode_sign(new_test)

# new_test = np.load(path2)
# new_test= new_test[0:1200,:]


Model_Severity = model_severity
testing_synthetic_severity(Data_syn_test , Model_Severity,'SingleCluster_DB_Test24Agos_Severity_F1')





path1 = os.path.join('Data', 'Testing_damage_scenarios02Agos22', 'Testing_data_03Agos022.npy' )
new_test = np.load(path1)
Data_syn_test = correct_mode_sign(new_test)


Model_Location = model_location
alpha = 0.05
Damaged_syn_test = Data_syn_test[Data_syn_test[:,33]>alpha,:]
testing_synthetic_location(Damaged_syn_test,std_model, Model_Location,'SingleCluster_DB_Test24Agos_Location_F')






#TESTING WITH DATA FROM SAME EOCs
#Generate random test points from the training data
import random
Ytest = std_model.inverse_transform(Ytest_std)
Xtest = np.hstack((Xtest,Ytest))
Xtest = Xtest[Xtest[:,32]>0.05,:]
# X_test_samples =  np.array(random.sample(range(0, Xtest.shape[0]), 1200))


Xtest_samples = Xtest
Damaged_NoEOCs_test = Xtest

# Damaged_NoEOCs_test = Xtest[Xtest_samples,:]
testing_synthetic_location(Damaged_NoEOCs_test,std_model, Model_Location,'NoEOCs_Clustered_DB_Test23Agos_Location_F')

testing_synthetic_severity(Damaged_NoEOCs_test, Model_Severity,'NoEOCs_SingleCluster_DB_Test23Agos_Severity_F')





#TESTING WITH DATA FROM SAME EOCs
Model_Location =  model_location
#Generate random test points from the training data
import random
Ytest = std_model.inverse_transform(Ytest_std)
Xtest = np.hstack((Xtest,Ytest))
Xtest = Xtest[Xtest[:,32]>0.015,:]
X_test_samples =  np.array(random.sample(range(0, Xtest.shape[0]), 1200))
Damaged_NoEOCs_test = Xtest[X_test_samples,:]
testing_synthetic_location(Damaged_NoEOCs_test,std_model, Model_Location,'NoEOCs_SingleCluster_DB_Test12Agos_Location_F')


testing_synthetic_severity(Damaged_NoEOCs_test, Model_Severity,'NoEOCs_SingleCluster_DB_Test12Agos_Severity_F')


# testing_synthetic_severity(Damaged_NoEOCs_test, Model_Severity,'NoEOCs_Original_DB_Test04Agos_Severity_F')
# 


































# #Build and Train the model for LOCATION ESTIMATION
# architecture_info = architecture_info_initializer(
#     PCA_enc = "arch_PCA_encoder",
#     PCA_dec = "arch_PCA_decoder",
#     Res_enc = "arch_Residual_encoder_Location",
#     Res_dec = "arch_Residual_decoder_Location",
#     Compact_Z24 = "arch_compact_Z24",
#     input_dim = Input_dim,
#     enc_dim = 8,
#     output_dim = Output_dim)

# Xtrain, Xval, Xtest, Ytrain_std, Yval_std, Ytest_std = preprocessing_interface(bridge_loc) #We have to make slight modification in the Data load to remove the low severity scenarios during training
# model_location, history_location = training_model(architecture_info, training_info, Xtrain, Ytrain_std[:,0], Xval, Yval_std[:,0], f_name)
# Loc_train_predictions, Loc_val_predictions, Loc_test_predictions, Loc_train_rec_error, Loc_val_rec_error, Loc_test_rec_error = predictions_plots(model_location, Xtrain, Ytrain_std[:,0], Xval, Yval_std[:,0], Xtest, Ytest_std[:,0],f_name)


# print(datetime.datetime.now() - begin_time)
# #POSTPROCESSING (predictions and crossplots)

# from tensorflow.keras.models import load_model
# import os
# loc_model = os.path.join("Output","best_model.hdf5")
# my_model.load_weights(loc_model)

# # from MODULES.POSTPROCESSING.postprocessing import plot_loss_evolution
# # plot_loss_evolution(history,f_name)
# train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error = predictions_plots(my_model, Xtrain, Ytrain_std, Xval, Yval_std, Xtest, Ytest_std,std_model,f_name)

# #EVALUATE EXPERIMENTAL DAMAGE SCENARIOS

# import numpy as np
# import pandas as pd

#For Z24 bridge
# loc = os.path.join("Data","Z24_Input_dam.xlsx")
# Z24_exp_dam = np.array(pd.read_excel(loc, header = 0))
# Z24_dam_prediction2 = my_model.predict(Z24_exp_dam)
# Z24_dam_prediction = std_model.inverse_transform(my_model.predict(Z24_exp_dam))
# print(Z24_dam_prediction2)
 
# print ('-----')
# print(Z24_dam_prediction)
# pred_Points = Z24_dam_prediction2
# gt_Points = 

# #Save Experimental predictoins
# Exp_preds = Z24_dam_prediction;
# np.save(os.path.join('Output','Experimental_predictions_'+str(f_name)+'.npy'), Exp_preds)

# #For Infante bridge
# loc = os.path.join("Data","Infante_exp_test_7p.xlsx")
# Porto_exp_dam = np.array(pd.read_excel(loc, header = 0))
# Porto_Exp_dam = Porto_exp_dam[:,(2,3,18,19,20,21,22,23,24,25,26,27,28,29,30,31)]
# # Porto_Exp_dam = Porto_exp_dam[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)]
# Porto_dam_prediction2 = my_model.predict(Porto_Exp_dam)
# Porto_dam_prediction = std_model.inverse_transform(my_model.predict(Porto_Exp_dam))
# print(Porto_dam_prediction2)
# print ('-----')
# print(Porto_dam_prediction)

# #PRint Modeshapes
# X = [140.0,175.0,210.0,245.0]
# my_xticks = ['140.0','175.0','210.0','245.0']
# M1 = Porto_exp_dam[2,(4,6,8,10)]
# M2 = Porto_exp_dam[2,(11,13,15,17)]
# M3 = Porto_exp_dam[2,(18,20,22,24)]
# M4= Porto_exp_dam[2,(25,27,29,31)]
# from matplotlib import pyplot as plt
# plt.plot(X,M4,'ro-')
# plt.xticks(X, my_xticks)
# plt.xlabel('Axial coordinate (m)')
# plt.ylabel('Amplitude')
# plt.savefig(os.path.join("Figures",str(f_name)+"Mode4"),dpi = 500, bbox_inches='tight')

