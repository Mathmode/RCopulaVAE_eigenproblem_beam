# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:37 2020

@author: 109457
"""
import numpy as np
import os
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration, plot_loss_evolution, plot_histogram, plot_predicted_values_vs_ground_truth, plot_crossplots, plot_outliers_dam
from MODULES.POSTPROCESSING.postprocessing_tools import cumulated_errors, calculate_errors, calculate_metrics  

#Llamar a la función de configruación de gráficos para tamaños/fuentes
plot_configuration()
########################################################################################################################################

class postprocessing_info_initializer():
    #definicion e inicialización de las variables que van dentro 
    def __init__(self, k = 1, dam_level = 0.1, percentile = 99):        
        self.k = k
        self.Dam_level = dam_level
        self.Percentile = percentile
        if k == None or dam_level == None or percentile == None:
            print("************************************************************************")
            print("Please initialize the info for POSTPROCESSING!")
            print("************************************************************************")
            quit()

##############################################################################################################################3

def make_predictions(my_model, Xtrain, Ytrain_std, Xval, Yval_std, Xtest, Ytest_std):
    #RECONSTRUCTION ERROR AS THE SINGLE VALUE DAMAGE INDICATOR 
    #FROM THIS DI WE THEN CALCULATE THE METRICS AND COMPARE METRICS TABLE
    train_predictions = my_model.predict(Xtrain)  
    train_rec_error = calculate_errors(Ytrain_std,train_predictions)
    captured = (1-np.mean(train_rec_error))*100
    val_predictions = my_model.predict(Xval)
    val_rec_error = calculate_errors(Yval_std, val_predictions)
    test_predictions = my_model.predict(Xtest)
    test_rec_error = calculate_errors(Ytest_std, test_predictions)
    print('% of information captured\n', captured)
    print('Train reconstrunction error\n', np.mean(train_rec_error))
    print('Test reconstrunction error\n', np.mean(test_rec_error), np.mean(val_rec_error))

    return train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error


def make_plots(Ytrain_std, Yval_std, Ytest_std, train_predictions, val_predictions, test_predictions,std_model, f_name):
    #Plot CROSSPLOTS (Ground truth vs Predictions)
    plot_crossplots(std_model.inverse_transform(Ytrain_std)[:,0], std_model.inverse_transform(Yval_std)[:,0], std_model.inverse_transform(Ytest_std)[:,0],std_model.inverse_transform(train_predictions)[:,0], std_model.inverse_transform(val_predictions)[:,0], std_model.inverse_transform(test_predictions)[:,0],f_name)
    # plot_crossplots(Ytrain_std[:,0], Yval_std[:,0], Ytest_std[:,0], train_predictions[:,0], val_predictions[:,0],test_predictions[:,0],f_name)


def predictions_plots(my_model, Xtrain, Ytrain_std, Xval, Yval_std, Xtest, Ytest_std,std_model,f_name):
    train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error = make_predictions(my_model, Xtrain, Ytrain_std, Xval, Yval_std, Xtest, Ytest_std)
    train_predictions = train_predictions.reshape(train_predictions.shape[0],)
    val_predictions = val_predictions.reshape(val_predictions.shape[0],)
    test_predictions = test_predictions.reshape(test_predictions.shape[0],)
    
    train_predictions = np.transpose(np.vstack((train_predictions,train_predictions)))
    val_predictions =  np.transpose(np.vstack((val_predictions,val_predictions)))
    test_predictions = np.transpose(np.vstack((test_predictions,test_predictions)))
    # Pred_train = np.concatenate((Ytrain_std, np.hstack((train_predictions,train_predictions))), axis = 0 )
    # Pred_val = np.concatenate((Yval_std,np.hstack((val_predictions))),axis = 0)
    # Pred_test = np.concatenate((Ytest_std,np.hstack((test_predictions))),axis = 0)
    make_plots(Ytrain_std, Yval_std, Ytest_std,train_predictions,val_predictions, test_predictions,std_model,f_name)
    # np.save(os.path.join('Output','train_preds'+str(f_name)+'.npy'), Pred_train)
    # np.save(os.path.join('Output','val_preds'+str(f_name)+'.npy'), Pred_val)
    # np.save(os.path.join('Output','test_preds'+str(f_name)+'.npy'),Pred_test)
    ##############################for Single Output######################################################################
    # Pred_train = np.concatenate((Ytrain_std, train_predictions), axis = 1)
    # Pred_val = np.concatenate((Yval_std,val_predictions),axis = 1)
    # Pred_test = np.concatenate((Ytest_std,test_predictions),axis = 1)
    # np.save(os.path.join('Output','train_preds'+str(f_name)+'.npy'), Pred_train)
    # np.save(os.path.join('Output','val_preds'+str(f_name)+'.npy'), Pred_val)
    # np.save(os.path.join('Output','test_preds'+str(f_name)+'.npy'),Pred_test)

    return  train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error


