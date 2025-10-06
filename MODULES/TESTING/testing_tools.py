# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:45:13 2022

@author: 110137
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration,plot_predicted_values_vs_ground_truth, plot_synthetic_testing_sev, plot_synthetic_testing_loc
import seaborn
#Llamar a la función de configruación de gráficos para tamaños/fuentes
plot_configuration()


def testing_synthetic_severity(Data, Model_Severity, fname):
    #IMPLEMENT THE METHOD TO FIRST ESTIMATE SEVERITY AND IN CASE S > THRESHOLD --> ESTIMATE LOCATION
    Sev_predictions =  Model_Severity.predict(Data[:,0:32])/2  #We divide by 2 for fast inverse transformation of severity scaling
    GT_severity = Data[:,33].reshape(Sev_predictions.shape)
    plot_synthetic_testing_sev(GT_severity,Sev_predictions,fname)
    seaborn.set(style = 'white', font_scale = 1.3)
    plot_predicted_values_vs_ground_truth(GT_severity[:,0],Sev_predictions[:,0], '', str(fname)+'_crossplotmap')

def testing_synthetic_location(Data,std_model, Model_Location, fname):
    #IMPLEMENT THE METHOD TO FIRST ESTIMATE SEVERITY AND IN CASE S > THRESHOLD --> ESTIMATE LOCATION
    Loc_predictions =  Model_Location.predict(Data[:,0:32]) 
    Loc_predictions = std_model.inverse_transform(np.hstack((Loc_predictions, Loc_predictions)))
    GT_Location = Data[:,32].reshape(Loc_predictions.shape[0],1)
    plot_synthetic_testing_loc(GT_Location,Loc_predictions,fname)
    seaborn.set(style = 'white', font_scale = 1.3)
    plot_predicted_values_vs_ground_truth(GT_Location[:,0],Loc_predictions[:,0], '', str(fname)+'_crossplotmap')

