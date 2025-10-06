# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:18:33 2020

@author: 109457
"""

#import packages
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd 

#IMPORT DATA
def import_data(loc):
   df = np.array(pd.read_excel(loc,nrows = None))
   return df
#1) remove zero elements (whole row) from the data
def rem_zeros(Data):
    Xdata  = Data[np.all(Data != 0.0, axis=1)]
    Xdata = np.array(Xdata)
    return Xdata

#3) Standardize the data to zero mean and unit standard deviation by simply using the standard Scaler Operator 
#inputs:
    #Xtrain: is the dataset to fit/train the standardization model (means and stds are calculated from this dataset)
    #X: is any dataset to BE standardized based on the scaler model (standardization)
#Outputs: 
    #Xstd: is the desired standardized dataset  X. It may be both Xtrain and Xtest in DL
def standardization(Xtrain): 
    scaler = StandardScaler()
    std_model = scaler.fit(Xtrain)
    return std_model
#4) This is a different type of scaling: rescaling into a certain range 
def rescaling(Xtrain,lb,ub):
    scaler = MinMaxScaler(feature_range = (lb,ub))
    rescaling_model = scaler.fit(Xtrain)
    return rescaling_model

def mode_UnitNormalizer(modes,nmodes,ncoord):
    #for nm = 4 eigenmodes of ncoord = 7 coordinates
     for i in range(modes.shape[0]):
         for j in range(nmodes):
             mode = modes[i,j*ncoord:ncoord*(j+1)]
             mode_abs = np.abs(mode)
             Max_coord = np.where(mode_abs == max(mode_abs))
             Max_value = mode[Max_coord[0][0]]
             mode_norm = mode/Max_value
             modes[i,j*ncoord:ncoord*(j+1)] = mode_norm
     modes_norm = modes
     return modes_norm

