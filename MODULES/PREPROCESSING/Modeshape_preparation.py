#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:20 2022

@author: ana
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.interpolate import UnivariateSpline

def Load_ModeFreq_data(path):
    #Load and merge all the acceleration signals in one single array with dimensions: [Number of samples, length of the signal, number of sensors/channels]
    All_files = []
    for i in os.listdir(path):
        path_i = os.path.join(path,i)
        data = np.load(path_i)
        All_files.append(data)
    
    ModesFreqs = np.array(All_files)
    return ModesFreqs

def Mode_shape_preparation(ModesFreqs):
    Modes = ModesFreqs[:,1:,:]
    Freqs = ModesFreqs[:,0,:]
    npoints = 7
    ntotal = Modes.shape[2]*npoints + Modes.shape[2]
    new_Modes = np.zeros((Modes.shape[0],npoints,Modes.shape[2]), dtype = complex)
    Exp_Input = np.zeros((Freqs.shape[0], ntotal))
    x = np.array([140,175,210,245])
    for i in range(Modes.shape[0]):
        for j in range(Modes.shape[2]):
            y = Modes[i,:,j]
            spl_model = UnivariateSpline(x,y)
            new_points = spl_model([157.5,192.5,227.5])
            new_mode = np.array([Modes[i,0,j],  new_points[0], Modes[i,1,j], new_points[1], Modes[i,2,j], new_points[2], Modes[i,3,j]])
            new_Modes[i,:,j] = new_mode
        # new_Modes = new_Modes.reshape((new_Modes.shape[0],new_Modes.shape[1]*new_Modes.shape[2]))   
        Exp_Input[i,:] = np.concatenate([Freqs[i,:], new_Modes[i,:,:].reshape((new_Modes[i,:,:].shape[0]*new_Modes[i,:,:].shape[1]))]) 
    
    
    return Exp_Input
      

# Exp_Input = Mode_shape_prepartion(Modes,Freqs)
### how to generate 7-points mode shapes from 4-dimensional eigenmodes
#WE use spline to add the extra intermediate points
def Mode_shapes_7points(Data_Modes):
    # Data_Modes = np.load(os.path.join('Data','Data_modes.npy'))
    Modes = Data_Modes[:,4:20]
    x = np.array([0,1,2,3])
    M1 = Modes[:,0:4]
    M2 = Modes[:,4:8]
    M3 = Modes[:,8:12]
    M4 = Modes[:,12:16]
    M1_7points = np.zeros((M1.shape[0],8)) #We initialize eight-dimensional to accomodate the loop operations and then remove the last point
    M2_7points = np.zeros((M1.shape[0],8))
    M3_7points = np.zeros((M1.shape[0],8))
    M4_7points = np.zeros((M1.shape[0],8))
    
    for i in range (M1_7points.shape[0]):
        for j in range (M1.shape[1]):
            spl = UnivariateSpline(x, M1[i,:])
            M1_7points[i,2*j] = M1[i,j]
            M1_7points[i,2*j+1] = spl(j+0.5)
    
    for i in range (M2_7points.shape[0]):
        for j in range (M2.shape[1]):
            spl = UnivariateSpline(x, M2[i,:])
            M2_7points[i,2*j] = M2[i,j]
            M2_7points[i,2*j+1] = spl(j+0.5)
    
    for i in range (M3_7points.shape[0]):
        for j in range (M3.shape[1]):
            spl = UnivariateSpline(x, M3[i,:])
            M3_7points[i,2*j] = M3[i,j]
            M3_7points[i,2*j+1] = spl(j+0.5)
                    
    
    for i in range (M4_7points.shape[0]):
        for j in range (M4.shape[1]):
            spl = UnivariateSpline(x, M4[i,:])
            M4_7points[i,2*j] = M4[i,j]
            M4_7points[i,2*j+1] = spl(j+0.5)
    
    Data_Modes_11Jul022 = np.concatenate((Data_Modes[:,0:4],M1_7points[:,0:7], M2_7points[:,0:7], M3_7points[:,0:7], M4_7points[:,0:7]), axis = 1)
    np.save(os.path.join('Data', 'Data_Modes_11Jul022.npy'), Data_Modes_11Jul022)
