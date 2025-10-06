# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:25:20 2022

@author: 110137
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

def my_MSE(y_actual,y_pred):
    loss = tf.math.reduce_mean(tf.math.square(y_actual - y_pred), axis= None)
    tf.print(loss)
    return loss


def custom_loss_Location(y_actual,y_pred):
    aux_var = tf.math.square(y_actual - y_pred)
    loss_zone1 = tf.math.reduce_mean(aux_var[:,0], axis= None)
    return loss_zone1/y_pred.shape[1]

def custom_loss_Severity(y_actual,y_pred):
    aux_var = tf.math.square(y_actual - y_pred)
    loss_zone2 = tf.math.reduce_mean(aux_var[:,1], axis= None)
    return loss_zone2/y_pred.shape[1]


def custom_loss(y_actual,y_pred):
    aux_var = tf.math.square(y_actual - y_pred)
    # tf.print(aux_var.shape)
    loss_Location = tf.math.reduce_mean(aux_var[:,0], axis= None)
    loss_Severity = tf.math.reduce_mean(aux_var[:,1], axis= None)
    loss = (loss_Location + loss_Severity)/y_pred.shape[1]
    return loss


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