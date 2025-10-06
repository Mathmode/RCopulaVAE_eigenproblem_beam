# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:45:54 2020

@author: 109457
"""
import tensorflow as tf
from tensorflow.keras import layers

#Create the autoencoder from the two Submodels (encoder, decoder)

def PCA_model_creation(input_layer, linear_encoder, linear_decoder):
    output_encoder = linear_encoder(input_layer)
    output_decoder = linear_decoder(output_encoder)
    modelPCA_autoencoder = tf.keras.Model(inputs = input_layer, outputs = output_decoder, name = 'autoencoder')
    return modelPCA_autoencoder

####################################################################################

#create a custom layer (that applies an activation inside)
aux_layer = tf.keras.layers.Dense(2,activation = 'sigmoid', name = 'scale_layer')
#We define the aux_layer to apply the activation before the subclassing (otherswise it raises error)
class Output_scale_layer(layers.Layer):
    def __init__(self):
        super(Output_scale_layer,self).__init__()

    def call(self, inputs):
        A = inputs
        B = aux_layer(A)
        C = B + 0.5
        print('HELLO')
        return C


#define autoencoder models for possibly sharing the encoder before entering the decoder

#Autoenocder residual that ouputs the complete autoencoder and the encoder 
# def ResNet_model_creation(input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder):
#     output_linear_encoder = linear_encoder(input_layer)
#     output_nonlinear_encoder = nonlinear_encoder(input_layer)
#     modelResNet_encoder = tf.keras.Model(inputs = input_layer, outputs = output_linear_encoder + output_nonlinear_encoder, name = 'Encoder')
#     output_encoder = modelResNet_encoder(input_layer)
#     output_linear_decoder = linear_decoder(output_encoder)
#     output_nonlinear_decoder = nonlinear_decoder(output_encoder)
#     output_autoencoder = output_linear_decoder + output_nonlinear_decoder
#     # output_autoencoder  = tf.clip_by_value(output_autoencoder, 0.5, 1.5, name="Clipping output")
#     modelResNet_autoencoder = tf.keras.Model(inputs = input_layer, outputs = output_autoencoder, name = 'autoencoder2')
#     return modelResNet_autoencoder, modelResNet_encoder



def ResNet_Severity_model_creation(input_layer,nonlinear_encoder,nonlinear_decoder):
    output_nonlinear_encoder = nonlinear_encoder(input_layer)
    output_autoencoder  = nonlinear_decoder(output_nonlinear_encoder)
    model_ResNet_Severity = tf.keras.Model(inputs = input_layer, outputs = output_autoencoder, name = "autoencoder_Sev")
    return model_ResNet_Severity


def ResNet_Location_model_creation(input_layer,nonlinear_encoder,nonlinear_decoder):
    output_nonlinear_encoder = nonlinear_encoder(input_layer)
    output_autoencoder  = nonlinear_decoder(output_nonlinear_encoder)
    model_ResNet_Location = tf.keras.Model(inputs = input_layer, outputs = output_autoencoder, name = "autoencoder_Sev")
    return model_ResNet_Location


def ResNet_model_creation(input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder):
    output_linear_encoder = linear_encoder(input_layer)
    output_nonlinear_encoder = nonlinear_encoder(input_layer)
    output_encoder = output_linear_encoder + output_nonlinear_encoder
    output_linear_decoder = linear_decoder(output_encoder)
    output_nonlinear_decoder = nonlinear_decoder(output_encoder)
    output_autoencoder = output_linear_decoder + output_nonlinear_decoder
    # output_autoencoder  = tf.clip_by_value(output_autoencoder1, 0, 0.5, name="Clipping output")
    output_autoencoder =  layers.Dense(1, activation = 'sigmoid', name = 'Oputput_layer_s')(output_autoencoder)
    modelResNet_autoencoder = tf.keras.Model(inputs = input_layer, outputs = output_autoencoder, name = 'autoencoder')
    return modelResNet_autoencoder

##########################################################################################################
def Compact_model_creation(input_layer, compact_model):
    output_model = compact_model(input_layer)
    Z24_compact_model = tf.keras.Model(inputs = input_layer, outputs = output_model, name = "compact_model")
    return Z24_compact_model
