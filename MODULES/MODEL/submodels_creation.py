# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:52:52 2020

@author: 109457
"""
import tensorflow as tf
from MODULES.MODEL.architectures import arch_dictionary

###########################################################################################
 
class architecture_info_initializer():
    #definicion e inicialización de las variables q van dentro 
    def __init__(self, PCA_enc = "arch_PCA_encoder", PCA_dec = "arch_PCA_decoder", 
                 Res_enc = "arch_Residual_encoder_Porto", Res_dec = "arch_Residual_decoder_Porto", Compact_Z24 = "arch_compact_Z24", 
                 input_dim = None, enc_dim = None, output_dim = None):        
        self.PCA_encoder = PCA_enc
        self.PCA_decoder = PCA_dec
        self.Residual_encoder = Res_enc
        self.Residual_decoder = Res_dec
        self.Compact_model = Compact_Z24
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.output_dim = output_dim
        if input_dim == None or enc_dim == None:
            print("************************************************************************")
            print("Please initialize input_dim and enc_dim in the architecture_info object!")
            print("************************************************************************")
            quit()
    # Definición de un método (operacion a hacer q es una funcion)
    # def function(self):
    #     print("PCA_enc = " , self.PCA_encoder)
    # return

############################################################################################

def submodel_features(arch_info):
    arch_d = arch_dictionary()
    if arch_info.PCA_encoder == None:
        Input_compact_model, Output_compact_model = arch_d[arch_info.Compact_model](arch_info.input_dim,arch_info.output_dim)
        architecture_list = Input_compact_model, Output_compact_model
    else:
        Input_linear_encoder, Output_linear_encoder = arch_d[arch_info.PCA_encoder](arch_info.input_dim,arch_info.enc_dim)
        Input_linear_decoder, Output_linear_decoder = arch_d[arch_info.PCA_decoder](arch_info.input_dim,arch_info.enc_dim)
        Input_residual_encoder, Output_residual_encoder = arch_d[arch_info.Residual_encoder](arch_info.input_dim,arch_info.enc_dim)
        Input_residual_decoder,Output_residual_decoder = arch_d[arch_info.Residual_decoder](arch_info.input_dim,arch_info.enc_dim)
        architecture_list = [Input_linear_encoder, Output_linear_encoder, Input_linear_decoder, Output_linear_decoder, 
                                 Input_residual_encoder, Output_residual_encoder, Input_residual_decoder, Output_residual_decoder]
    return architecture_list


############################################################################################

#create the submodels 
def submodels_creation(architecture_list):
    input_layer = architecture_list[0]
    linear_encoder =  tf.keras.Model(input_layer, architecture_list[1], name = 'encoder')
    linear_decoder = tf.keras.Model(architecture_list[2],architecture_list[3], name = 'decoder')
    nonlinear_encoder = tf.keras.Model(architecture_list[4], architecture_list[5], name = 'first_nonlinear_path')
    nonlinear_decoder = tf.keras.Model(architecture_list[6], architecture_list[7], name = 'second_nonlinear_path')
    return input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder

def submodels(arch_info):
    #llamar a submodel features
    architecture_list = submodel_features(arch_info)
    #llamar submodels_Creation
    input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder = submodels_creation(architecture_list)
    return input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder

def compact_models(arch_info):
    architecture_list = submodel_features(arch_info)
    input_layer = architecture_list[0]
    compact_model = tf.keras.Model(input_layer,architecture_list[1], name = 'Compact_model')
    return input_layer,compact_model