# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:48:49 2020

@author: 109457
"""
import tensorflow as tf
from tensorflow.keras import layers

#create a custom layer (that applies an activation inside)
aux_layer = tf.keras.layers.Dense(2,activation = 'linear', name = 'scale_layer')
#We define the aux_layer to apply the activation before the subclassing (otherswise it raises error)
class Output_scale_layer(layers.Layer):
    def __init__(self):
        super(Output_scale_layer,self).__init__()
    # def build():
        
    def call(self, inputs):
        A = inputs
        B = aux_layer(A)
        C = B
        # ax+b la operacion de la layer densa
        
        print('HELLO')
        return C
    

# class MyDenseLayer(layers.Layer):
#   def __init__(self, num_outputs):
#     super(MyDenseLayer, self).__init__()
#     self.num_outputs = num_outputs


#   def call(self, input):
#       A = input[0]
#       tf.print(A)
#       tf.print('HOLAAA')
#       return input

# def output_layer (input,maxx,minn):
    
    
#     return tf.clip_by_value(input, minn, maxx)

def architecture_PCA_encoder(input_dim,enc_dim):
    input1 = tf.keras.Input(shape =(input_dim,), name = 'input_layer')
    output1 = tf.keras.layers.Dense(enc_dim, activation = 'linear', name = 'output1')(input1)
    return input1, output1


def architecture_PCA_decoder(input_dim, enc_dim):
    input2 = tf.keras.Input(shape =(enc_dim,), name = 'decoder_input')
    output2 =tf.keras.layers.Dense(1,activation = 'linear', name = 'output2')(input2)
    # output2 = tf.clip_by_value(output2a, 0.5, 1.5, name=None)
    return input2, output2

def architecture_PCA_decoder_rev(input_dim, enc_dim):
    input2 = tf.keras.Input(shape =(enc_dim,), name = 'decoder_input')
    output2 =tf.keras.layers.Dense(32,activation = 'linear', name = 'output2')(input2)
    # output2 = tf.clip_by_value(output2a, 0.5, 1.5, name=None)
    return input2, output2


def architecture_PCA_decoder_rev(input_dim, enc_dim):
    input2 = tf.keras.Input(shape =(enc_dim,), name = 'decoder_input')
    output2 =tf.keras.layers.Dense(32,activation = 'linear', name = 'output2')(input2)
    # output2 = tf.clip_by_value(output2a, 0.5, 1.5, name=None)
    return input2, output2

def architecture_Residual_encoder_Z24(input_dim, enc_dim):
    input3 = tf.keras.Input(shape =(input_dim,), name = 'residual_input')
    lay1 = layers.Dense(32, activation = 'relu', name = 'H1')(input3) #Intermediate layers
    lay2 = layers.Dense(24, activation = 'relu', name = 'H2')(lay1) #Intermediate layers
    lay3 = layers.Dense(36, activation = 'relu', name = 'H3')(lay2) #Intermediate layers
    lay4 = layers.Dense(24, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(36, activation = 'relu')(lay4) #Intermediate layers
    lay6 = layers.Dense(24, activation = 'relu')(lay5) #Intermediate layers
    output3 =tf.keras.layers.Dense(enc_dim,activation = 'linear' , name = 'output3')(lay6)
    return input3, output3


def architecture_Residual_decoder_Z24(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(24, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(16, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(24, activation = 'relu')(lay2) #Intermediate layers
    lay4 = layers.Dense(36, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(24, activation = 'relu')(lay4) #Intermediate layers
    lay6 = layers.Dense(6, activation = 'relu')(lay5) #Intermediate layers
    output4 =tf.keras.layers.Dense(2,activation = 'linear', name = 'output4')(lay6)
    # output4 = tf.clip_by_value(output4a, 0.5, 1.5, name=None)
    return input4, output4

#SINGLE CLUSTER ARCH
# def architecture_Residual_encoder_Severity(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(24, activation = 'relu', name ='L1')(input_s)
#     lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
#     # lay3 = layers.Dense(48, activation = 'relu', name = 'L3')(lay2)
#     # lay4 = layers.Dense(6, activation = 'relu', name = 'L4')(lay2)
#     lay5 = layers.Dense(60, activation = 'relu', name = 'L3')(lay2)
#     # lay4b = layers.Dense(86, activation = 'relu', name = 'L3')(lay4a)
#     # lay5 = layers.Dense(54, activation = 'relu', name = 'L5')(lay3)
#     lay6 = layers.Dense(16, activation = 'relu', name = 'L6')(lay5)
#     output_s = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     return input_s, output_s

# def architecture_Residual_decoder_Severity(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(16, activation = 'relu', name ='L1')(input_s)
#     # lay2 = layers.Dense(54, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(36, activation = 'relu', name = 'L3')(lay1)
#     # lay3a = layers.Dense(86, activation = 'relu', name = 'L3')(lay3)
#     # lay3b = layers.Dense(86, activation = 'relu', name = 'L3')(lay3a)
#     # lay4 = layers.Dense(48, activation = 'relu', name = 'L4')(lay3)
#     lay5 = layers.Dense(36, activation = 'relu', name = 'L5')(lay3)
#     lay6 = layers.Dense(12, activation = 'relu', name = 'L6')(lay5)
#     output_s1 = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     output_s =  layers.Dense(1, activation = 'sigmoid', name = 'Oputput__layer')(output_s1)
#     # output_s = tf.clip_by_value(output_s1,0,0.5, name='Clipping_fun')

#     return input_s, output_s

# CLustered DAtabase
def architecture_Residual_encoder_Severity(input_dim,enc_dim):
    input_s = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
    lay1 =layers.Dense(24, activation = 'relu', name ='L1')(input_s)
    lay2 = layers.Dense(116, activation = 'relu', name = 'L2')(lay1)
    # lay3 = layers.Dense(48, activation = 'relu', name = 'L3')(lay2)
    # lay4 = layers.Dense(6, activation = 'relu', name = 'L4')(lay2)
    lay5 = layers.Dense(160, activation = 'relu', name = 'L3')(lay2)
    # lay4b = layers.Dense(86, activation = 'relu', name = 'L3')(lay4a)
    # lay5 = layers.Dense(54, activation = 'relu', name = 'L5')(lay3)
    lay6 = layers.Dense(16, activation = 'relu', name = 'L6')(lay5)
    output_s = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
    return input_s, output_s

def architecture_Residual_decoder_Severity(input_dim,enc_dim):
    input_s = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
    lay1 =layers.Dense(16, activation = 'relu', name ='L1')(input_s)
    # lay2 = layers.Dense(54, activation = 'relu', name = 'L2')(lay1)
    lay3 = layers.Dense(116, activation = 'relu', name = 'L3')(lay1)
    # lay3a = layers.Dense(86, activation = 'relu', name = 'L3')(lay3)
    # lay3b = layers.Dense(86, activation = 'relu', name = 'L3')(lay3a)
    # lay4 = layers.Dense(48, activation = 'relu', name = 'L4')(lay3)
    lay5 = layers.Dense(36, activation = 'relu', name = 'L5')(lay3)
    lay6 = layers.Dense(12, activation = 'relu', name = 'L6')(lay5)
    output_s1 = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
    output_s =  layers.Dense(1, activation = 'sigmoid', name = 'Oputput__layer')(output_s1)
    # output_s = tf.clip_by_value(output_s1,0,0.5, name='Clipping_fun')

    return input_s, output_s

# def architecture_Residual_encoder_Severity(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(64, activation = 'relu', name ='L1')(input_s)
#     lay2 = layers.Dense(86, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(148, activation = 'relu', name = 'L3')(lay2)
#     # lay4 = layers.Dense(26, activation = 'relu', name = 'L4')(lay3)
#     # lay5 = layers.Dense(54, activation = 'relu', name = 'L5')(lay4)
#     lay6 = layers.Dense(36, activation = 'relu', name = 'L6')(lay3)
#     output_s = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     return input_s, output_s

# def architecture_Residual_decoder_Severity(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(96, activation = 'relu', name ='L1')(input_s)
#     lay2 = layers.Dense(136, activation = 'relu', name = 'L2')(lay1)
#     # lay3 = layers.Dense(86, activation = 'relu', name = 'L3')(lay2)
#     lay5 = layers.Dense(86, activation = 'relu', name = 'L5')(lay2)
#     lay6 = layers.Dense(12, activation = 'relu', name = 'L6')(lay5)
#     output_s1 = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     output_s =  layers.Dense(1, activation = 'sigmoid', name = 'Oputput__layer')(output_s1)
#     # output_s = tf.clip_by_value(output_s1,0,0.5, name='Clipping_fun')

#     return input_s, output_s

############################################################################################################
#Old working architectue
# def architecture_Residual_encoder_Location(input_dim,enc_dim):
#     input_l = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(16, activation = 'relu', name ='L1')(input_l)
#     lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(56, activation = 'relu', name = 'L3')(lay2)
#     lay4 = layers.Dense(36, activation = 'relu', name = 'L4')(lay3)
#     lay5 = layers.Dense(16, activation = 'relu', name = 'L5')(lay4)
#     output_l = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay5)
#     return input_l, output_l

# def architecture_Residual_decoder_Location(input_dim,enc_dim):
#     input_l = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(36, activation = 'relu', name ='L1')(input_l)
#     # lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(24, activation = 'relu', name = 'L3')(lay1)
#     # lay4 = layers.Dense(56, activation = 'relu', name = 'L4')(lay1)#
#     lay5 = layers.Dense(48, activation = 'relu', name = 'L5')(lay3)
#     lay6 = layers.Dense(36, activation = 'relu', name = 'L6')(lay5)
#     # output_1 = layers.Dense(16, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     # output_1 = layers.Dense(46, activation="sigmoid")(lay6)
#     # output_l = layers.Dense(8, activation="softmax")(output_1)
#     output_l = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     output_l = layers.Dense(1, activation="sigmoid")(output_l)
#     # output_l  = tf.clip_by_value(output_l,0.5,1.5, name='Clipping_fun_loc')
#     return input_l, output_l


#new working architecture 23 Agosto
def architecture_Residual_encoder_Location(input_dim,enc_dim):
    input_l = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
    lay1 =layers.Dense(16, activation = 'relu', name ='L1')(input_l)
    lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
    # lay3 = layers.Dense(16, activation = 'relu', name = 'L3')(lay2)
    lay4 = layers.Dense(36, activation = 'relu', name = 'L4')(lay2)
    lay5 = layers.Dense(16, activation = 'relu', name = 'L5')(lay4)
    output_l = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay5)
    return input_l, output_l

def architecture_Residual_decoder_Location(input_dim,enc_dim):
    input_l = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
    lay1 =layers.Dense(36, activation = 'relu', name ='L1')(input_l)
    # lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
    lay3 = layers.Dense(12, activation = 'relu', name = 'L3')(lay1)
    # lay4 = layers.Dense(56, activation = 'relu', name = 'L4')(lay1)#
    lay5 = layers.Dense(48, activation = 'relu', name = 'L5')(lay3)
    lay6 = layers.Dense(36, activation = 'relu', name = 'L6')(lay5)
    # output_1 = layers.Dense(16, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
    # output_1 = layers.Dense(46, activation="sigmoid")(lay6)
    # output_l = layers.Dense(8, activation="softmax")(output_1)
    output_l = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
    output_l = layers.Dense(1, activation="sigmoid")(output_l)
    # output_l  = tf.clip_by_value(output_l,0.5,1.5, name='Clipping_fun_loc')
    return input_l, output_l







# def architecture_Residual_encoder_Location(input_dim,enc_dim): CLUSTERED DB
#     input_l = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(16, activation = 'relu', name ='L1')(input_l)
#     # lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
#     # lay3 = layers.Dense(56, activation = 'relu', name = 'L3')(lay1)
#     lay4 = layers.Dense(36, activation = 'relu', name = 'L4')(lay1)
#     lay5 = layers.Dense(16, activation = 'relu', name = 'L5')(lay4)
#     output_l = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay5)
#     return input_l, output_l


# def architecture_Residual_decoder_Location(input_dim,enc_dim):
#     input_l = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(36, activation = 'relu', name ='L1')(input_l)
#     # lay2 = layers.Dense(36, activation = 'relu', name = 'L2')(lay1)
#     # lay3 = layers.Dense(24, activation = 'relu', name = 'L3')(lay2)
#     # lay4 = layers.Dense(56, activation = 'relu', name = 'L4')(lay1)#
#     # lay5 = layers.Dense(48, activation = 'relu', name = 'L5')(lay4)
#     lay6 = layers.Dense(36, activation = 'relu', name = 'L6')(lay1)
#     # output_1 = layers.Dense(16, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     # output_1 = layers.Dense(46, activation="sigmoid")(lay6)
#     # output_l = layers.Dense(8, activation="softmax")(output_1)
#     output_l = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay6)
#     output_l = layers.Dense(1, activation="sigmoid")(output_l)
#     # output_l  = tf.clip_by_value(output_l,0.5,1.5, name='Clipping_fun_loc')
#     return input_l, output_l






# def architecture_Residual_encoder_Location(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (input_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(24, activation = 'relu', name ='L1')(input_s)
#     lay2 = layers.Dense(12, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(12, activation = 'relu', name = 'L3')(lay2)
#     output_s = layers.Dense(enc_dim, activation = 'linear', name = 'Oputput_residual_layer')(lay3)
#     return input_s, output_s

# def architecture_Residual_decoder_Location(input_dim,enc_dim):
#     input_s = tf.keras.Input(shape = (enc_dim,), name = "residual_input_layer")
#     lay1 =layers.Dense(12, activation = 'relu', name ='L1')(input_s)
#     lay2 = layers.Dense(12, activation = 'relu', name = 'L2')(lay1)
#     lay3 = layers.Dense(24, activation = 'relu', name = 'L3')(lay2)
#     output_s = layers.Dense(1, activation = 'linear', name = 'Oputput_residual_layer')(lay3)
#     return input_s, output_s



def architecture_Residual_encoder_Porto(input_dim, enc_dim):
    input3 = tf.keras.Input(shape =(input_dim,), name = 'residual_input')
    lay1 = layers.Dense(24, activation = 'relu', name = 'H1')(input3) #Intermediate layers
    # lay2 = layers.Dense(36, activation = 'relu', name = 'H2')(lay1) #Intermediate layers
    # lay3 = layers.Dense(24, activation = 'relu', name = 'H3')(lay2) #Intermediate layers
    lay4 = layers.Dense(36, activation = 'relu')(lay1) #Intermediate layers
    lay5 = layers.Dense(12, activation = 'relu')(lay4) #Intermediate layers
    # lay6 = layers.Dense(24, activation = 'relu')(lay5) #Intermediate layers
    output3 =tf.keras.layers.Dense(enc_dim,activation = 'linear' , name = 'output3')(lay5)
    return input3, output3


def architecture_Residual_decoder_Porto(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(12, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(36, activation = 'relu')(lay1) #Intermediate layers
    # lay3 = layers.Dense(24, activation = 'relu')(lay2) #Intermediate layers
    # lay4 = layers.Dense(36, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(24, activation = 'relu')(lay2) #Intermediate layers
    lay6 = layers.Dense(12, activation = 'relu')(lay5) #Intermediate layers
    output4 =tf.keras.layers.Dense(1,activation = 'linear', name = 'output4')(lay6)
    # output4 = tf.clip_by_value(output4a, 0.5, 1.5, name=None)
    return input4, output4

def architecture_Residual_decoder_Porto_rev(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(12, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(36, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(24, activation = 'relu')(lay2) #Intermediate layers
    lay4 = layers.Dense(36, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(24, activation = 'relu')(lay4) #Intermediate layers
    lay6 = layers.Dense(12, activation = 'relu')(lay5) #Intermediate layers
    output4 =tf.keras.layers.Dense(32,activation = 'linear', name = 'output4')(lay6)
    # output4 = tf.clip_by_value(output4a, 0.5, 1.5, name=None)
    return input4, output4


def architecture_Residual_decoder_Porto_rev(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(12, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(36, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(48, activation = 'relu')(lay2) #Intermediate layers
    lay4 = layers.Dense(56, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(42, activation = 'relu')(lay4) #Intermediate layers
    lay6 = layers.Dense(40, activation = 'relu')(lay5) #Intermediate layers
    output4 =tf.keras.layers.Dense(32,activation = 'linear', name = 'output4')(lay6)
    # output4 = tf.clip_by_value(output4a, 0.5, 1.5, name=None)
    return input4, output4

#####################################################################################################3


#################################################################################
def architecture_Residual_encoder_Mexico(input_dim, enc_dim):
    input3 = tf.keras.Input(shape =(input_dim,), name = 'residual_input')
    lay1 = layers.Dense(8, activation = 'relu')(input3) #Intermediate layers
    lay2 = layers.Dense(12, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(16, activation = 'relu')(lay2) #Intermediate layers
    output3 =tf.keras.layers.Dense(enc_dim,activation = 'linear' , name = 'output3')(lay3)
    return input3, output3


def architecture_Residual_decoder_Mexico(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(16, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(12, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(8, activation = 'relu')(lay2) #Intermediate layers
    output4 =tf.keras.layers.Dense(input_dim,activation = 'linear', name = 'output4')(lay3)
    return input4,output4

#####################################################################################################

def architecture_Residual_encoder_FEMU_Z24(input_dim, enc_dim):
    input3 = tf.keras.Input(shape =(input_dim,), name = 'residual_input')
    lay1 = layers.Dense(30, activation = 'relu')(input3) #Intermediate layers
    lay2 = layers.Dense(18, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(enc_dim, activation = 'relu')(lay2) #Intermediate layers
    output3 =tf.keras.layers.Dense(enc_dim,activation = 'linear' , name = 'output3')(lay3)
    return input3,output3

def architecture_Residual_decoder_FEMU_Z24(input_dim, enc_dim):
    input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
    lay1 = layers.Dense(enc_dim, activation = 'relu')(input4) #Intermediate layers
    lay2 = layers.Dense(12, activation = 'relu')(lay1) #Intermediate layers
    lay3 = layers.Dense(6, activation = 'relu')(lay2) #Intermediate layers
    output4 =tf.keras.layers.Dense(2,activation = 'linear', name = 'output4')(lay3)
    # output4 = Output_scale_layer()(lay4)
    # output4 = Output_scale_layer()(lay3)
    return input4, output4

    # output4_2 = tf.keras.layers.Lambda(custom_layer,arguments={'maxx': 0, 'minn': 1} ,name="lambda_layer")(sec_var)

    # return input4,output4

# optional concat
# clipped_output = tf.concat([temperature, humidity], axis=1)

# def architecture_Residual_decoder_FEMU_Z24(input_dim, enc_dim):
#     input4 = tf.keras.Input(shape =(enc_dim,), name = 'residual_input')
#     lay1 = layers.Dense(enc_dim, activation = 'relu')(input4) #Intermediate layers
#     lay2 = layers.Dense(12, activation = 'relu')(lay1) #Intermediate layers
#     lay3 = layers.Dense(6, activation = 'relu')(lay2) #Intermediate layers
#     first_var=tf.keras.layers.Dense(1,activation = 'linear', name = 'first_var')(lay3)
#     sec_var = tf.keras.layer.Dense(1, activation = 'linear', name = 'sec_var')(lay3)
#     output4_1 = first_var

# def output_layer (input,maxx,minn):
        
        
#     return tf.clip_by_value(input, minn, maxx)
#     output4_2 = tf.keras.layers.Lambda(custom_layer,arguments={'maxx': 0, 'minn': 1} ,name="lambda_layer")(sec_var)

#     model = Model(inputs=input_layer,outputs=[y1_output, y2_output])

#     return input4,output4



def architecture_FEMU_Z24(input_dim,output_dim):
    input1= tf.keras.Input(shape =(input_dim,), name = 'input_layer')
    lay1 = layers.Dense(36,activation = 'relu')(input1)
    lay2 = layers.Dense(8,activation = 'relu')(lay1)
    lay3 = layers.Dense(24,activation = 'relu')(lay2)
    lay4 = layers.Dense(136,activation = 'relu')(lay3)
    lay5 = layers.Dense(48,activation = 'relu')(lay4)
    lay6 = layers.Dense(36,activation = 'relu')(lay5)    
    lay7 = layers.Dense(12,activation = 'relu')(lay6)
    output1 = layers.Dense(2,activation = 'linear', name = 'output_layer')(lay7)
    return input1,output1


#function para que me devuelva diccionario de arquitecturas
def arch_dictionary():
    #Definir diccionariode arquitecturas
    architecture_dictionary = {
        "arch_Residual_encoder_Mexico" : architecture_Residual_encoder_Mexico,
        "arch_Residual_decoder_Mexico" : architecture_Residual_decoder_Mexico,
        "arch_Residual_encoder_Porto" : architecture_Residual_encoder_Porto,
        "arch_Residual_decoder_Porto" : architecture_Residual_decoder_Porto,
        "arch_Residual_decoder_Porto_rev" : architecture_Residual_decoder_Porto_rev,
        "arch_Residual_encoder_Z24": architecture_Residual_encoder_Z24,
        "arch_Residual_decoder_Z24": architecture_Residual_decoder_Z24,
        "arch_PCA_encoder" :  architecture_PCA_encoder,
        "arch_PCA_decoder" :  architecture_PCA_decoder,
        "arch_PCA_decoder_rev" :  architecture_PCA_decoder_rev,        
        "arch_Residual_encoder_FEMU_Z24" : architecture_Residual_encoder_FEMU_Z24,
        "arch_Residual_decoder_FEMU_Z24" : architecture_Residual_decoder_FEMU_Z24,
        "arch_Residual_encoder_Severity" : architecture_Residual_encoder_Severity,
        "arch_Residual_decoder_Severity" : architecture_Residual_decoder_Severity,
        "arch_Residual_encoder_Location" : architecture_Residual_encoder_Location,
        "arch_Residual_decoder_Location" : architecture_Residual_decoder_Location,
        "arch_compact_Z24" : architecture_FEMU_Z24
        }
    return architecture_dictionary


