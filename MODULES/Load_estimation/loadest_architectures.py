#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:23:42 2025

@author: afernandez
"""

import tensorflow as tf
import tensorflow.keras as K 
import numpy as np 


# Load Estimator Architecture architecture: 
def Fully_connected_arch(num_points_sim, n_modes):
    input1 = K.Input(shape =(num_points_sim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(50, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(input1) #Intermediate layers
    lay2 = K.layers.Dense(100, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay1) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(500, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    lay2 = K.layers.Dense(20, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers

    Qpred_output = K.layers.Dense(num_points_sim*n_modes, activation = 'linear')(lay2)
    return K.Model(inputs = input1, outputs = Qpred_output)

# Load Estimator Architecture architecture: 
def Fully_connected_arch_singlenode(num_points_sim, n_modes):
    input1 = K.Input(shape =(num_points_sim,), name = 'Innnputlayer')
    lay1 = K.layers.Dense(50, activation = 'relu', kernel_initializer="he_uniform", bias_initializer="zeros",  name='lay1',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(input1) #Intermediate layers
    lay2 = K.layers.Dense(100, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay1) #Intermediate layers
    # lay2 = K.layers.Dense(300, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    # lay2 = K.layers.Dense(500, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers
    lay2 = K.layers.Dense(20, activation = 'relu',  kernel_initializer="he_uniform", bias_initializer="zeros")(lay2) #Intermediate layers

    Qpred_output = K.layers.Dense(num_points_sim, activation = 'linear')(lay2)
    return K.Model(inputs = input1, outputs = Qpred_output)

# --- LSTM Qpred Estimator Architecture ---
def LSTM_Qpred_Estimator(num_points_sim, n_modes):
    """
    Defines an LSTM-based model to estimate the Qpred tensor.
    Input is expected to be the time vector t_vector.
    Output will be Qpred of shape (batch_size, num_points_sim, n_modes).
    """
    # Input layer expecting the time vector of shape (num_points_sim,)
    time_vector_input = K.Input(shape=(num_points_sim,), name='time_vector_input')

    # Reshape the input to be suitable for RNN: (batch_size, num_points_sim, 1 feature)
    # The batch_size dimension is implicitly handled by Keras.
    reshaped_input = K.layers.Reshape((num_points_sim, 1), name='reshape_for_rnn')(time_vector_input)

    # Layer 1: LSTM
    # return_sequences=True is crucial to get output for each time step for the next layer.
    lstm_layer1 = K.layers.LSTM(
        units=64,
        activation='tanh', # Default LSTM activation
        recurrent_activation='sigmoid', # Default LSTM recurrent activation
        kernel_initializer='glorot_uniform', # Default
        recurrent_initializer='orthogonal', # Default
        bias_initializer='zeros',
        return_sequences=True,
        name='lstm_layer1'
    )(reshaped_input)
    # Optional: Add BatchNormalization or Dropout here if needed
    # lstm_layer1 = K.layers.BatchNormalization()(lstm_layer1)
    # lstm_layer1 = K.layers.Dropout(0.1)(lstm_layer1)

    # Layer 2: LSTM
    lstm_layer2 = K.layers.LSTM(
        units=64,
        activation='tanh',
        recurrent_activation='sigmoid',
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        return_sequences=True,
        name='lstm_layer2'
    )(lstm_layer1)
    # lstm_layer2 = K.layers.BatchNormalization()(lstm_layer2)
    # lstm_layer2 = K.layers.Dropout(0.1)(lstm_layer2)

    # # Layer 3: LSTM (optional, deeper model)
    # # The last RNN layer before TimeDistributed Dense must also have return_sequences=True
    # lstm_layer3 = K.layers.LSTM(
    #     units=32,
    #     activation='tanh',
    #     recurrent_activation='sigmoid',
    #     kernel_initializer='glorot_uniform',
    #     recurrent_initializer='orthogonal',
    #     bias_initializer='zeros',
    #     return_sequences=True, # Must be true for TimeDistributed Dense
    #     name='lstm_layer3'
    # )(lstm_layer2)
    # lstm_layer3 = K.layers.BatchNormalization()(lstm_layer3)
    # lstm_layer3 = K.layers.Dropout(0.1)(lstm_layer3)

    # Output Layer: TimeDistributed Dense
    # This applies a Dense layer to each time step of the LSTM output.
    # The Dense layer will have n_modes units to predict modal forces for each mode.
    # We use kernel_initializer and bias_initializer as in your example FNN.
    q_pred_sequence_output = K.layers.TimeDistributed(
        K.layers.Dense(
            units=n_modes,
            activation='linear', # For regression of force values
            kernel_initializer="he_uniform", # As per your FNN example
            bias_initializer="zeros",        # As per your FNN example
            kernel_regularizer=tf.keras.regularizers.l2(0.001) # As per your FNN example
        ),
        name='q_pred_output_td'
    )(lstm_layer2) # Taking output from the last LSTM layer

    return K.Model(inputs=time_vector_input, outputs=q_pred_sequence_output, name="LSTM_Qpred_Model")

# --- GRU Qpred Estimator Architecture ---
def GRU_Qpred_Estimator(num_points_sim, n_modes):
    """
    Defines a GRU-based model to estimate the Qpred tensor.
    Input is expected to be the time vector t_vector.
    Output will be Qpred of shape (batch_size, num_points_sim, n_modes).
    """
    time_vector_input = K.Input(shape=(num_points_sim,), name='time_vector_input')

    reshaped_input = K.layers.Reshape((num_points_sim, 1), name='reshape_for_rnn')(time_vector_input)

    # Layer 1: GRU
    gru_layer1 = K.layers.GRU(
        units=128,
        activation='tanh', # Default GRU activation
        recurrent_activation='sigmoid', # Default GRU recurrent activation
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        return_sequences=True,
        name='gru_layer1'
    )(reshaped_input)

    # Layer 2: GRU
    gru_layer2 = K.layers.GRU(
        units=64,
        activation='tanh',
        recurrent_activation='sigmoid',
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        return_sequences=True,
        name='gru_layer2'
    )(gru_layer1)

    # Layer 3: GRU
    gru_layer3 = K.layers.GRU(
        units=32,
        activation='tanh',
        recurrent_activation='sigmoid',
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        return_sequences=True, # Must be true for TimeDistributed Dense
        name='gru_layer3'
    )(gru_layer2)

    # Output Layer: TimeDistributed Dense
    q_pred_sequence_output = K.layers.TimeDistributed(
        K.layers.Dense(
            units=n_modes,
            activation='linear',
            kernel_initializer="he_uniform",
            bias_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        name='q_pred_output_td'
    )(gru_layer3)

    return K.Model(inputs=time_vector_input, outputs=q_pred_sequence_output, name="GRU_Qpred_Model")


# # --- Example Usage ---
# num_points_sim_example = 100 # Example value from your Newmark_beta_solver context
# n_modes_example = 5         # Example value (n_m)

# # Create the LSTM model
# lstm_model = LSTM_Qpred_Estimator(num_points_sim_example, n_modes_example)
# print("\n--- LSTM Model Summary ---")
# lstm_model.summary()

# # Create the GRU model
# gru_model = GRU_Qpred_Estimator(num_points_sim_example, n_modes_example)
# print("\n--- GRU Model Summary ---")
# gru_model.summary()

# # --- How to call the model ---
# # 1. Prepare your t_vector:
# # It should be a 1D NumPy array or TensorFlow tensor of shape (num_points_sim_example,)
# t_vector_example_np = np.linspace(0, 1, num_points_sim_example, dtype=np.float32)

# # 2. If you have a single t_vector for prediction, add a batch dimension:
# t_vector_batch = tf.expand_dims(t_vector_example_np, axis=0) # Shape: (1, num_points_sim_example)
# print(f"\nShape of t_vector_batch input: {t_vector_batch.shape}")

# # 3. Make a prediction:
# Qpred_output_lstm = lstm_model(t_vector_batch) # Output shape: (1, num_points_sim_example, n_modes_example)
# Qpred_output_gru = gru_model(t_vector_batch)   # Output shape: (1, num_points_sim_example, n_modes_example)

# print(f"Shape of LSTM model output (Qpred_output_lstm): {Qpred_output_lstm.shape}")
# print(f"Shape of GRU model output (Qpred_output_gru): {Qpred_output_gru.shape}")

# # 4. If you need Qpred for your Newmark_beta_solver (which expects (num_points_sim, n_m)),
# #    and you predicted for a single t_vector (batch size 1), squeeze the batch dimension:
# Qpred_for_solver_lstm = tf.squeeze(Qpred_output_lstm, axis=0) # Shape: (num_points_sim_example, n_modes_example)
# Qpred_for_solver_gru = tf.squeeze(Qpred_output_gru, axis=0)   # Shape: (num_points_sim_example, n_modes_example)

# print(f"Shape of Qpred_for_solver_lstm (after squeeze): {Qpred_for_solver_lstm.shape}")
# print(f"Shape of Qpred_for_solver_gru (after squeeze): {Qpred_for_solver_gru.shape}")

# # Now Qpred_for_solver_lstm or Qpred_for_solver_gru can be used in your Newmark_beta_solver,
# # assuming it matches the expected dtype (e.g., tf.float64). You might need tf.cast.
# # Qpred_for_solver_lstm = tf.cast(Qpred_for_solver_lstm, dtype=tf.float64)