#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v1/GC_KSmix_architectures.py

Encoder of the PI-GC-KSmix-VAE (same 128-256-128 ReLU network as review29oct). It returns raw outputs;
Inverse_Copula_KS_Model (GC_KSmix_models.py) maps them to valid parameters.

Changes with respect to MODULES/review29oct/GC_KSmix_architectures.py:
  * a, b = 1 + exp(7 tanh(raw / 7)), in (1.001, 1100): smooth, no hard clip (the old clip_by_value at
    [1.01, 200] has zero gradient outside the box), and it reaches the large values that narrow marginals
    need (a std of 0.01 in z near z = 1 needs a ~ 100);
  * the copula head starts at zero, i.e. at the independence copula R = I.
"""

import tensorflow as tf

from MODULES.review29oct_fixed_v1 import tf_keras

K = tf_keras()


def Fully_connected_enc_GC_KS(input_dim, n_dims, num_KS):
    input1 = K.Input(shape=(input_dim,), name="InputLayer")

    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(input1)
    x = K.layers.Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = K.layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    a_raw = K.layers.Dense(n_dims * num_KS, activation="linear", name="a_ks")(x)
    b_raw = K.layers.Dense(n_dims * num_KS, activation="linear", name="b_ks")(x)
    weight_logits = K.layers.Dense(n_dims * num_KS, activation="linear", name="weights")(x)

    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L = K.layers.Dense(n_correlations, activation="linear", kernel_initializer="zeros", name="off_diag")(x)
    diag_raw = K.layers.Dense(n_dims, activation="linear", name="diag")(x)

    outputs = K.layers.Concatenate(axis=1)([a_raw, b_raw, weight_logits, off_diag_L, diag_raw])
    return K.Model(inputs=input1, outputs=outputs)


def positive_shape_parameter(raw):
    """1 + exp(7 tanh(raw / 7)): smooth map to (1.0009, 1097.6)."""
    return 1.0 + tf.math.exp(7.0 * tf.math.tanh(raw / 7.0))
