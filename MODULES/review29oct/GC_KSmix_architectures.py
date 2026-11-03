#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/GC_KSmix_architectures.py

Residual LayerNorm Encoder Architecture & Copula-Kumaraswamy Sampling Layer.
"""

import tensorflow as tf
import tensorflow.keras as K
from MODULES.review29oct.GC_KSmix_functions import (
    build_correlation_matrices_from_cholesky,
    gaussian_copula_samples,
    ks_mixture_marginal_samples,
)


def Fully_connected_enc_GC_KS(input_dim, n_dims, num_KS, lbound):
    """
    Encoder network mapping modal observations [freqs, vert_modes, rot_modes]
    to Kumaraswamy Mixture parameters (a, b, weight_logits) and Gaussian Copula
    Cholesky parameters (off_diag, diag).
    """
    inp = K.Input(shape=(input_dim,), name="Modal_Input_Layer")

    x = K.layers.Dense(256, activation="gelu", kernel_initializer="he_normal")(inp)
    x = K.layers.LayerNormalization()(x)

    # Residual block 1
    h1 = K.layers.Dense(256, activation="gelu", kernel_initializer="he_normal")(x)
    h1 = K.layers.Dense(256, activation="linear", kernel_initializer="he_normal")(h1)
    x = K.layers.Activation("gelu")(K.layers.Add()([x, h1]))
    x = K.layers.LayerNormalization()(x)

    # Bottleneck feature representation
    x = K.layers.Dense(192, activation="gelu", kernel_initializer="he_normal")(x)

    # 1. Kumaraswamy shape parameters a, b in [1.15, 45.0]
    # Keeping upper bound <= 45.0 prevents needle-like density collapse and ensures well-calibrated UQ
    a_raw = K.layers.Dense(n_dims * num_KS, activation="softplus", name="a_ks_raw")(x)
    b_raw = K.layers.Dense(n_dims * num_KS, activation="softplus", name="b_ks_raw")(x)

    a_params = K.layers.Lambda(lambda t: tf.clip_by_value(t + 1.15, 1.15, 45.0), output_shape=lambda s: s, name="a_ks")(a_raw)
    b_params = K.layers.Lambda(lambda t: tf.clip_by_value(t + 1.15, 1.15, 45.0), output_shape=lambda s: s, name="b_ks")(b_raw)

    # 2. Raw Mixture Logits (Softmax is applied ONCE in Inverse_Copula_KS_Model)
    weight_logits = K.layers.Dense(n_dims * num_KS, activation="linear", name="weight_logits")(x)

    # 3. Gaussian Copula Cholesky Parameters
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_raw = K.layers.Dense(n_correlations, activation="tanh", name="off_diag_raw")(x)
    off_diag_L = K.layers.Lambda(lambda t: 0.85 * t, output_shape=lambda s: s, name="off_diag")(off_diag_raw)

    diag_raw = K.layers.Dense(n_dims, activation="softplus", name="diag_raw")(x)
    diag_L = K.layers.Lambda(lambda t: t + 0.25, output_shape=lambda s: s, name="diag")(diag_raw)

    outputs = K.layers.Concatenate(axis=1)([a_params, b_params, weight_logits, off_diag_L, diag_L])
    return K.Model(inputs=inp, outputs=outputs, name="Encoder_GC_KSmix")


class Copula_KS_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_KS, num_samples, lbound, **kwargs):
        super(Copula_KS_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_KS = num_KS
        self.num_samples = num_samples
        self.lbound = lbound

    def call(self, inputs):
        a_ks, b_ks, weights_prob, offdiag, diag = inputs

        # Build Cholesky Matrices: [Batch, Dims, Dims]
        LT_matrices = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)

        # Gaussian Copula Samples: [Batch, Samples, Dims]
        u_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples, self.lbound)

        # Map to Physical Stiffness Space [lbound, 1.0]: [Batch, Samples, Dims]
        z_samples = ks_mixture_marginal_samples(a_ks, b_ks, weights_prob, u_samples, self.lbound)

        return z_samples, u_samples, LT_matrices