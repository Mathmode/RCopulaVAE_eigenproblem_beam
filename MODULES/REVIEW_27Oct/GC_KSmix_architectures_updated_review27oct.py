#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC_KSmix_architectures_updated_review27oct.py
=============================================
Full Updated Version of `MODULES/review_27oct/Last_used_scripts/GC_KSmix_architectures.py`
for the MS CPENG-8674 Major Revision.

Uses `GC_KSmix_functions_updated_review27oct` so that:
  - `u_samples` has shape `(Batch, Samples, Dims)` for any `num_samples >= 1`.
  - When `num_KS == 1` (unimodal Kumaraswamy), the exact analytical closed-form
    quantile is used; when `num_KS > 1` (multimodal Kumaraswamy mixture), the
    35-step bisection solver without double-softmax is used.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from .GC_KSmix_functions_updated_review27oct import (
    build_correlation_matrices_from_cholesky,
    gaussian_copula_samples,
    ks_mixture_marginal_samples,
)


class KumaraswamyParamLayer(K.layers.Layer):
    """
    Maps well-conditioned (median `m_sig`, relative concentration `c_sig`) in (0, 1)^2
    to exact Kumaraswamy shape parameters `(a, b)` via the analytical median identity:
        F(m; a, b) = 1 - (1 - m^a)^b = 0.5  <=>  b = -ln(2) / ln(1 - m^a).
    This completely eliminates dead `b_ks` neurons at 1.01 while preserving the exact
    Kumaraswamy (a, b) parameterization across the full domain [lbound, 1.0].
    """
    def __init__(self, **kwargs):
        super(KumaraswamyParamLayer, self).__init__(**kwargs)

    def call(self, inputs):
        m_sig, c_sig = inputs
        # Map sigmoid outputs to unit-space median m in [0.03, 0.985] and sharpness c in [0.06, 0.78]
        m = 0.03 + 0.955 * m_sig
        c = 0.06 + 0.720 * c_sig

        # Ensure m^a >= 2e-3 so (1 - m^a) never underflows in float32
        ln_m = tf.math.log(tf.clip_by_value(m, 0.02, 0.990))
        a_max = tf.minimum(32.0, tf.math.log(2e-3) / ln_m)
        a_max = tf.maximum(a_max, 2.0)

        a_params = 1.20 + c * (a_max - 1.20)
        m_pow_a = tf.math.exp(tf.clip_by_value(a_params * ln_m, -6.5, -1e-4))
        ln_one_minus_ma = tf.math.log(tf.clip_by_value(1.0 - m_pow_a, 1e-6, 1.0 - 1e-6))

        b_raw = -0.69314718 / ln_one_minus_ma
        b_params = tf.clip_by_value(b_raw, 1.02, 160.0)
        a_params = tf.clip_by_value(a_params, 1.10, 35.0)
        return a_params, b_params


def Fully_connected_enc_GC_KS(input_dim, n_dims, num_KS, lbound):
    input1 = K.Input(shape=(input_dim,), name="InputLayer")

    x = K.layers.Dense(256, activation="elu", kernel_initializer="he_normal")(input1)
    x = K.layers.LayerNormalization()(x)
    x = K.layers.Dense(256, activation="elu", kernel_initializer="he_normal")(x)
    x = K.layers.Dense(128, activation="elu", kernel_initializer="he_normal")(x)

    # Spread initial component medians across [0.35, 0.90] when num_KS > 1 so mixture components
    # naturally cover distinct modes (e.g., damaged vs. healthy/mirror modes) from epoch 1
    if num_KS > 1:
        init_m_logits = np.linspace(-0.4, 1.8, num_KS, dtype=np.float32)
        init_m_bias = np.repeat(init_m_logits, n_dims)
    else:
        init_m_bias = np.full((n_dims * num_KS,), 0.8, dtype=np.float32)

    m_sig = K.layers.Dense(
        n_dims * num_KS,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(init_m_bias),
        name="m_ks",
    )(x)
    c_sig = K.layers.Dense(
        n_dims * num_KS,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-0.3),
        name="c_ks",
    )(x)

    a_params, b_params = KumaraswamyParamLayer(name="ks_ab_map")([m_sig, c_sig])

    # Mixture logits (B, D * K)
    weight_vals = K.layers.Dense(n_dims * num_KS, activation="linear", name="weights")(x)

    # Copula Correlation Cholesky Parameters
    n_correlations = n_dims * (n_dims - 1) // 2
    off_diag_L = K.layers.Dense(n_correlations, activation="linear", name="off_diag")(x)
    diag_L = K.layers.Dense(
        n_dims,
        activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(1.0),
        name="diag",
    )(x)
    diag_L = K.layers.Lambda(lambda t: t + 0.20, name="diag_shift")(diag_L)

    outputs = K.layers.Concatenate(axis=1)([a_params, b_params, weight_vals, off_diag_L, diag_L])
    return K.Model(inputs=input1, outputs=outputs)


class Copula_KS_pdf_layer(tf.keras.layers.Layer):
    def __init__(self, n_dims, num_KS, num_samples, lbound, **kwargs):
        super(Copula_KS_pdf_layer, self).__init__(**kwargs)
        self.n_dims = n_dims
        self.num_KS = num_KS
        self.num_samples = num_samples
        self.lbound = lbound

    def call(self, inputs):
        a_ks, b_ks, weights, offdiag, diag = inputs

        # Build Cholesky Matrices: [Batch, Dims, Dims]
        LT_matrices = build_correlation_matrices_from_cholesky(offdiag, diag, self.n_dims)

        # Gaussian Copula Samples: [Batch, Samples, Dims]
        u_samples = gaussian_copula_samples(LT_matrices, self.n_dims, self.num_samples, self.lbound)

        # Map to Kumaraswamy (num_KS=1) or Kumaraswamy Mixture (num_KS>1) Space: [Batch, Samples, Dims]
        z_samples = ks_mixture_marginal_samples(a_ks, b_ks, weights, u_samples, self.lbound)

        return z_samples, u_samples, LT_matrices
