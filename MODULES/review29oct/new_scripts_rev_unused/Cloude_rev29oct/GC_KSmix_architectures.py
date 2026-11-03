# -*- coding: utf-8 -*-
"""
GC_KSmix_architectures.py -- Encoder of the PI-GC-KSmix-VAE.

Fully connected network (Dense-swish + LayerNorm, as in
MODULES/review29oct/GC_KSmix_architectures.py) mapping the standardized
modal observations to the parameters of q(theta | y):

  logits (D*K) -> mixture weights (one softmax, applied in GC_KSmix_models.py)
  a_raw, b_raw (D*K) -> Kumaraswamy shapes a = 1 + softplus(a_raw), b likewise
  off_diag (D(D-1)/2), diag_raw (D) -> unit-row-norm copula Cholesky factor

Initialization: output kernels are small, so the initial q is set by the
biases: the K components of every marginal start with modes spread over the
support (k + 0.5) / K, equal weights, and an identity copula.
"""

from __future__ import annotations

import config as C  # noqa: F401  (sets KERAS_BACKEND before tensorflow is imported)
import numpy as np
import tensorflow as tf

keras = tf.keras


def _inv_softplus(y):
    return np.log(np.expm1(y))


def spread_component_shapes(num_KS, a0=3.0):
    """(a, b) >= 1 such that component k has its mode at x = (k + 0.5) / K."""
    modes = (np.arange(num_KS) + 0.5) / num_KS
    a = np.full(num_KS, a0)
    b = ((a - 1.0) / modes ** a + 1.0) / a                 # Kumaraswamy mode formula solved for b
    return a, np.maximum(b, 1.05)


def build_encoder(n_features, n_dims, num_KS, hidden=(256, 256, 192)):
    D, K = n_dims, num_KS
    small = keras.initializers.VarianceScaling(scale=0.01)
    inp = keras.Input((n_features,), dtype="float64", name="features")
    x = inp
    for i, h in enumerate(hidden):
        x = keras.layers.Dense(h, activation="swish", kernel_initializer="he_normal",
                               dtype="float64", name=f"dense_{i}")(x)
        if i < len(hidden) - 1:
            x = keras.layers.LayerNormalization(dtype="float64", name=f"layer_norm_{i}")(x)
    heads = {"logits": D * K, "a_raw": D * K, "b_raw": D * K,
             "off_diag": D * (D - 1) // 2, "diag_raw": D}
    outs = [keras.layers.Dense(n, kernel_initializer=small, dtype="float64", name=name)(x)
            for name, n in heads.items()]
    enc = keras.Model(inp, outs, name="encoder")
    a0, b0 = spread_component_shapes(K)
    # outputs are reshaped to (D, K): the component index runs fastest
    enc.get_layer("a_raw").bias.assign(np.tile(_inv_softplus(a0 - 1.0), D))
    enc.get_layer("b_raw").bias.assign(np.tile(_inv_softplus(b0 - 1.0), D))
    enc.get_layer("diag_raw").bias.assign(np.full(D, _inv_softplus(1.0 - 1e-3)))
    return enc
