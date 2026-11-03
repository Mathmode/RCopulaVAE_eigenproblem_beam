#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/GC_KSmix_architectures.py

Encoder for the mixture of K Gaussian-copula / Kumaraswamy-marginal components.
Outputs, per observation:
    a, b       [K, D]   Kumaraswamy shapes (a >= a_min, b >= b_min)
    log_pi     [K]      log mixture weights
    L          [K, D, D] correlation Cholesky factor of each component's copula

Changes w.r.t. MODULES/review29oct/GC_KSmix_architectures.py:
  * b may go below 1 (b_min = 0.5). With the corrected (+log q) entropy sign there is
    no density-collapse incentive, and b < 1 lets a component put its mass right at the
    healthy bound z = 1, which is where the reference posteriors of undamaged elements
    concentrate. a, b >= 1.15 forced the density to be ZERO at z = 1.
  * No hard clip_by_value on a, b (it zeroes gradients); softplus + floor only.
  * Components are initialised at different random locations so they do not start
    identical (identical components receive identical gradients and never separate).
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K


def _inv_softplus(v):
    return np.log(np.expm1(v)).astype(np.float32)


def Fully_connected_enc_MixGC_KS(input_dim, n_dims, n_comp, a_min=1.0, b_min=0.5,
                                 hidden=(256, 256, 192), seed=1234):
    rng = np.random.default_rng(seed)
    n_corr = n_dims * (n_dims - 1) // 2

    inp = K.Input(shape=(input_dim,), name='InputLayer')
    x = inp
    for i, h in enumerate(hidden):
        x = K.layers.Dense(h, activation='swish', kernel_initializer='he_normal', name=f'hidden_{i}')(x)
        if i < len(hidden) - 1:
            x = K.layers.LayerNormalization(name=f'ln_{i}')(x)

    # Spread initial component locations: Kumaraswamy mode moves from damaged (a small,
    # b large) to healthy (a large, b small). One component starts at the healthy corner.
    a0 = rng.uniform(1.5, 6.0, size=(n_comp, n_dims))
    b0 = rng.uniform(1.0, 4.0, size=(n_comp, n_dims))
    a0[0], b0[0] = 4.0, 1.0
    small = K.initializers.RandomNormal(stddev=0.02, seed=seed)

    a_raw = K.layers.Dense(n_comp * n_dims, kernel_initializer=small,
                           bias_initializer=K.initializers.Constant(_inv_softplus(a0 - a_min).ravel()),
                           name='a_ks')(x)
    b_raw = K.layers.Dense(n_comp * n_dims, kernel_initializer=small,
                           bias_initializer=K.initializers.Constant(_inv_softplus(b0 - b_min).ravel()),
                           name='b_ks')(x)
    pi_logits = K.layers.Dense(n_comp, kernel_initializer=small, bias_initializer='zeros',
                               name='pi_logits')(x)
    off_diag = K.layers.Dense(n_comp * n_corr, kernel_initializer=small, name='off_diag')(x)
    diag_raw = K.layers.Dense(n_comp * n_dims, kernel_initializer=small,
                              bias_initializer=K.initializers.Constant(_inv_softplus(1.0)), name='diag')(x)

    a = K.layers.Reshape((n_comp, n_dims))(K.activations.softplus(a_raw) + a_min)
    b = K.layers.Reshape((n_comp, n_dims))(K.activations.softplus(b_raw) + b_min)
    log_pi = K.activations.log_softmax(pi_logits, axis=-1)
    off_diag = K.layers.Reshape((n_comp, n_corr))(off_diag)
    diag = K.layers.Reshape((n_comp, n_dims))(K.activations.softplus(diag_raw) + 1e-3)

    return K.Model(inputs=inp, outputs=[a, b, log_pi, off_diag, diag], name='FC_Encoder_MixGC_KS')
