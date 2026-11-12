#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/GCTN_surrogate_model.py

Gaussian copula VAE with a PRE-TRAINED, FROZEN surrogate decoder (the approach of Ref. 41), for the comparison of R1.4.

Everything is inherited from the proposed model (MODULES/baseline_GaussCopula_GaussMarginals/GCTN_model.py: encoder,
Gaussian copula with truncated Gaussian marginals, entropy, calibrated likelihood, negative ELBO with gamma = 1, warm-up and
training steps) EXCEPT the forward operator in the likelihood: nll_terms evaluates the frozen surrogate network
(surrogate_decoder.py) instead of the differentiable eigenvalue solver. The surrogate weights are not trainable, so the
trainable parameters are those of the encoder, as in the proposed model.
"""

import tensorflow as tf

from MODULES.review29oct_fixed import tf_keras
from MODULES.review29oct_fixed.GC_KSmix_models import calculate_MAC
from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import GaussCopulaTNVAE
from MODULES.comparison_surrogate_decoder.surrogate_decoder import load_surrogate

K = tf_keras()


class GaussCopulaTNSurrogateVAE(GaussCopulaTNVAE):
    """GC-TN posterior + frozen surrogate decoder. Set SURROGATE_DIR before building the model."""
    SURROGATE_DIR = None

    def __init__(self, *args, **kwargs):
        super(GaussCopulaTNSurrogateVAE, self).__init__(*args, **kwargs)
        if self.SURROGATE_DIR is None:
            raise ValueError("GaussCopulaTNSurrogateVAE.SURROGATE_DIR is not set")
        surrogate, cfg = load_surrogate(self.SURROGATE_DIR, K)
        if int(cfg["n_modes"]) != self.n_modes or int(cfg["n_in"]) != self.n_dims:
            raise ValueError(f"surrogate of {cfg['n_modes']} modes / {cfg['n_in']} inputs, model of {self.n_modes} / {self.n_dims}")
        if "r" in self.obs_types:
            raise ValueError("the surrogate decoder predicts frequencies and vertical mode shapes only (obs 'f' or 'fv')")
        surrogate.trainable = False                       # pre-trained and frozen (two-step training of Ref. 41)
        self.Surrogate = surrogate
        self.n_sens = int(cfg["n_sensors"])
        self.sur_lo = tf.constant(cfg["scaler_lo"], dtype=tf.float32)
        self.sur_span = tf.constant([h - l for h, l in zip(cfg["scaler_hi"], cfg["scaler_lo"])], dtype=tf.float32)

    def nll_terms(self, inputs, z):
        """Same likelihood as the proposed model (GC_KSmix_models.nll_terms), with log f(z) and the vertical shapes of the
        surrogate instead of the eigenvalue solver. Rotations are not observed: nll_r = 0."""
        [freq_data, _, vert_modes_data, _] = inputs
        B, S, M = tf.shape(z)[0], tf.shape(z)[1], self.n_modes
        u = (tf.reshape(z, (-1, self.n_dims)) - self.lbound) / (1.0 - self.lbound)
        y = self.Surrogate(u, training=False) * self.sur_span + self.sur_lo
        log_f_pred = tf.reshape(y[:, :M], (B, S, M))                           # log f(z) of the nominal model
        pred_vert = tf.reshape(y[:, M:], (B, S, M, self.n_sens))

        log_f_obs = freq_data * self.std_freq + self.mean_freq
        d = log_f_obs[:, None, :] - self.bias_logf - log_f_pred
        nll_f = 0.5 * tf.einsum("bsm,mn,bsn->bs", d, self.prec_logf, d)
        nll_v = tf.reduce_sum((1.0 - calculate_MAC(vert_modes_data[:, None], pred_vert)) * self.inv2var_v, axis=-1)
        nll_r = tf.zeros_like(nll_f)
        nll = tf.zeros_like(nll_f)
        for key, term in (("f", nll_f), ("v", nll_v)):
            if key in self.obs_types:
                nll += term
        return nll, nll_f, nll_v, nll_r
