#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/joint_NN_model.py

R1.4: the proposed GC-LogN VAE with the eigenvalue solver replaced by a neural-network decoder that is trained TOGETHER
with the encoder, in one step, from random weights (no pre-training, no damage labels): a purely data-driven VAE.

Everything else is the proposed model (encoder, Gaussian copula with truncated log-normal marginals, entropy, calibrated
likelihood, negative ELBO with gamma = 1, warm-up): only nll_terms changes. The decoder has the architecture of Ref. 41
(10-30-50-70-80, Tanh/ReLU, sigmoid output); its outputs are scaled to the range of the MEASURED training data
(log f - b and the vertical mode shapes), so no simulation at known damage states is used.
"""

import numpy as np
import tensorflow as tf

from MODULES.review29oct_fixed import tf_keras
from MODULES.review29oct_fixed.GC_KSmix_models import calculate_MAC
from MODULES.baseline_GaussCopula_GaussMarginals.GCLogN_model import GaussCopulaLogNVAE
from MODULES.comparison_surrogate_decoder.surrogate_decoder import build_surrogate, layer_spec

K = tf_keras()


class GaussCopulaLogNJointNN(GaussCopulaLogNVAE):
    """GC-LogN posterior + trainable NN decoder (trained jointly). Set SCALER_LO/HI, N_SENS and DEC_HIDDEN first."""
    SCALER_LO = None
    SCALER_HI = None
    N_SENS = None
    DEC_HIDDEN = (10, 30, 50, 70, 80)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.SCALER_LO is None or self.N_SENS is None:
            raise ValueError("GaussCopulaLogNJointNN: set SCALER_LO, SCALER_HI and N_SENS before building the model")
        if "r" in self.obs_types:
            raise ValueError("the NN decoder predicts frequencies and vertical mode shapes only (obs 'f' or 'fv')")
        self.n_sens = int(self.N_SENS)
        acts, inits = layer_spec(self.DEC_HIDDEN)
        self.Decoder = build_surrogate(K, self.n_dims, self.n_modes * (1 + self.n_sens), self.DEC_HIDDEN, acts, inits)
        self.Decoder.trainable = True                     # trained with the encoder, in the same step
        self.dec_lo = tf.constant(np.asarray(self.SCALER_LO, np.float32))
        self.dec_span = tf.constant(np.asarray(self.SCALER_HI, np.float32) - np.asarray(self.SCALER_LO, np.float32))

    def decode(self, z):
        """log f (B*S, M) and vertical shapes (B*S, M, n_sens) for flat z (B*S, D)."""
        u = (z - self.lbound) / (1.0 - self.lbound)
        y = self.Decoder(u) * self.dec_span + self.dec_lo
        return y[:, :self.n_modes], tf.reshape(y[:, self.n_modes:], (-1, self.n_modes, self.n_sens))

    def nll_terms(self, inputs, z):
        """Same likelihood as the proposed model, with the jointly trained NN as forward operator."""
        [freq_data, _, vert_modes_data, _] = inputs
        B, S, M = tf.shape(z)[0], tf.shape(z)[1], self.n_modes
        log_f, vert = self.decode(tf.reshape(z, (-1, self.n_dims)))
        log_f_pred = tf.reshape(log_f, (B, S, M))
        pred_vert = tf.reshape(vert, (B, S, M, self.n_sens))
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


class LearnedDecoderBridge:
    """NumPy interface of fe_model.BridgeModel (solve(z) -> f [Hz], vertical shapes, rotations; attribute lik) for the
    current weights of the jointly trained decoder: used by the validation monitor, so that no eigenvalue solver is used
    to select the weights of this approach."""

    def __init__(self, model, pcfg):
        self.model, self.pcfg, self.lik = model, pcfg, pcfg.get("likelihood")

    def solve(self, z):
        out_f, out_v = [], []
        for i in range(0, len(z), 4096):
            lf, v = self.model.decode(tf.convert_to_tensor(np.asarray(z[i:i + 4096], np.float32)))
            out_f.append(lf.numpy().astype(np.float64))
            out_v.append(v.numpy().astype(np.float64))
        lf, v = np.concatenate(out_f), np.concatenate(out_v)
        v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        return np.exp(lf), v, v.copy()
