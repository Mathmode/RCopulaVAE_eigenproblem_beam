#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/surrogate_decoder.py

Data-driven surrogate of the forward operator (decoder), for the comparison requested in R1.4 with the Gaussian copula VAE
of Ref. 41 (Fernandez-Navamuel et al., arXiv:2510.02013). In Ref. 41 the decoder is a fully connected neural network that is
pre-trained to approximate the forward operator and then frozen while the encoder is trained (two-step training). Its
architecture (Ref. 41, Table 3) is used here unchanged:

    hidden layers 10, 30, 50, 70, 80 units; activations Tanh, ReLU, ReLU, ReLU, ReLU;
    Glorot uniform (Tanh) and He uniform (ReLU) initialisation; sigmoid output on targets rescaled to [0, 1].

Input:   u = (z - lbound) / (1 - lbound), the stiffness factors in the unit box [0, 1]^7.
Outputs: exactly the quantities the likelihood uses (fe_model.neg_log_likelihood):
         * log f(z), the log-frequencies of the nominal model. The training targets are log f_obs - b, where b is the mean
           modelling error of the calibrated likelihood (problem_config.json), so the surrogate replaces the decoder output
           log f(z) and the likelihood is unchanged;
         * the vertical mode shapes at the sensors (n_modes x n_sensors, signs aligned as in the data set). The likelihood
           only uses their direction (MAC), so the scale of the regression target does not matter.
Data:    the damage states z* of the SAME training split as the VAE (validation split for early stopping); targets: the
         noise-free modal data of the nominal model at z* (default), or the noisy observations of the split.

The scaling to [0, 1] adds a 5 % margin on both sides of the training range, so that the sigmoid never has to reach 0 or 1.
"""

import json
import os

import numpy as np

HIDDEN = (10, 30, 50, 70, 80)
ACTIVATIONS = ("tanh", "relu", "relu", "relu", "relu")
INITIALIZERS = ("glorot_uniform", "he_uniform", "he_uniform", "he_uniform", "he_uniform")
MARGIN = 0.05
WEIGHTS_FILE = "surrogate_decoder.weights.h5"
CONFIG_FILE = "surrogate_config.json"


def layer_spec(hidden):
    """Activations and initialisers of the hidden layers: those of Ref. 41 for its architecture; otherwise the same pattern
    (Tanh with Glorot uniform in the first layer, ReLU with He uniform in the others)."""
    hidden = tuple(int(h) for h in hidden)
    if hidden == HIDDEN:
        return list(ACTIVATIONS), list(INITIALIZERS)
    acts = ["tanh"] + ["relu"] * (len(hidden) - 1)
    return acts, ["glorot_uniform" if a == "tanh" else "he_uniform" for a in acts]


def build_surrogate(K, n_in, n_out, hidden=HIDDEN, activations=ACTIVATIONS, initializers=INITIALIZERS):
    """Fully connected surrogate of Ref. 41 (Table 3): u in [0, 1]^n_in -> scaled outputs in (0, 1)^n_out."""
    x_in = K.Input(shape=(int(n_in),), name="u")
    x = x_in
    for i, (units, act, init) in enumerate(zip(hidden, activations, initializers)):
        x = K.layers.Dense(int(units), activation=act, kernel_initializer=init, name=f"sur_hidden_{i}")(x)
    y = K.layers.Dense(int(n_out), activation="sigmoid", kernel_initializer="glorot_uniform", name="sur_output")(x)
    return K.Model(x_in, y, name="surrogate_decoder")


def to_unit(z, lbound):
    return (np.asarray(z, np.float64) - float(lbound)) / (1.0 - float(lbound))


def make_targets(f_hz, vert, bias_logf):
    """Regression targets (N, n_modes + n_modes * n_sensors): log f_obs - b and the vertical mode shapes."""
    f = np.asarray(f_hz, np.float64)
    v = np.asarray(vert, np.float64)
    return np.concatenate([np.log(f) - np.asarray(bias_logf, np.float64)[None], v.reshape(len(f), -1)], axis=1)


class MinMaxScaler:
    """Min-max scaling of every output to [MARGIN', 1 - MARGIN'] (training range widened by MARGIN on each side)."""

    def __init__(self, lo=None, hi=None):
        self.lo = None if lo is None else np.asarray(lo, np.float64)
        self.hi = None if hi is None else np.asarray(hi, np.float64)

    def fit(self, y):
        lo, hi = np.min(y, axis=0), np.max(y, axis=0)
        span = np.maximum(hi - lo, 1e-9)
        self.lo, self.hi = lo - MARGIN * span, hi + MARGIN * span
        return self

    def transform(self, y):
        return (np.asarray(y, np.float64) - self.lo) / (self.hi - self.lo)

    def inverse(self, s):
        return np.asarray(s, np.float64) * (self.hi - self.lo) + self.lo


def save_surrogate(folder, model, scaler, meta, hidden=HIDDEN, activations=ACTIVATIONS, initializers=INITIALIZERS):
    """Writes the weights, the architecture, the scaling and `meta` (data, training, accuracy) of a trained surrogate."""
    os.makedirs(folder, exist_ok=True)
    model.save_weights(os.path.join(folder, WEIGHTS_FILE))
    try:                                   # full Keras model as well (convenient, but the weights file is what is loaded)
        model.save(os.path.join(folder, "surrogate_decoder.keras"))
    except Exception as err:               # pragma: no cover - depends on the Keras version
        print(f"  (surrogate_decoder.keras not written: {err})")
    cfg = dict(meta)
    cfg.update({"hidden": [int(h) for h in hidden], "activations": list(activations), "initializers": list(initializers),
                "output_activation": "sigmoid", "scaler_lo": scaler.lo.tolist(), "scaler_hi": scaler.hi.tolist(),
                "scaling_margin": MARGIN, "weights_file": WEIGHTS_FILE})
    with open(os.path.join(folder, CONFIG_FILE), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def read_config(folder):
    with open(os.path.join(folder, CONFIG_FILE), "r", encoding="utf-8") as fh:
        return json.load(fh)


def update_config(folder, **items):
    cfg = read_config(folder)
    cfg.update(items)
    with open(os.path.join(folder, CONFIG_FILE), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg


def load_surrogate(folder, K):
    """Rebuilds the surrogate from its configuration and loads the weights. Returns (Keras model, config)."""
    cfg = read_config(folder)
    model = build_surrogate(K, cfg["n_in"], cfg["n_out"], cfg["hidden"], cfg["activations"], cfg["initializers"])
    model.load_weights(os.path.join(folder, cfg.get("weights_file", WEIGHTS_FILE)))
    return model, cfg


def _act(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(name)


class SurrogateBridge:
    """
    Same interface as fe_model.BridgeModel (solve(z) -> f [Hz], vertical shapes, rotational shapes; attribute lik), with the
    surrogate in place of the eigenvalue solver. NumPy forward pass in float64 (used by the validation monitor of the
    train-to-convergence protocol and by the accuracy checks).
    """

    def __init__(self, pcfg, folder, K=None):
        if K is None:
            from MODULES.review29oct_fixed import tf_keras
            K = tf_keras()
        model, cfg = load_surrogate(folder, K)
        self.cfg = cfg
        self.layers = [(w[0].astype(np.float64), w[1].astype(np.float64))
                       for w in (layer.get_weights() for layer in model.layers) if len(w) == 2]
        self.acts = list(cfg["activations"]) + [cfg.get("output_activation", "sigmoid")]
        self.scaler = MinMaxScaler(cfg["scaler_lo"], cfg["scaler_hi"])
        self.n_modes, self.n_sens = int(cfg["n_modes"]), int(cfg["n_sensors"])
        self.lbound = float(pcfg["lbound"])
        self.lik = pcfg.get("likelihood")
        self.pcfg = pcfg

    def predict(self, z):
        """Surrogate outputs in physical units: log f(z) (N, M) and vertical shapes (N, M, n_sens) (not normalised)."""
        x = to_unit(np.atleast_2d(z), self.lbound)
        for (W, b), act in zip(self.layers, self.acts):
            x = _act(act, x @ W + b)
        y = self.scaler.inverse(x)
        M = self.n_modes
        return y[:, :M], y[:, M:].reshape(len(y), M, self.n_sens)

    def solve(self, z):
        logf, vert = self.predict(z)
        vert = vert / np.maximum(np.linalg.norm(vert, axis=-1, keepdims=True), 1e-12)
        return np.exp(logf), vert, vert.copy()          # rotations are not observed (obs "fv"): placeholder
