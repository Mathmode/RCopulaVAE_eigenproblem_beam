# -*- coding: utf-8 -*-
"""
ks28_tf_model.py -- Physics-informed copula VAE with Kumaraswamy marginals (extended).

Encoder  E_theta(m) -> parameters of q(theta | m) = sum_c pi_c q_c(theta)   (ks28_tf_distributions)
Sampling stratified: H samples from EVERY component c
Decoder  D(theta)   -> modal properties (ks28_tf_physics, no trainable weights)

Objective (per observation m; minimised as a loss = -objective / n_obs)
------------------------------------------------------------------------
    log w_ch = log p(m | theta_ch) + beta * [ log p(theta_ch) - log q(theta_ch | m) ]

  "elbo"  : stratified ELBO      sum_c pi_c (1/H) sum_h log w_ch
            (Roeder et al. 2017; Morningstar et al. 2021). Exact ELBO of the mixture,
            differentiable w.r.t. the mixture weights without sampling the component index.
  "siwae" : stratified importance-weighted bound   log (1/H) sum_h sum_c pi_c w_ch
            (Morningstar et al. 2021). Tighter bound, less mode-seeking than the ELBO
            -> better at keeping several modes (symmetric damage solutions).

beta = 1 is the true ELBO. beta < 1 is used only as a short warm-up (annealing), after
which beta = 1 for the rest of training and for every reported number; no weight
is tuned on the test set (Reviewer 3 c.2). Note the SIGN: the loss contains
+E_q[log q], i.e. it rewards entropy (the old code had the opposite sign).

The likelihood is the one of ks28_likelihood.py (noise model), the prior is the scenario
prior (or uniform), both exactly as in the ground-truth sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Tuple

import numpy as np
from . import ks28_tf_setup  # noqa: F401  (forces the TensorFlow backend of Keras)
import tensorflow as tf

from .ks28_tf_distributions import (raw_sizes, raw_to_params, sample_stratified, log_q,
                                    log_q_all_components, ScenarioPriorTF, UniformPriorTF)
from .ks28_tf_physics import ModalDecoder


# =============================================================================
@dataclass
class ModelConfig:
    n_components: int = 3          # C: joint (copula) components; 1 = paper model
    n_ks: int = 1                  # K: Kumaraswamy terms per marginal (K>1 -> bisection)
    copula: str = "full"           # "full" | "independent"
    hidden: Tuple[int, ...] = (256, 256, 256)
    activation: str = "relu"
    l2: float = 1e-6
    n_samples: int = 4             # H per component during training
    objective: str = "elbo"        # "elbo" | "siwae"
    prior: str = "scenario"        # "scenario" | "uniform"
    n_modes_used: int = 5          # modes used in encoder input AND likelihood (R3.4 study)
    use_mode_shapes: bool = True
    eigh_device: str = ""          # "" (default placement) or "/CPU:0"
    physics_dtype: str = "float64"

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


# =============================================================================
def build_encoder(input_dim, D, mc: ModelConfig, seed=0):
    """Fully connected encoder -> raw parameter vector."""
    C, K = mc.n_components, mc.n_ks
    sizes = raw_sizes(C, K, D)
    reg = tf.keras.regularizers.l2(mc.l2) if mc.l2 > 0 else None
    inp = tf.keras.Input(shape=(input_dim,), name="modal_data", dtype="float32")
    h = inp
    for i, n in enumerate(mc.hidden):
        h = tf.keras.layers.Dense(n, activation=mc.activation, kernel_initializer="he_uniform",
                                  kernel_regularizer=reg, name=f"hidden_{i}", dtype="float32")(h)
    rng = np.random.default_rng(seed)

    heads, init_bias = [], {}

    def head(name, n, bias):
        layer = tf.keras.layers.Dense(
            n, kernel_initializer=tf.keras.initializers.VarianceScaling(0.1, seed=int(rng.integers(1e6))),
            bias_initializer="zeros", name=name, dtype="float32")
        heads.append(layer)
        init_bias[name] = np.asarray(bias, dtype=np.float32)
        return layer(h)

    # Different bias per component breaks the symmetry between components from the start.
    # a, b start around 2.5 (smooth unimodal marginals): sigmoid^-1 of the normalised log-shape.
    a0 = (math.log(2.5) - math.log(0.3)) / (math.log(3000.0) - math.log(0.3))
    a_bias = np.log(a0 / (1 - a0)) + 0.3 * rng.standard_normal(sizes["a"])
    b_bias = np.log(a0 / (1 - a0)) + 0.3 * rng.standard_normal(sizes["b"])
    out = tf.keras.layers.Concatenate(name="raw_params", dtype="float32")([
        head("logits_pi", sizes["logits_pi"], np.zeros(sizes["logits_pi"])),
        head("ks_a", sizes["a"], a_bias),
        head("ks_b", sizes["b"], b_bias),
        head("ks_w", sizes["w"], 0.1 * rng.standard_normal(sizes["w"])),
        head("copula_offdiag", sizes["off"], np.zeros(sizes["off"])),
        head("copula_diag", sizes["diag"], np.full(sizes["diag"], 1.0)),
    ])
    model = tf.keras.Model(inp, out, name="KS28_encoder")
    for layer in heads:                       # set the initial biases (Keras 2 and 3 safe)
        layer.bias.assign(init_bias[layer.name])
    return model


# =============================================================================
class KS28CopulaVAE:
    """Holds encoder, decoder, prior and likelihood; exposes loss and posterior utilities."""

    def __init__(self, cfg, noise, prior_cfg, basis, input_dim, mc: ModelConfig, seed=0):
        self.cfg, self.noise, self.mc = cfg, noise, mc
        self.D = cfg.n_latent
        self.lo = cfg.latent_lower.astype(np.float32)
        self.hi = cfg.latent_upper.astype(np.float32)
        self.encoder = build_encoder(input_dim, self.D, mc, seed)
        dt = tf.float64 if mc.physics_dtype == "float64" else tf.float32
        self.decoder = ModalDecoder(basis, cfg, n_modes=mc.n_modes_used, dtype=dt,
                                    eigh_device=mc.eigh_device or None)
        self.log_prior = ScenarioPriorTF(cfg, prior_cfg) if mc.prior == "scenario" else UniformPriorTF(cfg)
        nm, ns = mc.n_modes_used, cfg.n_sensors
        self.sf = tf.constant(noise.sigma_f_lik(nm), tf.float32)        # (nm,) per-mode std
        self.sp = tf.constant(noise.sigma_phi_lik(nm), tf.float32)
        self.n_obs = float(nm + (nm * (ns - 1) if mc.use_mode_shapes else 0))   # loss normaliser

    # ------------------------------------------------------------------ q parameters
    def q_params(self, x, training=False):
        raw = self.encoder(x, training=training)
        return raw_to_params(raw, self.mc.n_components, self.mc.n_ks, self.D, self.lo, self.hi,
                             self.mc.copula)

    # ------------------------------------------------------------------ likelihood
    def neg_log_lik(self, f_obs, phi_obs, f_pred, phi_pred):
        """Same expression as ks28_likelihood.neg_log_likelihood. Shapes broadcast."""
        nm = self.mc.n_modes_used
        r = tf.math.log(f_obs[..., :nm]) - tf.math.log(f_pred[..., :nm])
        nll = 0.5 * tf.reduce_sum(r ** 2 / self.sf ** 2, -1)
        if self.mc.use_mode_shapes:
            po, pp = phi_obs[..., :nm, :], phi_pred[..., :nm, :]
            num = tf.reduce_sum(po * pp, -1) ** 2
            no = tf.reduce_sum(po ** 2, -1)
            den = no * tf.reduce_sum(pp ** 2, -1) + 1e-30
            nll += 0.5 * tf.reduce_sum(no * (1.0 - num / den) / self.sp ** 2, -1)
        return nll

    # ------------------------------------------------------------------ objective
    def log_weights(self, x, f_obs, phi_obs, H, beta=1.0, training=False):
        """log w_ch (B, C, H) and the pieces, for a batch of observations."""
        p = self.q_params(x, training=training)
        theta, xs = sample_stratified(p, H)                                         # (B, C, H, D)
        B, C = tf.shape(theta)[0], tf.shape(theta)[1]
        f_pred, phi_pred = self.decoder(tf.reshape(theta, [-1, self.D]))
        f_pred = tf.reshape(f_pred, tf.concat([tf.stack([B, C, H]), tf.shape(f_pred)[1:]], 0))
        phi_pred = tf.reshape(phi_pred, tf.concat([tf.stack([B, C, H]), tf.shape(phi_pred)[1:]], 0))
        nll = self.neg_log_lik(f_obs[:, None, None], phi_obs[:, None, None], f_pred, phi_pred)
        lq = log_q(theta, p)                                                         # (B, C, H)
        lp = self.log_prior(theta)
        logw = -nll + beta * (lp - lq)
        return logw, dict(nll=nll, log_q=lq, log_prior=lp, log_pi=p["log_pi"])

    def objective(self, logw, log_pi, H):
        if self.mc.objective == "siwae":
            return tf.reduce_logsumexp(logw + log_pi[:, :, None], axis=[1, 2]) - math.log(H)
        pi = tf.exp(log_pi)
        return tf.reduce_sum(pi * tf.reduce_mean(logw, axis=2), axis=1)

    def loss(self, x, f_obs, phi_obs, H, beta=1.0, training=False):
        logw, parts = self.log_weights(x, f_obs, phi_obs, H, beta, training)
        obj = self.objective(logw, parts["log_pi"], H)                              # (B,)
        loss = -tf.reduce_mean(obj) / self.n_obs
        if self.encoder.losses:
            loss += tf.cast(tf.add_n([tf.cast(l, tf.float32) for l in self.encoder.losses]), tf.float32)
        pi = tf.exp(parts["log_pi"])
        mets = dict(
            loss=loss,
            nll=tf.reduce_mean(tf.reduce_sum(pi * tf.reduce_mean(parts["nll"], 2), 1)),
            entropy=-tf.reduce_mean(tf.reduce_sum(pi * tf.reduce_mean(parts["log_q"], 2), 1)),
            log_prior=tf.reduce_mean(tf.reduce_sum(pi * tf.reduce_mean(parts["log_prior"], 2), 1)),
            pi_max=tf.reduce_mean(tf.reduce_max(pi, 1)),
        )
        return loss, mets

    # ------------------------------------------------------------------ export
    def export_params(self, x, batch=1024):
        """q parameters for a dataset as NumPy (for the TF-free postprocessing)."""
        out = {k: [] for k in ("log_pi", "a", "b", "log_w", "L")}
        for i in range(0, x.shape[0], batch):
            p = self.q_params(tf.constant(x[i:i + batch]))
            for k in out:
                out[k].append(p[k].numpy())
        res = {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}
        res["lo"], res["hi"] = self.lo.astype(np.float64), self.hi.astype(np.float64)
        return res
