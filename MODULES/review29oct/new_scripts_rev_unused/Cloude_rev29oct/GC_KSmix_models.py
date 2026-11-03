# -*- coding: utf-8 -*-
"""
GC_KSmix_models.py -- Physics-informed Gaussian-copula VAE with Kumaraswamy-mixture
marginals (PI-GC-KSmix-VAE).

  encoder  q_phi(theta | y)    : GC_KSmix_architectures.build_encoder
  decoder  p(y | theta)        : modal solver of the inference FE model + the
                                 Gaussian measurement model (preprocessing_and_eigen)
  prior    p(theta)            : scenario mixture (priors.build_prior)

Training objective (per observation y, averaged over the batch):

  loss = -tau * E_q[log p(y | theta)] + beta * KL(q(theta | y) || p(theta))
         [+ w_fwd * E_data[-(p/rho)(theta*) log q(theta* | y)]]

  * tau = 1 and beta = 1: the negative ELBO; its minimizer is the exact
    posterior p(theta | y). Labels theta* are not used (physics-informed,
    label-free training, as in the paper).
  * tau < 1 only during the first anneal_epochs (likelihood tempering); model
    selection and early stopping use tau = 1.
  * beta is the gamma^2 of the paper; it is selected on the validation set.
  * w_fwd > 0 adds the amortized forward-KL term, which uses the labels and is
    importance-weighted by p/rho so that it has the same minimizer. Off by default.

The KL term is estimated by Monte Carlo with reparameterized samples:
KL = E_q[log q(theta | y) - log p(theta)].
"""

from __future__ import annotations

from pathlib import Path

import config as C  # noqa: F401  (sets KERAS_BACKEND before tensorflow is imported)
import numpy as np
import tensorflow as tf

import GC_KSmix_functions as KS
from GC_KSmix_architectures import build_encoder
from preprocessing_and_eigen import ModalDecoderTF, log_likelihood_tf
from priors import build_prior

DT = tf.float64


class PIGCVAE:
    def __init__(self, cfg: C.TrainConfig, n_features: int, coarse: dict):
        self.cfg = cfg
        self.n_dims = C.N_DIMS
        self.lo = tf.constant(coarse["lower"], DT)
        self.hi = tf.constant(coarse["upper"], DT)
        self.encoder = build_encoder(n_features, self.n_dims, cfg.num_KS, cfg.hidden)
        self.decoder = ModalDecoderTF(coarse, cfg.n_modes, cfg.eigh_on_cpu)
        self.sig_f = float(coarse["sig_f"])
        self.sig_v = tf.constant(coarse["sig_v"][:cfg.n_modes], DT)
        self.sig_r = tf.constant(coarse["sig_r"][:cfg.n_modes], DT)
        self.prior_log_prob = build_prior().tf_log_prob_fn()

    # ---------------------------------------------------------------- q(theta|y)
    def params(self, feats, training=False):
        logits, a_raw, b_raw, off, diag = self.encoder(feats, training=training)
        shape = (-1, self.n_dims, self.cfg.num_KS)
        return {
            "logw": tf.nn.log_softmax(tf.reshape(logits, shape), axis=-1),
            "a": tf.minimum(1.0 + tf.nn.softplus(tf.reshape(a_raw, shape)), 1e4),
            "b": tf.minimum(1.0 + tf.nn.softplus(tf.reshape(b_raw, shape)), 1e4),
            "L": KS.copula_cholesky(off, diag, self.n_dims),
        }

    def log_q(self, params, theta):
        return KS.log_q(params, theta, self.lo, self.hi)

    def sample(self, params, n, seed=None, return_log_q=False):
        return KS.sample(params, n, self.lo, self.hi, seed=seed, return_log_q=return_log_q)

    # ------------------------------------------------------------- p(y|theta)
    def log_likelihood(self, theta, y_logf, y_v, y_r):
        """theta (N, D) and observations with the same leading dimension N."""
        logf, v, r = self.decoder(theta)
        return log_likelihood_tf(y_logf, y_v, y_r, logf, v, r, self.sig_f, self.sig_v, self.sig_r,
                                 self.cfg.obs)

    # ------------------------------------------------------------------- losses
    def elbo_terms(self, params, y_logf, y_v, y_r, n_samples, seed=None):
        theta, lq = self.sample(params, n_samples, seed=seed, return_log_q=True)     # (B, H, D)
        rep = lambda t: tf.repeat(t, n_samples, axis=0)                              # noqa: E731
        ll = self.log_likelihood(tf.reshape(theta, (-1, self.n_dims)), rep(y_logf), rep(y_v), rep(y_r))
        return tf.reshape(ll, (-1, n_samples)), self.prior_log_prob(theta), lq

    def losses(self, batch, tau, training=True, n_samples=None, seed=None):
        feats, theta_true, y_logf, y_v, y_r, w_is = batch
        p = self.params(feats, training=training)
        ll, lp, lq = self.elbo_terms(p, y_logf, y_v, y_r, n_samples or self.cfg.n_samples, seed)
        exp_ll = tf.reduce_mean(ll)
        kl = tf.reduce_mean(lq - lp)
        out = {"neg_elbo": -exp_ll + kl, "exp_log_lik": exp_ll, "kl": kl,
               "loss": -tau * exp_ll + self.cfg.beta * kl}
        nll = -self.log_q(p, theta_true)                      # label-based score (monitoring)
        out["nll"] = tf.reduce_mean(nll)
        if self.cfg.w_fwd > 0:
            out["loss"] += self.cfg.w_fwd * tf.reduce_mean(w_is * nll)
        return out

    # ---------------------------------------------------------------- utilities
    def predict_params(self, feats, batch=4096):
        outs = []
        for i in range(0, len(feats), batch):
            p = self.params(tf.constant(feats[i:i + batch], DT))
            outs.append({k: v.numpy() for k, v in p.items()})
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def save(self, run_dir):
        self.encoder.save_weights(str(Path(run_dir) / "encoder.weights.h5"))


def params_to_tf(params_np, idx=None):
    sl = slice(None) if idx is None else idx
    return {k: tf.constant(np.asarray(params_np[k])[sl], DT) for k in ("logw", "a", "b", "L")}


def load_run(run_dir):
    """Rebuild a trained model from Output/<run>/ (config.json + encoder weights + feature scaler)."""
    from preprocessing_and_eigen import FeatureScaler, load_coarse_model
    run_dir = Path(run_dir)
    meta = C.load_json(run_dir / "config.json")
    tc = dict(meta["train"])
    tc["hidden"] = tuple(tc["hidden"])
    cfg = C.TrainConfig(**tc)
    scaler = FeatureScaler.load(run_dir / "feature_scaler.npz")
    model = PIGCVAE(cfg, len(scaler.mean), load_coarse_model(Path(meta["data_dir"])))
    model.encoder.load_weights(str(run_dir / "encoder.weights.h5"))
    return model, scaler, meta
