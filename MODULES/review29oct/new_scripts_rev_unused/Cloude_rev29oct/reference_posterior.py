# -*- coding: utf-8 -*-
"""
reference_posterior.py -- Reference ("true") Bayesian posterior by adaptive tempered SMC.

    p(theta | y)  ∝  p(y | theta) p(theta)

with EXACTLY the prior (priors.build_prior), the inference FE model
(coarse_model.npz) and the likelihood (preprocessing_and_eigen.log_likelihood_np)
of the VAE objective, so the two posteriors can be compared directly. For
fine-model observations it is the best posterior attainable with the
inference model. The sampler does not use the true parameter values.

Algorithm (adaptive SMC sampler, Del Moral, Doucet & Jasra 2006):
  * N particles drawn from the prior; temperatures 0 = t_0 < ... < t_S = 1
    chosen so that the effective sample size of each reweighting is
    ess_target * N; systematic resampling;
  * after each resampling, n_mcmc Metropolis-Hastings moves targeting
    p(theta) p(y | theta)^t: random walk with the particle covariance
    (adaptive scale, target acceptance 0.25) and the mirror move
    theta -> P theta (P = left/right permutation, an involution with unit
    Jacobian, so the acceptance ratio is pi(P theta) / pi(theta)).
"""

from __future__ import annotations

import time

import numpy as np

import config as C
from preprocessing_and_eigen import ModalDecoderTF, log_likelihood_np
from priors import build_prior


class LogLikelihood:
    """
    log p(y | theta) for many theta. The modal solution uses the TensorFlow
    decoder (multi-threaded batched eigh, about 10x faster than NumPy on CPU);
    check_implementation.py verifies it equals the NumPy model.
    """

    def __init__(self, coarse, n_modes, obs):
        import tensorflow as tf
        self._tf = tf
        decoder = ModalDecoderTF(coarse, n_modes)
        self._modal = tf.function(decoder.__call__, reduce_retracing=True)
        self.sig_f = float(coarse["sig_f"])
        self.sig_v, self.sig_r = coarse["sig_v"][:n_modes], coarse["sig_r"][:n_modes]
        self.obs = obs

    def __call__(self, theta, y_logf, y_v, y_r, chunk=20000):
        out = np.empty(len(theta))
        for i in range(0, len(theta), chunk):
            logf, v, r = (t.numpy() for t in self._modal(self._tf.constant(theta[i:i + chunk], self._tf.float64)))
            out[i:i + chunk] = log_likelihood_np(y_logf, y_v, y_r, logf, v, r, self.sig_f, self.sig_v,
                                                 self.sig_r, self.obs)
        return out


def _ess(logw):
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / np.sum(w ** 2)


def _systematic_resample(logw, rng):
    w = np.exp(logw - logw.max())
    cdf = np.cumsum(w / w.sum())
    pos = (rng.random() + np.arange(len(w))) / len(w)
    return np.minimum(np.searchsorted(cdf, pos), len(w) - 1)


def _mh_update(prop, th, lp, ll, t, prior, loglik, y, rng):
    """In-place Metropolis-Hastings update for a symmetric proposal; returns the acceptance rate."""
    lp_p = prior.log_prob(prop)
    ok = np.isfinite(lp_p)
    ll_p = np.full(len(prop), -np.inf)
    if ok.any():
        ll_p[ok] = loglik(prop[ok], *y)
    log_a = np.where(ok, lp_p + t * ll_p - lp - t * ll, -np.inf)
    acc = np.log(rng.random(len(th))) < log_a
    th[acc], lp[acc], ll[acc] = prop[acc], lp_p[acc], ll_p[acc]
    return float(acc.mean())


def smc_posterior(y_logf, y_v, y_r, loglik: LogLikelihood, prior=None, n_particles=4000, n_mcmc=10,
                  ess_target=0.5, mirror_move=True, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    prior = prior or build_prior()
    y = (y_logf, y_v, y_r)
    perm = np.array(C.MIRROR)
    th = prior.sample(n_particles, rng)[0]
    lp, ll = prior.log_prob(th), loglik(th, *y)
    t, scale, n_stages, acc_hist, t0 = 0.0, 0.5, 0, [], time.time()
    while t < 1.0:
        dt_max = 1.0 - t
        if _ess(dt_max * ll) >= ess_target * n_particles:
            dt = dt_max
        else:
            lo, hi = 0.0, dt_max
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _ess(mid * ll) >= ess_target * n_particles else (lo, mid)
            dt = max(lo, 1e-12)
        idx = _systematic_resample(dt * ll, rng)
        th, lp, ll = th[idx], lp[idx], ll[idx]
        t = min(1.0, t + dt)
        n_stages += 1
        chol = np.linalg.cholesky(np.cov(th.T) + 1e-12 * np.eye(th.shape[1]))
        for _ in range(n_mcmc if t < 1.0 else 3 * n_mcmc):
            step = scale * 2.38 / np.sqrt(th.shape[1])
            acc = _mh_update(th + step * rng.standard_normal(th.shape) @ chol.T, th, lp, ll, t,
                             prior, loglik, y, rng)
            acc_hist.append(acc)
            scale = float(np.clip(scale * np.exp(acc - 0.25), 0.02, 3.0))
            if mirror_move:
                _mh_update(th[:, perm], th, lp, ll, t, prior, loglik, y, rng)
        if verbose:
            print(f"   stage {n_stages:3d}  t={t:.4f}  acceptance={acc_hist[-1]:.2f}")
    return th, dict(n_stages=n_stages, mean_acceptance=float(np.mean(acc_hist)),
                    time_s=time.time() - t0, n_particles=n_particles)


def importance_refine(samples, log_q, y_logf, y_v, y_r, loglik: LogLikelihood, prior=None):
    """
    Self-normalized importance sampling with the VAE posterior as proposal:
    w ∝ p(y | theta) p(theta) / q(theta | y). Exact as the number of samples
    grows when q covers the posterior; the effective sample size (ESS) tells
    whether it does. Returns normalized weights and ESS / N.
    """
    prior = prior or build_prior()
    logw = loglik(samples, y_logf, y_v, y_r) + prior.log_prob(samples) - log_q
    logw = np.where(np.isfinite(logw), logw, -np.inf)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return w, float(1.0 / np.sum(w ** 2) / len(w))
