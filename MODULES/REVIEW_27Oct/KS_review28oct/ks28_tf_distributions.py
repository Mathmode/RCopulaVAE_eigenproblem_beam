# -*- coding: utf-8 -*-
"""
ks28_tf_distributions.py -- TensorFlow version of q(theta | m) and of the scenario prior.

Line-by-line mirror of ks28_q_numpy.py and ks28_priors.py (the smoke test checks that
both give the same numbers). Pure TensorFlow (no tensorflow_probability needed).

Shapes used throughout
    B : batch of observations       C : copula (joint) components
    H : samples per component        K : Kumaraswamy terms per marginal
    D : latent dimension (zones + bearings)

Reparameterisation (gradients flow to EVERY parameter, including the copula):
    eps ~ N(0, I)  ->  v = L_c eps  ->  u = Phi(v)  ->  x = F_c^{-1}(u)  ->  theta = lo + (hi-lo) x
  * K = 1: F^{-1} is closed form,  x = (1 - (1-u)^(1/b))^(1/a).
  * K > 1: x* = bisection(u) with no gradient, then the IMPLICIT reparameterisation
        x = x* - (F(x*; a,b,w) - u) / stop_grad(f(x*))
    whose value is x* and whose gradient is dx/dpsi = -(dF/dpsi)/f,  dx/du = 1/f
    (implicit function theorem). This replaces the custom-gradient of the old code,
    which returned None for u, so the copula correlation received no gradient from
    the physics terms.
"""

from __future__ import annotations

import math

import numpy as np
from . import ks28_tf_setup  # noqa: F401  (forces the TensorFlow backend of Keras)
import tensorflow as tf

EPS = 1e-6
LOG_A_MIN, LOG_A_MAX = math.log(0.3), math.log(3000.0)    # Kumaraswamy shape range


# =============================================================================
# 1. Encoder raw output  ->  distribution parameters
# =============================================================================
def raw_sizes(C, K, D):
    n_off = D * (D - 1) // 2
    return dict(logits_pi=C, a=C * K * D, b=C * K * D, w=C * K * D, off=C * n_off, diag=C * D)


def _tril_index(D):
    """Index matrix to build L from [offdiag (D(D-1)/2), diag (D), 0]."""
    n_off = D * (D - 1) // 2
    idx = np.full((D, D), n_off + D, dtype=np.int32)        # upper part -> the trailing zero
    r, c = np.tril_indices(D, k=-1)
    idx[r, c] = np.arange(n_off)
    idx[np.arange(D), np.arange(D)] = n_off + np.arange(D)
    return idx


def raw_to_params(raw, C, K, D, lo, hi, copula="full"):
    """Split and constrain the encoder output (B, n_raw) -> dict of tensors."""
    s = raw_sizes(C, K, D)
    raw = tf.cast(raw, tf.float32)                     # independent of the Keras floatx setting
    parts = tf.split(raw, [s["logits_pi"], s["a"], s["b"], s["w"], s["off"], s["diag"]], axis=-1)
    logits_pi, ra, rb, rw, roff, rdiag = parts
    B = tf.shape(raw)[0]

    log_pi = tf.nn.log_softmax(logits_pi, axis=-1)                                   # (B, C)
    span = LOG_A_MAX - LOG_A_MIN
    a = tf.exp(LOG_A_MIN + span * tf.sigmoid(tf.reshape(ra, [B, C, K, D])))         # (B, C, K, D)
    b = tf.exp(LOG_A_MIN + span * tf.sigmoid(tf.reshape(rb, [B, C, K, D])))
    log_w = tf.nn.log_softmax(tf.reshape(rw, [B, C, K, D]), axis=2)

    if copula == "independent" or D == 1:
        L = tf.eye(D, batch_shape=tf.stack([B, C]), dtype=raw.dtype)
    else:
        n_off = D * (D - 1) // 2
        off = tf.reshape(roff, [B, C, n_off])
        diag = tf.nn.softplus(tf.reshape(rdiag, [B, C, D])) + 1e-3
        vec = tf.concat([off, diag, tf.zeros([B, C, 1], raw.dtype)], axis=-1)
        L = tf.gather(vec, tf.constant(_tril_index(D)), axis=-1)                     # (B, C, D, D)
        L = L / tf.norm(L, axis=-1, keepdims=True)                                    # unit rows -> correlation
    return dict(log_pi=log_pi, a=a, b=b, log_w=log_w, L=L,
                lo=tf.constant(lo, raw.dtype), hi=tf.constant(hi, raw.dtype))


# =============================================================================
# 2. Kumaraswamy (mixture) marginals in x-space [0, 1]
# =============================================================================
def _log1m_xa(x, a):
    """log(1 - x^a), stable."""
    return tf.math.log(-tf.math.expm1(a * tf.math.log(x)))


def ks_mix_logpdf(x, a, b, log_w):
    """x (..., D); a, b, log_w (..., K, D)  ->  (..., D)."""
    xk = tf.clip_by_value(x, EPS, 1.0 - EPS)[..., None, :]
    lp = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * tf.math.log(xk) + (b - 1.0) * _log1m_xa(xk, a)
    return tf.reduce_logsumexp(log_w + lp, axis=-2)


def ks_mix_cdf(x, a, b, log_w):
    xk = tf.clip_by_value(x, EPS, 1.0 - EPS)[..., None, :]
    cdf_k = -tf.math.expm1(b * _log1m_xa(xk, a))
    return tf.reduce_sum(tf.exp(log_w) * cdf_k, axis=-2)


def ks_mix_quantile(u, a, b, log_w, n_iter=40):
    """Reparameterised inverse CDF (see module docstring). u (..., D)."""
    if a.shape[-2] == 1:
        a1, b1 = a[..., 0, :], b[..., 0, :]
        t = -tf.math.expm1(tf.math.log1p(-u) / b1)
        t = tf.clip_by_value(t, 1e-30, 1.0)
        return tf.exp(tf.math.log(t) / a1)
    # ---- bisection without gradient
    a_s, b_s, w_s, u_s = (tf.stop_gradient(t) for t in (a, b, log_w, u))
    lo = tf.zeros_like(u_s)
    hi = tf.ones_like(u_s)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        less = ks_mix_cdf(mid, a_s, b_s, w_s) < u_s
        lo = tf.where(less, mid, lo)
        hi = tf.where(less, hi, mid)
    x_star = tf.stop_gradient(0.5 * (lo + hi))
    # ---- implicit reparameterisation
    pdf = tf.stop_gradient(tf.exp(ks_mix_logpdf(x_star, a_s, b_s, w_s)))
    pdf = tf.maximum(pdf, 1e-3)
    return x_star - (ks_mix_cdf(x_star, a, b, log_w) - u) / pdf


# =============================================================================
# 3. Gaussian copula
# =============================================================================
_SQRT2 = math.sqrt(2.0)


def std_normal_cdf(v):
    return 0.5 * tf.math.erfc(-v / _SQRT2)


def gauss_copula_logpdf(u, L):
    """u (..., D), L (..., D, D) broadcastable -> log c(u) (...)."""
    v = tf.math.ndtri(tf.clip_by_value(u, EPS, 1.0 - EPS))
    D = tf.shape(L)[-1]
    bshape = tf.broadcast_dynamic_shape(tf.shape(v)[:-1], tf.shape(L)[:-2])
    Lb = tf.broadcast_to(L, tf.concat([bshape, [D, D]], 0))
    vb = tf.broadcast_to(v, tf.concat([bshape, [D]], 0))
    y = tf.linalg.triangular_solve(Lb, vb[..., None], lower=True)[..., 0]       # L y = v
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lb)), axis=-1)
    return -0.5 * tf.reduce_sum(y ** 2, -1) - logdet + 0.5 * tf.reduce_sum(vb ** 2, -1)


# =============================================================================
# 4. Mixture of copulas: stratified sampling and log-density
# =============================================================================
def sample_stratified(p, H, seed=None):
    """H samples from EVERY component -> theta (B, C, H, D) and x (B, C, H, D)."""
    L = p["L"]
    B, C, D = tf.shape(L)[0], tf.shape(L)[1], tf.shape(L)[-1]
    eps = tf.random.normal(tf.stack([B, C, H, D]), dtype=L.dtype, seed=seed)
    v = tf.einsum("bcij,bchj->bchi", L, eps)
    u = tf.clip_by_value(std_normal_cdf(v), EPS, 1.0 - EPS)
    a, b, lw = (t[:, :, None] for t in (p["a"], p["b"], p["log_w"]))          # (B, C, 1, K, D)
    x = ks_mix_quantile(u, a, b, lw)
    theta = p["lo"] + (p["hi"] - p["lo"]) * x
    return theta, x


def log_q_all_components(x, p):
    """log q_c'(x) for x (B, S1, S2, D) under every component c' -> (B, S1, S2, C) (x-space)."""
    xe = x[:, :, :, None, :]                                                     # (B, S1, S2, 1, D)
    a, b, lw = (t[:, None, None] for t in (p["a"], p["b"], p["log_w"]))       # (B, 1, 1, C, K, D)
    lf = tf.reduce_sum(ks_mix_logpdf(xe, a, b, lw), axis=-1)                    # (B, S1, S2, C)
    u = ks_mix_cdf(xe, a, b, lw)                                                 # (B, S1, S2, C, D)
    lc = gauss_copula_logpdf(u, p["L"][:, None, None])                          # (B, S1, S2, C)
    return lf + lc


def log_q(theta, p):
    """log q(theta) for theta (B, S1, S2, D) -> (B, S1, S2) (theta-space density)."""
    x = (theta - p["lo"]) / (p["hi"] - p["lo"])
    lc = log_q_all_components(x, p) + p["log_pi"][:, None, None, :]
    return tf.reduce_logsumexp(lc, axis=-1) - tf.reduce_sum(tf.math.log(p["hi"] - p["lo"]))


# =============================================================================
# 5. Priors (mirror of ks28_priors.py)
# =============================================================================
_LOG0 = -1e4          # finite stand-in for log(0), keeps gradients finite


def _beta_logpdf(x, a, b):
    inside = (x >= 0.0) & (x <= 1.0)          # closed support: float32 samples can hit z = 1
    xs = tf.clip_by_value(x, 1e-7, 1.0 - 1e-7)
    lbeta = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
    out = (a - 1.0) * tf.math.log(xs) + (b - 1.0) * tf.math.log1p(-xs) - lbeta
    return tf.where(inside, out, _LOG0 * tf.ones_like(out))


class ScenarioPriorTF:
    """log p(theta) of the scenario prior (PriorConfig) for theta (..., D)."""

    def __init__(self, cfg, prior, dtype=tf.float32):
        self.nz = cfg.n_zones
        self.support = cfg.support_latent
        self.dt = dtype
        c = lambda v: tf.constant(v, dtype)
        self.band, self.hb = c(prior.healthy_band), c(prior.healthy_beta)
        self.zlb, self.zw = c(cfg.z_lb), c(cfg.z_ub - cfg.z_lb)
        self.da, self.db = c(prior.dam_a), c(prior.dam_b)
        pn = prior.probs_n_damaged(cfg.n_zones)
        with np.errstate(divide="ignore"):
            lw = np.log(pn) - np.log([math.comb(cfg.n_zones, n) for n in range(cfg.n_zones + 1)])
        self.log_wn = c(np.maximum(lw, _LOG0))
        self.eta_lo, self.eta_hi = cfg.eta_lb, cfg.eta_ub
        s, pa = prior.eta_nominal_std, prior.p_support_anomaly
        from scipy.special import ndtr
        Z = ndtr(cfg.eta_ub / s) - ndtr(cfg.eta_lb / s)
        self.eta_s = c(s)
        self.log_tn_const = c(np.log1p(-pa) - np.log(s * np.sqrt(2 * np.pi)) - np.log(Z))
        self.log_u = c(np.log(pa) - np.log(cfg.eta_ub - cfg.eta_lb))

    def log_h(self, z):
        return _beta_logpdf((1.0 - z) / self.band, tf.constant(1.0, self.dt), self.hb) - tf.math.log(self.band)

    def log_d(self, z):
        return _beta_logpdf((z - self.zlb) / self.zw, self.da, self.db) - tf.math.log(self.zw)

    def log_eta(self, e):
        out = tf.reduce_logsumexp(tf.stack([self.log_tn_const - 0.5 * (e / self.eta_s) ** 2,
                                            self.log_u * tf.ones_like(e)], -1), -1)
        inside = (e >= self.eta_lo) & (e <= self.eta_hi)
        return tf.where(inside, out, _LOG0 * tf.ones_like(out))

    def __call__(self, theta):
        z = theta[..., :self.nz]
        lh, ld = self.log_h(z), self.log_d(z)
        # dp[n]: log-sum over subsets of size n (list of tensors, shape (...))
        neg = _LOG0 * tf.ones_like(z[..., 0])
        dp = [tf.zeros_like(z[..., 0])] + [neg] * self.nz
        for k in range(self.nz):
            new = [dp[0] + lh[..., k]]
            for n in range(1, self.nz + 1):
                new.append(tf.reduce_logsumexp(tf.stack([dp[n] + lh[..., k], dp[n - 1] + ld[..., k]], -1), -1))
            dp = new
        lp = tf.reduce_logsumexp(tf.stack(dp, -1) + self.log_wn, axis=-1)
        if self.support:
            lp = lp + self.log_eta(theta[..., self.nz]) + self.log_eta(theta[..., self.nz + 1])
        return lp


class UniformPriorTF:
    """Uniform prior on the latent box (constant; samples of q never leave the box)."""

    def __init__(self, cfg, dtype=tf.float32):
        self.c = tf.constant(-float(np.sum(np.log(cfg.latent_upper - cfg.latent_lower))), dtype)

    def __call__(self, theta):
        return self.c * tf.ones_like(theta[..., 0])
