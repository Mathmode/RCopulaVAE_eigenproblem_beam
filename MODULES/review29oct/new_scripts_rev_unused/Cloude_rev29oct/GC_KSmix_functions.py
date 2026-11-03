# -*- coding: utf-8 -*-
"""
GC_KSmix_functions.py -- Gaussian copula with Kumaraswamy-mixture marginals (TensorFlow).

q(theta | y) = c_R(u) * prod_i f_i(theta_i),   u_i = F_i(theta_i)

  marginal i : theta_i = lo_i + (hi_i - lo_i) x_i,  x_i ~ sum_k w_ik Kum(a_ik, b_ik),
               Kum CDF F(x) = 1 - (1 - x^a)^b  (closed form), a, b >= 1
  copula     : Gaussian with correlation R = L L^T, L lower triangular with
               unit-norm rows (diag(R) = 1)

All functions are pure, use float64, and take a parameter dictionary
  {"logw": (B, D, K), "a": (B, D, K), "b": (B, D, K), "L": (B, D, D)}.

Sampling uses an exact implicit reparameterization: the quantile x is found by
bisection and its gradient with respect to (a, b, w) and to the copula sample
g follows from the implicit-function theorem,
    dx/dpsi = -(dF/dpsi) / f(x),     dx/dg = phi(g) / f(x),
implemented without custom gradients or clipping (see reparam_quantile).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.special import ndtri

DT = tf.float64
_EPS_X = 1e-12
_SQRT2 = np.sqrt(2.0)


def std_normal_cdf(g):
    return 0.5 * tf.math.erfc(-g / _SQRT2)


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0, accurate in both regimes."""
    return tf.where(x > -0.693, tf.math.log(-tf.math.expm1(x)), tf.math.log1p(-tf.exp(x)))


def _component_terms(x, a, b):
    """x (..., D) -> per-component log-pdf and log-survival (..., D, K)."""
    x = tf.clip_by_value(x, _EPS_X, 1.0 - _EPS_X)[..., None]
    logx = tf.math.log(x)
    log1m_xa = _log1mexp(a * logx)
    log_pdf = tf.math.log(a) + tf.math.log(b) + (a - 1.0) * logx + (b - 1.0) * log1m_xa
    return log_pdf, b * log1m_xa


def mixture_eval(x, logw, a, b):
    """Mixture log-pdf, CDF and survival function at x (..., D) in [0, 1]."""
    log_pdf_k, log_surv_k = _component_terms(x, a, b)
    w = tf.exp(logw)
    log_pdf = tf.reduce_logsumexp(logw + log_pdf_k, axis=-1)
    surv = tf.reduce_sum(w * tf.exp(log_surv_k), axis=-1)
    cdf = tf.reduce_sum(w * (-tf.math.expm1(log_surv_k)), axis=-1)
    return log_pdf, cdf, surv


def _with_sample_axis(params):
    return params["logw"][:, None], params["a"][:, None], params["b"][:, None]


def mixture_quantile(g, logw, a, b, n_iter=48):
    """x with F(x) = Phi(g) by bisection, no gradient. The upper tail solves S(x) = Phi(-g)."""
    g = tf.stop_gradient(g)
    logw, a, b = tf.stop_gradient(logw), tf.stop_gradient(a), tf.stop_gradient(b)
    upper = g > 0.0
    target = tf.where(upper, std_normal_cdf(-g), std_normal_cdf(g))
    lo, hi = tf.zeros_like(g), tf.ones_like(g)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        _, cdf, surv = mixture_eval(mid, logw, a, b)
        go_right = tf.where(upper, surv > target, cdf < target)
        lo = tf.where(go_right, mid, lo)
        hi = tf.where(go_right, hi, mid)
    return 0.5 * (lo + hi)


def reparam_quantile(g, logw, a, b):
    """
    x = F^-1(Phi(g)) with exact implicit gradients: x = x0 + r / stop_gradient(f(x0)),
    where r = Phi(g) - F(x0) (or S(x0) - Phi(-g) in the upper tail) is zero in value.
    """
    x0 = mixture_quantile(g, logw, a, b)
    log_pdf, cdf, surv = mixture_eval(x0, logw, a, b)
    r = tf.where(g > 0.0, surv - std_normal_cdf(-g), std_normal_cdf(g) - cdf)
    x = x0 + r / tf.maximum(tf.stop_gradient(tf.exp(log_pdf)), 1e-10)
    return tf.clip_by_value(x, _EPS_X, 1.0 - _EPS_X)


def copula_cholesky(off_diag_raw, diag_raw, n_dims):
    """Unit-row-norm lower-triangular L from unconstrained network outputs (B, D(D-1)/2), (B, D)."""
    B = tf.shape(diag_raw)[0]
    tril = np.stack(np.tril_indices(n_dims, -1), axis=-1)
    L = tf.scatter_nd(tril, tf.transpose(off_diag_raw), (n_dims, n_dims, B))
    L = tf.transpose(L, (2, 0, 1)) + tf.linalg.diag(tf.nn.softplus(diag_raw) + 1e-3)
    return L / tf.norm(L, axis=-1, keepdims=True)


def copula_log_density(g, L):
    """log c_R(u) for g = Phi^-1(u), g (B, H, D), L (B, D, D) -> (B, H)."""
    Lb = tf.broadcast_to(L[:, None], tf.concat([tf.shape(g)[:2], tf.shape(L)[1:]], 0))
    eps = tf.linalg.triangular_solve(Lb, g[..., None], lower=True)[..., 0]
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), -1)[:, None]
    return -0.5 * tf.reduce_sum(eps ** 2, -1) + 0.5 * tf.reduce_sum(g ** 2, -1) - logdet


def log_q(params, theta, lo, hi):
    """log q(theta) for theta (B, D) or (B, H, D)."""
    single = len(theta.shape) == 2
    th = theta[:, None] if single else theta
    logw, a, b = _with_sample_axis(params)
    log_pdf, cdf, surv = mixture_eval((th - lo) / (hi - lo), logw, a, b)
    tiny = 1e-300
    g = tf.where(cdf < 0.5, tf.math.ndtri(tf.maximum(cdf, tiny)), -tf.math.ndtri(tf.maximum(surv, tiny)))
    out = copula_log_density(g, params["L"]) + tf.reduce_sum(log_pdf - tf.math.log(hi - lo), -1)
    return out[:, 0] if single else out


def sample(params, n, lo, hi, seed=None, return_log_q=False):
    """
    Reparameterized samples theta (B, n, D) and optionally log q(theta) (B, n).
    seed: None -> stateful RNG; int -> stateless (reproducible, e.g. validation).
    """
    L = params["L"]
    shape = tf.stack([tf.shape(L)[0], n, tf.shape(L)[1]])
    if seed is None:
        eps = tf.random.normal(shape, dtype=DT)
    else:
        eps = tf.random.stateless_normal(shape, seed=[seed, 0], dtype=DT)
    g = tf.einsum("bij,bhj->bhi", L, eps)
    logw, a, b = _with_sample_axis(params)
    x = reparam_quantile(g, logw, a, b)
    theta = lo + (hi - lo) * x
    if not return_log_q:
        return theta
    log_pdf, _, _ = mixture_eval(x, logw, a, b)
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), -1)[:, None]
    log_c = -0.5 * tf.reduce_sum(eps ** 2, -1) + 0.5 * tf.reduce_sum(g ** 2, -1) - logdet
    return theta, log_c + tf.reduce_sum(log_pdf - tf.math.log(hi - lo), -1)


def marginal_quantile(params, prob, lo, hi):
    """Exact marginal quantile of every parameter (B, D) for a probability prob in (0, 1)."""
    g = tf.fill(tf.shape(params["a"])[:2], tf.constant(ndtri(prob), DT))
    return lo + (hi - lo) * mixture_quantile(g, params["logw"], params["a"], params["b"])


def marginal_pdf_grid(params, grid, lo, hi):
    """Marginal densities on a theta grid (G, D) -> (B, G, D)."""
    logw, a, b = _with_sample_axis(params)
    log_pdf, _, _ = mixture_eval((grid[None] - lo) / (hi - lo), logw, a, b)
    return tf.exp(log_pdf) / (hi - lo)


def correlation(params):
    return tf.matmul(params["L"], params["L"], transpose_b=True)
