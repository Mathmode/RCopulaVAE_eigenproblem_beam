# -*- coding: utf-8 -*-
"""
ks28_q_numpy.py -- NumPy implementation of the approximate posterior q(theta | m).

q is a MIXTURE OF GAUSSIAN COPULAS with Kumaraswamy(-mixture) marginals:

    q(theta) = sum_c pi_c  q_c(theta)
    q_c(theta) = c_Gauss( F_c1(x_1), ..., F_cD(x_D) ; R_c )  prod_d f_cd(x_d) / (hi_d - lo_d)
    x_d = (theta_d - lo_d) / (hi_d - lo_d)  in [0, 1]
    f_cd = sum_k w_cdk Kumaraswamy(x; a_cdk, b_cdk)      (K = 1 -> single Kumaraswamy)

With C = 1 this is exactly the model of the paper (one Gaussian copula, Kumaraswamy-
mixture marginals). C > 1 adds JOINT modes: e.g. "damage in zone 2" OR "damage in zone 5"
(mirror solution of a symmetric bridge) is represented by two components, each with its
own marginals AND its own correlation matrix. A single Gaussian copula with bimodal
marginals cannot do this: it necessarily puts mass in the two "wrong" corners
(both zones damaged / both healthy).

This file is the reference implementation used by the postprocessing (no TensorFlow
needed) and by the smoke test, which checks that the TensorFlow version
(ks28_tf_distributions.py) gives the same numbers.

Parameter dictionary (all NumPy arrays, N = number of observations):
    log_pi (N, C)   a, b, log_w (N, C, K, D)   L (N, C, D, D)   lo, hi (D,)
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri, logsumexp

EPS = 1e-6


# ----------------------------------------------------------------------------- Kumaraswamy
def ks_log_cdf_terms(x, a, b):
    """log(1 - x^a) for x in (0,1) (stable)."""
    return np.log(-np.expm1(a * np.log(x)))


def ks_mix_logpdf(x, a, b, log_w):
    """x (..., D); a, b, log_w (..., K, D) -> log f(x) (..., D)."""
    xk = np.clip(x, EPS, 1 - EPS)[..., None, :]
    l1mxa = ks_log_cdf_terms(xk, a, b)
    lp = np.log(a) + np.log(b) + (a - 1) * np.log(xk) + (b - 1) * l1mxa
    return logsumexp(log_w + lp, axis=-2)


def ks_mix_cdf(x, a, b, log_w):
    xk = np.clip(x, EPS, 1 - EPS)[..., None, :]
    cdf_k = -np.expm1(b * ks_log_cdf_terms(xk, a, b))      # 1 - (1 - x^a)^b
    return np.sum(np.exp(log_w) * cdf_k, axis=-2)


def ks_mix_quantile(u, a, b, log_w, n_iter=50):
    """Inverse CDF. Closed form for K = 1, bisection otherwise. u (..., D)."""
    if a.shape[-2] == 1:
        a1, b1 = a[..., 0, :], b[..., 0, :]
        t = -np.expm1(np.log1p(-u) / b1)                       # 1 - (1-u)^(1/b)
        return np.exp(np.log(np.clip(t, 1e-300, 1.0)) / a1)
    lo = np.zeros_like(u)
    hi = np.ones_like(u)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        less = ks_mix_cdf(mid, a, b, log_w) < u
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- copula
def gauss_copula_logpdf(u, L):
    """u (..., D) in (0,1), L (..., D, D) correlation Cholesky -> log c(u)."""
    v = ndtri(np.clip(u, EPS, 1 - EPS))
    y = np.linalg.solve(L, v[..., None])[..., 0]            # L y = v  (L lower-triangular)
    logdet = np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * np.sum(y ** 2, -1) - logdet + 0.5 * np.sum(v ** 2, -1)


# ----------------------------------------------------------------------------- q
def _component_logpdf(x, p, c):
    """log q_c in x-space for x (N, S, D) -> (N, S)."""
    a, b, lw, L = (p["a"][:, c][:, None], p["b"][:, c][:, None], p["log_w"][:, c][:, None],
                   p["L"][:, c][:, None])
    lf = ks_mix_logpdf(x, a, b, lw).sum(-1)
    u = ks_mix_cdf(x, a, b, lw)
    return lf + gauss_copula_logpdf(u, L)


def log_q(theta, p):
    """log q(theta | m) for theta (N, S, D) -> (N, S)."""
    lo, hi = p["lo"], p["hi"]
    x = (theta - lo) / (hi - lo)
    C = p["log_pi"].shape[1]
    comps = np.stack([p["log_pi"][:, c][:, None] + _component_logpdf(x, p, c) for c in range(C)], -1)
    return logsumexp(comps, axis=-1) - np.sum(np.log(hi - lo))


def sample_q(p, S, rng):
    """Draw S samples per observation -> theta (N, S, D), component index (N, S)."""
    N, C = p["log_pi"].shape
    D = p["L"].shape[-1]
    pi = np.exp(p["log_pi"])
    pi = pi / pi.sum(1, keepdims=True)
    comp = np.stack([rng.choice(C, size=S, p=pi[n]) for n in range(N)])          # (N, S)
    eps = rng.standard_normal((N, S, D))
    L = np.take_along_axis(p["L"], comp[:, :, None, None], axis=1)                   # (N, S, D, D)
    v = np.einsum("nsij,nsj->nsi", L, eps)
    u = np.clip(ndtr(v), EPS, 1 - EPS)
    a = np.take_along_axis(p["a"], comp[:, :, None, None], axis=1)                   # (N, S, K, D)
    b = np.take_along_axis(p["b"], comp[:, :, None, None], axis=1)
    lw = np.take_along_axis(p["log_w"], comp[:, :, None, None], axis=1)
    x = ks_mix_quantile(u, a, b, lw)
    return p["lo"] + (p["hi"] - p["lo"]) * x, comp


def subset_params(p, idx):
    """Parameters of a subset of observations."""
    out = {k: (v[idx] if k not in ("lo", "hi") else v) for k, v in p.items()}
    return out
