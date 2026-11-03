# -*- coding: utf-8 -*-
"""
priors.py -- Scenario-mixture densities: Bayesian prior p(theta) and DoE rho(theta).

    density(theta) = sum_c w_c prod_i ScaledBeta(theta_i; lo_ci, hi_ci, a_ci, b_ci)

A component c is one damage scenario with a fixed set of damaged parameters
(e.g. "zone 3 damaged", "left bearing + zone 1 damaged"); damaged parameters
use the damaged family, the others the healthy family (config.py).

* The PRIOR enters the ELBO and the reference posterior. All its component
  densities are continuous inside the support [LBOUND, UBOUND] (alpha, beta > 1
  at interior edges), which the reparameterization gradient of E_q[log p]
  requires.
* The DoE only decides which observations the encoder is trained on. When
  the optional amortized forward-KL term is used, its samples are weighted by
  p/rho; both densities share the component table, so the ratio is bounded.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.special import betaln, xlogy, logsumexp
from scipy.stats import beta as beta_dist

import config as C

SCENARIOS = list(C.PRIOR_SCENARIO_WEIGHTS.keys())


def _damaged_sets():
    """List of (scenario name, tuple of damaged parameter indices)."""
    zones = range(C.N_ZONES)
    bearings = range(C.N_ZONES, C.N_DIMS)
    out = [("healthy", ())]
    out += [("single_zone", (z,)) for z in zones]
    out += [("single_bearing", (b,)) for b in bearings]
    out += [("zone_and_bearing", (z, b)) for z in zones for b in bearings]
    out += [("two_zones", pair) for pair in combinations(zones, 2)]
    out += [("three_sites", tri) for tri in combinations(range(C.N_DIMS), 3)]
    out += [("broad", tuple(range(C.N_DIMS)))]
    return out


class MixtureDensity:
    def __init__(self, scenario_weights: dict, damaged_family, healthy_family=C.HEALTHY):
        sets = _damaged_sets()
        self.scen = np.array([s for s, _ in sets])
        self.damaged = np.array([[d in idx for d in range(C.N_DIMS)] for _, idx in sets])
        tab = np.where(self.damaged[..., None], np.array(damaged_family), np.array(healthy_family))
        self.lo, self.hi, self.a, self.b = (tab[..., k] for k in range(4))        # (C, D)
        counts = {s: np.sum(self.scen == s) for s in set(self.scen)}
        w = np.array([scenario_weights.get(s, 0.0) / counts[s] for s in self.scen])
        self.w = w / w.sum()
        with np.errstate(divide="ignore"):
            self.logw = np.log(self.w)
        self.log_norm = betaln(self.a, self.b) + np.log(self.hi - self.lo)
        self.n_dims = C.N_DIMS

    # ------------------------------------------------------------------ numpy
    def log_prob(self, theta):
        th = np.atleast_2d(theta)[:, None, :]                       # (N, 1, D)
        x = (th - self.lo) / (self.hi - self.lo)
        inside = np.all((x > 0) & (x < 1), axis=-1)                 # (N, C)
        xs = np.clip(x, 1e-12, 1 - 1e-12)
        lp = (xlogy(self.a - 1, xs) + xlogy(self.b - 1, 1 - xs) - self.log_norm).sum(-1)
        return logsumexp(np.where(inside, lp, -np.inf) + self.logw, axis=-1)

    def sample(self, n, rng, stratified=False):
        """
        i.i.d. draws, or stratified draws: exact number of samples per component
        (largest remainder) and a Latin hypercube inside each component mapped
        through the Beta quantile functions, so every damage location and every
        severity range is evenly represented.
        Returns theta (n, D), scenario name (n,), damaged mask (n, D).
        """
        if not stratified:
            c = rng.choice(len(self.w), size=n, p=self.w)
            x = rng.beta(self.a[c], self.b[c])
        else:
            raw = self.w * n
            counts = np.floor(raw).astype(int)
            counts[np.argsort(raw - counts)[::-1][:n - counts.sum()]] += 1
            c = np.repeat(np.arange(len(self.w)), counts)
            x = np.empty((n, self.n_dims))
            start = 0
            for k, nk in enumerate(counts):
                if nk == 0:
                    continue
                strata = rng.permuted(np.tile(np.arange(nk), (self.n_dims, 1)), axis=1).T
                u = (strata + rng.random((nk, self.n_dims))) / nk
                x[start:start + nk] = beta_dist.ppf(u, self.a[k], self.b[k])
                start += nk
            perm = rng.permutation(n)
            c, x = c[perm], x[perm]
        theta = self.lo[c] + (self.hi[c] - self.lo[c]) * x
        return theta, self.scen[c], self.damaged[c]

    # ------------------------------------------------------------- tensorflow
    def tf_log_prob_fn(self):
        import tensorflow as tf
        dt = tf.float64
        keep = self.w > 0
        lo, hi = tf.constant(self.lo[keep], dt), tf.constant(self.hi[keep], dt)
        am1, bm1 = tf.constant(self.a[keep] - 1, dt), tf.constant(self.b[keep] - 1, dt)
        lnorm, logw = tf.constant(self.log_norm[keep], dt), tf.constant(self.logw[keep], dt)

        def log_prob(theta):
            x = (theta[..., None, :] - lo) / (hi - lo)
            inside = tf.reduce_all((x > 0.0) & (x < 1.0), axis=-1)
            xs = tf.clip_by_value(x, 1e-12, 1.0 - 1e-12)          # safe values for the masked branch
            lp = tf.reduce_sum(tf.math.xlogy(am1, xs) + tf.math.xlogy(bm1, 1.0 - xs) - lnorm, -1)
            lp = tf.where(inside, lp, tf.constant(-1e30, dt)) + logw
            return tf.reduce_logsumexp(lp, axis=-1)

        return log_prob


def build_prior():
    return MixtureDensity(C.PRIOR_SCENARIO_WEIGHTS, C.DAMAGED_PRIOR)


def build_doe():
    return MixtureDensity(C.DOE_SCENARIO_WEIGHTS, C.DAMAGED_DOE)


def sample_four_sites(n, rng):
    """Out-of-distribution scenario (R1.5): four simultaneously damaged parameters."""
    lo, hi, a, b = C.HEALTHY
    theta = lo + (hi - lo) * rng.beta(a, b, size=(n, C.N_DIMS))
    damaged = np.zeros((n, C.N_DIMS), dtype=bool)
    for i in range(n):
        idx = rng.choice(C.N_DIMS, size=4, replace=False)
        theta[i, idx] = rng.uniform(0.50, 0.90, size=4)
        damaged[i, idx] = True
    return theta, np.array(["four_sites"] * n), damaged
