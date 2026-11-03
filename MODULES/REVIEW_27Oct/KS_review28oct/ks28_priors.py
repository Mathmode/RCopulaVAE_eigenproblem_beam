# -*- coding: utf-8 -*-
"""
ks28_priors.py -- Scenario prior p(theta): sampling AND exact log-density (NumPy).

Why a scenario prior? (Reviewer 1 c.6, Reviewer 3 c.3, and the supervisor's remark)
-------------------------------------------------------------------------------
Under U[0.5, 1]^n almost every training structure is damaged everywhere:
P(all 6 zones > 0.97) = 0.06^6 ~ 5e-8.  Real bridges are mostly healthy and,
when damaged, the damage is usually localised.  The prior below encodes that
through a hierarchical story (see PriorConfig):

    n_d ~ Cat(p_n_damaged) ;  S ~ Uniform{subsets of size n_d}
    z_k ~ h(.)  if k not in S   (healthy band [1-band, 1])
    z_k ~ d(.)  if k in S       (damage density on [z_lb, 1], mild damage favoured)
    eta_j ~ (1-p_a) TN(0, s_eta) + p_a U[eta_lb, eta_ub]

Its density is a finite mixture over the subsets S:

    p(z) = sum_n  P(n)/C(nz, n)  sum_{|S| = n}  prod_{k in S} d(z_k)  prod_{k notin S} h(z_k)

which we evaluate exactly with an O(nz^2) dynamic programme in log space
(elementary symmetric sums), so it can be used
  * to SAMPLE the training scenarios (design of experiments),
  * as the PRIOR inside the ELBO           (ks28_tf_distributions.log_prior_tf),
  * as the PRIOR of the ground-truth posterior (ks28_ground_truth).
The same density in the three places is what makes the comparison fair.

Decoupling (Reviewer 3 c.3): the density used to generate training data and the
Bayesian prior in the ELBO are independent choices in this code. The amortised
ELBO targets p(theta | m) proportional to p(m | theta) p(theta) for EVERY m, whatever
distribution the m's were drawn from; the data distribution only decides where
the encoder is accurate. `ks28_dataset` can therefore generate data from a
broader "design" density (e.g. uniform) while the ELBO uses this prior.
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.special import betaln, ndtr

from .ks28_config import BridgeConfig, PriorConfig


# =============================================================================
# 1-D component densities
# =============================================================================
def _beta_logpdf(x, a, b):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b)
    return np.where((x > 0) & (x < 1), out, -np.inf)


def log_h(z, prior: PriorConfig):
    """Healthy-band density: z = 1 - band * Beta(1, healthy_beta)."""
    x = (1.0 - np.asarray(z)) / prior.healthy_band
    return _beta_logpdf(x, 1.0, prior.healthy_beta) - np.log(prior.healthy_band)


def log_d(z, prior: PriorConfig, cfg: BridgeConfig):
    """Damage density: z = z_lb + (1 - z_lb) * Beta(dam_a, dam_b)."""
    w = cfg.z_ub - cfg.z_lb
    x = (np.asarray(z) - cfg.z_lb) / w
    return _beta_logpdf(x, prior.dam_a, prior.dam_b) - np.log(w)


def log_eta(eta, prior: PriorConfig, cfg: BridgeConfig):
    """Bearing factor density: (1-p_a) truncated N(0, s) + p_a Uniform on [eta_lb, eta_ub]."""
    eta = np.asarray(eta, dtype=np.float64)
    lo, hi, s, pa = cfg.eta_lb, cfg.eta_ub, prior.eta_nominal_std, prior.p_support_anomaly
    Z = ndtr(hi / s) - ndtr(lo / s)
    log_tn = -0.5 * (eta / s) ** 2 - np.log(s * np.sqrt(2 * np.pi)) - np.log(Z)
    log_u = -np.log(hi - lo) * np.ones_like(eta)
    out = np.logaddexp(np.log1p(-pa) + log_tn, np.log(pa) + log_u)
    return np.where((eta >= lo) & (eta <= hi), out, -np.inf)


# =============================================================================
# Exact joint log-density
# =============================================================================
def log_prior(theta: np.ndarray, cfg: BridgeConfig, prior: PriorConfig) -> np.ndarray:
    """log p(theta) for a batch theta (N, D)."""
    theta = np.atleast_2d(theta)
    nz = cfg.n_zones
    z = theta[:, :nz]
    lh = log_h(z, prior)                 # (N, nz)
    ld = log_d(z, prior, cfg)            # (N, nz)

    # DP over zones: dp[:, n] = log sum over subsets S (of the zones seen so far) with
    # |S| = n of prod_{S} d * prod_{not S} h
    N = z.shape[0]
    dp = np.full((N, nz + 1), -np.inf)
    dp[:, 0] = 0.0
    for k in range(nz):
        new = np.full_like(dp, -np.inf)
        new[:, 0] = dp[:, 0] + lh[:, k]
        new[:, 1:] = np.logaddexp(dp[:, 1:] + lh[:, k:k + 1], dp[:, :-1] + ld[:, k:k + 1])
        dp = new
    pn = prior.probs_n_damaged(nz)
    with np.errstate(divide="ignore"):
        log_w = np.log(pn) - np.log([comb(nz, n) for n in range(nz + 1)])
    lp = np.logaddexp.reduce(dp + log_w[None, :], axis=1)

    if cfg.support_latent:
        lp = lp + log_eta(theta[:, nz], prior, cfg) + log_eta(theta[:, nz + 1], prior, cfg)
    return lp


def log_prior_uniform(theta: np.ndarray, cfg: BridgeConfig) -> np.ndarray:
    """Uniform prior on the latent box (the prior of the original paper)."""
    theta = np.atleast_2d(theta)
    lo, hi = cfg.latent_lower, cfg.latent_upper
    inside = np.all((theta >= lo) & (theta <= hi), axis=1)
    return np.where(inside, -np.sum(np.log(hi - lo)), -np.inf)


# =============================================================================
# Sampling
# =============================================================================
def sample_prior(N: int, cfg: BridgeConfig, prior: PriorConfig, rng: np.random.Generator,
                 force_n_damaged=None, severity_range=None):
    """Draw N scenarios from the scenario prior.

    force_n_damaged : int or None. If given, every scenario has exactly this number of
                      damaged zones (used to build the evaluation families).
    severity_range  : (zmin, zmax) or None. If given, damaged zones are drawn uniformly in
                      this range instead of from d(.) (used for "mild damage" families).
    Returns theta (N, D) and a dict with the scenario metadata.
    """
    nz = cfg.n_zones
    pn = prior.probs_n_damaged(nz)
    if force_n_damaged is None:
        n_dam = rng.choice(nz + 1, size=N, p=pn)
    else:
        n_dam = np.full(N, int(force_n_damaged))

    damaged = np.zeros((N, nz), dtype=bool)
    for i in range(N):
        if n_dam[i] > 0:
            damaged[i, rng.choice(nz, size=n_dam[i], replace=False)] = True

    z_h = 1.0 - prior.healthy_band * rng.beta(1.0, prior.healthy_beta, size=(N, nz))
    if severity_range is None:
        z_d = cfg.z_lb + (cfg.z_ub - cfg.z_lb) * rng.beta(prior.dam_a, prior.dam_b, size=(N, nz))
    else:
        z_d = rng.uniform(severity_range[0], severity_range[1], size=(N, nz))
    z = np.where(damaged, z_d, z_h)

    meta = dict(n_damaged=n_dam, damaged_mask=damaged)
    if cfg.support_latent:
        eta, anom = sample_eta(N, cfg, prior, rng)
        theta = np.concatenate([z, eta], axis=1)
        meta["support_anomaly"] = anom
    else:
        theta = z
    return theta, meta


def sample_eta(N, cfg: BridgeConfig, prior: PriorConfig, rng: np.random.Generator,
               force_anomaly=None):
    """Bearing log2-factors (N, 2) and the anomaly mask."""
    lo, hi, s = cfg.eta_lb, cfg.eta_ub, prior.eta_nominal_std
    if force_anomaly is None:
        anom = rng.random((N, 2)) < prior.p_support_anomaly
    else:
        anom = np.broadcast_to(np.asarray(force_anomaly, dtype=bool), (N, 2)).copy()
    # truncated normal by rejection (truncation is mild)
    eta_n = rng.normal(0.0, s, size=(N, 2))
    bad = (eta_n < lo) | (eta_n > hi)
    while np.any(bad):
        eta_n[bad] = rng.normal(0.0, s, size=bad.sum())
        bad = (eta_n < lo) | (eta_n > hi)
    eta_u = rng.uniform(lo, hi, size=(N, 2))
    return np.where(anom, eta_u, eta_n), anom


def sample_uniform_box(N: int, cfg: BridgeConfig, rng: np.random.Generator):
    """theta ~ U(box): the design density of the original paper."""
    return rng.uniform(cfg.latent_lower, cfg.latent_upper, size=(N, cfg.n_latent))


def prior_summary(cfg: BridgeConfig, prior: PriorConfig, rng=None, N=200000) -> dict:
    """A few numbers that describe the prior (for the paper / sanity checks)."""
    rng = np.random.default_rng(0) if rng is None else rng
    th, meta = sample_prior(N, cfg, prior, rng)
    z = th[:, :cfg.n_zones]
    return {
        "P(n_damaged)": prior.probs_n_damaged(cfg.n_zones).round(3).tolist(),
        "P(all zones > 0.97)": float(np.mean(np.all(z > 0.97, axis=1))),
        "P(min z < 0.7)": float(np.mean(z.min(axis=1) < 0.7)),
        "mean of damaged z": float(z[meta["damaged_mask"]].mean()),
        "uniform-box P(all zones > 0.97)": float((0.03 / (cfg.z_ub - cfg.z_lb)) ** cfg.n_zones),
    }
