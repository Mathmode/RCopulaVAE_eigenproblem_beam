#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/posterior_quality.py

How close is the amortised posterior q(z | m) to the EXACT posterior p(z | m), without reference samples? For every
observation m and S samples z_s ~ q(z | m) (any family of posterior_families.py):

    log w_s  = log p(m | z_s) + log p(z_s) - log q(z_s | m)    (likelihood of training and of the SMC reference)
    ELBO     = mean_s log w_s                                    -ELBO = KL(q || p(.|m)) - log p(m)
    IW-ELBO  = log mean_s w_s                                   ELBO <= IW-ELBO <= log p(m)
    KL_lower = IW-ELBO - ELBO                                   lower bound of KL(q || p(.|m)) [nats]
    k_hat    = Pareto-k of the importance weights w_s           (Yao et al. 2018, ICML; Vehtari et al. 2024, JMLR)
               < 0.5: q covers the exact posterior well; 0.5-0.7: acceptable; > 0.7: q misses posterior mass

Used by MODULES/model_comparison/evaluate_posterior_quality.py (model selection) and ood_study.py (R1.5).
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from MODULES.review29oct_fixed.fe_model import BridgeModel, neg_log_likelihood


def _gpd_k(x):
    """Shape k of a generalized Pareto fitted to the exceedances x (ascending) by Zhang & Stephens (2009), with the
    weakly informative prior of PSIS (Vehtari et al. 2024); same estimator as arviz.psislw."""
    prior_bs, prior_k = 3, 10
    n = len(x)
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    bs /= prior_bs * x[int(n / 4 + 0.5) - 1]
    bs += 1.0 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x), axis=1)
    L = n * (np.log(-(bs / ks)) - ks - 1.0)
    w = 1.0 / np.sum(np.exp(L - L[:, None]), axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, bs = w[keep] / np.sum(w[keep]), bs[keep]
    k = np.mean(np.log1p(-np.sum(bs * w) * x))
    return (n * k + prior_k * 0.5) / (n + prior_k)


def psis_khat(log_w):
    """Pareto-k of the largest importance weights (tail of min(0.2 S, 3 sqrt(S)) draws)."""
    x = np.asarray(log_w, np.float64)
    x = x - np.max(x)
    n = len(x)
    n_tail = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n))))
    xs = np.sort(x)
    cutoff = max(xs[-n_tail - 1], np.log(np.finfo(float).tiny))
    tail = xs[xs > cutoff]
    if len(tail) <= 4:
        return np.inf
    exceed = np.exp(tail) - np.exp(cutoff)
    if np.all(exceed <= 0):
        return 0.0
    return float(_gpd_k(np.sort(exceed)))


def _log_weights(fam, obs, bridge, lik, obs_types, gamma2, n_samples, seed, chunk):
    """Yields (idx, log_w, nll, log_q) for every chunk of observations; arrays (n, S) of S samples of q in unit space."""
    f, v, r = obs[:3]
    rng = np.random.default_rng(seed)
    for i0 in range(0, fam.n, chunk):
        idx = np.arange(i0, min(i0 + chunk, fam.n))
        u = fam.sample_unit(n_samples, rng, idx)                                    # (n, S, D)
        log_q = fam.log_density_unit(u, idx)                                        # unit space: prior density 1
        fp, vp, rp = bridge.solve(fam.to_z(u).reshape(-1, fam.d))
        shape = (len(idx), n_samples)
        fp, vp, rp = fp.reshape(shape + fp.shape[1:]), vp.reshape(shape + vp.shape[1:]), rp.reshape(shape + rp.shape[1:])
        nll, _ = neg_log_likelihood(f[idx][:, None], v[idx][:, None], r[idx][:, None], fp, vp, rp, lik, obs_types)
        yield idx, -nll / gamma2 - log_q, nll, log_q


def mean_neg_elbo(fam, obs, bridge, obs_types, gamma2, n_samples=20, seed=0, chunk=200):
    """Mean -ELBO of a split with the estimator of evaluate_split (log w, float64), without its diagnostics: the monitor
    of the train-to-convergence protocol. The fixed seed repeats the random numbers, so two evaluations during a
    training differ by the change of q and not by Monte Carlo noise."""
    total = sum(-np.sum(np.mean(log_w, axis=1)) for _, log_w, _, _ in
                _log_weights(fam, obs, bridge, bridge.lik, obs_types, gamma2, n_samples, seed, chunk))
    return total / fam.n


def evaluate_split(fam, obs, pcfg, obs_types, gamma2, n_samples=500, seed=0, chunk=40):
    """Per-observation -ELBO, IW-ELBO, KL lower bound and Pareto-k (DataFrame). obs = (f [Hz], v, r, ...)."""
    rows = []
    for idx, log_w, nll, log_q in _log_weights(fam, obs, BridgeModel(pcfg), pcfg["likelihood"], obs_types, gamma2,
                                               n_samples, seed, chunk):
        elbo = log_w.mean(axis=1)
        iw = logsumexp(log_w, axis=1) - np.log(n_samples)
        for j, i in enumerate(idx):
            rows.append({"obs": int(i), "neg_elbo": -elbo[j], "iw_elbo": iw[j], "kl_lower": iw[j] - elbo[j],
                         "k_hat": psis_khat(log_w[j]), "E_q_NLL": float(np.mean(nll[j])) / gamma2,
                         "entropy_unit": -float(np.mean(log_q[j])),
                         "mc_se_elbo": float(np.std(log_w[j]) / np.sqrt(n_samples))})
    return pd.DataFrame(rows)


def summarize(df):
    n = len(df)
    k = df["k_hat"].to_numpy()
    return {"n_obs": n, "neg_elbo_mean": float(df["neg_elbo"].mean()),
            "neg_elbo_se": float(df["neg_elbo"].std(ddof=1) / np.sqrt(n)),
            "kl_lower_mean": float(df["kl_lower"].mean()), "kl_lower_median": float(df["kl_lower"].median()),
            "k_hat_median": float(np.median(k)), "frac_k_hat_below_0.5": float(np.mean(k < 0.5)),
            "frac_k_hat_below_0.7": float(np.mean(k < 0.7)), "frac_k_hat_above_1": float(np.mean(k > 1.0)),
            "E_q_NLL_mean": float(df["E_q_NLL"].mean()), "entropy_unit_mean": float(df["entropy_unit"].mean())}
