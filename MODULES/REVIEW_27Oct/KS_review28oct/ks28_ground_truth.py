# -*- coding: utf-8 -*-
"""
ks28_ground_truth.py -- Reference ("ground truth") posterior by tempered Sequential Monte Carlo.

Why not the QMC + KDE of the paper?
  * In 8 dimensions a fixed grid of 4096 QMC points has almost no points inside a
    posterior that is a few % wide per coordinate -> the KDE is driven by a handful of
    weights.
  * The paper's likelihood used the training gamma as a temperature, so the "truth"
    depended on the hyper-parameter being evaluated.
Here the reference posterior is

    p(theta | m)  proportional to  p(m | theta) p(theta)

with EXACTLY the likelihood (ks28_likelihood, from the noise model) and the prior
(ks28_priors) used in the ELBO, and the decoder (training) model as forward model.
For the shifted test families this is the posterior the VAE should reproduce;
the distance between this posterior and the true theta then measures the effect of
the model-form error itself, separately from the amortisation error of the VAE.

Algorithm (adaptive tempered SMC, e.g. Del Moral et al. 2006; Chopin & Papaspiliopoulos 2020)
  pi_t(theta) ~ p(theta) p(m|theta)^t,   t: 0 -> 1
  1. N particles from the prior (exact sampler), t = 0
  2. choose the next t by bisection so that the ESS of the incremental weights = ess_target*N
  3. systematic resampling
  4. MCMC moves that leave pi_t invariant:
       - random walk with the particle covariance (scale adapted to ~30 % acceptance)
       - single-coordinate independence proposals (healthy band / full range), which
         let particles jump between "healthy" and "damaged" values of a zone and
         between mirror-symmetric solutions
  5. repeat until t = 1.  Output: equally weighted particles + log-evidence estimate.
Tempering is what makes SMC reliable for MULTIMODAL posteriors: particles populate all
modes at t = 0 and the relative mass of the modes is set by the weights, not by a
chain that has to jump between them.
"""

from __future__ import annotations

import time

import numpy as np

from .ks28_config import BridgeConfig, NoiseConfig, PriorConfig
from .ks28_fe_model import FEBasis, forward
from .ks28_likelihood import neg_log_likelihood
from .ks28_priors import log_prior, log_prior_uniform, sample_prior, sample_uniform_box


def _systematic_resample(logw, rng):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    N = len(w)
    pos = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)


def _ess(logw):
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / np.sum(w ** 2)


class GroundTruthSMC:
    def __init__(self, cfg: BridgeConfig, noise: NoiseConfig, prior_cfg: PriorConfig, basis: FEBasis,
                 prior_kind="scenario", n_modes_used=None, n_particles=4000, n_mcmc=10,
                 ess_target=0.5, seed=0):
        self.cfg, self.noise, self.prior_cfg, self.basis = cfg, noise, prior_cfg, basis
        self.prior_kind = prior_kind
        self.nm = cfg.n_modes if n_modes_used is None else int(n_modes_used)
        self.N, self.n_mcmc, self.ess_target = n_particles, n_mcmc, ess_target
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = cfg.latent_lower, cfg.latent_upper

    # --------------------------------------------------------------- model pieces
    def log_prior(self, th):
        if self.prior_kind == "scenario":
            return log_prior(th, self.cfg, self.prior_cfg)
        return log_prior_uniform(th, self.cfg)

    def log_lik(self, th, f_obs, phi_obs):
        out = np.full(th.shape[0], -np.inf)
        ok = np.all((th >= self.lo) & (th <= self.hi), axis=1)
        if np.any(ok):
            f, p = forward(th[ok], self.basis, self.cfg, n_modes=self.nm)
            out[ok] = -neg_log_likelihood(f_obs[None], phi_obs[None], f, p, self.noise, self.nm)
        return out

    def sample_prior(self, N):
        if self.prior_kind == "scenario":
            return sample_prior(N, self.cfg, self.prior_cfg, self.rng)[0]
        return sample_uniform_box(N, self.cfg, self.rng)

    def _log_g(self, d, v):
        """Log density of the single-coordinate independence proposal for coordinate d."""
        if d < self.cfg.n_zones:
            band = self.prior_cfg.healthy_band
            return np.log(0.5 * (v > 1 - band) / band + 0.5 / (self.cfg.z_ub - self.cfg.z_lb))
        return -np.log(self.hi[d] - self.lo[d]) * np.ones_like(v)

    def _draw_g(self, d, n):
        if d < self.cfg.n_zones:
            band = self.prior_cfg.healthy_band
            u = self.rng.random(n) < 0.5
            return np.where(u, self.rng.uniform(1 - band, 1.0, n), self.rng.uniform(self.cfg.z_lb, 1.0, n))
        return self.rng.uniform(self.lo[d], self.hi[d], n)

    # --------------------------------------------------------------- MCMC kernel
    def _mutate(self, th, lp, ll, t, f_obs, phi_obs, scale):
        N, D = th.shape
        cov = np.cov(th.T) + 1e-10 * np.eye(D)
        Lc = np.linalg.cholesky(cov)
        acc_rw, acc_ind = [], []
        for _ in range(self.n_mcmc):
            # (a) random walk
            prop = th + scale * (self.rng.standard_normal((N, D)) @ Lc.T) * (2.38 / np.sqrt(D))
            lp_p = self.log_prior(prop)
            ll_p = np.where(np.isfinite(lp_p), self.log_lik(prop, f_obs, phi_obs), -np.inf)
            la = (lp_p + t * ll_p) - (lp + t * ll)
            acc = np.log(self.rng.random(N)) < la
            th[acc], lp[acc], ll[acc] = prop[acc], lp_p[acc], ll_p[acc]
            acc_rw.append(acc.mean())
            # (b) single-coordinate independence proposal
            d_idx = self.rng.integers(0, D, N)
            prop = th.copy()
            lg_old = np.zeros(N); lg_new = np.zeros(N)
            for d in range(D):
                m = d_idx == d
                if not np.any(m):
                    continue
                new = self._draw_g(d, m.sum())
                lg_old[m] = self._log_g(d, th[m, d])
                lg_new[m] = self._log_g(d, new)
                prop[m, d] = new
            lp_p = self.log_prior(prop)
            ll_p = np.where(np.isfinite(lp_p), self.log_lik(prop, f_obs, phi_obs), -np.inf)
            la = (lp_p + t * ll_p) - (lp + t * ll) + lg_old - lg_new
            acc = np.log(self.rng.random(N)) < la
            th[acc], lp[acc], ll[acc] = prop[acc], lp_p[acc], ll_p[acc]
            acc_ind.append(acc.mean())
        a = float(np.mean(acc_rw))
        scale = float(np.clip(scale * np.exp(2.0 * (a - 0.3)), 0.05, 2.0))
        return th, lp, ll, scale, a, float(np.mean(acc_ind))

    # --------------------------------------------------------------- run
    def run(self, f_obs, phi_obs, verbose=False, max_stages=300):
        t0 = time.time()
        th = self.sample_prior(self.N)
        lp = self.log_prior(th)
        ll = self.log_lik(th, f_obs, phi_obs)
        t, logZ, scale, stages = 0.0, 0.0, 0.5, 0
        log = []
        while t < 1.0 and stages < max_stages:
            lo_d, hi_d = 0.0, 1.0 - t
            if _ess(hi_d * ll) >= self.ess_target * self.N:
                delta = hi_d
            else:
                for _ in range(60):
                    mid = 0.5 * (lo_d + hi_d)
                    if _ess(mid * ll) >= self.ess_target * self.N:
                        lo_d = mid
                    else:
                        hi_d = mid
                delta = max(lo_d, 1e-12)
            logw = delta * ll
            m = logw.max()
            logZ += m + np.log(np.mean(np.exp(logw - m)))
            t = min(1.0, t + delta)
            idx = _systematic_resample(logw, self.rng)
            th, lp, ll = th[idx], lp[idx], ll[idx]
            th, lp, ll, scale, a_rw, a_ind = self._mutate(th, lp, ll, t, f_obs, phi_obs, scale)
            stages += 1
            log.append((t, a_rw, a_ind, len(np.unique(th[:, 0]))))
            if verbose:
                print(f"  stage {stages:3d} t={t:.4g} acc_rw={a_rw:.2f} acc_ind={a_ind:.2f}")
        return dict(particles=th, log_lik=ll, log_prior=lp, log_evidence=logZ, n_stages=stages,
                    time_s=time.time() - t0, log=np.array(log))


def run_ground_truth_for_cases(posterior_npz: dict, case_ids, cfg, noise, prior_cfg, basis,
                               n_particles=4000, n_mcmc=10, prior_kind="scenario", n_modes_used=None,
                               seed=0, verbose=True):
    """Ground truth for selected cases of one split -> dict case_id -> result."""
    out = {}
    for i, c in enumerate(case_ids):
        smc = GroundTruthSMC(cfg, noise, prior_cfg, basis, prior_kind, n_modes_used, n_particles,
                             n_mcmc, seed=seed + int(c))
        r = smc.run(posterior_npz["f_obs"][c], posterior_npz["phi_obs"][c])
        out[int(c)] = r
        if verbose:
            print(f"[GT] case {c}: {r['n_stages']} stages, {r['time_s']:.1f}s, log Z = {r['log_evidence']:.1f}")
    return out
