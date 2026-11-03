#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/reference_posterior_smc.py

Reference ("ground-truth") posterior for ONE observation, computed with adaptive
tempered Sequential Monte Carlo (SMC) under EXACTLY the target the VAE is trained for:

    p_gamma(z | m) ∝ exp(-E(z) / gamma^2) * prod_d Beta(x_d; alpha, beta)

with E(z) = L_freq + lambda_vert * L_MAC,vert + lambda_rot * L_MAC,rot computed with the
same 5-element decoder (Ke_matrix.npy, L_inv.npy) and the same frequency scaling.

Why this replaces sample_smooth_ground_truth_posterior_7d in postprocessing_7D_bridge.py:
  * That sampler starts 256 chains AT z_true, its mirror and a hand-made trade-off point,
    and runs only 28 post-burn-in random-walk steps, so its output mostly reflects the
    initialisation (it "knows" the answer) rather than the posterior.
  * It used a different prior (Beta(1.8, 1.05) with weight 0.08) and a different MAC weight
    for some test cases (rot_weight 0.15) than the training loss, so even a perfect VAE
    could not match it.
  * It re-weighted MCMC draws by exp(0.25 * logp), which is not a valid posterior estimate.
SMC starts from the prior (no knowledge of z_true), anneals the likelihood in, and finds
all modes that carry posterior mass.
"""

import numpy as np


class NumpyBridgeForward:
    """Batched numpy copy of the TF decoder (assembly + Cholesky-standardised eigensolve)."""
    def __init__(self, K_basis, L_inv, n_modes, mean_f, std_f, lambda_vert=1.5, lambda_rot=0.5):
        self.K_basis = np.asarray(K_basis, np.float64)
        self.L_inv = np.asarray(L_inv, np.float64)
        self.n_modes = int(n_modes)
        self.mean_f = np.asarray(mean_f, np.float64)
        self.std_f = np.asarray(std_f, np.float64)
        self.lambda_vert = float(lambda_vert)
        self.lambda_rot = float(lambda_rot)

    def modal(self, z):
        K_g = np.einsum('nd,dij->nij', z, self.K_basis)
        A = self.L_inv @ K_g @ self.L_inv.T
        lam, V = np.linalg.eigh(0.5 * (A + np.swapaxes(A, 1, 2)))
        f = np.sqrt(np.maximum(lam[:, :self.n_modes], 1e-12)) / (2.0 * np.pi)
        phi = self.L_inv.T @ V[:, :, :self.n_modes]                 # [N, dofs, M]
        vert = np.swapaxes(phi[:, 0::2, :], 1, 2)                   # [N, M, nodes]
        rot = np.swapaxes(phi[:, 1::2, :], 1, 2)
        return f, rot, vert

    @staticmethod
    def _mac(t, p):
        t = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-12)
        p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
        return np.sum(t * p, axis=-1) ** 2

    def energy(self, z, obs_f_scaled, obs_rot, obs_vert, batch=4096):
        out = np.empty(len(z))
        for i in range(0, len(z), batch):
            f, rot, vert = self.modal(z[i:i + batch])
            f_s = (np.log(np.maximum(f, 1e-12)) - self.mean_f) / self.std_f
            e_f = np.mean((obs_f_scaled[None] - f_s) ** 2, axis=-1)
            e_v = np.mean(1.0 - self._mac(obs_vert[None], vert), axis=-1)
            e_r = np.mean(1.0 - self._mac(obs_rot[None], rot), axis=-1)
            out[i:i + batch] = e_f + self.lambda_vert * e_v + self.lambda_rot * e_r
        return out


def _logit(x):
    return np.log(x) - np.log1p(-x)


def _sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def _systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), pos)
    return np.minimum(idx, n - 1)


def smc_reference_posterior(forward, obs_f_scaled, obs_rot, obs_vert, gamma, lbound,
                            prior_alpha=2.2, prior_beta=1.1, n_particles=8192,
                            n_mh=12, target_ess=0.5, n_final_mh=30, seed=0, verbose=False):
    """
    Adaptive tempered SMC. Particles live in t = logit(x) space (x = unit-scaled z) so the
    Gaussian random-walk moves are unconstrained; the log-Jacobian is included.
    Returns equally weighted samples z [n_particles, D] and a diagnostics dict.
    """
    rng = np.random.default_rng(seed)
    D = forward.K_basis.shape[0]
    g2 = gamma ** 2

    def log_prior_t(t):
        # Beta(alpha, beta) on x plus log|dx/dt| = log x + log(1-x)
        x = _sigmoid(t)
        return np.sum(prior_alpha * np.log(x) + prior_beta * np.log1p(-x), axis=-1)

    def neg_energy(t):
        x = np.clip(_sigmoid(t), 1e-9, 1 - 1e-9)
        z = lbound + (1.0 - lbound) * x
        return -forward.energy(z, obs_f_scaled, obs_rot, obs_vert) / g2

    x0 = np.clip(rng.beta(prior_alpha, prior_beta, size=(n_particles, D)), 1e-6, 1 - 1e-6)
    t = _logit(x0)
    lp, ll = log_prior_t(t), neg_energy(t)
    phi, n_stages, acc_hist = 0.0, 0, []

    def mh_moves(t, lp, ll, phi, n_steps):
        cov = np.cov(t, rowvar=False) + 1e-6 * np.eye(D)
        chol = np.linalg.cholesky(cov)
        scale = 2.38 / np.sqrt(D)
        acc = 0.0
        for _ in range(n_steps):
            prop = t + scale * rng.standard_normal(t.shape) @ chol.T
            lp_p, ll_p = log_prior_t(prop), neg_energy(prop)
            log_a = (lp_p + phi * ll_p) - (lp + phi * ll)
            ok = np.log(rng.random(len(t))) < log_a
            t[ok], lp[ok], ll[ok] = prop[ok], lp_p[ok], ll_p[ok]
            rate = ok.mean()
            acc += rate
            scale *= np.exp(rate - 0.25)          # keep acceptance near 25%
        return t, lp, ll, acc / n_steps

    while phi < 1.0:
        # Next temperature: bisection so that the ESS of the incremental weights = target_ess * N
        def ess(dphi):
            lw = dphi * ll
            w = np.exp(lw - lw.max())
            return w.sum() ** 2 / np.sum(w ** 2) / n_particles
        lo, hi = 0.0, 1.0 - phi
        if ess(hi) >= target_ess:
            dphi = hi
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ess(mid) >= target_ess else (lo, mid)
            dphi = lo
        lw = dphi * ll
        w = np.exp(lw - lw.max())
        w /= w.sum()
        idx = _systematic_resample(w, rng)
        t, lp, ll = t[idx].copy(), lp[idx].copy(), ll[idx].copy()
        phi = min(1.0, phi + dphi)
        t, lp, ll, acc = mh_moves(t, lp, ll, phi, n_mh)
        acc_hist.append(acc)
        n_stages += 1
        if verbose:
            print(f"  SMC stage {n_stages:3d}: phi = {phi:.4e}, MH acceptance = {acc:.2f}")

    t, lp, ll, acc = mh_moves(t, lp, ll, 1.0, n_final_mh)
    acc_hist.append(acc)
    z = lbound + (1.0 - lbound) * _sigmoid(t)
    return z, {'n_stages': n_stages, 'acceptance': acc_hist}
