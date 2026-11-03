#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/posterior_families.py

The approximate posteriors q(z | m) of every model compared in the revision (R1.4), in NumPy and with one
interface, so that postprocessing.py computes every metric with the same code for all of them:

  KSCopula      proposed method: Gaussian copula + mixtures of K Kumaraswamy distributions (GC_KSmix_models.py)
  TNCopula      Ref. 41 / submitted version: Gaussian copula + one truncated Gaussian per marginal
                (MODULES/baseline_GaussCopula_GaussMarginals)
  GaussMixture  Ref. 39: mixture of K Gaussians with full covariance, written in the logit space of the prior box
                (MODULES/baseline_GaussMixture_K5)

Densities live on the unit cube u = (z - lbound) / (1 - lbound) in [0, 1]^D; methods that return z say so.
family_from_preds() recognises the family from the keys of a Test_predicted_props*.npy file. The formulas of
KSCopula are those of postprocessing.py (same operations), so its metrics are unchanged.
"""

import numpy as np
from scipy.special import betaln, expit, logsumexp, ndtr, ndtri

LOG_2PI = np.log(2.0 * np.pi)


def correlation_from_factor(offdiag, diag):
    """Row-normalised lower factor L (N, D, D) and R = L L^T, as build_correlation_matrices_from_cholesky."""
    off = np.asarray(offdiag, np.float64)
    dg = np.asarray(diag, np.float64)
    N, D = dg.shape
    r, c = np.tril_indices(D, k=-1)
    L = np.zeros((N, D, D))
    L[:, r, c] = off
    L[:, np.arange(D), np.arange(D)] = dg
    L /= np.linalg.norm(L, axis=-1, keepdims=True)
    return L, L @ np.swapaxes(L, -1, -2)


class PosteriorFamily:
    """Common part. A family implements cdf, pdf (N, D), cdf_grid, pdf_grid (n, D, G), moments, sample_unit,
    log_density_unit, marginal_ppf and dependence."""
    name = ""
    label = ""
    n_components = 1

    def __init__(self, lbound):
        self.lbound = float(lbound)

    def to_z(self, u):
        return self.lbound + (1.0 - self.lbound) * u

    def quantiles(self, probs, n_grid=4001, chunk=100):
        """Quantiles (N, D, P) and means (N, D) in z units of every marginal (exact CDF on a fine grid)."""
        x = np.linspace(0.0, 1.0, n_grid)
        q = np.zeros((self.n, self.d, len(probs)))
        for i0 in range(0, self.n, chunk):
            cdf = self.cdf_grid(slice(i0, i0 + chunk), x)
            for i in range(cdf.shape[0]):
                for d in range(self.d):
                    q[i0 + i, d] = np.interp(probs, cdf[i, d], x)
        return self.to_z(q), self.to_z(self.moments()[0])

    def hpd_level(self, u_true, n_grid=2000, chunk=50):
        """Smallest level (N, D) of the highest-density region of every marginal that contains u*: the q-mass of
        {u : q(u) > q(u*)}. u* is inside the p-HPD region iff hpd_level <= p."""
        g = (np.arange(n_grid) + 0.5) / n_grid
        fz = self.pdf(u_true)
        out = np.zeros((self.n, self.d))
        for i0 in range(0, self.n, chunk):
            fg = self.pdf_grid(slice(i0, i0 + chunk), g)
            out[i0:i0 + chunk] = np.mean(fg * (fg > fz[i0:i0 + chunk, :, None]), axis=-1)
        return out

    def sample(self, n, seed=0, idx=None):
        """Joint samples (len(idx), n, D) of q in z units."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        return self.to_z(self.sample_unit(n, np.random.default_rng(seed), idx))

    def sample_marginals(self, n, seed=0):
        """n independent samples (N, n, D) of every marginal in z units (dependence not used): shape indicators."""
        rng = np.random.default_rng(seed)
        v = rng.uniform(1e-6, 1 - 1e-6, size=(self.n, n, self.d))
        return self.to_z(self.marginal_ppf(v, np.arange(self.n)))

    def log_density(self, z, idx):
        """Joint log q(z) (len(idx), S) in z units of samples z (len(idx), S, D)."""
        u = (np.asarray(z, np.float64) - self.lbound) / (1.0 - self.lbound)
        return self.log_density_unit(u, idx) - self.d * np.log(1.0 - self.lbound)

    def marginal_pdf_z(self, pos):
        """Callable (grid in z, d) -> exact marginal density in z units of observation pos (for the 1-D plots)."""
        def f(grid, d):
            u = (np.asarray(grid, np.float64) - self.lbound) / (1.0 - self.lbound)
            return self.pdf_grid(slice(pos, pos + 1), u)[0, d] / (1.0 - self.lbound)
        return f


class _GaussianCopulaFamily(PosteriorFamily):
    """Gaussian copula with correlation R = L L^T and any marginals (cdf, pdf, marginal_ppf of the subclass)."""
    dependence_label = "copula correlation"

    def _set_copula(self, offdiag, diag):
        self.L, self.R = correlation_from_factor(offdiag, diag)
        self._Rinv_m_I = np.linalg.inv(self.R) - np.eye(self.R.shape[-1])[None]
        self._logdetR = np.linalg.slogdet(self.R)[1]

    def dependence(self):
        return self.R

    def copula_logpdf(self, v, idx):
        """log c(v) (len(idx), S) of the Gaussian copula at uniforms v (len(idx), S, D)."""
        y = ndtri(np.clip(v, 1e-9, 1.0 - 1e-9))
        quad = np.einsum("nsd,nde,nse->ns", y, self._Rinv_m_I[idx], y)
        return -0.5 * self._logdetR[idx][:, None] - 0.5 * quad

    def log_joint_at(self, u):
        """log q(u) (N,) at one point u (N, D) per observation (copula + marginals; as postprocessing.py)."""
        y = ndtri(np.clip(self.cdf(u), 1e-9, 1.0 - 1e-9))
        quad = np.einsum("nd,nde,ne->n", y, self._Rinv_m_I, y)
        return -0.5 * self._logdetR - 0.5 * quad + np.sum(np.log(self.pdf(u) + 1e-12), axis=1)

    def sample_unit(self, n, rng, idx):
        eps = rng.standard_normal((len(idx), n, self.d))
        v = np.clip(ndtr(np.einsum("nsd,nfd->nsf", eps, self.L[idx])), 1e-9, 1.0 - 1e-9)
        return self.marginal_ppf(v, idx)

    def log_density_unit(self, u, idx):
        u = np.clip(u, 1e-9, 1.0 - 1e-9)
        v = self.marginal_cdf_samples(u, idx)
        return self.copula_logpdf(v, idx) + np.sum(np.log(self.marginal_pdf_samples(u, idx) + 1e-300), axis=-1)


# =============================================================================
# 1. PROPOSED: GAUSSIAN COPULA + KUMARASWAMY-MIXTURE MARGINALS
# =============================================================================
class KSCopula(_GaussianCopulaFamily):
    name = "GC-KS"

    def __init__(self, a, b, w, offdiag, diag, lbound):
        super().__init__(lbound)
        self.a, self.b = (np.asarray(v, np.float64) for v in (a, b))                       # (N, K, D)
        w = np.asarray(w, np.float64)
        self.w = w / np.sum(w, axis=1, keepdims=True)
        self.n, self.n_components, self.d = self.a.shape
        self.label = f"Gaussian copula + Kumaraswamy-mixture marginals (K = {self.n_components})"
        self.short = f"KS-Copula VAE ($K={self.n_components}$)"
        self._set_copula(offdiag, diag)

    def cdf(self, u):
        a, b, w = self.a, self.b, self.w
        return np.sum(w * (1.0 - (1.0 - u[:, None, :] ** a) ** b), axis=1)

    def pdf(self, u):
        a, b, w = self.a, self.b, self.w
        return np.sum(w * a * b * u[:, None, :] ** (a - 1) * (1.0 - u[:, None, :] ** a) ** (b - 1), axis=1)

    def _grid(self, sl):
        return (v[sl].transpose(0, 2, 1)[:, :, None, :] for v in (self.a, self.b, self.w))   # (n, D, 1, K)

    def cdf_grid(self, sl, x):
        aa, bb, ww = self._grid(sl)
        xx = x[None, None, :, None]
        return np.sum(ww * (1.0 - (1.0 - xx ** aa) ** bb), axis=-1) / np.sum(ww, axis=-1)

    def pdf_grid(self, sl, x):
        aa, bb, ww = self._grid(sl)
        xx = np.clip(x, 1e-12, 1 - 1e-12)[None, None, :, None]
        pdf = aa * bb * xx ** (aa - 1.0) * np.clip(1.0 - xx ** aa, 1e-300, 1.0) ** (bb - 1.0)
        return np.sum(ww * pdf, axis=-1) / np.sum(ww, axis=-1)

    def moments(self):
        m1 = np.sum(self.w * np.exp(np.log(self.b) + betaln(1.0 + 1.0 / self.a, self.b)), axis=1)
        m2 = np.sum(self.w * np.exp(np.log(self.b) + betaln(1.0 + 2.0 / self.a, self.b)), axis=1)
        return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))

    def _samples_params(self, idx):
        return (v[idx][:, None] for v in (self.a, self.b, self.w))                          # (n, 1, K, D)

    def marginal_cdf_samples(self, u, idx):
        a, b, w = self._samples_params(idx)
        return np.sum(w * (1.0 - (1.0 - u[:, :, None, :] ** a) ** b), axis=2)

    def marginal_pdf_samples(self, u, idx):
        a, b, w = self._samples_params(idx)
        uu = u[:, :, None, :]
        return np.sum(w * a * b * uu ** (a - 1) * np.clip(1.0 - uu ** a, 1e-300, 1.0) ** (b - 1), axis=2)

    def marginal_ppf(self, v, idx, n_iter=50):
        lo, hi = np.zeros_like(v), np.ones_like(v)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            less = self.marginal_cdf_samples(mid, idx) < v
            lo, hi = np.where(less, mid, lo), np.where(less, hi, mid)
        return 0.5 * (lo + hi)

    def sample_marginals(self, n, seed=0):
        """As sample_ks_marginals of postprocessing.py (same random numbers)."""
        rng = np.random.default_rng(seed)
        N, K, D = self.a.shape
        cw = np.cumsum(self.w.transpose(0, 2, 1), axis=-1)                                  # (N, D, K)
        k = np.minimum(np.sum(cw[:, :, None, :] < rng.random((N, D, n, 1)), axis=-1), K - 1)
        aa = np.take_along_axis(self.a.transpose(0, 2, 1), k, axis=-1)
        bb = np.take_along_axis(self.b.transpose(0, 2, 1), k, axis=-1)
        u = rng.uniform(1e-6, 1 - 1e-6, size=aa.shape)
        x = (1.0 - (1.0 - u) ** (1.0 / bb)) ** (1.0 / aa)
        return self.to_z(x.transpose(0, 2, 1))


# =============================================================================
# 2. BASELINE (Ref. 41 / submitted version): GAUSSIAN COPULA + TRUNCATED-GAUSSIAN MARGINALS
# =============================================================================
class TNCopula(_GaussianCopulaFamily):
    """Marginal d: N(mu_d, sigma_d^2) truncated to [0, 1] (unit space), with mu_d inside [0, 1] as in the submitted
    version, so that alpha = -mu / sigma <= 0 <= beta = (1 - mu) / sigma and the normaliser Z is never tiny."""
    name = "GC-TN"
    label = "Gaussian copula + truncated-Gaussian marginals"
    short = "Gaussian-copula VAE (truncated-Gaussian marginals)"

    def __init__(self, loc, scale, offdiag, diag, lbound):
        super().__init__(lbound)
        self.mu, self.sig = np.asarray(loc, np.float64), np.asarray(scale, np.float64)       # (N, D)
        self.n, self.d = self.mu.shape
        self.alpha, self.beta = -self.mu / self.sig, (1.0 - self.mu) / self.sig
        self.Pa, self.Qb = ndtr(self.alpha), ndtr(-self.beta)                                # mass below / above
        self.Z = 1.0 - self.Pa - self.Qb
        self._set_copula(offdiag, diag)

    @staticmethod
    def _phi(x):
        return np.exp(-0.5 * x ** 2 - 0.5 * LOG_2PI)

    def cdf(self, u):
        return np.clip((ndtr((u - self.mu) / self.sig) - self.Pa) / self.Z, 0.0, 1.0)

    def pdf(self, u):
        return self._phi((u - self.mu) / self.sig) / (self.sig * self.Z)

    def cdf_grid(self, sl, x):
        mu, sig, Pa, Z = (v[sl][..., None] for v in (self.mu, self.sig, self.Pa, self.Z))
        return np.clip((ndtr((x - mu) / sig) - Pa) / Z, 0.0, 1.0)

    def pdf_grid(self, sl, x):
        mu, sig, Z = (v[sl][..., None] for v in (self.mu, self.sig, self.Z))
        return self._phi((x - mu) / sig) / (sig * Z)

    def moments(self):
        pa, pb = self._phi(self.alpha), self._phi(self.beta)
        r = (pa - pb) / self.Z
        var = self.sig ** 2 * (1.0 + (self.alpha * pa - self.beta * pb) / self.Z - r ** 2)
        return self.mu + self.sig * r, np.sqrt(np.maximum(var, 1e-12))

    def marginal_cdf_samples(self, u, idx):
        mu, sig, Pa, Z = (v[idx][:, None] for v in (self.mu, self.sig, self.Pa, self.Z))
        return np.clip((ndtr((u - mu) / sig) - Pa) / Z, 0.0, 1.0)

    def marginal_pdf_samples(self, u, idx):
        mu, sig, Z = (v[idx][:, None] for v in (self.mu, self.sig, self.Z))
        return self._phi((u - mu) / sig) / (sig * Z)

    def marginal_ppf(self, v, idx):
        """Inverse CDF; the lower or the upper tail formula is used so that ndtri never sees 1 - tiny."""
        mu, sig, Pa, Qb, Z = (v_[idx][:, None] for v_ in (self.mu, self.sig, self.Pa, self.Qb, self.Z))
        p_low = Pa + v * Z
        p_up = Qb + (1.0 - v) * Z
        xi = np.where(p_low < 0.5, ndtri(np.minimum(p_low, 0.5)), -ndtri(np.minimum(p_up, 0.5)))
        return np.clip(mu + sig * xi, 0.0, 1.0)


# =============================================================================
# 3. BASELINE (Ref. 39): GAUSSIAN MIXTURE (FULL COVARIANCE) IN THE LOGIT SPACE OF THE BOX
# =============================================================================
class GaussMixture(PosteriorFamily):
    """q(y) = sum_k pi_k N(y; mu_k, L_k L_k^T) with y = logit(u): the prior box [lbound, 1]^D is mapped onto R^D as in
    ADVI (Kucukelbir et al. 2017), so q has the support of the prior. No copula: the dependence comes from the
    covariances and from the mixture."""
    name = "GMM"
    dependence_label = "normal-score (Gaussian-rank) correlation"

    def __init__(self, means, chol, weights, lbound):
        super().__init__(lbound)
        self.mu = np.asarray(means, np.float64)                                             # (N, K, D)
        self.Lc = np.asarray(chol, np.float64)                                              # (N, K, D, D)
        w = np.asarray(weights, np.float64)
        self.pi = w / np.sum(w, axis=1, keepdims=True)                                      # (N, K)
        self.n, self.n_components, self.d = self.mu.shape
        self.s = np.linalg.norm(self.Lc, axis=-1)                                           # marginal std (N, K, D)
        self.Linv = np.linalg.inv(self.Lc)
        self.logdetL = np.sum(np.log(np.abs(np.diagonal(self.Lc, axis1=-2, axis2=-1))), axis=-1)   # (N, K)
        self.label = f"Gaussian mixture (K = {self.n_components}, full covariance)"
        self.short = f"Gaussian-mixture VAE ($K={self.n_components}$)"
        self._dep = None

    @staticmethod
    def _logit(u):
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        return np.log(u) - np.log1p(-u)

    def _mix_cdf(self, y, mu, s, pi):
        return np.sum(pi * ndtr((y - mu) / s), axis=-2)

    def _mix_pdf_y(self, y, mu, s, pi):
        t = (y - mu) / s
        return np.sum(pi * np.exp(-0.5 * t ** 2 - 0.5 * LOG_2PI) / s, axis=-2)

    @staticmethod
    def _jac(u):
        return np.clip(u, 1e-300, 1.0) * np.clip(1.0 - u, 1e-300, 1.0)

    def cdf(self, u):            # (N, D); components on axis -2
        return self._mix_cdf(self._logit(u)[:, None, :], self.mu, self.s, self.pi[..., None])

    def pdf(self, u):
        y = self._logit(u)
        return self._mix_pdf_y(y[:, None, :], self.mu, self.s, self.pi[..., None]) / self._jac(u)

    def _grid(self, sl):
        return (self.mu[sl].transpose(0, 2, 1)[:, :, None, :], self.s[sl].transpose(0, 2, 1)[:, :, None, :],
                self.pi[sl][:, None, None, :])                                               # (n, D, 1, K)

    def cdf_grid(self, sl, x):
        mu, s, pi = self._grid(sl)
        return np.sum(pi * ndtr((self._logit(x)[None, None, :, None] - mu) / s), axis=-1)

    def pdf_grid(self, sl, x):
        mu, s, pi = self._grid(sl)
        t = (self._logit(x)[None, None, :, None] - mu) / s
        f = np.sum(pi * np.exp(-0.5 * t ** 2 - 0.5 * LOG_2PI) / s, axis=-1) / self._jac(x)[None, None, :]
        return np.where((x > 0) & (x < 1), f, 0.0)

    def moments(self, n_nodes=48):
        t, wq = np.polynomial.hermite_e.hermegauss(n_nodes)
        wq = wq / np.sqrt(2.0 * np.pi)
        u = expit(self.mu[..., None] + self.s[..., None] * t)                             # (N, K, D, nodes)
        m1 = np.sum(self.pi[..., None] * np.sum(wq * u, axis=-1), axis=1)
        m2 = np.sum(self.pi[..., None] * np.sum(wq * u ** 2, axis=-1), axis=1)
        return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))

    def _components(self, rng, idx, shape):
        cw = np.cumsum(self.pi[idx], axis=-1)                                                # (n, K)
        r = rng.random((len(idx),) + shape)
        k = np.sum(cw.reshape((len(idx),) + (1,) * len(shape) + (-1,)) < r[..., None], axis=-1)
        return np.minimum(k, self.n_components - 1)

    def sample_unit(self, n, rng, idx):
        k = self._components(rng, idx, (n,))                                                 # (n_obs, S)
        rows = np.arange(len(idx))[:, None]
        mu = self.mu[idx][rows, k]                                                           # (n_obs, S, D)
        L = self.Lc[idx][rows, k]                                                            # (n_obs, S, D, D)
        y = mu + np.einsum("nsij,nsj->nsi", L, rng.standard_normal((len(idx), n, self.d)))
        return expit(y)

    def sample_marginals(self, n, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.arange(self.n)
        k = self._components(rng, idx, (n, self.d))                                          # (N, S, D)
        mu = np.take_along_axis(self.mu.transpose(0, 2, 1)[:, None], k[..., None], axis=-1)[..., 0]
        s = np.take_along_axis(self.s.transpose(0, 2, 1)[:, None], k[..., None], axis=-1)[..., 0]
        return self.to_z(expit(mu + s * rng.standard_normal(mu.shape)))

    def marginal_ppf(self, v, idx, n_iter=60):
        y = None
        lo, hi = np.full(v.shape, -40.0), np.full(v.shape, 40.0)
        mu, s, pi = self.mu[idx][:, None], self.s[idx][:, None], self.pi[idx][:, None, :, None]
        for _ in range(n_iter):
            y = 0.5 * (lo + hi)
            less = self._mix_cdf(y[:, :, None, :], mu, s, pi) < v
            lo, hi = np.where(less, y, lo), np.where(less, hi, y)
        return expit(0.5 * (lo + hi))

    def log_density_unit(self, u, idx):
        y = self._logit(u)                                                                   # (n, S, D)
        diff = y[:, :, None, :] - self.mu[idx][:, None]                                      # (n, S, K, D)
        sol = np.einsum("nkij,nskj->nski", self.Linv[idx], diff)
        log_comp = -0.5 * np.sum(sol ** 2, axis=-1) - self.logdetL[idx][:, None] - 0.5 * self.d * LOG_2PI
        log_qy = logsumexp(np.log(np.maximum(self.pi[idx], 1e-300))[:, None] + log_comp, axis=-1)
        uc = np.clip(u, 1e-15, 1.0 - 1e-15)
        return log_qy - np.sum(np.log(uc) + np.log1p(-uc), axis=-1)

    def log_joint_at(self, u):
        return self.log_density_unit(u[:, None, :], np.arange(self.n))[:, 0]

    def dependence(self, n=1000, seed=0):
        """Correlation (N, D, D) of the normal scores Phi^-1(F_d(u_d)) of q: the copula correlation of a Gaussian
        copula fitted to q (comparable with the copula correlations of the other models)."""
        if self._dep is None:
            rng = np.random.default_rng(seed)
            out = np.zeros((self.n, self.d, self.d))
            for i0 in range(0, self.n, 100):
                idx = np.arange(i0, min(i0 + 100, self.n))
                u = self.sample_unit(n, rng, idx)
                y = self._logit(u)[:, :, None, :]
                v = self._mix_cdf(y, self.mu[idx][:, None], self.s[idx][:, None], self.pi[idx][:, None, :, None])
                sc = ndtri(np.clip(v, 1e-9, 1 - 1e-9))
                sc = (sc - sc.mean(axis=1, keepdims=True)) / (sc.std(axis=1, keepdims=True) + 1e-12)
                out[idx] = np.einsum("nsd,nse->nde", sc, sc) / n
            self._dep = out
        return self._dep


def family_from_preds(preds, lbound):
    """Posterior family of a Test_predicted_props*.npy / Val_predicted_props.npy dictionary."""
    p = preds
    if "test_a" in p:
        return KSCopula(p["test_a"], p["test_b"], p["test_weight_vals"], p["test_offdiag_elems"],
                        p["test_diag_elems"], lbound)
    if "test_loc" in p:
        return TNCopula(p["test_loc"], p["test_scale"], p["test_offdiag_elems"], p["test_diag_elems"], lbound)
    if "test_means" in p:
        return GaussMixture(p["test_means"], p["test_chol"], p["test_weights"], lbound)
    raise KeyError(f"unknown posterior family, keys: {sorted(p)}")


def as_family(preds, lbound):
    return preds if isinstance(preds, PosteriorFamily) else family_from_preds(preds, lbound)
