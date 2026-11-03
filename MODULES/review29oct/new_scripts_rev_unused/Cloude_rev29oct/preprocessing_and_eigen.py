# -*- coding: utf-8 -*-
"""
preprocessing_and_eigen.py -- Modal model, likelihood and data loading.

Modal model of the inference FE model (affine in theta, nominal mass M = C C^T):

    K(theta) = sum_p theta_p K_p,     A(theta) = C^-1 K(theta) C^-T / lam_ref,
    A psi = (omega^2 / lam_ref) psi,  vertical / rotational components at the
    sensors: S_v psi, S_r psi (each normalized to unit norm, as in the data).

Likelihood (the measurement model of dataset_generation.py):

    log f_obs = log f(theta) + N(0, s_f^2)
    v_obs     = s * v(theta) + N(0, s_v,m^2 I),   r_obs = s * r(theta) + N(0, s_r,m^2 I)

with s = +-1 the (unobservable) eigenvector sign, marginalized with
probability 1/2 each:  log[ 0.5 N(y; phi) + 0.5 N(y; -phi) ].

The NumPy and TensorFlow implementations below compute the same formulas;
check_implementation.py verifies that they agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg as sla
import tensorflow as tf

import config as C

DT = tf.float64
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


# =============================================================================
# Affine modal model
# =============================================================================
def modal_arrays(K_basis, M, sensor_nodes):
    """Mass-normalized, scaled affine arrays of the inference model."""
    n = M.shape[0]
    Cinv = sla.solve_triangular(np.linalg.cholesky(M), np.eye(n), lower=True)
    K_h = K_basis.sum(0)                                         # healthy: theta = 1
    lam_ref = float(np.linalg.eigvalsh(Cinv @ K_h @ Cinv.T)[0])
    A_basis = np.einsum("ij,pjk,lk->pil", Cinv, K_basis, Cinv) / lam_ref
    return dict(A_basis=A_basis, lam_ref=lam_ref,
                S_v=Cinv.T[2 * sensor_nodes, :], S_r=Cinv.T[2 * sensor_nodes + 1, :])


class ModalModelNP:
    """theta (N, P) -> log f (N, M), unit vertical / rotational shapes (N, M, S)."""

    def __init__(self, arrays, n_modes):
        self.A_basis, self.lam_ref = arrays["A_basis"], float(arrays["lam_ref"])
        self.S_v, self.S_r, self.n_modes = arrays["S_v"], arrays["S_r"], n_modes

    def __call__(self, theta):
        A = np.einsum("np,pij->nij", np.atleast_2d(theta), self.A_basis)
        lam, psi = np.linalg.eigh(A)
        lam, psi = np.maximum(lam[:, :self.n_modes], 1e-300), psi[:, :, :self.n_modes]
        logf = 0.5 * np.log(lam * self.lam_ref) - np.log(2.0 * np.pi)
        v = np.einsum("sd,ndm->nms", self.S_v, psi)
        r = np.einsum("sd,ndm->nms", self.S_r, psi)
        return (logf, v / np.linalg.norm(v, axis=-1, keepdims=True),
                r / np.linalg.norm(r, axis=-1, keepdims=True))


def align_to_reference(v, r, v_ref, r_ref, sig_v, sig_r):
    """Choose the sign of each mode so that it agrees with the healthy reference."""
    c = np.sum(v * v_ref, -1) / sig_v ** 2 + np.sum(r * r_ref, -1) / sig_r ** 2
    s = np.where(c >= 0, 1.0, -1.0)[..., None]
    return v * s, r * s


def log_likelihood_np(y_logf, y_v, y_r, logf, v, r, sig_f, sig_v, sig_r, obs):
    """
    y_logf (M,), y_v / y_r (M, S); predictions logf (N, M), v / r (N, M, S);
    sig_f scalar, sig_v / sig_r (M,). Returns (N,).
    """
    M = logf.shape[-1]
    ll = -0.5 * np.sum(((y_logf - logf) / sig_f) ** 2, -1) - M * (LOG_SQRT_2PI + np.log(sig_f))
    if "v" in obs or "r" in obs:
        quad, cross, const = 0.0, 0.0, 0.0
        for key, y, p, s in (("v", y_v, v, sig_v), ("r", y_r, r, sig_r)):
            if key in obs:
                quad = quad + (np.sum(y ** 2, -1) + np.sum(p ** 2, -1)) / s ** 2
                cross = cross + np.sum(y * p, -1) / s ** 2
                const = const + p.shape[-1] * (LOG_SQRT_2PI + np.log(s))
        mode_ll = -0.5 * quad + np.logaddexp(cross, -cross) - np.log(2.0) - const
        ll = ll + np.sum(mode_ll, -1)
    return ll


class ModalDecoderTF:
    """TensorFlow twin of ModalModelNP (differentiable through tf.linalg.eigh)."""

    def __init__(self, arrays, n_modes, eigh_on_cpu=False):
        self.A_basis = tf.constant(arrays["A_basis"], DT)
        self.log_lam_ref = tf.constant(np.log(float(arrays["lam_ref"])), DT)
        self.S_v, self.S_r = tf.constant(arrays["S_v"], DT), tf.constant(arrays["S_r"], DT)
        self.n_modes, self.eigh_on_cpu = n_modes, eigh_on_cpu

    def __call__(self, theta):
        A = tf.einsum("np,pij->nij", theta, self.A_basis)
        if self.eigh_on_cpu:
            with tf.device("/CPU:0"):
                lam, psi = tf.linalg.eigh(A)
        else:
            lam, psi = tf.linalg.eigh(A)
        lam = tf.maximum(lam[:, :self.n_modes], 1e-300)
        psi = psi[:, :, :self.n_modes]
        logf = 0.5 * (tf.math.log(lam) + self.log_lam_ref) - np.log(2.0 * np.pi)
        v = tf.math.l2_normalize(tf.einsum("sd,ndm->nms", self.S_v, psi), axis=-1)
        r = tf.math.l2_normalize(tf.einsum("sd,ndm->nms", self.S_r, psi), axis=-1)
        return logf, v, r


def log_likelihood_tf(y_logf, y_v, y_r, logf, v, r, sig_f, sig_v, sig_r, obs):
    """Same formula as log_likelihood_np, batched: y_* have the batch dimension of the predictions."""
    M = tf.cast(tf.shape(logf)[-1], DT)
    ll = -0.5 * tf.reduce_sum(((y_logf - logf) / sig_f) ** 2, -1) - M * (LOG_SQRT_2PI + np.log(sig_f))
    if "v" in obs or "r" in obs:
        quad, cross, const = 0.0, 0.0, 0.0
        for key, y, p, s in (("v", y_v, v, sig_v), ("r", y_r, r, sig_r)):
            if key in obs:
                S = tf.cast(tf.shape(p)[-1], DT)
                quad += (tf.reduce_sum(y ** 2, -1) + tf.reduce_sum(p ** 2, -1)) / s ** 2
                cross += tf.reduce_sum(y * p, -1) / s ** 2
                const += S * (LOG_SQRT_2PI + tf.math.log(s))
        mode_ll = (-0.5 * quad + tf.reduce_logsumexp(tf.stack([cross, -cross]), axis=0)
                   - np.log(2.0) - const)
        ll += tf.reduce_sum(mode_ll, -1)
    return ll


# =============================================================================
# Data loading and encoder features
# =============================================================================
def load_coarse_model(data_dir: Path = C.DATA_DIR):
    path = Path(data_dir) / "coarse_model.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run dataset_generation.py first.")
    return dict(np.load(path))


def _select(d, n_modes, baseline_correction):
    out = dict(d)
    shift = d.get("logf_baseline_shift")
    for k in ("y_logf", "logf_clean"):
        out[k] = d[k][..., :n_modes]
    if baseline_correction and shift is not None:
        out["y_logf"] = out["y_logf"] - shift[:n_modes]
    for k in ("y_v", "y_r", "v_clean", "r_clean"):
        out[k] = d[k][..., :n_modes, :]
    return out


def load_set(name, n_modes, data_dir: Path = C.DATA_DIR, n_max=0, baseline_correction=False):
    path = Path(data_dir) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run dataset_generation.py first.")
    raw = dict(np.load(path))
    sl = slice(None) if n_max <= 0 else slice(0, n_max)
    d = {k: (v[sl] if v.ndim > 0 and k != "logf_baseline_shift" else v) for k, v in raw.items()}
    return _select(d, n_modes, baseline_correction)


def load_benchmarks(variant, n_modes, data_dir: Path = C.DATA_DIR, baseline_correction=False):
    raw = dict(np.load(Path(data_dir) / "benchmarks.npz"))
    d = {k[len(variant) + 1:]: v for k, v in raw.items() if k.startswith(variant + "_")}
    d["names"] = list(raw["names"])
    return _select(d, n_modes, baseline_correction)


def features(y_logf, y_v, y_r, obs):
    feats = [y_logf]
    if "v" in obs:
        feats.append(y_v.reshape(len(y_v), -1))
    if "r" in obs:
        feats.append(y_r.reshape(len(y_r), -1))
    return np.concatenate(feats, axis=1)


class FeatureScaler:
    """Standardization fitted on the training set only."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"])
