# -*- coding: utf-8 -*-
"""
ks28_likelihood.py -- The measurement likelihood p(m | theta) (NumPy version).

It is derived from the noise model used to generate the data (NoiseConfig), so no
weight (lambda) or temperature (gamma) has to be tuned (Reviewer 3 c.2, Reviewer 1 c.3):

Frequencies (log-normal noise)
    -log p(f | theta) = sum_j (log f_obs_j - log f_j(theta))^2 / (2 sigma_f^2) + const

Mode shapes (additive Gaussian noise, unknown scale)
    phi_obs_j = alpha_j * phi_j(theta) + eps,  eps ~ N(0, sigma_phi^2 I)
    (phi_obs is scaled to unit peak, so the noise std is ~sigma_phi.)
    Integrating out the unknown scale alpha_j with a flat prior (phi_j(theta) taken with
    unit norm) gives
    -log p(phi_j | theta) = ||phi_obs_j||^2 (1 - MAC(phi_obs_j, phi_j(theta))) / (2 sigma_phi^2) + const
    i.e. the MAC misfit of the paper comes out of the noise model, with a weight
    ||phi_obs||^2 / (2 sigma_phi^2) that replaces the empirical lambda = 10.
    It is invariant to the sign and the scale of the predicted modes.

The TensorFlow loss (ks28_tf_model) and the ground truth (ks28_ground_truth) use
exactly this expression.
"""

from __future__ import annotations

import numpy as np

from .ks28_config import NoiseConfig


def mac_matrix_diag(phi_obs: np.ndarray, phi_pred: np.ndarray) -> np.ndarray:
    """MAC of corresponding modes. phi_*: (..., n_modes, n_sensors) -> (..., n_modes)."""
    num = np.sum(phi_obs * phi_pred, axis=-1) ** 2
    den = np.sum(phi_obs ** 2, axis=-1) * np.sum(phi_pred ** 2, axis=-1) + 1e-300
    return num / den


def neg_log_likelihood(f_obs, phi_obs, f_pred, phi_pred, noise: NoiseConfig,
                       n_modes_used=None, return_parts=False):
    """-log p(m | theta) (up to a constant) for broadcastable batches.

    f_obs (..., nm), phi_obs (..., nm, ns), f_pred, phi_pred same trailing shapes.
    n_modes_used: use only the first n modes (identifiability study, Reviewer 3 c.4).
    """
    nm = f_obs.shape[-1] if n_modes_used is None else int(n_modes_used)
    sf, sp = noise.sigma_f_lik(nm), noise.sigma_phi_lik(nm)          # per-mode std
    r = np.log(f_obs[..., :nm]) - np.log(f_pred[..., :nm])
    nll_f = 0.5 * np.sum(r ** 2 / sf ** 2, axis=-1)
    if noise.use_mode_shapes:
        mac = mac_matrix_diag(phi_obs[..., :nm, :], phi_pred[..., :nm, :])
        w = np.sum(phi_obs[..., :nm, :] ** 2, axis=-1)
        nll_p = 0.5 * np.sum(w * (1.0 - mac) / sp ** 2, axis=-1)
    else:
        nll_p = np.zeros_like(nll_f)
    if return_parts:
        return nll_f + nll_p, nll_f, nll_p
    return nll_f + nll_p


def estimate_model_discrepancy(folder, cfg, noise, n_modes_used=None):
    """Per-mode model-discrepancy std from the SHIFTED VALIDATION set (never test data).

    For val_shift the true theta is known, so the residual between the data and the
    decoder (training) model evaluated at the truth contains noise + model-form error:
        var(log f_obs - log f_dec(theta))          = sigma_f^2   + sigma_f_model^2
        E[||phi_obs||^2 (1 - MAC)] / (ns - 1)        = sigma_phi^2 + sigma_phi_model^2
    (the second identity follows from the scale-marginalised mode-shape likelihood).
    This is a simple, transparent version of a Kennedy-O'Hagan discrepancy term with an
    i.i.d. Gaussian discrepancy per mode (Reviewer 3 c.1, Reviewer 1 c.2-3).
    """
    import os
    from .ks28_fe_model import build_basis, forward
    from .ks28_dataset import load_npz
    d = load_npz(os.path.join(folder, "val_shift.npz"))
    nm = cfg.n_modes if n_modes_used is None else n_modes_used
    f_dec, p_dec = forward(d["theta"], build_basis(cfg), cfg)
    r = np.log(d["f_obs"][:, :nm]) - np.log(f_dec[:, :nm])
    var_f = np.mean(r ** 2, axis=0)
    mac = mac_matrix_diag(d["phi_obs"][:, :nm], p_dec[:, :nm])
    w = np.sum(d["phi_obs"][:, :nm] ** 2, -1)
    var_p = np.mean(w * (1 - mac), axis=0) / (cfg.n_sensors - 1)
    sf_model = np.sqrt(np.maximum(var_f - noise.sigma_f ** 2, 0.0))
    sp_model = np.sqrt(np.maximum(var_p - noise.sigma_phi ** 2, 0.0))
    return dict(sigma_f_model=sf_model.round(5).tolist(), sigma_phi_model=sp_model.round(5).tolist(),
                rms_logf_residual=np.sqrt(var_f).round(5).tolist(),
                rms_phi_residual=np.sqrt(var_p).round(5).tolist())
