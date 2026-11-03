#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v3/fe_model.py

NumPy (float64) physics of the 7-parameter bridge on elastic bearings.

  * Euler-Bernoulli beam of 5 damage zones, `el_per_zone` elements per zone, vertical + rotational springs at
    both ends. Sensors at the 6 zone-boundary nodes, whatever the mesh.
  * Decoder model (VAE and reference posterior): 2 elements per zone, nominal mass and modulus.
  * "True structure" generator (data): any mesh, plus what the decoder ignores: temperature (deck modulus),
    element mass scatter and within-zone stiffness scatter.
  * Measurement noise and THE likelihood. The TensorFlow version in GC_KSmix_models.py is the same formula;
    check_consistency.py verifies that both agree.

Likelihood (constants dropped), d = log f_obs - b - log f(z):
    NLL(z) = 1/2 d^T P d                                     (P = inverse of the calibrated 5x5 covariance)
           + sum_j (1 - MAC(v_j^obs, v_j(z))) / (2 s_v,j^2)  [+ the same for rotations if observed]
The covariance of log f is not diagonal: temperature shifts all frequencies together. b is the mean
discretisation bias of the decoder with respect to the 50-element model. For a unit-norm shape with Gaussian
noise, 1 - MAC is the squared noise component perpendicular to the shape, so the shape term is the Gaussian
log-density of that noise and does not depend on the mode sign.
"""

import numpy as np


def get_beam_element_matrices(E, I, rho, Area, L_e):
    """Returns 4x4 Euler-Bernoulli stiffness and consistent mass matrices (with rot. inertia)."""
    k_const = E * I / (L_e ** 3)
    Ke = k_const * np.array([
        [12.0,      6.0 * L_e,     -12.0,      6.0 * L_e],
        [6.0 * L_e, 4.0 * L_e**2,  -6.0 * L_e, 2.0 * L_e**2],
        [-12.0,    -6.0 * L_e,      12.0,     -6.0 * L_e],
        [6.0 * L_e, 2.0 * L_e**2,  -6.0 * L_e, 4.0 * L_e**2]
    ], dtype=np.float64)

    m_const = rho * Area * L_e
    r2 = I / Area
    Me = m_const * np.array([
        [13/35 + 6*r2/(5*L_e**2),      11*L_e/210 + r2/(10*L_e),    9/70 - 6*r2/(5*L_e**2),      -13*L_e/420 + r2/(10*L_e)],
        [11*L_e/210 + r2/(10*L_e),     L_e**2/105 + 2*r2/15,        13*L_e/420 - r2/(10*L_e),    -L_e**2/140 - r2/30],
        [9/70 - 6*r2/(5*L_e**2),       13*L_e/420 - r2/(10*L_e),    13/35 + 6*r2/(5*L_e**2),     -11*L_e/210 - r2/(10*L_e)],
        [-13*L_e/420 + r2/(10*L_e),   -L_e**2/140 - r2/30,         -11*L_e/210 - r2/(10*L_e),     L_e**2/105 + 2*r2/15]
    ], dtype=np.float64)
    return Ke, Me


class BeamMesh:
    """Element-level matrices of a beam with `el_per_zone` elements in each of the `n_zones` damage zones."""

    def __init__(self, n_zones=5, el_per_zone=1, L_total=10.0, E=210e9, I=8.33e-6, rho=7850.0, Area=0.005,
                 kv0=1.5e8, kr0=5.0e6, sensor_positions=None):
        self.n_zones, self.el_per_zone = n_zones, el_per_zone
        self.n_el = n_zones * el_per_zone
        self.n_dofs = 2 * (self.n_el + 1)
        self.zone_of_el = np.repeat(np.arange(n_zones), el_per_zone)
        if sensor_positions is None:                                   # default: the zone boundaries
            self.sensor_nodes = np.arange(n_zones + 1) * el_per_zone
        else:
            s = np.asarray(sensor_positions, dtype=float) / (L_total / self.n_el)
            if np.any(np.abs(s - np.round(s)) > 1e-6):
                raise ValueError(f"sensor positions {sensor_positions} are not nodes of a {self.n_el}-element mesh")
            self.sensor_nodes = np.round(s).astype(int)
        Ke, Me = get_beam_element_matrices(E, I, rho, Area, L_total / self.n_el)

        nd = self.n_dofs
        self.K_el = np.zeros((self.n_el, nd, nd))
        self.M_el = np.zeros((self.n_el, nd, nd))
        for e in range(self.n_el):
            s = slice(2 * e, 2 * e + 4)
            self.K_el[e, s, s] = Ke
            self.M_el[e, s, s] = Me
        self.K_bear = np.zeros((2, nd, nd))
        self.K_bear[0, 0, 0], self.K_bear[0, 1, 1] = kv0, kr0
        self.K_bear[1, nd - 2, nd - 2], self.K_bear[1, nd - 1, nd - 1] = kv0, kr0

    def zone_basis(self):
        """K(z) = sum_d z_d K_basis[d] (5 zones + 2 bearings) and the nominal mass matrix."""
        K_zone = np.stack([self.K_el[self.zone_of_el == i].sum(axis=0) for i in range(self.n_zones)])
        return np.concatenate([K_zone, self.K_bear]), self.M_el.sum(axis=0)

    def assemble(self, el_stiff, bear_stiff, el_mass=None):
        """Batched K and M from element stiffness factors (N, n_el), bearing factors (N, 2), mass factors (N, n_el)."""
        nd = self.n_dofs
        K = (el_stiff @ self.K_el.reshape(self.n_el, -1) + bear_stiff @ self.K_bear.reshape(2, -1)).reshape(-1, nd, nd)
        if el_mass is None:
            return K, self.M_el.sum(axis=0)
        return K, (el_mass @ self.M_el.reshape(self.n_el, -1)).reshape(-1, nd, nd)


def build_baseline_matrices(n_zones=5, el_per_zone=1, **structure):
    """Decoder matrices: M (nd, nd), K_basis (7, nd, nd), sensor_nodes, L_inv = inv(chol(M))."""
    mesh = BeamMesh(n_zones, el_per_zone, **structure)
    K_basis, M = mesh.zone_basis()
    return M, K_basis, mesh.sensor_nodes, np.linalg.inv(np.linalg.cholesky(M))


def mac(x, y):
    """MAC between the last axes of x and y (broadcast over leading axes)."""
    num = np.sum(x * y, axis=-1) ** 2
    return num / (np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1) + 1e-30)


def align_signs(vert, rot, ref):
    """Flips every mode so that vert . ref_vert + rot . ref_rot > 0 (ref = (ref_vert, ref_rot), unit-norm shapes)."""
    s = np.sign(np.sum(vert * ref[0][None], axis=-1, keepdims=True) + np.sum(rot * ref[1][None], axis=-1, keepdims=True))
    s[s == 0] = 1.0
    return vert * s, rot * s


def modal_solve(K, M, sensor_nodes, n_modes, ref=None):
    """
    Batched modal analysis in float64. K (N, nd, nd); M (nd, nd) shared or (N, nd, nd).
    Returns f (N, n_modes) in Hz and unit-norm vertical / rotational shapes at the sensors (N, n_modes, n_sensors).
    """
    L_inv = np.linalg.inv(np.linalg.cholesky(M))
    Lt = np.swapaxes(L_inv, -1, -2)
    A = L_inv @ K @ Lt
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    lam, Y = np.linalg.eigh(A)
    phi = Lt @ Y[:, :, :n_modes]                                          # (N, nd, n_modes)

    f = np.sqrt(np.maximum(lam[:, :n_modes], 1e-12)) / (2.0 * np.pi)
    vert = np.swapaxes(phi[:, 2 * sensor_nodes, :], 1, 2)
    rot = np.swapaxes(phi[:, 2 * sensor_nodes + 1, :], 1, 2)
    vert = vert / np.linalg.norm(vert, axis=-1, keepdims=True)
    rot = rot / np.linalg.norm(rot, axis=-1, keepdims=True)
    if ref is not None:
        vert, rot = align_signs(vert, rot, ref)
    return f, vert, rot


def simulate_true_structure(z, mesh, rng, n_modes, temp_std=0.0, mass_std=0.0, stiff_var=0.0, ref=None, chunk=500):
    """
    Noise-free modal data of the "true" structures z (N, 7) on `mesh`, with what the decoder ignores:
      deck modulus factor 1 + N(0, temp_std) common to all deck elements (temperature),
      element mass factors 1 + N(0, mass_std),
      element flexibility scatter (1 + N(0, stiff_var)) with zero mean inside every zone, so that the
      zone's equivalent (harmonic-mean) stiffness factor is exactly z.
    """
    z = np.atleast_2d(z)
    out = []
    for i0 in range(0, len(z), chunk):
        zz = z[i0:i0 + chunk]
        n = len(zz)
        eps = rng.normal(0.0, stiff_var, size=(n, mesh.n_el))
        for k in range(mesh.n_zones):
            m = mesh.zone_of_el == k
            eps[:, m] -= eps[:, m].mean(axis=1, keepdims=True)
        temp = 1.0 + rng.normal(0.0, temp_std, size=(n, 1))
        el_stiff = temp * zz[:, mesh.zone_of_el] / (1.0 + eps)
        el_mass = 1.0 + rng.normal(0.0, mass_std, size=(n, mesh.n_el))
        K, M = mesh.assemble(el_stiff, zz[:, mesh.n_zones:], el_mass)
        out.append(modal_solve(K, M, mesh.sensor_nodes, n_modes, ref))
    return tuple(np.concatenate([o[i] for o in out]) for i in range(3))


def add_measurement_noise(f, vert, rot, rng, freq_noise, mode_noise, ref=None):
    """
    f * (1 + N(0, freq_noise)); every mode-shape component gets N(0, (mode_noise * peak|component|)^2);
    vertical and rotational parts are then re-normalised (as in the original noisy data generator).
    """
    f_n = f * (1.0 + rng.normal(0.0, freq_noise, size=f.shape))
    v_n = vert + rng.normal(size=vert.shape) * mode_noise * np.max(np.abs(vert), axis=-1, keepdims=True)
    r_n = rot + rng.normal(size=rot.shape) * mode_noise * np.max(np.abs(rot), axis=-1, keepdims=True)
    v_n /= np.linalg.norm(v_n, axis=-1, keepdims=True)
    r_n /= np.linalg.norm(r_n, axis=-1, keepdims=True)
    if ref is not None:
        v_n, r_n = align_signs(v_n, r_n, ref)
    return f_n, v_n, r_n


def frequency_precision(lik):
    """(bias, precision matrix) of the log-frequency term; diagonal fallback for older problem_config.json files."""
    if "prec_logf" in lik:
        return np.asarray(lik["bias_logf"]), np.asarray(lik["prec_logf"])
    s = np.asarray(lik["sig_logf"])
    return np.zeros_like(s), np.diag(1.0 / s ** 2)


def neg_log_likelihood(f_obs, vert_obs, rot_obs, f_pred, vert_pred, rot_pred, lik, obs_types="fv"):
    """
    NLL (constants dropped) of one or many observations under predicted modal data.
      f_obs (..., n_modes) Hz,  vert_obs / rot_obs (..., n_modes, n_sensors); predictions broadcastable.
      lik: likelihood dict of problem_config.json.
    Returns total (...,) and a dict with the 'f', 'v', 'r' parts.
    """
    parts = {}
    if "f" in obs_types:
        bias, prec = frequency_precision(lik)
        d = np.log(f_obs) - bias - np.log(f_pred)
        parts["f"] = 0.5 * np.einsum("...m,mn,...n->...", d, prec, d)
    if "v" in obs_types:
        parts["v"] = np.sum((1.0 - mac(vert_obs, vert_pred)) / (2.0 * np.asarray(lik["sig_v"]) ** 2), axis=-1)
    if "r" in obs_types:
        parts["r"] = np.sum((1.0 - mac(rot_obs, rot_pred)) / (2.0 * np.asarray(lik["sig_r"]) ** 2), axis=-1)
    return sum(parts.values()), parts


class BridgeModel:
    """Decoder of one problem_config.json (mesh, reference shapes, likelihood)."""

    def __init__(self, pcfg):
        self.pcfg = pcfg
        self.n_modes = int(pcfg["n_modes"])
        self.lbound = float(pcfg["lbound"])
        _, self.K_basis, self.sensor_nodes, self.L_inv = build_baseline_matrices(
            n_zones=int(pcfg["n_zones"]), el_per_zone=int(pcfg.get("el_per_zone_decoder", 1)),
            sensor_positions=pcfg.get("sensor_positions"), **pcfg["structure"])
        self.A_basis = self.L_inv[None] @ self.K_basis @ self.L_inv.T[None]
        self.ref = (np.asarray(pcfg["ref_vert"]), np.asarray(pcfg["ref_rot"])) if "ref_vert" in pcfg else None
        self.lik = pcfg.get("likelihood")

    def solve(self, z):
        """Decoder modal data (nominal mass); A(z) = sum_d z_d L^-1 K_d L^-T is one matrix product per batch."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        nd = self.K_basis.shape[-1]
        A = (z @ self.A_basis.reshape(len(self.A_basis), -1)).reshape(len(z), nd, nd)
        lam, Y = np.linalg.eigh(0.5 * (A + np.swapaxes(A, -1, -2)))
        phi = self.L_inv.T[None] @ Y[:, :, :self.n_modes]
        f = np.sqrt(np.maximum(lam[:, :self.n_modes], 1e-12)) / (2.0 * np.pi)
        vert = np.swapaxes(phi[:, 2 * self.sensor_nodes, :], 1, 2)
        rot = np.swapaxes(phi[:, 2 * self.sensor_nodes + 1, :], 1, 2)
        vert = vert / np.linalg.norm(vert, axis=-1, keepdims=True)
        rot = rot / np.linalg.norm(rot, axis=-1, keepdims=True)
        if self.ref is not None:
            vert, rot = align_signs(vert, rot, self.ref)
        return f, vert, rot

    def log_likelihood_fn(self, f_obs, vert_obs, rot_obs, gamma2=1.0, obs_types=None, chunk=8192):
        """Returns z (N,7) -> log L(z) / gamma2 for one observation."""
        obs_types = obs_types or self.pcfg["obs_types"]

        def log_lik(z):
            out = np.empty(len(z))
            for i in range(0, len(z), chunk):
                f, v, r = self.solve(z[i:i + chunk])
                nll, _ = neg_log_likelihood(f_obs, vert_obs, rot_obs, f, v, r, self.lik, obs_types)
                out[i:i + chunk] = -nll / gamma2
            return out
        return log_lik
