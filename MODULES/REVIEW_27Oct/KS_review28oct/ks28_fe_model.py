# -*- coding: utf-8 -*-
"""
ks28_fe_model.py -- NumPy finite-element engine of the extended case study.

Model
-----
Euler-Bernoulli beam, cubic Hermite elements, DOFs per node [v, theta]
(dof = 2*node + 0 for the vertical displacement, 2*node + 1 for the rotation).
NO DOF IS FIXED: the beam rests on elastic bearings at x=0 and x=L, modelled as a
vertical spring kv and a rotational spring kr at each end node (Reviewer 2, c.2-3).

Stiffness is AFFINE in the latent parameters:

    K(theta) = sum_k z_k * K_zone[k]  +  sum_j 2**eta_j * K_sup[j]  (+ K_fixed)

    K_zone[k] : sum of the (global-size) element matrices of the elements of zone k
    K_sup[j]  : bearing matrix of support j (kv at the vertical DOF, kr at the rotation)
    K_fixed   : nominal bearings when they are not latent (support_latent=False)

The same affine "basis" {K_zone, K_sup, K_fixed, M} is saved with the dataset and
used by the TensorFlow decoder (einsum + eigh), so the decoder works for any mesh,
zone layout and support model, instead of the hard-coded element scatter.

The engine can also build a different "truth" model for test data (finer mesh,
perturbed masses, element-level crack patterns, temperature factor on E, different
rotational restraint), which is how the test samples come from a slightly different
model than the one used in training (Reviewer 1 c.1, Reviewer 3 c.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ks28_config import BridgeConfig


# =============================================================================
# 1. ELEMENT MATRICES
# =============================================================================
def beam_element_matrices(EI: float, m: float, Le: float, rotary_inertia: bool = False,
                          rhoI: float = 0.0):
    """Hermite Euler-Bernoulli element: stiffness and consistent mass (4x4).

    m    : mass per unit length [kg/m]
    rhoI : rotary inertia per unit length [kg m] (used only if rotary_inertia)
    """
    k = EI / Le ** 3
    Ke = k * np.array([[12, 6 * Le, -12, 6 * Le],
                       [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
                       [-12, -6 * Le, 12, -6 * Le],
                       [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]], dtype=np.float64)
    c = m * Le / 420.0
    Me = c * np.array([[156, 22 * Le, 54, -13 * Le],
                       [22 * Le, 4 * Le ** 2, 13 * Le, -3 * Le ** 2],
                       [54, 13 * Le, 156, -22 * Le],
                       [-13 * Le, -3 * Le ** 2, -22 * Le, 4 * Le ** 2]], dtype=np.float64)
    if rotary_inertia and rhoI > 0:
        r = rhoI / (30.0 * Le)
        Me = Me + r * np.array([[36, 3 * Le, -36, 3 * Le],
                                [3 * Le, 4 * Le ** 2, -3 * Le, -Le ** 2],
                                [-36, -3 * Le, 36, -3 * Le],
                                [3 * Le, -Le ** 2, -3 * Le, 4 * Le ** 2]], dtype=np.float64)
    return Ke, Me


# =============================================================================
# 2. SYSTEM "BASIS" (everything the forward model needs, independent of theta)
# =============================================================================
@dataclass
class FEBasis:
    n_elements: int
    n_nodes: int
    n_dofs: int
    node_x: np.ndarray            # (n_nodes,)
    zone_of_element: np.ndarray   # (n_el,)
    K_el: np.ndarray              # (n_el, n_dofs, n_dofs) element stiffness in global size
    M_el: np.ndarray              # (n_el, n_dofs, n_dofs) element mass in global size
    K_sup: np.ndarray             # (2, n_dofs, n_dofs) nominal bearing matrices (left, right)
    sensor_nodes: np.ndarray      # (n_sensors,)
    sensor_dofs: np.ndarray       # vertical DOF of every sensor node
    n_zones: int

    @property
    def K_zone(self) -> np.ndarray:
        """(n_zones, n_dofs, n_dofs): sum of element matrices per zone."""
        Kz = np.zeros((self.n_zones, self.n_dofs, self.n_dofs))
        for e, k in enumerate(self.zone_of_element):
            Kz[k] += self.K_el[e]
        return Kz

    @property
    def M(self) -> np.ndarray:
        return self.M_el.sum(axis=0)


def build_basis(cfg: BridgeConfig, elements_per_zone: Optional[int] = None) -> FEBasis:
    """Assemble element-level global matrices for a mesh with `elements_per_zone`."""
    epz = cfg.elements_per_zone if elements_per_zone is None else int(elements_per_zone)
    n_el = cfg.n_zones * epz
    n_nodes = n_el + 1
    n_dofs = 2 * n_nodes
    Le = cfg.L / n_el
    node_x = np.linspace(0.0, cfg.L, n_nodes)

    # rotary inertia of the cross-section (only used if cfg.rotary_inertia)
    rhoI = cfg.rho * cfg.I
    Ke, Me = beam_element_matrices(cfg.EI, cfg.mass_per_length, Le, cfg.rotary_inertia, rhoI)

    K_el = np.zeros((n_el, n_dofs, n_dofs))
    M_el = np.zeros((n_el, n_dofs, n_dofs))
    for e in range(n_el):
        idx = np.arange(2 * e, 2 * e + 4)
        K_el[e][np.ix_(idx, idx)] = Ke
        M_el[e][np.ix_(idx, idx)] = Me
    zone_of_element = np.repeat(np.arange(cfg.n_zones), epz)

    # elastic bearings (vertical + rotational spring) at the two end nodes
    K_sup = np.zeros((2, n_dofs, n_dofs))
    for j, node in enumerate([0, n_nodes - 1]):
        K_sup[j, 2 * node, 2 * node] = cfg.kv_nominal
        K_sup[j, 2 * node + 1, 2 * node + 1] = cfg.kr_nominal

    # sensors -> nodes (must coincide with nodes of this mesh)
    sx = np.asarray(cfg.sensor_x_over_L) * cfg.L
    sensor_nodes = np.array([int(np.argmin(np.abs(node_x - x))) for x in sx])
    if np.max(np.abs(node_x[sensor_nodes] - sx)) > 1e-9 * cfg.L:
        raise ValueError("Sensor positions do not coincide with FE nodes for this mesh "
                         f"(elements_per_zone={epz}). Use an even elements_per_zone or "
                         "sensors at zone boundaries.")
    sensor_dofs = 2 * sensor_nodes

    return FEBasis(n_el, n_nodes, n_dofs, node_x, zone_of_element, K_el, M_el, K_sup,
                   sensor_nodes, sensor_dofs, cfg.n_zones)


# =============================================================================
# 3. THETA  ->  MATRICES
# =============================================================================
def split_theta(theta: np.ndarray, cfg: BridgeConfig):
    """theta (N, D) -> z (N, nz), support factors s (N, 2) (= 2**eta, or ones)."""
    theta = np.atleast_2d(theta)
    z = theta[:, :cfg.n_zones]
    if cfg.support_latent:
        s = np.power(2.0, theta[:, cfg.n_zones:cfg.n_zones + 2])
    else:
        s = np.ones((theta.shape[0], 2))
    return z, s


def stiffness_from_factors(basis: FEBasis, element_factors: np.ndarray,
                           support_factors: np.ndarray) -> np.ndarray:
    """K = sum_e f_e K_el[e] + sum_j s_j K_sup[j]   for a batch (N, n_el), (N, 2)."""
    K = np.einsum("ne,eij->nij", element_factors, basis.K_el, optimize=True)
    K += np.einsum("ns,sij->nij", support_factors, basis.K_sup, optimize=True)
    return K


def stiffness_from_theta(basis: FEBasis, cfg: BridgeConfig, theta: np.ndarray) -> np.ndarray:
    z, s = split_theta(theta, cfg)
    f_el = z[:, basis.zone_of_element]                    # zone factor applied to its elements
    return stiffness_from_factors(basis, f_el, s)


# =============================================================================
# 4. MODAL ANALYSIS (batched)
# =============================================================================
def modal_analysis(K: np.ndarray, M: np.ndarray, n_modes: int, sensor_dofs: np.ndarray,
                   return_full: bool = False):
    """Generalised eigenproblem K phi = w^2 M phi for a batch of K (and M).

    K : (N, n, n);  M : (n, n) shared, or (N, n, n) per sample.
    Returns f [Hz] (N, n_modes) and vertical mode shapes at the sensors (N, n_modes, ns),
    mass-normalised (phi^T M phi = 1), before any noise or rescaling.
    """
    K = np.asarray(K, dtype=np.float64)
    if M.ndim == 2:
        C = np.linalg.cholesky(M)
        Cinv = np.linalg.inv(C)
        A = Cinv @ K @ Cinv.T
        lam, V = np.linalg.eigh(A)
        Phi = Cinv.T @ V[..., :n_modes]
    else:
        C = np.linalg.cholesky(M)
        Cinv = np.linalg.inv(C)
        A = Cinv @ K @ np.swapaxes(Cinv, -1, -2)
        lam, V = np.linalg.eigh(A)
        Phi = np.swapaxes(Cinv, -1, -2) @ V[..., :n_modes]
    lam = np.maximum(lam[..., :n_modes], 0.0)
    f = np.sqrt(lam) / (2.0 * np.pi)
    phi_s = np.swapaxes(Phi[:, sensor_dofs, :], 1, 2)     # (N, n_modes, n_sensors)
    if return_full:
        return f, phi_s, Phi
    return f, phi_s


def normalise_modes(phi: np.ndarray) -> np.ndarray:
    """Scale every mode to unit peak amplitude with a positive peak (N, nm, ns)."""
    idx = np.argmax(np.abs(phi), axis=-1)
    peak = np.take_along_axis(phi, idx[..., None], axis=-1)
    peak = np.where(np.abs(peak) < 1e-300, 1.0, peak)
    return phi / peak


def forward(theta: np.ndarray, basis: FEBasis, cfg: BridgeConfig, n_modes: Optional[int] = None,
            batch: int = 4096):
    """Noise-free forward model of the TRAINING model: theta -> (f, unit-peak modes)."""
    n_modes = cfg.n_modes if n_modes is None else n_modes
    theta = np.atleast_2d(theta)
    M = basis.M
    fs, ps = [], []
    for i in range(0, theta.shape[0], batch):
        K = stiffness_from_theta(basis, cfg, theta[i:i + batch])
        f, p = modal_analysis(K, M, n_modes, basis.sensor_dofs)
        fs.append(f)
        ps.append(normalise_modes(p))
    return np.concatenate(fs), np.concatenate(ps)


# =============================================================================
# 5. "TRUTH" MODEL FOR SHIFTED TEST DATA
# =============================================================================
def forward_truth(theta: np.ndarray, cfg: BridgeConfig, rng: np.random.Generator,
                  elements_per_zone: Optional[int] = None,
                  element_damage: Optional[np.ndarray] = None,
                  mass_cov: float = 0.0, mass_corr_elements: int = 1,
                  E_global: Optional[np.ndarray] = None,
                  kr_factor: float = 1.0, n_modes: Optional[int] = None,
                  basis: Optional[FEBasis] = None):
    """Forward model of a (possibly) different 'real' structure.

    theta            : (N, D) latent values (zone factors / bearing factors).
    elements_per_zone: mesh of the truth model (finer than the decoder -> model-form error).
    element_damage   : (N, n_el_truth) extra element-level stiffness factors (crack-like
                       damage inside a zone). Multiplies the zone factor.
    mass_cov         : CoV of a random element mass field (mass modelling error, R3.1).
                       mass_corr_elements > 1 makes the field piecewise constant over
                       blocks of that many elements (spatially correlated error).
    E_global         : (N,) global stiffness factor (temperature / EOV, R1.2).
    kr_factor        : multiplies the nominal rotational bearing stiffness (unmodelled
                       rotational restraint).
    Returns f (N, nm), unit-peak modes (N, nm, ns), and the element mass factors used.
    """
    n_modes = cfg.n_modes if n_modes is None else n_modes
    theta = np.atleast_2d(theta)
    N = theta.shape[0]
    if basis is None:
        basis = build_basis(cfg, elements_per_zone)
    n_el = basis.n_elements

    z, s = split_theta(theta, cfg)
    f_el = z[:, basis.zone_of_element]
    if element_damage is not None:
        f_el = f_el * element_damage
    if E_global is not None:
        f_el = f_el * np.asarray(E_global).reshape(-1, 1)

    K_sup = basis.K_sup.copy()
    if kr_factor != 1.0:
        for j, node in enumerate([0, basis.n_nodes - 1]):
            K_sup[j, 2 * node + 1, 2 * node + 1] *= kr_factor
    K = np.einsum("ne,eij->nij", f_el, basis.K_el, optimize=True)
    K += np.einsum("ns,sij->nij", s, K_sup, optimize=True)

    if mass_cov > 0:
        n_blocks = int(np.ceil(n_el / mass_corr_elements))
        r_block = 1.0 + mass_cov * rng.standard_normal((N, n_blocks))
        r_block = np.clip(r_block, 0.5, 1.5)
        r_el = np.repeat(r_block, mass_corr_elements, axis=1)[:, :n_el]
        M = np.einsum("ne,eij->nij", r_el, basis.M_el, optimize=True)
    else:
        r_el = np.ones((N, n_el))
        M = basis.M

    f, p = modal_analysis(K, M, n_modes, basis.sensor_dofs)
    return f, normalise_modes(p), r_el


# =============================================================================
# 6. MEASUREMENT NOISE
# =============================================================================
def add_measurement_noise(f: np.ndarray, phi: np.ndarray, sigma_f: float, sigma_phi: float,
                          rng: np.random.Generator):
    """f_obs = f exp(sigma_f xi);  phi_obs = phi + sigma_phi max|phi| xi, then unit peak."""
    f_obs = f * np.exp(sigma_f * rng.standard_normal(f.shape))
    peak = np.max(np.abs(phi), axis=-1, keepdims=True)
    phi_obs = phi + sigma_phi * peak * rng.standard_normal(phi.shape)
    return f_obs, normalise_modes(phi_obs)
