# -*- coding: utf-8 -*-
"""
fe_beam_model.py  --  Generalised Euler-Bernoulli beam FE engine for the
                       CPENG-8674 major revision (review_27oct).

Why a new engine?
-----------------
The original generators (PREPROCESSING/*dataset_generation.py) hard-wire:
  * FE mesh == damage parametrisation (one z per FE element),
  * pinned-pinned supports (fixed vertical DOFs),
  * a known, constant mass matrix.
Most reviewer comments (R1.1, R1.2, R1.6, R1.7, R2.2-R2.4, R3.1) require
relaxing exactly these assumptions.  This module decouples them:

  K(z, s) = K_0 + sum_k z_k K_k^zone + sum_j s_j K_j^support        (affine)
  M(r)    = sum_e r_e M_e                                          (mass factors)

  * z_k : stiffness reduction factor of damage ZONE k (latent variable)
  * s_j : factor on the nominal stiffness of support spring j (optional latent)
  * r_e : element mass (density) factor (nuisance / modelling error)

With n_elements == n_zones, pinned supports and r_e == 1 the engine reproduces
the original dataset generator bit-for-bit (see verify_fe_model.py).

The affine structure is also what the revised physics-informed decoder
should use: it only needs the constant matrices {K_0, K_k^zone, K_j^support}
(saved with every dataset), so the same TF code handles any mesh/zone/support
configuration (einsum + eigh), instead of the element-scatter assembly.

DOF ordering per node: [v (vertical), theta (rotation)]  -> dof = 2*node + {0,1}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence

import numpy as np
import scipy.linalg as sla


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class BeamConfig:
    """Geometry, material, mesh, damage zones, supports and sensors."""
    name: str = "toy_beam_paper"
    L: float = 10.0            # span [m]
    E: float = 210e9           # Young modulus [Pa]
    I: float = 8.33e-6         # second moment of area [m^4]
    rho: float = 7850.0        # density [kg/m^3]
    A: float = 0.005           # cross-section area [m^2]
    n_elements: int = 5        # FE mesh (elements)
    n_zones: int = 5           # damage zones (latent stiffness dims); n_elements % n_zones == 0
    rotary_inertia: bool = True  # legacy consistent mass includes rotary-inertia terms

    # --- supports ---------------------------------------------------------
    # "pinned"  : vertical DOFs at both ends removed (original paper)
    # "elastic" : vertical (and optional rotational) springs at both ends
    support: str = "pinned"
    kv_nom: float = 0.0        # nominal vertical spring stiffness per support [N/m]
    kr_nom: float = 0.0        # nominal rotational spring stiffness per support [N m/rad]
    support_latent: bool = False  # if True, the kv springs are scaled by latent factors s_L, s_R

    # --- sensors ----------------------------------------------------------
    # FE node indices carrying sensors. None -> zone-boundary nodes
    # (reproduces the paper: 6 sensors at x = 0, 2, 4, 6, 8, 10 m).
    sensor_nodes: Optional[List[int]] = None

    # ---------------------------------------------------------------------
    def __post_init__(self):
        if self.n_elements % self.n_zones != 0:
            raise ValueError("n_elements must be a multiple of n_zones")
        if self.support not in ("pinned", "elastic"):
            raise ValueError("support must be 'pinned' or 'elastic'")
        if self.support == "elastic" and self.kv_nom <= 0:
            raise ValueError("elastic support requires kv_nom > 0")
        if self.sensor_nodes is None:
            step = self.n_elements // self.n_zones
            self.sensor_nodes = list(range(0, self.n_elements + 1, step))

    # derived quantities ---------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes

    @property
    def Le(self) -> float:
        return self.L / self.n_elements

    @property
    def EI(self) -> float:
        return self.E * self.I

    @property
    def elements_per_zone(self) -> int:
        return self.n_elements // self.n_zones

    @property
    def zone_of_element(self) -> np.ndarray:
        return np.arange(self.n_elements) // self.elements_per_zone

    @property
    def fixed_dofs(self) -> List[int]:
        if self.support == "pinned":
            return [0, self.n_dofs - 2]
        return []

    @property
    def free_dofs(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n_dofs), self.fixed_dofs)

    @property
    def n_support_latents(self) -> int:
        return 2 if (self.support == "elastic" and self.support_latent) else 0

    def latent_names(self) -> List[str]:
        names = [f"z_zone{k + 1}" for k in range(self.n_zones)]
        if self.n_support_latents:
            names += ["s_kv_left", "s_kv_right"]
        return names

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(n_nodes=self.n_nodes, n_dofs=self.n_dofs, Le=self.Le,
                 fixed_dofs=self.fixed_dofs, free_dofs=self.free_dofs.tolist(),
                 latent_names=self.latent_names(),
                 zone_of_element=self.zone_of_element.tolist())
        return d


# =============================================================================
# 2. ELEMENT MATRICES  (identical to the original generator)
# =============================================================================
def element_stiffness(EI: float, Le: float) -> np.ndarray:
    k = EI / Le ** 3
    return k * np.array([[12, 6 * Le, -12, 6 * Le],
                         [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
                         [-12, -6 * Le, 12, -6 * Le],
                         [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]], dtype=np.float64)


def element_mass(rho: float, A: float, I: float, Le: float, rotary_inertia: bool = True) -> np.ndarray:
    """Consistent mass matrix. With rotary_inertia=True it is the exact matrix
    used in the original dataset generator (translational + rotary inertia)."""
    m = rho * A * Le
    r = (I / A) if rotary_inertia else 0.0
    return m * np.array([
        [13 / 35 + 6 * r / (5 * Le ** 2), 11 * Le / 210 + r / (10 * Le), 9 / 70 - 6 * r / (5 * Le ** 2), -13 * Le / 420 + r / (10 * Le)],
        [11 * Le / 210 + r / (10 * Le), Le ** 2 / 105 + 2 * r / 15, 13 * Le / 420 - r / (10 * Le), -Le ** 2 / 140 - r / 30],
        [9 / 70 - 6 * r / (5 * Le ** 2), 13 * Le / 420 - r / (10 * Le), 13 / 35 + 6 * r / (5 * Le ** 2), -11 * Le / 210 - r / (10 * Le)],
        [-13 * Le / 420 + r / (10 * Le), -Le ** 2 / 140 - r / 30, -11 * Le / 210 - r / (10 * Le), Le ** 2 / 105 + 2 * r / 15],
    ], dtype=np.float64)


def _element_dofs(e: int) -> np.ndarray:
    return np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])


# =============================================================================
# 3. THE MODEL
# =============================================================================
class BeamFEModel:
    """Affine-parametrised beam model: K(z,s) = K0 + sum z_k Kzone_k + sum s_j Ksup_j."""

    def __init__(self, cfg: BeamConfig):
        self.cfg = cfg
        c = cfg
        self.Ke = element_stiffness(c.EI, c.Le)
        self.Me = element_mass(c.rho, c.A, c.I, c.Le, c.rotary_inertia)

        nd = c.n_dofs
        # expanded element matrices (full DOFs) ------------------------------
        self.Ke_full = np.zeros((c.n_elements, nd, nd))
        self.Me_full = np.zeros((c.n_elements, nd, nd))
        for e in range(c.n_elements):
            d = _element_dofs(e)
            self.Ke_full[e][np.ix_(d, d)] = self.Ke
            self.Me_full[e][np.ix_(d, d)] = self.Me

        # zone matrices -------------------------------------------------------
        self.Kzone_full = np.zeros((c.n_zones, nd, nd))
        for e, k in enumerate(c.zone_of_element):
            self.Kzone_full[k] += self.Ke_full[e]

        # supports --------------------------------------------------------------
        self.K0_full = np.zeros((nd, nd))
        self.Ksup_full = np.zeros((c.n_support_latents, nd, nd))
        if c.support == "elastic":
            vL, vR = 0, nd - 2           # vertical DOFs at the ends
            tL, tR = 1, nd - 1           # rotational DOFs at the ends
            if c.support_latent:
                self.Ksup_full[0, vL, vL] = c.kv_nom
                self.Ksup_full[1, vR, vR] = c.kv_nom
            else:
                self.K0_full[vL, vL] += c.kv_nom
                self.K0_full[vR, vR] += c.kv_nom
            # rotational springs are kept as fixed (non-latent) stiffness
            self.K0_full[tL, tL] += c.kr_nom
            self.K0_full[tR, tR] += c.kr_nom

        self.free = c.free_dofs
        self.M_nominal_full = self.Me_full.sum(axis=0)

    # ------------------------------------------------------------------ utils
    def _free(self, A: np.ndarray) -> np.ndarray:
        idx = self.free
        if A.ndim == 2:
            return A[np.ix_(idx, idx)]
        return A[:, idx][:, :, idx]

    # ------------------------------------------------------------- matrices
    def stiffness(self, z_zone: np.ndarray, s_support: Optional[np.ndarray] = None,
                  z_element: Optional[np.ndarray] = None,
                  global_factor: float = 1.0) -> np.ndarray:
        """Free-DOF stiffness matrix.

        z_zone     : (n_zones,) zone factors (ignored if z_element is given)
        z_element  : (n_elements,) element-level factors (localised damage,
                     model-form tests). Overrides z_zone.
        s_support  : (n_support_latents,) support spring factors.
        global_factor : multiplies the beam stiffness (EOV / temperature nuisance).
        """
        if z_element is not None:
            Kb = np.tensordot(z_element, self.Ke_full, axes=1)
        else:
            Kb = np.tensordot(z_zone, self.Kzone_full, axes=1)
        K = global_factor * Kb + self.K0_full
        if self.cfg.n_support_latents:
            s = np.ones(self.cfg.n_support_latents) if s_support is None else s_support
            K = K + np.tensordot(s, self.Ksup_full, axes=1)
        return self._free(K)

    def mass(self, r_element: Optional[np.ndarray] = None) -> np.ndarray:
        if r_element is None:
            return self._free(self.M_nominal_full)
        return self._free(np.tensordot(r_element, self.Me_full, axes=1))

    # --------------------------------------------------------------- solve
    def modal(self, K_free: np.ndarray, M_free: np.ndarray, n_modes: int):
        """Generalised eigenproblem K phi = w^2 M phi (mass-normalised modes).

        Returns
        -------
        f_hz  : (n_modes,)
        vert  : (n_modes, n_nodes)   vertical mode components at ALL FE nodes
        rot   : (n_modes, n_nodes)   rotational components at ALL FE nodes
        """
        w2, phi = sla.eigh(K_free, M_free, subset_by_index=[0, n_modes - 1])
        w2 = np.maximum(w2, 0.0)
        f_hz = np.sqrt(w2) / (2 * np.pi)
        full = np.zeros((self.cfg.n_dofs, n_modes))
        full[self.free, :] = phi
        vert = full[0::2, :].T
        rot = full[1::2, :].T
        return f_hz, vert, rot

    # -------------------------------------------------------- constant data
    def decoder_matrices(self) -> dict:
        """Constant matrices required by an affine physics-informed decoder."""
        M = self.mass()
        L = np.linalg.cholesky(M)
        return dict(
            Kzone_free=self._free(self.Kzone_full),
            Ksupport_free=self._free(self.Ksup_full) if self.cfg.n_support_latents else np.zeros((0,) + M.shape),
            K0_free=self._free(self.K0_full),
            Mass_matrix=M,
            L_inv=np.linalg.inv(L),
            Ke_matrix=self.Ke,                 # legacy decoder (element == zone, pinned)
        )


# =============================================================================
# 4. MEASUREMENT OPERATOR: sensor extraction, noise, sign convention
# =============================================================================
def observe(f_hz: np.ndarray, vert: np.ndarray, rot: np.ndarray, sensor_nodes: Sequence[int],
            rng: np.random.Generator, sigma_f: float = 0.025, sigma_phi: float = 0.05,
            noise_rotations: bool = True):
    """Replicates the original noise model (Noisy_random_dataset_generation.py):

    * frequencies : multiplicative zero-mean Gaussian, f~ = f (1 + eps), eps ~ N(0, sigma_f)
    * mode shapes : additive Gaussian with std = sigma_phi * max|phi_m| (per mode,
                    computed over the sensor nodes)
    * sign convention: largest |vertical| component positive (applied after noise)

    Returns clean and noisy (f, vert, rot) at the sensor nodes.
    """
    sn = np.asarray(sensor_nodes)
    v_c, r_c = vert[:, sn].copy(), rot[:, sn].copy()
    n_modes, n_s = v_c.shape

    # clean copy with sign convention
    sgn = np.sign(v_c[np.arange(n_modes), np.argmax(np.abs(v_c), axis=1)])
    sgn[sgn == 0] = 1.0
    clean = (f_hz.copy(), v_c * sgn[:, None], r_c * sgn[:, None])

    f_n = f_hz * (1.0 + rng.normal(0.0, sigma_f, size=n_modes)) if sigma_f > 0 else f_hz.copy()
    v_n, r_n = v_c.copy(), r_c.copy()
    if sigma_phi > 0:
        for m in range(n_modes):
            vmax = np.max(np.abs(v_n[m]))
            v_n[m] += rng.normal(0.0, sigma_phi * vmax, size=n_s)
            if noise_rotations:
                rmax = np.max(np.abs(r_n[m]))
                if rmax > 1e-12:
                    r_n[m] += rng.normal(0.0, sigma_phi * rmax, size=n_s)
    sgn = np.sign(v_n[np.arange(n_modes), np.argmax(np.abs(v_n), axis=1)])
    sgn[sgn == 0] = 1.0
    noisy = (f_n, v_n * sgn[:, None], r_n * sgn[:, None])
    return clean, noisy


def zone_equivalent_factor(z_element: np.ndarray, zone_of_element: np.ndarray, n_zones: int) -> np.ndarray:
    """Equivalent zone stiffness factor for element-level (localised) damage.

    Bending flexibilities of sub-segments add up in series, so the zone factor
    that preserves the zone's mean flexibility is the HARMONIC mean of the
    element factors.  Used as ground-truth label when the data come from a
    finer damage description than the one identified by the VAE.
    """
    z_eq = np.empty(n_zones)
    for k in range(n_zones):
        zk = z_element[zone_of_element == k]
        z_eq[k] = 1.0 / np.mean(1.0 / zk)
    return z_eq


def save_json(obj: dict, path: str):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
