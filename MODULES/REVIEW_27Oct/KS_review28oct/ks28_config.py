# -*- coding: utf-8 -*-
"""
ks28_config.py -- Configuration objects for the EXTENDED case study (review 28 Oct).

Everything that defines the problem lives here, so that the dataset generator,
the TensorFlow decoder, the ground-truth sampler and the postprocessing all read
the SAME numbers (they are saved as JSON next to every dataset).

Three groups of parameters
--------------------------
1. BridgeConfig  : geometry, material, mesh, damage zones, elastic bearings,
                   sensors and the definition of the latent vector theta.
2. NoiseConfig   : measurement-noise model (also defines the likelihood).
3. PriorConfig   : the "scenario prior" (healthy / single-zone damage favoured).
                   It is BOTH the density used to sample training scenarios and
                   the prior p(theta) inside the ELBO and the ground truth.

Latent vector (what the VAE infers)
-----------------------------------
    theta = [ z_1, ..., z_nz,  eta_L, eta_R ]
      z_k    in [z_lb, 1]      stiffness factor of damage zone k (1 = healthy)
      eta_j  in [eta_lb, eta_ub]  log2 factor of the bearing stiffness at support j
                               (0 = nominal, <0 softened/deteriorated, >0 stiffened/seized)
    k_bearing_j = 2**eta_j * k_bearing_nominal

The support factors are latent only if `support_latent=True`; otherwise the
bearings are fixed at their nominal value (theta = z only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


# =============================================================================
@dataclass
class BridgeConfig:
    """Equivalent Euler-Bernoulli beam of a single-span short bridge on elastic bearings.

    Default values: a 20 m single-span concrete slab-girder deck (equivalent beam).
      E  = 34 GPa (C40/50 concrete)            I = 0.90 m^4 (whole deck section)
      A  = 4.8 m^2, rho = 2500 kg/m^3           extra non-structural mass 2000 kg/m
      -> m = 14 t/m, f1(pinned) ~ 5.8 Hz (in line with the empirical f1 ~ 100/L)
    Bearings: laminated elastomeric bearings at both abutments, modelled as a
    vertical spring kv and a rotational spring kr per support (rotational restraint
    from the elastomer and the eccentric shear stiffness). They are given in
    non-dimensional form so that the model stays meaningful if E, I or L change:
      kv = kv_ratio * EI / L^3        kr = kr_ratio * EI / L
    kv_ratio = 1500 gives kv ~ 5.7e9 N/m per abutment (several laminated elastomeric
    bearings), ~30x the midspan bending stiffness 48EI/L^3. Even so, the bearings
    change f2..f5 by 2-30 % w.r.t. a pinned beam (pinned: 5.8/23.2/52.3/92.9/145 Hz;
    nominal bearings: 6.7/22.8/46.4/73.2/102.5 Hz), so they cannot be ignored and their
    condition is identifiable from the modal data (Reviewer 2, c.2-3).
    """
    name: str = "KS28_bridge_20m_elastic_bearings"

    # --- geometry & material ---------------------------------------------
    L: float = 20.0
    E: float = 34.0e9
    I: float = 0.90
    A: float = 4.8
    rho: float = 2500.0
    m_nonstructural: float = 2000.0      # kg/m (pavement, barriers, ...)
    rotary_inertia: bool = False          # include rotary inertia in the consistent mass

    # --- damage zones and FE mesh -----------------------------------------
    n_zones: int = 6                      # latent stiffness zones (even -> symmetric pairs)
    elements_per_zone: int = 2            # mesh of the decoder (model used for training)
    z_lb: float = 0.50                    # lower bound of zone stiffness factors
    z_ub: float = 1.00

    # --- elastic bearings ---------------------------------------------------
    kv_ratio: float = 1500.0              # kv * L^3 / EI  (per support)
    kr_ratio: float = 1.0                 # kr * L / EI    (per support), 0 -> free rotation
    support_latent: bool = True           # infer the bearing factors eta_L, eta_R
    eta_lb: float = -2.0                  # log2 bounds: k in [0.25, 2] x nominal
    eta_ub: float = 1.0

    # --- sensors / observations --------------------------------------------
    # sensor positions given as fractions of L. None -> zone boundaries (incl. both
    # bearings). They must coincide with nodes of every mesh used (true for zone
    # boundaries and zone mid-points when elements_per_zone is even).
    sensor_x_over_L: Optional[List[float]] = None
    sensor_layout: str = "boundaries"     # "boundaries" | "boundaries+mid"
    n_modes: int = 5

    # ------------------------------------------------------------------------
    def __post_init__(self):
        if self.sensor_x_over_L is None:
            nz = self.n_zones
            if self.sensor_layout == "boundaries":
                xs = [k / nz for k in range(nz + 1)]
            elif self.sensor_layout == "boundaries+mid":
                xs = [k / (2 * nz) for k in range(2 * nz + 1)]
            else:
                raise ValueError("sensor_layout must be 'boundaries' or 'boundaries+mid'")
            self.sensor_x_over_L = [float(x) for x in xs]

    # derived quantities ------------------------------------------------------
    @property
    def EI(self) -> float:
        return self.E * self.I

    @property
    def mass_per_length(self) -> float:
        return self.rho * self.A + self.m_nonstructural

    @property
    def kv_nominal(self) -> float:
        return self.kv_ratio * self.EI / self.L ** 3

    @property
    def kr_nominal(self) -> float:
        return self.kr_ratio * self.EI / self.L

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_x_over_L)

    @property
    def n_latent(self) -> int:
        return self.n_zones + (2 if self.support_latent else 0)

    @property
    def latent_names(self) -> List[str]:
        names = [f"z{k + 1}" for k in range(self.n_zones)]
        if self.support_latent:
            names += ["etaL", "etaR"]
        return names

    @property
    def latent_lower(self) -> np.ndarray:
        lo = [self.z_lb] * self.n_zones
        if self.support_latent:
            lo += [self.eta_lb] * 2
        return np.array(lo, dtype=np.float64)

    @property
    def latent_upper(self) -> np.ndarray:
        hi = [self.z_ub] * self.n_zones
        if self.support_latent:
            hi += [self.eta_ub] * 2
        return np.array(hi, dtype=np.float64)

    @property
    def healthy_latent(self) -> np.ndarray:
        h = [1.0] * self.n_zones
        if self.support_latent:
            h += [0.0, 0.0]
        return np.array(h, dtype=np.float64)


# =============================================================================
@dataclass
class NoiseConfig:
    """Measurement noise model = likelihood model.

    Frequencies : f_obs = f * exp(sigma_f * xi)          (log-normal, CoV ~ sigma_f)
    Mode shapes : phi_obs = phi + sigma_phi * max|phi| * xi   (additive, at sensors)
    The observed mode shapes are then scaled to unit peak amplitude.

    Defaults are typical of OMA on short-span bridges (frequency CoV ~1 %,
    mode-shape noise a few %). The paper used sigma_f = 0.025, sigma_phi = 0.05;
    set those values to reproduce the original noise level.

    sigma_f_model / sigma_phi_model: optional model-discrepancy std added in
    quadrature in the LIKELIHOOD ONLY (not when generating data). They are meant
    to be estimated on the validation-shift set, never on the test set.
    """
    sigma_f: float = 0.010
    sigma_phi: float = 0.030
    sigma_f_model: object = 0.0       # float or list (one value per mode)
    sigma_phi_model: object = 0.0     # float or list (one value per mode)
    use_mode_shapes: bool = True

    def sigma_f_lik(self, n_modes: int) -> np.ndarray:
        """Likelihood std of log f, per mode (noise and model discrepancy in quadrature)."""
        m = np.broadcast_to(np.asarray(self.sigma_f_model, dtype=np.float64), (n_modes,)) \
            if np.ndim(self.sigma_f_model) == 0 else np.asarray(self.sigma_f_model, dtype=np.float64)[:n_modes]
        return np.sqrt(self.sigma_f ** 2 + m ** 2)

    def sigma_phi_lik(self, n_modes: int) -> np.ndarray:
        m = np.broadcast_to(np.asarray(self.sigma_phi_model, dtype=np.float64), (n_modes,)) \
            if np.ndim(self.sigma_phi_model) == 0 else np.asarray(self.sigma_phi_model, dtype=np.float64)[:n_modes]
        return np.sqrt(self.sigma_phi ** 2 + m ** 2)


# =============================================================================
@dataclass
class PriorConfig:
    """Scenario prior: most bridges are healthy; damage is usually localised.

    Hierarchical generative story (per scenario):
      1. number of damaged zones  n_d ~ Categorical(p_n_damaged)      (0, 1, 2, ..., nz)
      2. which zones              uniformly among the C(nz, n_d) subsets
      3. healthy zone   z ~ 1 - healthy_band * Beta(1, healthy_beta)   ("as-built" scatter)
         damaged zone   z ~ z_lb + (1 - z_lb) * Beta(dam_a, dam_b)     (mild > severe)
      4. each bearing independently anomalous with prob p_support_anomaly:
         nominal  eta ~ N(0, eta_nominal_std) truncated to [eta_lb, eta_ub]
         anomalous eta ~ U[eta_lb, eta_ub]
    The resulting density p(theta) is a finite mixture and is evaluated EXACTLY
    (see ks28_priors.py). It is continuous on the whole box, so it is a valid
    prior for the ELBO and for the ground-truth posterior.

    With the defaults: P(healthy)=0.30, P(one zone)=0.44, P(two)=0.15, P(>=3)=0.11.
    Every number of damaged zones keeps a non-zero probability, so the prior has full
    support on the box (widespread damage is rare, not impossible).
    """
    p_n_damaged: List[float] = field(default_factory=lambda: [0.30, 0.44, 0.15, 0.06, 0.03, 0.015, 0.005])
    healthy_band: float = 0.03            # healthy zones lie in [0.97, 1.0]
    healthy_beta: float = 2.0             # Beta(1, 2): more mass close to 1
    dam_a: float = 2.0                    # Beta(2, 1.2) on [z_lb, 1]: mild damage favoured,
    dam_b: float = 1.2                    #   severe damage (z -> z_lb) penalised (R3.3)
    p_support_anomaly: float = 0.10
    eta_nominal_std: float = 0.10         # ~7 % scatter of the nominal bearing stiffness

    def probs_n_damaged(self, n_zones: int) -> np.ndarray:
        p = np.zeros(n_zones + 1)
        q = np.asarray(self.p_n_damaged, dtype=np.float64)
        p[: min(len(q), n_zones + 1)] = q[: n_zones + 1]
        return p / p.sum()


# =============================================================================
# JSON helpers
# =============================================================================
def save_configs(path: str, bridge: BridgeConfig, noise: NoiseConfig, prior: PriorConfig,
                 extra: Optional[dict] = None) -> None:
    payload = dict(bridge=asdict(bridge), noise=asdict(noise), prior=asdict(prior))
    if extra:
        payload["extra"] = extra
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_configs(path: str):
    with open(path) as fh:
        payload = json.load(fh)
    bridge = BridgeConfig(**payload["bridge"])
    noise = NoiseConfig(**payload["noise"])
    prior = PriorConfig(**payload["prior"])
    return bridge, noise, prior, payload.get("extra", {})
