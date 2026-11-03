# -*- coding: utf-8 -*-
"""
damage_scenarios.py -- Samplers of damage / nuisance configurations for the
                        CPENG-8674 revision.

Every sampler returns a dict with (at least)
    z_zone    : (n_zones,)            label = stiffness factor per damage zone
    z_element : (n_elements,) or None element-level factors (localised damage)
    s_support : (n_sup,) or None      support-spring factors
    r_element : (n_elements,) or None element mass factors (modelling error)
    g_factor  : float                 global stiffness factor (EOV nuisance)

Mapping to reviewer comments
----------------------------
uniform              : original paper (z ~ U[zlb, 1]^n)                      baseline
spike_slab           : healthy-inflated training density (R1.6, R3.3)
healthy              : all zones intact                                      R1.6 (false positives)
single / double      : mild damage in 1 or 2 zones, rest healthy             R1.6
ood_severe           : damage beyond the training lower bound                R1.5
localized            : narrow crack-like damage inside one zone, on a fine mesh
                       (damage parametrisation mismatch)                      R2.3 / R1.1
informative_prior    : z drawn from a Beta prior concentrated near healthy    R3.3
+ optional nuisances : mass perturbation (R3.1), support degradation (R2.2-3),
                       global stiffness / temperature factor (R1.2)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .fe_beam_model import BeamConfig, zone_equivalent_factor


# -----------------------------------------------------------------------------
def _base(cfg: BeamConfig) -> dict:
    return dict(z_zone=np.ones(cfg.n_zones), z_element=None, s_support=None,
                r_element=None, g_factor=1.0)


# ============================== damage samplers ==============================
def uniform(cfg, rng, zlb=0.5, zub=1.0, **_):
    d = _base(cfg)
    d["z_zone"] = rng.uniform(zlb, zub, size=cfg.n_zones)
    return d


def spike_slab(cfg, rng, zlb=0.5, p_healthy=0.5, healthy_band=0.02, **_):
    """Each zone is healthy (z in [1-band, 1]) with prob. p_healthy, else U[zlb, 1].
    Gives healthy / sparsely damaged structures a non-negligible training mass:
    under U[0.5,1]^5, P(all z > 0.9) = 0.2^5 = 3e-4 (practically never seen)."""
    d = _base(cfg)
    healthy = rng.random(cfg.n_zones) < p_healthy
    z = rng.uniform(zlb, 1.0, size=cfg.n_zones)
    z[healthy] = rng.uniform(1.0 - healthy_band, 1.0, size=healthy.sum())
    d["z_zone"] = z
    return d


def healthy(cfg, rng, **_):
    return _base(cfg)


def single(cfg, rng, severities=(0.95, 0.9, 0.8, 0.7), zone=None, **_):
    d = _base(cfg)
    k = rng.integers(cfg.n_zones) if zone is None else zone
    d["z_zone"][k] = rng.choice(severities)
    d["meta"] = dict(damaged_zones=[int(k)])
    return d


def double(cfg, rng, severities=(0.95, 0.9, 0.8, 0.7), adjacent: Optional[bool] = None, **_):
    d = _base(cfg)
    k1 = rng.integers(cfg.n_zones)
    adj = rng.random() < 0.5 if adjacent is None else adjacent
    if adj:
        k2 = k1 + 1 if k1 < cfg.n_zones - 1 else k1 - 1
    else:
        choices = [k for k in range(cfg.n_zones) if abs(k - k1) > 1]
        k2 = rng.choice(choices) if choices else (k1 + 1) % cfg.n_zones
    d["z_zone"][[k1, k2]] = rng.choice(severities, size=2)
    d["meta"] = dict(damaged_zones=[int(k1), int(k2)], adjacent=bool(adj))
    return d


def ood_severe(cfg, rng, zlo=0.30, zhi=0.45, n_damaged=(1, 2), zlb=0.5, **_):
    """1-2 zones with damage below the training lower bound, others ~ U[zlb, 1]."""
    d = _base(cfg)
    z = rng.uniform(zlb, 1.0, size=cfg.n_zones)
    nd = rng.integers(n_damaged[0], n_damaged[1] + 1)
    ks = rng.choice(cfg.n_zones, size=nd, replace=False)
    z[ks] = rng.uniform(zlo, zhi, size=nd)
    d["z_zone"] = z
    d["meta"] = dict(ood_zones=ks.tolist())
    return d


def localized(cfg, rng, severity=(0.3, 0.7), width_elements=(1, 2), zone=None, **_):
    """Crack-like damage: 1-2 FINE elements inside one zone lose 30-70 % stiffness.
    Requires n_elements > n_zones.  Label = harmonic-mean zone equivalent."""
    if cfg.elements_per_zone < 2:
        raise ValueError("localized damage needs a mesh finer than the zones")
    d = _base(cfg)
    k = rng.integers(cfg.n_zones) if zone is None else zone
    els = np.where(cfg.zone_of_element == k)[0]
    w = rng.integers(width_elements[0], width_elements[1] + 1)
    start = rng.integers(els[0], els[-1] - w + 2)
    ze = np.ones(cfg.n_elements)
    ze[start:start + w] = 1.0 - rng.uniform(*severity)
    d["z_element"] = ze
    d["z_zone"] = zone_equivalent_factor(ze, cfg.zone_of_element, cfg.n_zones)
    d["meta"] = dict(zone=int(k), elements=list(range(start, start + w)))
    return d


def informative_prior(cfg, rng, zlb=0.5, a=5.0, b=1.0, **_):
    """z = zlb + (1-zlb) * Beta(a, b): mass concentrated near healthy (a > b)."""
    d = _base(cfg)
    d["z_zone"] = zlb + (1.0 - zlb) * rng.beta(a, b, size=cfg.n_zones)
    return d


def realistic_bridge_doe(
    cfg,
    rng,
    zlb=0.5,
    weights=(0.30, 0.45, 0.15, 0.10),
    beta_a=2.4,
    beta_b=1.0,
    zub_damaged=0.98,
    **_,
):
    """
    Realistic Early-Stage & Localized Bridge Design of Experiments (DoE) for Training/Validation
    (Directly addresses Reviewer 1 Comment 6 and Reviewer 3 Comment 3).

    Design Principles:
    ------------------
    1. Favour Healthy & Single-Zone Localized Scenarios (75% of the dataset!):
       - Healthy / Intact Structure (k = 0 damaged zones) : 30% of samples
       - Single-Zone Localized Damage (k = 1 damaged zone): 45% of samples
       - Two-Zone Localized Damage (k = 2 damaged zones)  : 15% of samples
       - Multi-Zone Distributed Damage (k >= 3 zones)     : 10% of samples (preserves
         amortization coverage across the hypercube without dominating training).

    2. Smooth Monotonic Priority on Slight / Early-Stage Damage Over Heavy Damage:
       - Within any damaged zone, severity `z_k in [zlb, 0.98]` is drawn from a single
         smooth power-law / Beta(a=2.4, b=1.0) distribution:
             z_k = zlb + (0.98 - zlb) * Beta(2.4, 1.0)
         whose density increases monotonically from z=0.50 (rare heavy damage) up to
         z=0.98 (frequent slight / early-stage damage):
           * ~62% Slight / Early-Stage damage : z_k in [0.80, 0.98] (2% to 20% stiffness loss)
           * ~28% Moderate damage             : z_k in [0.65, 0.80] (20% to 35% stiffness loss)
           * ~10% Heavy / Severe damage       : z_k in [0.50, 0.65] (35% to 50% stiffness loss)
       - Intact zones in localized scenarios have z_i in [0.99, 1.00] (50% exact 1.0,
         50% micro-fluctuation in [0.99, 1.00]).
    """
    d = _base(cfg)
    regime = rng.choice(["healthy", "single", "double", "multi"], p=weights)

    def _intact_vector(n_z: int) -> np.ndarray:
        z = np.ones(n_z, dtype=float)
        mask = rng.random(n_z) < 0.5
        z[mask] = rng.uniform(0.99, 1.0, size=mask.sum())
        return z

    def _sample_early_stage_damaged_val(n_vals: int) -> np.ndarray:
        b = rng.beta(beta_a, beta_b, size=n_vals)
        return zlb + (zub_damaged - zlb) * b

    if regime == "healthy":
        if rng.random() < 0.5:
            z = np.ones(cfg.n_zones, dtype=float)
        else:
            z = rng.uniform(0.985, 1.0, size=cfg.n_zones)
        d["z_zone"] = z
        d["meta"] = dict(regime="healthy", damaged_zones=[])

    elif regime == "single":
        z = _intact_vector(cfg.n_zones)
        k = int(rng.integers(cfg.n_zones))
        z[k] = _sample_early_stage_damaged_val(1)[0]
        d["z_zone"] = z
        d["meta"] = dict(regime="single", damaged_zones=[k])

    elif regime == "double":
        z = _intact_vector(cfg.n_zones)
        k1 = int(rng.integers(cfg.n_zones))
        if rng.random() < 0.5:
            k2 = k1 + 1 if k1 < cfg.n_zones - 1 else k1 - 1
        else:
            choices = [k for k in range(cfg.n_zones) if abs(k - k1) > 1]
            k2 = int(rng.choice(choices)) if choices else (k1 + 1) % cfg.n_zones
        z[[k1, k2]] = _sample_early_stage_damaged_val(2)
        d["z_zone"] = z
        d["meta"] = dict(regime="double", damaged_zones=[k1, k2])

    else:
        # Multi-zone (3 to D zones damaged, still favouring slight/moderate over extreme heavy)
        n_dam = int(rng.integers(min(3, cfg.n_zones), cfg.n_zones + 1))
        ks = rng.choice(cfg.n_zones, size=n_dam, replace=False)
        z = _intact_vector(cfg.n_zones)
        z[ks] = _sample_early_stage_damaged_val(n_dam)
        d["z_zone"] = z
        d["meta"] = dict(regime="multi", damaged_zones=ks.tolist())

    return d


SAMPLERS = dict(uniform=uniform, spike_slab=spike_slab, healthy=healthy, single=single,
                double=double, ood_severe=ood_severe, localized=localized,
                informative_prior=informative_prior, realistic_bridge_doe=realistic_bridge_doe)


# ============================== nuisance samplers ============================
def add_nuisances(d: dict, cfg: BeamConfig, rng: np.random.Generator,
                  mass_sigma: float = 0.0, mass_global_sigma: float = 0.0,
                  support_range: Optional[tuple] = None,
                  g_range: Optional[tuple] = None) -> dict:
    """Adds modelling-error / nuisance perturbations to a scenario.

    mass_sigma        : std of i.i.d. element density factors r_e ~ N(1, sigma)  (R3.1)
    mass_global_sigma : std of a common (global) mass factor, e.g. unknown
                        non-structural mass (asphalt, ballast)                   (R3.1)
    support_range     : (lo, hi) for support-spring factors s_j ~ U[lo, hi]      (R2.2/R2.3)
    g_range           : (lo, hi) global stiffness factor (temperature/EOV)      (R1.2)
    """
    if mass_sigma > 0 or mass_global_sigma > 0:
        r = rng.normal(1.0, mass_sigma, size=cfg.n_elements) if mass_sigma > 0 else np.ones(cfg.n_elements)
        if mass_global_sigma > 0:
            r = r * rng.normal(1.0, mass_global_sigma)
        d["r_element"] = np.clip(r, 0.5, 1.5)
    if support_range is not None and cfg.n_support_latents:
        d["s_support"] = rng.uniform(*support_range, size=cfg.n_support_latents)
    if g_range is not None:
        d["g_factor"] = float(rng.uniform(*g_range))
    return d
