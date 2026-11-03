# -*- coding: utf-8 -*-
"""
config.py -- Every setting of the Cloude_rev29oct workflow.

All scripts in this folder read their defaults from here; edit this file (or
use the command-line overrides of each script) instead of editing the scripts.

Structure (values taken from MODULES/review29oct/dataset_generation.py)
-----------------------------------------------------------------------
10 m beam, E = 210 GPa, I = 8.33e-6 m^4, rho = 7850 kg/m^3, A = 0.005 m^2,
5 damage zones of 2 m, elastic bearings at both ends: vertical spring
kv0 = 1.5e8 N/m and rotational spring kr0 = 5e6 N m/rad; the bearing factor
multiplies both springs of its support. 6 sensors at the zone boundaries
(x = 0, 2, ..., 10 m) measuring vertical and rotational mode-shape components.

theta = [Z1, Z2, Z3, Z4, Z5, BL, BR] in [0.45, 1.0]^7
  Zk = EI of zone k / nominal EI;  BL, BR = bearing stiffness / nominal.

Inference model : 2 Euler-Bernoulli elements per zone (22 DOFs). With 1
                  element per zone the 5th frequency is 11.8 % below the
                  converged value (2.4 % for the 4th), far above the 1 %
                  measurement noise; with 2 elements per zone it is <= 0.5 %.
                  Set COARSE_EL_PER_ZONE = 1 to reproduce the 12-DOF model.
Test models     : 10 elements per zone (50 elements), element mass
                  perturbations, within-zone stiffness variability.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Keras 3 must run on the TensorFlow backend (some environments default to JAX).
# This module is imported before tensorflow by every script of the folder.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = ROOT / "Output"


# -----------------------------------------------------------------------------
# Structure
# -----------------------------------------------------------------------------
@dataclass
class BeamConfig:
    L_total: float = 10.0       # [m]
    E: float = 210e9            # [Pa]
    I: float = 8.33e-6          # [m^4]
    rho: float = 7850.0         # [kg/m^3]
    Area: float = 0.005         # [m^2]
    n_zones: int = 5
    kv0: float = 1.5e8          # [N/m]     vertical bearing stiffness per support
    kr0: float = 5.0e6          # [N m/rad] rotational bearing stiffness per support
    n_modes_max: int = 5        # modes stored in the datasets


COARSE_EL_PER_ZONE = 2          # inference (training) model
FINE_EL_PER_ZONE = 10           # test model (50 elements)

PARAM_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "BL", "BR"]
PARAM_LABELS = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]
N_ZONES, N_BEARINGS = 5, 2
N_DIMS = N_ZONES + N_BEARINGS
LBOUND, UBOUND = 0.45, 1.00     # support of every parameter (prior and posterior)

# Left/right mirror of theta (used by the reference sampler and the figures)
MIRROR = [4, 3, 2, 1, 0, 6, 5]


def param_bounds():
    return np.full(N_DIMS, LBOUND), np.full(N_DIMS, UBOUND)


# -----------------------------------------------------------------------------
# Measurement noise = likelihood (identical in data generation, ELBO and the
# reference posterior). Levels taken from dataset_generation.py.
# -----------------------------------------------------------------------------
@dataclass
class NoiseConfig:
    freq_std: float = 0.010       # std of log-frequency error (1 % frequency error)
    vert_rel_std: float = 0.025   # vertical mode-shape noise, fraction of the mode's peak
    rot_rel_std: float = 0.065    # rotational mode-shape noise, fraction of the mode's peak
    # Mode-shape noise std per mode = rel_std * max|component| of the healthy
    # unit-norm mode shape (computed once from the inference model).


# -----------------------------------------------------------------------------
# Prior p(theta) and design of experiments rho(theta)   (R1.6, R3.3)
# -----------------------------------------------------------------------------
# Both are mixtures over damage scenarios. A scenario fixes which parameters
# are damaged; every other parameter is drawn from the HEALTHY family.
# Families are scaled Beta(alpha, beta) on [lo, hi]:
HEALTHY = (0.94, 1.00, 2.0, 1.0)            # mean 0.98 (dataset_generation.py: U(0.96, 1.00));
                                            # alpha > 1 keeps the density continuous at 0.94
DAMAGED_PRIOR = (LBOUND, 1.00, 2.2, 1.1)    # Beta(2.2, 1.1) of main_KS_29Oct.py: mild damage
                                            # favoured, severe damage penalized
DAMAGED_DOE = (LBOUND, 1.00, 1.0, 1.0)      # uniform: every severity equally represented

# Scenario probabilities. "three_sites" = any 3 of the 7 parameters;
# "broad" = all 7 parameters drawn from the damaged family (keeps p > 0 on the
# whole support, which a finite KL(q || p) requires).
PRIOR_SCENARIO_WEIGHTS = {        # realistic: healthy-dominant, localized damage
    "healthy": 0.50,
    "single_zone": 0.25,
    "single_bearing": 0.10,
    "zone_and_bearing": 0.05,
    "two_zones": 0.05,
    "three_sites": 0.02,
    "broad": 0.03,
}
DOE_SCENARIO_WEIGHTS = {          # training data: proportions of dataset_generation.py
    "healthy": 0.12,              # (12 / 33 / 18 / 22 (dual) / 15 (triple) %), plus 3 % broad
    "single_zone": 0.33,
    "single_bearing": 0.18,
    "zone_and_bearing": 0.12,
    "two_zones": 0.10,
    "three_sites": 0.12,
    "broad": 0.03,
}

# Evaluation thresholds
DAMAGE_THRESHOLD = 0.90           # theta < 0.90 counts as damaged (ROC, P(damage))
INTACT_THRESHOLD = 0.94           # theta >= 0.94 counts as intact (false-positive rate)
SEVERITY_BINS = [0.45, 0.60, 0.75, 0.85, 0.94, 1.00]


# -----------------------------------------------------------------------------
# FE variants used to simulate data
# -----------------------------------------------------------------------------
@dataclass
class FEVariant:
    name: str
    el_per_zone: int
    mass_std: float = 0.0           # element mass factor 1 + N(0, mass_std)
    stiffness_jitter: float = 0.0   # zero-mean within-zone flexibility variability
    noise_factor: float = 1.0       # multiplies all measurement-noise levels


VARIANTS = {
    "coarse": FEVariant("coarse", COARSE_EL_PER_ZONE),
    "fine": FEVariant("fine", FINE_EL_PER_ZONE, mass_std=0.02, stiffness_jitter=0.025),
    "fine_mass0": FEVariant("fine_mass0", FINE_EL_PER_ZONE, mass_std=0.0, stiffness_jitter=0.025),
    "fine_mass5": FEVariant("fine_mass5", FINE_EL_PER_ZONE, mass_std=0.05, stiffness_jitter=0.025),
    "fine_noise2": FEVariant("fine_noise2", FINE_EL_PER_ZONE, mass_std=0.02, stiffness_jitter=0.025,
                             noise_factor=2.0),
}

# Datasets: FE variant, size, sampling density of theta, seed
DATASETS = {
    "train":              dict(variant="coarse", n=30000, sampler="doe", seed=101),
    "val":                dict(variant="coarse", n=4000, sampler="doe", seed=202),
    "test_coarse":        dict(variant="coarse", n=1000, sampler="prior", seed=303),
    "test_fine":          dict(variant="fine", n=2000, sampler="prior", seed=404),
    "test_fine_balanced": dict(variant="fine", n=3000, sampler="doe", seed=505),
    "test_fine_mass0":    dict(variant="fine_mass0", n=1000, sampler="prior", seed=606),
    "test_fine_mass5":    dict(variant="fine_mass5", n=1000, sampler="prior", seed=707),
    "ood_four_sites":     dict(variant="fine", n=500, sampler="four_sites", seed=808),
    "ood_noise2":         dict(variant="fine_noise2", n=1000, sampler="prior", seed=909),
}
TEST_SETS = [k for k in DATASETS if k not in ("train", "val")]

# Benchmark scenarios of dataset_generation.py (positions 0..9), simulated with
# the coarse and with the fine model. Mode shapes are NOT blended with mirrored
# data: every observation is a plain noisy simulation.
BENCHMARKS = {
    "P0_healthy":         [0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.98],
    "P1_Z1_severe":       [0.56, 0.98, 0.99, 0.98, 0.99, 0.98, 0.99],
    "P2_Z3_moderate":     [0.98, 0.99, 0.65, 0.98, 0.99, 0.99, 0.98],
    "P3_Z2_mild":         [0.99, 0.78, 0.98, 0.99, 0.98, 0.98, 0.99],
    "P4_BL":              [0.98, 0.99, 0.98, 0.99, 0.98, 0.55, 0.98],
    "P5_Z1_BL":           [0.68, 0.98, 0.99, 0.98, 0.99, 0.60, 0.98],
    "P6_Z2_Z4_BR":        [0.98, 0.54, 0.98, 0.72, 0.98, 0.98, 0.64],
    "P7_five_sites":      [0.62, 0.82, 0.55, 0.98, 0.74, 0.68, 0.98],
    "P8_Z1_Z5_BL":        [0.64, 0.98, 0.99, 0.98, 0.76, 0.72, 0.98],
    "P9_Z3_BL_BR":        [0.98, 0.98, 0.70, 0.98, 0.98, 0.58, 0.74],
}
BENCHMARK_SEED = 999


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    run_name: str = ""                # empty -> automatic name
    n_modes: int = 5                  # modes used (<= n_modes_max)            (R3.4)
    obs: str = "fvr"                  # f = frequencies, v = vertical, r = rotational modes
    num_KS: int = 3                   # Kumaraswamy components per marginal
    hidden: tuple = (256, 256, 192)
    batch_size: int = 256
    n_samples: int = 4                # Monte Carlo samples H per observation   (R3.7)
    epochs: int = 400
    anneal_epochs: int = 100          # likelihood weight tau: tau_min -> 1 (geometric)
    tau_min: float = 0.01
    beta: float = 1.0                 # weight of KL(q || prior) (gamma^2 of the paper)
    w_fwd: float = 0.0                # optional amortized forward-KL term (uses labels)
    lr: float = 1e-3
    lr_final: float = 1e-5
    clipnorm: float = 1.0             # global gradient-norm clipping in the optimizer
    patience: int = 80                # early stopping on validation -ELBO (after annealing)
    val_every: int = 2
    n_val_elbo: int = 1024            # validation cases used for the -ELBO estimate
    n_samples_val: int = 16
    seed: int = 1234
    n_train: int = 0                  # 0 -> all; > 0 -> subsample (quick tests)
    n_post_samples: int = 4000        # posterior samples stored per benchmark case
    eigh_on_cpu: bool = True          # small batched eigh is faster on CPU than on GPU
    jit_compile: bool = False


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------
def to_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
