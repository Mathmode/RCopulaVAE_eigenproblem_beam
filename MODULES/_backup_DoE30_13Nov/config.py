#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/config.py

Every setting of the 7D bridge-on-bearings benchmark, in one place.

dataset_generation.py writes these values, plus the calibrated likelihood standard
deviations, to <data_dir>/problem_config.json. Training, the reference ("ground-truth")
posterior and the post-processing all read that JSON back, so the three of them use
exactly the same prior, forward model and likelihood.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# --- Structure: FINAL case study (3 Nov 2026), a realistic single-span short bridge ---
N_ZONES = 5
N_DIMS = N_ZONES + 2          # 5 deck zones + left/right bearing
N_MODES = 5                   # identified modes (with 5 sensors at the zone centres: no spatial aliasing)
LBOUND = 0.45                 # every stiffness factor z lives in [LBOUND, 1]
# Bridge S10 of Moliner, Martinez-Rodrigo and Museros (2017), Eng. Struct. 132, 98-109, Tables 2 and 3: reinforced concrete
# slab deck derived from existing structures, L = 10 m, B = 11.6 m, h = 0.77 m, E = 3.4849e10 Pa (nominal), m = 31 301 kg/m
# (nominal, with ballast, sleepers and rails). Modelled as a beam on vertical elastic supports (their Eqs., and Erduran et
# al. 2022), with I = B h^3 / 12 and free end rotations, so no rotational springs. rho = m / (B h) is an equivalent density,
# used only so that rho * Area = m and the radius of gyration is that of the slab.
# Support: kappa = E I pi^3 / (Kv L^3) = 0.05, the intermediate value of their range {0, 0.05, 0.1}.
_S10 = dict(L=10.0, B=11.6, h=0.77, E=3.4849e10, m=31301.0, kappa=0.05)
_S10_I = _S10["B"] * _S10["h"] ** 3 / 12.0
_S10_A = _S10["B"] * _S10["h"]
STRUCTURE = dict(L_total=_S10["L"], E=_S10["E"], I=_S10_I, rho=_S10["m"] / _S10_A, Area=_S10_A,
                 kv0=_S10["E"] * _S10_I * 3.141592653589793 ** 3 / (_S10["kappa"] * _S10["L"] ** 3), kr0=0.0)

# --- Finite-element models (reviewers R1.1, R1.7) ---
EL_PER_ZONE_DECODER = 2       # decoder of the VAE and reference posterior, training, validation and the "same-model"
                              # test set: 10 elements (22 DOFs) (1 element per zone: 13 % error on mode 5; 2: 0.5 %)
EL_PER_ZONE_TEST = 10         # "advanced" test set: 50 elements, same 5 zones and sensors, same damage states
# Test set B (test_fine) also has a realistic damage MORPHOLOGY that the decoder cannot represent: in every damaged
# zone the loss of stiffness is concentrated in a contiguous 0.2-2.0 m sub-region at a random position, with the same
# equivalent (harmonic-mean) zone factor z* (fe_model.localize_zone_damage). A finer mesh alone changes the
# frequencies by <= 0.4 %; the localization adds a discrepancy of up to ~2 %, of the order of the noise.
TEST_FINE_LOCALIZED_DAMAGE = True
LOCAL_DAMAGE_MIN_STIFFNESS = 0.15  # stiffness factor of the damaged cells >= 0.15 (at most 85 % local loss)
# Test sets: name -> elements per zone. Both contain the SAME as-built structures (damage state z, mass and
# stiffness fields) and the SAME measurement-noise draws; only the FE model that generates the data differs.
# The first N_MODES modes of the 50-element model are the same physical modes as those of the decoder
# (paired MAC >= 0.998 over the whole parameter space), so both keep the first N_MODES modes.
TEST_SETS = {"test": EL_PER_ZONE_DECODER, "test_fine": EL_PER_ZONE_TEST}

# --- Observation model: realistic vibration test ---
OBS_TYPES = "fv"              # f = frequencies, v = vertical mode shapes, r = rotational shapes. Accelerometers
                              # measure vertical motion only, so rotations are NOT observed by default
SENSOR_POSITIONS = [1.0, 3.0, 5.0, 7.0, 9.0]  # [m] 5 vertical accelerometers at the zone centres; every
                              # position must be a node of the decoder mesh. The same sensors are used for every
                              # mesh, so the mode shapes always have len(SENSOR_POSITIONS) components
FREQ_NOISE = 0.025            # relative std of every identified frequency (as in the submitted manuscript)
MODE_NOISE = 0.050            # std of every mode-shape component, relative to the peak |component| (as in the
                              # submitted manuscript)

# --- Modelling error: how the as-built structure that produces the measurements differs from the NOMINAL model
#     of the decoder (R1.2, R3.1). Every measurement (train, val, test, test_fine) comes from an as-built structure
#     with its own random realisation; the VAE and the reference posterior only know the nominal model. The
#     realisations are never used by them: only the statistical size of the error enters the likelihood ---
MASS_PERTURB = 0.050          # mass factors 1 + N(0, 5 %), one per 1 m segment (= one decoder element; the five
                              # 0.2 m elements of the 50-element model inside a segment share it): non-structural
                              # mass, asphalt, as-built cross-sections
WITHIN_ZONE_STIFF_VAR = 0.0   # flexibility scatter N(0, 2.5 %) of 0.2 m cells inside every zone, zero zone mean (the
                              # zone value is the harmonic mean, so z keeps its meaning); each element takes the mean
                              # flexibility of its cells, so both meshes see the same cells

# --- Prior of the Bayesian model: uniform on [LBOUND, 1]^7 (implicit in the ELBO and in the reference sampler) ---

# --- Design of experiments (training distribution; independent of the prior) ---
HEALTHY_RANGE = (0.95, 1.00)  # an undamaged parameter
DAMAGE_RANGE = (LBOUND, 0.95) # a damaged parameter: continuous, no gaps, down to the lower bound
DAMAGE_THRESHOLD = HEALTHY_RANGE[0]
# Exact fractions; inside every category each location (5 zones + 2 bearings) or combination
# of locations gets the same number of samples, and severities are Latin-hypercube stratified.
SCENARIO_FRACTIONS = {
    "healthy": 0.10,          # all 7 parameters in HEALTHY_RANGE
    "single": 0.30,           # 1 damaged parameter   (7 locations, equal counts)
    "double": 0.25,           # 2 damaged parameters  (21 pairs, equal counts)
    "triple": 0.15,           # 3 damaged parameters  (35 triples, equal counts)
    "multi45": 0.10,          # 4 or 5 damaged parameters (56 combinations, equal counts)
    "uniform": 0.10,          # all 7 parameters uniform on [LBOUND, 1] (covers the whole prior box)
}
N_TRAIN, N_VAL, N_TEST = 14000, 2400, 1200
SEEDS = dict(train=101, val=202, test=303, calibration=404, localization=505)

# FINAL data set of the revision (the only one in Data/ once the old ones are removed). The first '_'-separated
# token of DATASET_VERSION names the training runs (KSVAE_fixed_06Nov-FINAL5modes_7D_...).
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "06Nov_FINAL_SlabS10_7D_5sensors_5modes")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "review29oct_fixed")
DATASET_VERSION = "06Nov-FINAL_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB"
GT_EXAMPLES = [2, 4, 5, 7]    # benchmarks whose reference posteriors are plotted by dataset_generation.py

# --- Realistic design of experiments (13 Nov 2026, "DoE2"; R1.6): set the environment variable CVAE_DOE=realistic before
#     running ANY script (generation, training, post-processing). Mostly healthy structures and one damaged component, few
#     with two, very few with three; damage spread over the structure only through the uniform category. Structure,
#     sensors, noise, modelling errors, prior and model are unchanged: only the damage scenarios of the data change.
#     Without the variable: unchanged. ---
if os.environ.get("CVAE_DOE", "") == "realistic":
    SCENARIO_FRACTIONS = {"healthy": 0.20, "single": 0.45, "double": 0.20, "triple": 0.05, "multi45": 0.0,
                          "uniform": 0.10}
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "13Nov_FINAL_SlabS10_7D_DoE2")
    DATASET_VERSION = "13Nov-DoE2_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB"

# --- Benchmark scenarios: positions 0..10 of the test set (0..9 keep the z* of review29oct) ---
BENCHMARKS = [
    ([0.99, 0.98, 0.99, 0.99, 0.98, 0.99, 0.98], "Healthy baseline"),
    ([0.56, 0.98, 0.99, 0.98, 0.99, 0.98, 0.99], "Severe damage, deck zone 1 (z1=0.56)"),
    ([0.98, 0.99, 0.65, 0.98, 0.99, 0.99, 0.98], "Moderate damage, mid-span zone 3 (z3=0.65)"),
    ([0.99, 0.78, 0.98, 0.99, 0.98, 0.98, 0.99], "Mild damage, deck zone 2 (z2=0.78)"),
    ([0.98, 0.99, 0.98, 0.99, 0.98, 0.55, 0.98], "Left bearing degradation (zBL=0.55)"),
    ([0.60, 0.98, 0.99, 0.98, 0.98, 0.58, 0.98], "Deck zone 1 + left bearing (z1=0.60, zBL=0.58)"),
    ([0.98, 0.54, 0.98, 0.72, 0.98, 0.98, 0.64], "Triple damage (z2=0.54, z4=0.72, zBR=0.64)"),
    ([0.62, 0.82, 0.55, 0.98, 0.74, 0.68, 0.98], "Five damaged parameters (z1, z2, z3, z5, zBL)"),
    ([0.56, 0.98, 0.99, 0.98, 0.98, 0.98, 0.98], "Severe damage, deck zone 1 (z1=0.56), new noise draw"),
    ([0.98, 0.58, 0.98, 0.98, 0.98, 0.62, 0.98], "Deck zone 2 + left bearing (z2=0.58, zBL=0.62)"),
    ([0.99, 0.98, 0.99, 0.90, 0.98, 0.99, 0.98], "Mild damage, deck zone 4 (z4=0.90)"),
]

PARAM_NAMES = ["z1", "z2", "z3", "z4", "z5", "zBL", "zBR"]
LABELS_7D = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$", r"$z_5$", r"$z_{b,L}$", r"$z_{b,R}$"]

# --- Finer damage zoning (R1.7, scalability of the latent space): set the environment variable CVAE_ZONES=10 before
#     running ANY script (generation, training, post-processing). The FE models, sensors and modes are unchanged: the
#     same 10-element decoder (now 1 element per zone) and 50-element test model (5 elements per zone), 5 sensors and
#     5 modes, but 10 deck zones of 1 m + 2 bearings = 12 latent variables. The benchmarks are the same structures as
#     in 7D (each 2 m zone split into two 1 m zones with the same stiffness factor). Without the variable: unchanged. ---
if os.environ.get("CVAE_ZONES", "5") == "10":
    N_ZONES = 10
    N_DIMS = N_ZONES + 2
    EL_PER_ZONE_DECODER = 1
    EL_PER_ZONE_TEST = 5
    TEST_SETS = {"test": EL_PER_ZONE_DECODER, "test_fine": EL_PER_ZONE_TEST}
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "11Nov_FINAL_SlabS10_12D_5sensors_5modes")
    DATASET_VERSION = "11Nov-FINAL12D_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB"
    BENCHMARKS = [([z for z in zb[:5] for _ in (0, 1)] + list(zb[5:]), desc + " [zone k of 7D = zones 2k-1, 2k]")
                  for zb, desc in BENCHMARKS]
    PARAM_NAMES = [f"z{i + 1}" for i in range(N_ZONES)] + ["zBL", "zBR"]
    LABELS_7D = [rf"$z_{{{i + 1}}}$" for i in range(N_ZONES)] + [r"$z_{b,L}$", r"$z_{b,R}$"]

# --- Denser instrumentation for the 12D case (R1.7): CVAE_ZONES=10 together with CVAE_INSTR=10 -> 10 sensors and 10
#     modes. Sensors at the nodes of the 10-element decoder, symmetric, at both bearings and without the mid-span node
#     (10 nodes of 11): max off-diagonal auto-MAC of the 10 healthy modes 0.32. Discretization error of the 10-element
#     decoder against the 50-element test model: <= 1.9 % (healthy) and <= 2.9 % (any damage state) on mode 10, paired
#     MAC >= 0.995; it enters the calibrated likelihood as modelling error. Without the variable: unchanged. ---
if os.environ.get("CVAE_INSTR", "5") == "10":
    N_MODES = 10
    SENSOR_POSITIONS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    _tag = f"{N_DIMS}D_10sensors_10modes"
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data", f"12Nov_FINAL_SlabS10_{_tag}")
    DATASET_VERSION = f"12Nov-FINAL{N_DIMS}D_MolinerS10_kappa005_10sensors_10modes_mass5_localizedTestB"
