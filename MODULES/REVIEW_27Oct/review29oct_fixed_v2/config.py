#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v2/config.py

Every setting of the 7D bridge-on-bearings benchmark, in one place.

dataset_generation.py writes these values, plus the calibrated likelihood standard
deviations, to <data_dir>/problem_config.json. Training, the reference ("ground-truth")
posterior and the post-processing all read that JSON back, so the three of them use
exactly the same prior, forward model and likelihood.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# --- Structure (unchanged from MODULES/review29oct) ---
N_ZONES = 5
N_DIMS = N_ZONES + 2          # 5 deck zones + left/right bearing
N_MODES = 5
LBOUND = 0.45                 # every stiffness factor z lives in [LBOUND, 1]
STRUCTURE = dict(L_total=10.0, E=210e9, I=8.33e-6, rho=7850.0, Area=0.005, kv0=1.5e8, kr0=5.0e6)

# --- Finite-element models (reviewers R1.1, R1.7) ---
EL_PER_ZONE_DECODER = 2       # decoder of the VAE and reference posterior, training, validation and the "same-model"
                              # test set: 10 elements (22 DOFs) (1 element per zone: 13 % error on mode 5; 2: 0.5 %)
EL_PER_ZONE_TEST = 10         # "advanced" test set: 50 elements, same 5 zones and 6 sensors, same damage states
# Test sets: name -> elements per zone. Both use the same damage states z; only the model that generates the data
# differs. The first N_MODES modes of the 50-element model are the same physical modes as those of the decoder
# (paired MAC >= 0.998 over the whole parameter space), so both keep the first 5 modes.
TEST_SETS = {"test": EL_PER_ZONE_DECODER, "test_fine": EL_PER_ZONE_TEST}

# --- Observation model: realistic vibration test ---
OBS_TYPES = "fv"              # f = frequencies, v = vertical mode shapes, r = rotational shapes. Accelerometers
                              # measure vertical motion only, so rotations are NOT observed by default
SENSOR_POSITIONS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # [m] vertical accelerometers (here the zone boundaries); every
                              # position must be a node of the decoder mesh. The same sensors are used for every
                              # mesh, so the mode shapes always have len(SENSOR_POSITIONS) components
FREQ_NOISE = 0.025            # relative std of every identified frequency (as in the submitted manuscript)
MODE_NOISE = 0.050            # std of every mode-shape component, relative to the peak |component| (as in the
                              # submitted manuscript)

# --- Modelling error: what the decoder does NOT model (R1.2, R3.1); calibrated into the likelihood ---
TEMP_E_STD = 0.0              # temperature effect on the deck modulus: switched off (left for future work with
                              # experimental data); set e.g. 0.03 to include it
MASS_PERTURB = 0.020          # element mass factors 1 + N(0, 2 %) (non-structural mass, asphalt, ...)
WITHIN_ZONE_STIFF_VAR = 0.025 # 50-element model: element flexibility scatter inside each zone (zone value = harmonic
                              # mean, so z keeps its meaning)

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
SEEDS = dict(train=101, val=202, test=303, calibration=404)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "BridgeBearings_7D_v4_noTemp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "review29oct_fixed")
DATASET_VERSION = "v4_noTemp_paper_noise"
GT_EXAMPLES = [3, 6, 7, 9]    # benchmarks whose reference posteriors are plotted by dataset_generation.py

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
