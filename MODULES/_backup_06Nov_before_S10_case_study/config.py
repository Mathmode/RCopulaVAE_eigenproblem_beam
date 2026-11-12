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
# One girder line of a steel-concrete composite deck, L = 10 m: HEB 600 + 2.0 m x 0.22 m C30/37 slab (n = 210/33,
# full interaction): I = 5.25e-3 m4 in steel units (E = 210 GPa). Mass 1 800 kg/m = steel 212 + slab 1 100 + 80 mm
# surfacing 368 + edges/barriers 120 (rho * Area with rho = 7 850 kg/m3). First frequency ~11.9 Hz, inside the range
# of EN 1991-2 for a 10 m span (80/L = 8.0 to 94.76 L^-0.748 = 16.9 Hz). The previous academic beam (39 kg/m,
# I = 8.33e-6 m4, f1 = 3.6 Hz) made the bearings effectively rigid (dead-load pad compression 0.05 mm vs 28.7 mm of
# mid-span deflection); here 0.13 mm vs 2.1 mm.
# Bearings: laminated elastomeric pad sized for the girder reaction (88 kN dead load, ~350 kN with traffic: 5.8 MPa)
# per EN 1337-3: 200 mm (along the bridge) x 300 mm, 3 x 8 mm rubber layers, G = 0.9 MPa, shape factor S = 7.5;
# kv0 = Ec A / t_r with 1/Ec = 1/(6 G S^2) + 1/K (K = 2 000 MPa) -> 6.6e8 N/m;
# kr0 = Er (b a^3 / 12) / t_r with 1/Er = 1/(2 G S^2) + 1/K -> 8.0e5 N m/rad (the pad lets the girder rotate:
# kr0 L / EI = 0.007). The vertical springs carry 6-56 % of the modal strain energy of modes 1-5.
STRUCTURE = dict(L_total=10.0, E=210e9, I=5.25e-3, rho=7850.0, Area=1800.0 / 7850.0, kv0=6.6e8, kr0=8.0e5)

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
WITHIN_ZONE_STIFF_VAR = 0.025 # flexibility scatter N(0, 2.5 %) of 0.2 m cells inside every zone, zero zone mean (the
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
# token of DATASET_VERSION names the training runs (KSVAE_fixed_03Nov-FINAL5modes_7D_...).
DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "03Nov_FINAL_GirderBearings_7D_5sensors_5modes")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output", "review29oct_fixed")
DATASET_VERSION = "03Nov-FINAL_girderline_padBearings_5sensors_5modes_mass5_localizedTestB"
GT_EXAMPLES = [2, 4, 5, 7]    # benchmarks whose reference posteriors are plotted by dataset_generation.py

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
