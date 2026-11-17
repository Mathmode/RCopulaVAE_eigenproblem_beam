#!/usr/bin/env python3
"""
Test set A of the letter (13 Nov 2026): the SAME 1,200 test structures as 'test' and 'test_fine' (damage states,
as-built mass and stiffness deviations, benchmarks P0-P10) and the SAME noise draws, computed with the 50-element model
WITHOUT localizing the damage: same design of experiments as the training and validation data, finer discretization.
('test_fine', with the damage of every damaged zone localized, is test set B, the out-of-distribution set of R1.5.)

Writes <data>/{freqs,vertmodes,rotmodes,alphas,labels}_test_fine_uniform.npy and adds 'test_fine_uniform' to
problem_config.json["test_sets"], so that later trainings also predict it.
Usage (project root): python MODULES/review29oct_fixed/add_test_fine_uniform.py --data-dir Data/<data set>
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.dataset_generation import build_meshes, simulate_observations

NAME = "test_fine_uniform"

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=C.DATA_DIR)
a = ap.parse_args()
d = a.data_dir if os.path.isabs(a.data_dir) else os.path.join(ROOT, a.data_dir)
pcfg_path = os.path.join(d, "problem_config.json")
pcfg = json.load(open(pcfg_path))
ref = (np.array(pcfg["ref_vert"]), np.array(pcfg["ref_rot"]))
C.N_MODES = int(pcfg["n_modes"])   # same number of modes as the data set (e.g. 3 for the R3.4 data set)
z = np.load(os.path.join(d, "alphas_test.npy")).astype(np.float64)
cats = np.load(os.path.join(d, "labels_test.npy"))
fd = os.path.join(d, "asbuilt_fields")
fields = (np.load(os.path.join(fd, "mass_factors_test.npy")).astype(np.float64),
          np.load(os.path.join(fd, "flexibility_deviations_test.npy")).astype(np.float64))
_, fine = build_meshes()
# same noise stream as 'test' and 'test_fine' (dataset_generation: seed SEEDS["test"] + 1000, same array shapes)
f, v, r = simulate_observations(z, fine, fields, ref, np.random.default_rng(C.SEEDS["test"] + 1000))
for k, x in (("freqs", f), ("vertmodes", v), ("rotmodes", r), ("alphas", z)):
    np.save(os.path.join(d, f"{k}_{NAME}.npy"), x.astype(np.float32))
np.save(os.path.join(d, f"labels_{NAME}.npy"), cats)
pcfg.setdefault("test_sets", {})[NAME] = C.EL_PER_ZONE_TEST
json.dump(pcfg, open(pcfg_path, "w"), indent=2)
fa = np.load(os.path.join(d, "freqs_test.npy"))
print(f"{NAME}: {len(z)} structures with the {C.EL_PER_ZONE_TEST * C.N_ZONES}-element model, uniform damage in each zone; "
      f"median |f/f_test - 1| = {np.median(np.abs(f / fa - 1)):.4f}")
