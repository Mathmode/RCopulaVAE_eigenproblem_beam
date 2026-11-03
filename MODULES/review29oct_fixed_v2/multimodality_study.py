#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed_v2/multimodality_study.py

Does a poorer measurement setup (fewer sensors, fewer modes) produce wider and multimodal exact posteriors?
For each setup (sensor layout, number of modes) and bearing option, it simulates the benchmark observations with the
data generator (10-element model, mass and within-zone variability, manuscript noise), samples the exact posterior with
SMC (likelihood = measurement noise, uniform prior) and reports:
  * the posterior std of every parameter and the widest 95 % credible interval,
  * the number of separate peaks of every 1-D marginal (KDE peaks above 15 % of the maximum, at least 0.08 apart),
  * the mass assigned to the mirror location of the damage (e.g. P(z5 damaged) when z1 is the damaged zone).
Corner plots of every case are saved, so the shapes can be inspected before generating the definitive data set.

    python MODULES/review29oct_fixed_v2/multimodality_study.py
Results: Output/review29oct_fixed/multimodality_study/
"""

import os
import sys

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from multiprocessing import Pool

from MODULES.review29oct_fixed_v2 import config as C
from MODULES.review29oct_fixed_v2.fe_model import (BeamMesh, modal_solve, simulate_true_structure, add_measurement_noise,
                                                mac, align_signs)
from MODULES.review29oct_fixed_v2.reference_posterior import smc_sample
from MODULES.review29oct_fixed_v2.plotting import corner_single, kde_1d

SETUPS = {   # name: (sensor positions [m], number of modes)
    "7 sensors, 5 modes": ([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0], 5),
    "5 sensors, 4 modes": ([1.0, 3.0, 5.0, 7.0, 9.0], 4),
    "4 sensors, 3 modes": ([2.0, 4.0, 6.0, 8.0], 3),
    "3 sensors, 3 modes": ([3.0, 5.0, 7.0], 3),
}
BEARINGS = {"elastomeric pad": (3.6e7, 6.8e4)}
CASES = [1, 3, 6, 9]                            # benchmark positions
MIRROR = [4, 3, 2, 1, 0, 6, 5]
OUT_DIR = os.path.join(C.OUTPUT_DIR, "multimodality_study")


def count_peaks(samples, lb=C.LBOUND):
    grid = np.linspace(lb, 1.0, 400)
    y = kde_1d(samples, grid, lb)
    peaks, _ = find_peaks(np.concatenate([[0], y, [0]]), height=0.15 * y.max(), distance=int(0.08 / (grid[1] - grid[0])))
    return len(peaks)


def run_case(args):
    setup, bear, pos = args
    sensors, n_modes = SETUPS[setup]
    kv, kr = BEARINGS[bear]
    structure = dict(C.STRUCTURE, kv0=kv, kr0=kr)
    mesh = BeamMesh(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=sensors, **structure)
    K_basis, M = mesh.zone_basis()
    nd = K_basis.shape[-1]

    def decoder(z):
        K = (np.atleast_2d(z) @ K_basis.reshape(7, -1)).reshape(-1, nd, nd)
        return modal_solve(K, M, mesh.sensor_nodes, n_modes)

    _, v_h, r_h = decoder(np.ones(7))
    ref = (v_h[0], r_h[0])
    s_v = C.MODE_NOISE * np.abs(v_h[0]).max(axis=1)
    z_true = np.array(C.BENCHMARKS[pos][0])
    rng = np.random.default_rng(100 + pos)
    f, v, r = simulate_true_structure(z_true[None], mesh, rng, n_modes, mass_std=C.MASS_PERTURB,
                                      stiff_var=C.WITHIN_ZONE_STIFF_VAR, ref=ref)
    f_o, v_o, _ = add_measurement_noise(f, v, r, rng, C.FREQ_NOISE, C.MODE_NOISE, ref=ref)

    def log_lik(z):
        fp, vp, _ = decoder(z)
        nll = np.sum((np.log(f_o) - np.log(fp)) ** 2, axis=1) / (2 * C.FREQ_NOISE ** 2)
        nll += np.sum((1 - mac(v_o, vp)) / (2 * s_v ** 2), axis=1)
        return -nll

    s, _ = smc_sample(log_lik, 7, C.LBOUND, n_particles=4000, rng=np.random.default_rng(pos))
    lo, hi = np.quantile(s, [0.025, 0.975], axis=0)
    damaged = [d for d in range(7) if z_true[d] < C.DAMAGE_THRESHOLD]
    mirror_mass = [float(np.mean(s[:, MIRROR[d]] < 0.85)) for d in damaged if MIRROR[d] != d]
    true_mass = [float(np.mean(s[:, d] < 0.85)) for d in damaged if MIRROR[d] != d]
    tag = f"{setup}, {bear}"
    os.makedirs(OUT_DIR, exist_ok=True)
    corner_single(s, z_true, C.LBOUND, f"Sample {pos} ({C.BENCHMARKS[pos][1]}): exact posterior, {tag}",
                  "relative posterior density",
                  os.path.join(OUT_DIR, f"P{pos}_{setup.replace(' ', '').replace(',', '_')}_{bear.replace(' ', '')}.png"))
    return {"setup": setup, "bearings": bear, "pos": pos, "max_std": float(s.std(0).max()),
            "mean_std": float(s.std(0).mean()), "widest_95": float(np.max(hi - lo)),
            "peaks_per_param": [count_peaks(s[:, d]) for d in range(7)],
            "P(true location damaged)": true_mass, "P(mirror location damaged)": mirror_mass}


def main():
    jobs = [(st, b, p) for st in SETUPS for b in BEARINGS for p in CASES]
    with Pool(min(len(jobs), max(1, (os.cpu_count() or 2) // 2))) as pool:
        rows = pool.map(run_case, jobs)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "multimodality_study.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(df.round(3).to_string(index=False))
    print(f"\nCorner plots and CSV in {OUT_DIR}")


if __name__ == "__main__":
    main()
