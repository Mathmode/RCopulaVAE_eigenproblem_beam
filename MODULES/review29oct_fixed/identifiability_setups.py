#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/identifiability_setups.py

R3.4: how the number of modes (and of sensors) changes the identifiability of the stiffness factors. For several
measurement setups, the data generator of the final case study (as-built 10-element structures of the test set,
calibrated likelihood of THAT setup) produces the observations of a few benchmarks, and their EXACT posteriors are
computed by adaptive SMC (reference_posterior.smc_sample). Reported per setup: mean posterior std, widest 95 %
credible interval, and the posterior probability that the true damaged zone and its mirror zone are damaged.

    python MODULES/review29oct_fixed/identifiability_setups.py
Results: Output/review29oct_fixed/identifiability_setups/ (CSV per benchmark and summary).
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

from MODULES.review29oct_fixed import config as C

SETUPS = {   # name: (sensor positions [m], number of modes)
    "7 sensors (zone centres + supports), 5 modes": ([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0], 5),
    "5 sensors, 5 modes (adopted)": ([1.0, 3.0, 5.0, 7.0, 9.0], 5),
    "5 sensors, 4 modes": ([1.0, 3.0, 5.0, 7.0, 9.0], 4),
    "5 sensors, 3 modes": ([1.0, 3.0, 5.0, 7.0, 9.0], 3),
    "3 sensors, 3 modes": ([3.0, 5.0, 7.0], 3),
}
BENCHMARKS = [1, 3, 4, 6, 9]          # positions in config.BENCHMARKS
MIRROR = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 6, 6: 5}
THRESHOLD = 0.85                      # "damaged" for the location probabilities


def setup_problem(sensors, n_modes):
    """Calibrated likelihood and test-set observations (10-element as-built structures) of one setup."""
    from MODULES.review29oct_fixed import dataset_generation as G
    from MODULES.review29oct_fixed.fe_model import build_baseline_matrices
    C.SENSOR_POSITIONS, C.N_MODES = list(sensors), int(n_modes)    # read by dataset_generation at call time
    M, K_basis, nodes, _ = build_baseline_matrices(C.N_ZONES, C.EL_PER_ZONE_DECODER, sensor_positions=sensors,
                                                    **C.STRUCTURE)
    dec = (K_basis, M, nodes)
    dec_mesh, _ = G.build_meshes()
    ref = G.reference_shapes(dec)
    lik = G.calibrate_likelihood(dec, dec_mesh, ref, np.random.default_rng(C.SEEDS["calibration"]))
    rng = np.random.default_rng(C.SEEDS["test"])                   # the test structures of the data set
    z, _ = G.sample_balanced_doe(C.N_TEST, rng)
    for pos, (zb, _) in enumerate(C.BENCHMARKS):
        z[pos] = zb
    fields = G.draw_fields(C.N_TEST, rng)
    f, v, r = G.simulate_observations(z, dec_mesh, fields, ref, np.random.default_rng(C.SEEDS["test"] + 1000))
    pcfg = {"n_modes": n_modes, "lbound": C.LBOUND, "n_zones": C.N_ZONES, "n_dims": C.N_DIMS,
            "el_per_zone_decoder": C.EL_PER_ZONE_DECODER, "sensor_positions": list(sensors), "structure": C.STRUCTURE,
            "ref_vert": ref[0].tolist(), "ref_rot": ref[1].tolist(), "likelihood": lik, "obs_types": C.OBS_TYPES}
    return pcfg, z, f, v, r


def _exact(args):
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.review29oct_fixed.reference_posterior import smc_sample
    key, pcfg, f, v, r, seed = args
    log_lik = BridgeModel(pcfg).log_likelihood_fn(f, v, r, gamma2=1.0, obs_types=pcfg["obs_types"])
    s, _ = smc_sample(log_lik, int(pcfg["n_dims"]), float(pcfg["lbound"]), n_particles=4000,
                      rng=np.random.default_rng(seed))
    return key, s


def main():
    out_dir = os.path.join(C.OUTPUT_DIR, "identifiability_setups")
    os.makedirs(out_dir, exist_ok=True)
    jobs, truth = [], {}
    for name, (sensors, n_modes) in SETUPS.items():
        pcfg, z, f, v, r = setup_problem(sensors, n_modes)
        print(f"  {name}: sigma log f {np.round(pcfg['likelihood']['sig_logf'], 4)}")
        for pos in BENCHMARKS:
            jobs.append(((name, pos), pcfg, f[pos], v[pos], r[pos], 30_000 + pos))
            truth[pos] = z[pos]
    from multiprocessing import Pool
    with Pool(max(1, (os.cpu_count() or 2) // 2)) as pool:
        results = dict(pool.map(_exact, jobs))
    rows = []
    for (name, pos), s in results.items():
        zt = truth[pos]
        lo, hi = np.quantile(s, [0.025, 0.975], axis=0)
        dam = int(np.argmin(zt))                                     # the most damaged parameter of the benchmark
        rows.append({"setup": name, "benchmark": f"P{pos}", "mean_std": float(s.std(0).mean()),
                     "widest_95": float(np.max(hi - lo)), "widest_param": C.PARAM_NAMES[int(np.argmax(hi - lo))],
                     "damaged_param": C.PARAM_NAMES[dam], "P_true_damaged": float(np.mean(s[:, dam] < THRESHOLD)),
                     "P_mirror_damaged": float(np.mean(s[:, MIRROR[dam]] < THRESHOLD)),
                     "zstar_in_95": int(np.sum((zt >= lo) & (zt <= hi))),
                     **{f"std_{p}": float(s[:, j].std()) for j, p in enumerate(C.PARAM_NAMES)}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exact_posteriors_per_benchmark.csv"), index=False, float_format="%.4f")
    summ = []
    for name in SETUPS:
        d = df[df.setup == name].set_index("benchmark")
        summ.append({"setup": name, "mean_std_range": f"{d.mean_std.min():.3f}-{d.mean_std.max():.3f}",
                     "widest_95_range": f"{d.widest_95.min():.2f}-{d.widest_95.max():.2f}",
                     **{f"{b}: true/mirror": f"{d.loc[b, 'P_true_damaged']:.3f} / {d.loc[b, 'P_mirror_damaged']:.3f}"
                        for b in ("P1", "P4", "P9")},
                     "std_zBL_P4": f"{d.loc['P4', 'std_zBL']:.3f}", "zstar_in_95": int(d.zstar_in_95.sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
