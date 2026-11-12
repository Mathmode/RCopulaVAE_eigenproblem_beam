#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/forward_operator_table.py

R1.4: one table, with the same definitions in every row, of the difference between the natural frequencies computed by the
surrogate decoder (architecture of Ref. 41) and by the eigenvalue solver of the proposed method (exact for the FE model):
  * inside the training domain (20 000 states drawn uniformly over [lbound, 1]^7), for surrogates trained with
    14 000 (the comparison), 5 000, 2 000, 1 000 and 500 simulations (surrogate_data_budget.py);
  * outside the training domain (surrogate of 14 000 simulations): one deck zone in [0.20, lbound) with the rest undamaged,
    and bearing factors in (1, 3] with the deck zones in the prior box (5 000 states each, as in decoder_checks.py).
Columns: mean |error| (all modes and states), 95th percentile, maximum, and share of damage states in which the error of
at least one frequency exceeds the identification noise (2.5 %).

    python MODULES/comparison_surrogate_decoder/forward_operator_table.py
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

OUT = os.path.join(PROJECT_ROOT, "Output", "comparison_surrogate_decoder")
FULL = os.path.join(OUT, "surrogate_decoder_06Nov-FINAL5modes_targets-nominal_H10-30-50-70-80")


def main():
    from MODULES.review29oct_fixed import config as C
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge

    pcfg = load_problem_config(C.DATA_DIR)
    lb, noise = float(pcfg["lbound"]), 100.0 * C.FREQ_NOISE
    bridge = BridgeModel(pcfg)
    rng = np.random.default_rng(0)
    D, nz = 7, C.N_ZONES
    box = rng.uniform(lb, 1.0, size=(20000, D))
    deck = rng.uniform(0.95, 1.0, size=(5000, D))
    deck[np.arange(5000), rng.integers(0, nz, 5000)] = rng.uniform(0.20, lb, 5000)
    bear = rng.uniform(lb, 1.0, size=(5000, D))
    bear[:, nz:] = rng.uniform(1.0, 3.0, size=(5000, D - nz))

    cases = [("inside the training domain", "14,000", FULL, box)]
    for n in (5000, 2000, 1000, 500):
        folder = os.path.join(OUT, "data_budget", f"surrogate_ntrain{n}")
        if os.path.exists(os.path.join(folder, "surrogate_config.json")):
            cases.append(("inside the training domain", f"{n:,}", folder, box))
    cases += [("one deck zone with z_d in [0.20, 0.45)", "14,000", FULL, deck),
              ("bearings with z_b in (1, 3]", "14,000", FULL, bear)]

    f_ref = {}
    rows = []
    for states, n_sim, folder, z in cases:
        key = id(z)
        if key not in f_ref:
            f_ref[key] = bridge.solve(z)[0]
        err = 100.0 * np.abs(SurrogateBridge(pcfg, folder).solve(z)[0] / f_ref[key] - 1.0)       # (N, modes), %
        rows.append({"damage states": states, "training simulations": n_sim, "mean_pct": err.mean(),
                     "p95_pct": np.percentile(err, 95), "max_pct": err.max(),
                     "states_above_noise_pct": 100.0 * np.mean(err.max(axis=1) > noise)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "decoder_checks_06Nov-FINAL5modes_H10-30-50-70-80"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "decoder_checks_06Nov-FINAL5modes_H10-30-50-70-80", "forward_operator_table.csv"),
              index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
