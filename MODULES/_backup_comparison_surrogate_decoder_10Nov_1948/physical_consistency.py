#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/physical_consistency.py

R1.4: physical laws that the eigenvalue solver satisfies by construction and a surrogate decoder does not have to satisfy:

  1. Monotonicity (Rayleigh): K(z) = sum_d z_d K_d + ... with every K_d positive semi-definite, so
     d(lambda_i)/d(z_d) = phi_i^T K_d phi_i >= 0: more damage (lower z_d) can never raise a natural frequency. Test: for
     damage states z and a stiffness increment dz = 0.05 of one parameter (z_d + dz <= 1, inside the training domain),
     count the cases in which the decoder predicts a LOWER frequency for the stiffer structure.
  2. Symmetry: the nominal bridge is symmetric about mid-span (zones, bearings, sensors), so a damage state and its mirror
     image (z1..z5 reversed, left and right bearings swapped) have identical frequencies (and mirrored mode shapes).
  3. Order of the frequencies: f_1 < f_2 < ... < f_M.

Decoders: eigenvalue solver (fe_model.BridgeModel, nominal model) and the surrogates of the comparison (14 000 training
simulations) and of the data-budget study (fewer simulations), if present. 20 000 damage states drawn uniformly over the
training domain [lbound, 1]^7. Outputs: physical_consistency.csv and, for the figure, monotonicity_sweep.csv.

    python MODULES/comparison_surrogate_decoder/physical_consistency.py
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
MIRROR = [4, 3, 2, 1, 0, 6, 5]           # z1..z5 reversed, bearings swapped


def checks(name, solve, z, dz=0.05):
    f0, v0, _ = solve(z)
    N, M = f0.shape
    D = z.shape[1]
    # 1. monotonicity: stiffer parameter -> no frequency may decrease
    viol, drop, n_pairs = 0, [], 0
    for d in range(D):
        ok = z[:, d] + dz <= 1.0
        zp = z[ok].copy()
        zp[:, d] += dz
        df = 100.0 * (solve(zp)[0] / f0[ok] - 1.0)          # % change of every frequency, (n, M)
        bad = df < 0.0
        viol += int(bad.sum())
        n_pairs += bad.size
        drop += list(-df[bad])
    states_with_violation = None
    # 2. symmetry
    fm, vm, _ = solve(z[:, MIRROR])
    sym = 100.0 * np.abs(fm / f0 - 1.0)
    v_ref = v0[..., ::-1]                                    # mirrored mode shapes: sensor order reversed
    mac = np.sum(vm * v_ref, -1) ** 2 / (np.sum(vm ** 2, -1) * np.sum(v_ref ** 2, -1))
    # 3. order of the frequencies
    order_bad = np.mean(np.any(np.diff(f0, axis=1) <= 0.0, axis=1))
    return {"decoder": name,
            "monotonicity_violations_pct": 100.0 * viol / n_pairs,
            "violation_size_median_pct": float(np.median(drop)) if drop else 0.0,
            "violation_size_max_pct": float(np.max(drop)) if drop else 0.0,
            "mirror_freq_diff_mean_pct": sym.mean(), "mirror_freq_diff_p95_pct": np.percentile(sym, 95),
            "mirror_freq_diff_max_pct": sym.max(), "mirror_states_diff_above_0.5pct_pct": 100.0 * np.mean(sym.max(1) > 0.5),
            "mirror_shape_1_minus_MAC_mean": float(np.mean(1.0 - mac)),
            "frequency_order_violations_pct": 100.0 * order_bad}


def main():
    from MODULES.review29oct_fixed import config as C
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge

    pcfg = load_problem_config(C.DATA_DIR)
    lb = float(pcfg["lbound"])
    z = np.random.default_rng(0).uniform(lb, 1.0, size=(20000, 7))
    decoders = [("eigenvalue solver", BridgeModel(pcfg).solve), ("surrogate, 14,000 simulations", SurrogateBridge(pcfg, FULL).solve)]
    for n in (2000, 500):
        folder = os.path.join(OUT, "data_budget", f"surrogate_ntrain{n}")
        if os.path.exists(os.path.join(folder, "surrogate_config.json")):
            decoders.append((f"surrogate, {n:,} simulations", SurrogateBridge(pcfg, folder).solve))
    rows = [checks(name, solve, z) for name, solve in decoders]
    df = pd.DataFrame(rows)
    out = os.path.join(OUT, "decoder_checks_06Nov-FINAL5modes_H10-30-50-70-80")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "physical_consistency.csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(df.round(4).T.to_string(header=False))

    # sweep for a figure: every frequency vs one stiffness factor, the others fixed (state with the most violations)
    eig, sur = decoders[0][1], decoders[1][1]
    best, best_count = None, -1
    rng = np.random.default_rng(1)
    for _ in range(300):
        z0 = rng.uniform(lb, 1.0, size=7)
        for d in range(7):
            g = np.repeat(z0[None], 111, 0)
            g[:, d] = np.linspace(lb, 1.0, 111)
            fs = sur(g)[0]
            cnt = int(np.sum(np.diff(fs, axis=0) < 0))
            if cnt > best_count:
                best, best_count = (z0.copy(), d), cnt
    z0, d = best
    g = np.repeat(z0[None], 111, 0)
    g[:, d] = np.linspace(lb, 1.0, 111)
    fe, fs = eig(g)[0], sur(g)[0]
    sweep = pd.DataFrame({"z_d": g[:, d], **{f"f{i + 1}_eigen": fe[:, i] for i in range(fe.shape[1])},
                          **{f"f{i + 1}_surrogate": fs[:, i] for i in range(fs.shape[1])}})
    sweep.attrs["param"] = d
    sweep.to_csv(os.path.join(out, "monotonicity_sweep.csv"), index=False, float_format="%.6g")
    print(f"\nsweep: parameter {C.PARAM_NAMES[d]} from {lb} to 1, other parameters {np.round(z0, 3).tolist()}; "
          f"decreasing steps of the surrogate frequencies: {best_count} of {110 * fs.shape[1]}")


if __name__ == "__main__":
    main()
