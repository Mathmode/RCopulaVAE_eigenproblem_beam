#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/decoder_checks.py

R1.4: the forward operator of both approaches, compared directly (no VAE involved). The eigenvalue solver of the decoder
(fe_model.BridgeModel, nominal model) is exact by construction; the surrogate of Ref. 41 approximates it. This script
measures the error of the surrogate with respect to the eigenvalue solver

  1. inside its training domain: 20 000 damage states drawn uniformly over the prior box [lbound, 1]^7 (reference);
  2. outside its training domain:
       - deck damage beyond the lower bound: the true states of the shifted test set 1 of ood_study.py (damaged deck
         zones with z in [0.30, 0.45)) and 5 000 states with one deck zone in [0.20, lbound) and the rest undamaged;
       - stiffer bearings: the true states of the shifted test set 6 (bearings three times stiffer than nominal) and
         5 000 states with both bearing factors in (1, 3] and the deck zones in the prior box;
  3. in its sensitivities: Jacobian d(log f)/dz at 2 000 states of the prior box (central differences for both).

It also counts the states whose exact frequencies fall outside the output range of the surrogate (its sigmoid output is
scaled to the range of the training targets, so it cannot return a frequency outside that range).

Outputs (--out-dir): decoder_checks.csv (per set and mode), jacobian_check.csv, decoder_checks_summary.txt.

    python MODULES/comparison_surrogate_decoder/decoder_checks.py
"""
import argparse
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

from MODULES.review29oct_fixed import config as C

OUT = os.path.join(C.PROJECT_ROOT, "Output", "comparison_surrogate_decoder")
parser = argparse.ArgumentParser(description="R1.4: surrogate decoder vs eigenvalue solver, inside and outside the "
                                             "training domain, and their sensitivities.")
parser.add_argument("--data-dir", default=C.DATA_DIR)
parser.add_argument("--surrogate-dir", default=os.path.join(OUT, "surrogate_decoder_06Nov-FINAL5modes_targets-nominal_H10-30-50-70-80"))
parser.add_argument("--ood-dir", default=None, help="shifted test sets of ood_study.py (default: <data dir>_OOD)")
parser.add_argument("--out-dir", default=os.path.join(OUT, "decoder_checks_06Nov-FINAL5modes_H10-30-50-70-80"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-jac", type=int, default=2000)
args = None


def main():
    global args
    args = parser.parse_args()
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(C.PROJECT_ROOT, args.data_dir)
    ood_dir = args.ood_dir or data_dir.rstrip("\\/") + "_OOD"
    pcfg = load_problem_config(data_dir)
    lb, M = float(pcfg["lbound"]), int(pcfg["n_modes"])
    bridge, sur = BridgeModel(pcfg), SurrogateBridge(pcfg, args.surrogate_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    nz, D = C.N_ZONES, 7

    # ---------------------------------------------------------------- damage states
    sets = {"inside: prior box [lb, 1]^7": rng.uniform(lb, 1.0, size=(20000, D))}
    f1 = os.path.join(ood_dir, "alphas_OOD1_severity.npy")
    if os.path.exists(f1):
        sets["outside: shifted set 1 (deck zones in [0.30, 0.45))"] = np.load(f1).astype(np.float64)
    z = rng.uniform(0.95, 1.0, size=(5000, D))
    z[np.arange(5000), rng.integers(0, nz, 5000)] = rng.uniform(0.20, lb, 5000)
    sets["outside: one deck zone in [0.20, lb), rest undamaged"] = z
    f6 = os.path.join(ood_dir, "alphas_OOD6_stiffbearings.npy")
    if os.path.exists(f6):
        sets["outside: shifted set 6 (bearings x3)"] = np.load(f6).astype(np.float64)
    z = rng.uniform(lb, 1.0, size=(5000, D))
    z[:, nz:] = rng.uniform(1.0, 3.0, size=(5000, D - nz))
    sets["outside: bearings in (1, 3], deck in the prior box"] = z

    # output range of the surrogate for log f (scaler of its training targets)
    lo, hi = np.asarray(sur.scaler.lo, np.float64)[:M], np.asarray(sur.scaler.hi, np.float64)[:M]

    rows, summary = [], []
    for name, zz in sets.items():
        f_ex, v_ex, _ = bridge.solve(zz)
        f_su, v_su, _ = sur.solve(zz)
        rel = 100.0 * np.abs(f_su / f_ex - 1.0)
        mac = np.sum(v_ex * v_su, -1) ** 2 / (np.sum(v_ex ** 2, -1) * np.sum(v_su ** 2, -1))
        out_range = np.any((np.log(f_ex) < lo) | (np.log(f_ex) > hi), axis=1)
        for j in range(M):
            rows.append({"set": name, "n": len(zz), "mode": j + 1, "freq_err_mean_pct": rel[:, j].mean(),
                         "freq_err_p95_pct": np.percentile(rel[:, j], 95), "freq_err_max_pct": rel[:, j].max(),
                         "one_minus_MAC_mean": float(np.mean(1.0 - mac[:, j]))})
        summary.append(f"{name:55s} n={len(zz):6d} | frequency error: mean {rel.mean():6.2f} %, p95 "
                       f"{np.percentile(rel, 95):6.2f} %, max {rel.max():6.2f} % | states with an error > 2.5 % (noise): "
                       f"{100 * np.mean(rel.max(1) > 2.5):5.1f} % | exact frequencies outside the output range of the "
                       f"surrogate: {100 * out_range.mean():5.1f} % | mean 1-MAC {np.mean(1 - mac):.2e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "decoder_checks.csv"), index=False, float_format="%.5g")

    # ---------------------------------------------------------------- sensitivities d(log f)/dz in the prior box
    z0 = rng.uniform(lb + 1e-3, 1.0 - 1e-3, size=(args.n_jac, D))
    h = 1e-4

    def jac(model):
        J = np.zeros((len(z0), M, D))
        for k in range(D):
            zp, zm = z0.copy(), z0.copy()
            zp[:, k] += h
            zm[:, k] -= h
            J[:, :, k] = (np.log(model.solve(zp)[0]) - np.log(model.solve(zm)[0])) / (2 * h)
        return J

    J_ex, J_su = jac(bridge), jac(sur)
    err = np.linalg.norm(J_su - J_ex, axis=-1) / np.linalg.norm(J_ex, axis=-1)          # (N, M) relative error
    cos = np.sum(J_su * J_ex, -1) / (np.linalg.norm(J_su, axis=-1) * np.linalg.norm(J_ex, axis=-1))
    jrows = [{"mode": j + 1, "rel_err_median_pct": 100 * np.median(err[:, j]),
              "rel_err_p95_pct": 100 * np.percentile(err[:, j], 95), "rel_err_max_pct": 100 * err[:, j].max(),
              "angle_median_deg": float(np.degrees(np.median(np.arccos(np.clip(cos[:, j], -1, 1)))))} for j in range(M)]
    pd.DataFrame(jrows).to_csv(os.path.join(args.out_dir, "jacobian_check.csv"), index=False, float_format="%.4g")
    summary.append(f"\nSensitivities d(log f)/dz at {len(z0)} states of the prior box (relative error of the gradient of "
                   f"each frequency): median {100 * np.median(err):.1f} %, p95 {100 * np.percentile(err, 95):.1f} %, "
                   f"max {100 * err.max():.1f} %")

    text = "\n".join([f"Surrogate: {args.surrogate_dir}", f"Data: {data_dir}", ""] + summary)
    with open(os.path.join(args.out_dir, "decoder_checks_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print("\nPer mode:\n" + df.round(3).to_string(index=False))
    print("\nJacobian per mode:\n" + pd.DataFrame(jrows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
