#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/shape_error_structure.py

Is the mode-shape error of the as-built data isotropic across sensors, as the likelihood assumes? (Results_analysis,
Sec. 4.3, Table 4.) The shape term (1 - MAC) / (2 sig_v^2) is an isotropic Gaussian on the residual orthogonal to the
predicted shape, with one variance per mode. On the training structures (as-built + noise, the generator of test
set A) the script reports
  1. the tails of the shape terms at z* against chi2_{n_s-1} / (n_s - 1) (isotropic Gaussian errors), and the same for
     noise-only simulations of the decoder, which must follow that distribution,
  2. for the healthy structures, per mode: the share of the largest eigenvalue of the residual covariance (isotropic:
     1/4), the residual variance along the direction in which every parameter changes the shape divided by sig_v^2
     (1: the likelihood weighs that evidence correctly; > 1: it over-weights it), and the share of the mode-shape
     information about every parameter carried by each mode.
No training, a few seconds:
    python MODULES/review29oct_fixed/shape_error_structure.py
Results: Output/review29oct_fixed/shape_error_structure_<M>modes/shape_error_structure.csv
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
from scipy.stats import chi2

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BridgeModel, mac, add_measurement_noise
from MODULES.review29oct_fixed.reference_posterior import load_problem_config
from MODULES.review29oct_fixed.prior_scenarios_coverage import data_terms_at_truth


def perp(v, ref):
    """Component of the unit shapes v orthogonal to the unit shapes ref (last axis)."""
    return v - np.sum(v * ref, axis=-1, keepdims=True) * ref


def main(data_dir=C.DATA_DIR, eps=1e-4):
    pcfg = load_problem_config(data_dir)
    lik = pcfg["likelihood"]
    model = BridgeModel(pcfg)
    sig_v, noise_v = np.asarray(lik["sig_v"]), np.asarray(lik["parts"]["noise_v"])
    z = np.load(os.path.join(data_dir, "alphas_train.npy")).astype(np.float64)
    labels = np.load(os.path.join(data_dir, "labels_train.npy"))
    v_obs = np.load(os.path.join(data_dir, "vertmodes_train.npy")).astype(np.float64)
    f_p, v_p, r_p = model.solve(z)
    _, v_n, _ = add_measurement_noise(f_p, v_p, r_p, np.random.default_rng(0), C.FREQ_NOISE, C.MODE_NOISE, ref=model.ref)
    n, n_m, n_s = v_obs.shape

    # --- 1. tails of the data terms at z* ---
    q95, q99 = chi2.ppf([0.95, 0.99], n_s - 1) / (n_s - 1)
    r_obs = (1 - mac(v_obs, v_p)) / (2 * sig_v ** 2) / ((n_s - 1) / 2)
    r_noise = (1 - mac(v_n, v_p)) / (2 * noise_v ** 2) / ((n_s - 1) / 2)
    ratio_f, _, _ = data_terms_at_truth(pcfg, data_dir, "train", np.arange(n), z)
    f95, f99 = chi2.ppf([0.95, 0.99], n_m) / n_m
    print(f"1) data terms at z* / expected, {n} training structures: mean | P(> 95 % quantile) | P(> 99 % quantile)")
    for m in range(n_m):
        print(f"   shape mode {m + 1}: as-built + noise {r_obs[:, m].mean():.2f} | {np.mean(r_obs[:, m] > q95):.3f} | "
              f"{np.mean(r_obs[:, m] > q99):.3f}    noise only {r_noise[:, m].mean():.2f} | "
              f"{np.mean(r_noise[:, m] > q95):.3f} | {np.mean(r_noise[:, m] > q99):.3f}")
    print(f"   frequencies:  {ratio_f.mean():.2f} | {np.mean(ratio_f > f95):.3f} | {np.mean(ratio_f > f99):.3f}"
          f"    (isotropic Gaussian: 1 | 0.05 | 0.01)")

    # --- 2. structure of the residual near the healthy state ---
    h = labels == list(C.SCENARIO_FRACTIONS).index("healthy")
    R_obs, R_noise = perp(v_obs[h], v_p[h]), perp(v_n[h], v_p[h])
    z0 = np.full((1, C.N_DIMS), 0.975)                     # middle of the healthy range
    v0 = model.solve(z0)[1][0]
    J = np.zeros((C.N_DIMS, n_m, n_s))
    for d in range(C.N_DIMS):
        zp, zm = z0.copy(), z0.copy()
        zp[0, d] += eps
        zm[0, d] -= eps
        J[d] = (model.solve(zp)[1][0] - model.solve(zm)[1][0]) / (2 * eps)
    J = perp(J, v0[None])                                  # shape change orthogonal to the shape itself
    info = np.sum(J ** 2, axis=-1) / sig_v[None] ** 2      # (7, n_m) mode-shape information per parameter and mode
    info = info / info.sum(axis=1, keepdims=True)

    rows = []
    for m in range(n_m):
        S_obs = R_obs[:, m].T @ R_obs[:, m] / h.sum()
        S_noise = R_noise[:, m].T @ R_noise[:, m] / h.sum()
        e_obs = np.sort(np.linalg.eigvalsh(S_obs))[::-1][:n_s - 1]
        e_noise = np.sort(np.linalg.eigvalsh(S_noise))[::-1][:n_s - 1]
        row = {"mode": m + 1, "largest_eigenvalue_share_asbuilt": e_obs[0] / e_obs.sum(),
               "largest_eigenvalue_share_noise_only": e_noise[0] / e_noise.sum(),
               "tail_above_chi2_99_asbuilt": np.mean(r_obs[:, m] > q99),
               "tail_above_chi2_99_noise_only": np.mean(r_noise[:, m] > q99)}
        for d, p in enumerate(C.PARAM_NAMES):
            u = J[d, m] / (np.linalg.norm(J[d, m]) + 1e-30)
            row[f"variance_ratio_{p}"] = u @ S_obs @ u / sig_v[m] ** 2
        for d, p in enumerate(C.PARAM_NAMES):
            row[f"info_share_{p}"] = info[d, m]
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = os.path.join(C.OUTPUT_DIR, f"shape_error_structure_{n_m}modes")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "shape_error_structure.csv"), index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    print(f"\n2) healthy structures ({h.sum()}): largest-eigenvalue share of the shape residual (isotropic 0.25) and "
          f"variance along each parameter's shape direction / sig_v^2")
    cols = ["mode", "largest_eigenvalue_share_asbuilt", "largest_eigenvalue_share_noise_only"] + \
           [f"variance_ratio_{p}" for p in C.PARAM_NAMES]
    print(df[cols].round(2).to_string(index=False))
    print("\n   share of the mode-shape information about each parameter carried by each mode")
    print(df[["mode"] + [f"info_share_{p}" for p in C.PARAM_NAMES]].round(2).to_string(index=False))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
