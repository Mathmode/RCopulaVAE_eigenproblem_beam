#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/metrics_by_case.py

R1.4: the performance metrics of the paper (R1.3) for both approaches, per test case and by groups of test cases:
  * by the error of the surrogate at the true damage state (mean |frequency error| with respect to the eigenvalue solver,
    nominal 10-element model of the decoder): is the surrogate approach worse where its decoder is less accurate?
  * by the number of damaged parameters (z* < 0.95);
  * paired: share of test cases in which each approach has the smaller absolute error of the posterior mean.
The overall values reproduce the comparison table (Table R3).

    python MODULES/comparison_surrogate_decoder/metrics_by_case.py
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

OUT = os.path.join(PROJECT_ROOT, "Output", "comparison_surrogate_decoder")


def per_case(fam, z, lb):
    """Per test case: mean over the D parameters of |error|, squared error, 95 % coverage and log density at z*."""
    u = np.clip((z - lb) / (1.0 - lb), 1e-6, 1.0 - 1e-6)
    mean_u, _ = fam.moments()
    mean = lb + (1.0 - lb) * mean_u
    pit = fam.cdf(u)
    logq = np.log(fam.pdf(u) + 1e-12) - np.log(1.0 - lb)          # physical units, as ALL in R1.3
    inside = np.abs(2.0 * pit - 1.0) <= 0.95
    return {"abs_err": np.abs(mean - z), "sq_err": (mean - z) ** 2, "inside95": inside, "logq": logq}


def summarise(m, mask):
    return {"MSE": m["sq_err"][mask].mean(), "MAE": m["abs_err"][mask].mean(), "ALL": m["logq"][mask].mean(),
            "C95": m["inside95"][mask].mean()}


def main():
    from MODULES.review29oct_fixed import config as C
    from MODULES.review29oct_fixed.posterior_families import family_from_preds
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge

    p = argparse.ArgumentParser(description="R1.4: metrics of both approaches by groups of test cases.")
    p.add_argument("--eigen-run", default=os.path.join(PROJECT_ROOT, "Output", "baseline_GaussCopula_GaussMarginals",
                                                       "GCTN_06Nov-5modes_7D_Gamma1.0_Conv6000"))
    p.add_argument("--surrogate-run", default=os.path.join(OUT, "VAE_surrogate"))
    p.add_argument("--surrogate-dir", default=os.path.join(OUT, "surrogate"))
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--split", default="test_fine", help="test split (test_fine = the test dataset of the revision)")
    p.add_argument("--out-dir", default=os.path.join(OUT, "metrics_by_case"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    pcfg = load_problem_config(a.data_dir)
    lb, thr = float(pcfg["lbound"]), float(pcfg.get("damage_threshold", 0.95))
    z = np.load(os.path.join(a.data_dir, f"alphas_{a.split}.npy")).astype(np.float64)
    suffix = "" if a.split == "test" else f"_{a.split}"
    m = {}
    for key, run in (("eigen", a.eigen_run), ("surrogate", a.surrogate_run)):
        preds = np.load(os.path.join(run, f"Test_predicted_props{suffix}.npy"), allow_pickle=True).item()
        m[key] = per_case(family_from_preds(preds, lb), z, lb)

    # error of the surrogate decoder at the true damage state (eigenvalue solver = reference)
    f_ex = BridgeModel(pcfg).solve(z)[0]
    f_su = SurrogateBridge(pcfg, a.surrogate_dir).solve(z)[0]
    sur_err = 100.0 * np.mean(np.abs(f_su / f_ex - 1.0), axis=1)          # (N,) mean over the modes, %
    sur_err_max = 100.0 * np.max(np.abs(f_su / f_ex - 1.0), axis=1)
    n_dam = np.sum(z < thr, axis=1)

    rows = []
    allD = np.ones_like(z, bool)

    def add(group, sel):
        mask = allD & sel[:, None]
        r = {"group": group, "n_cases": int(sel.sum())}
        for key in ("eigen", "surrogate"):
            r.update({f"{k}_{key}": v for k, v in summarise(m[key], mask).items()})
        ae = {k: m[k]["abs_err"].mean(axis=1) for k in m}
        r["eigen_smaller_abs_err_share"] = float(np.mean(ae["eigen"][sel] < ae["surrogate"][sel]))
        rows.append(r)

    add("all test cases", np.ones(len(z), bool))
    q = np.quantile(sur_err, [0.5, 0.9, 0.95, 0.99])
    add(f"surrogate error at z* below the median ({q[0]:.2f} %)", sur_err <= q[0])
    add(f"surrogate error at z* in the top 10 % (> {q[1]:.2f} %)", sur_err > q[1])
    add(f"surrogate error at z* in the top 5 % (> {q[2]:.2f} %)", sur_err > q[2])
    add(f"surrogate error at z* in the top 1 % (> {q[3]:.2f} %)", sur_err > q[3])
    for k, sel in (("0 damaged", n_dam == 0), ("1 damaged", n_dam == 1), ("2 damaged", n_dam == 2),
                   ("3 damaged", n_dam == 3), ("4 to 7 damaged", n_dam >= 4)):
        add(k, sel)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "metrics_by_case.csv"), index=False, float_format="%.5g")
    pd.DataFrame({"sur_err_mean_pct": sur_err, "sur_err_max_pct": sur_err_max, "n_damaged": n_dam,
                  **{f"MAE_{k}": m[k]["abs_err"].mean(axis=1) for k in m},
                  **{f"C95_{k}": m[k]["inside95"].mean(axis=1) for k in m}}).to_csv(
        os.path.join(a.out_dir, "per_case.csv"), index=False, float_format="%.5g")

    pd.set_option("display.width", 250)
    print(f"surrogate error at the true states (mean over the modes): median {np.median(sur_err):.3f} %, "
          f"max {sur_err.max():.3f} %; largest single-mode error {sur_err_max.max():.3f} %")
    cols = ["group", "n_cases", "MAE_eigen", "MAE_surrogate", "C95_eigen", "C95_surrogate", "ALL_eigen", "ALL_surrogate",
            "eigen_smaller_abs_err_share"]
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
