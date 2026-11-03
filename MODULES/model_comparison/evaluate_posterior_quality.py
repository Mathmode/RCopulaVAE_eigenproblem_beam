#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/model_comparison/evaluate_posterior_quality.py

How close is the amortised posterior q(z | m) of a trained run to the EXACT posterior p(z | m), without reference
samples? Works for every model: proposed KS-Copula VAE (any K) and the baselines (GC-TN, Ref. 41; GMM, Ref. 39).
For every observation m of a split and S samples z_s ~ q(z | m):

    log w_s  = log p(m | z_s) + log p(z_s) - log q(z_s | m)    (likelihood of training and of the SMC reference)
    ELBO     = mean_s log w_s                                    -ELBO = KL(q || p(.|m)) - log p(m)
    IW-ELBO  = log mean_s w_s                                   ELBO <= IW-ELBO <= log p(m)
    KL_lower = IW-ELBO - ELBO                                   lower bound of KL(q || p(.|m)) [nats]
    k_hat    = Pareto-k of the importance weights w_s           (Yao et al. 2018, ICML; Vehtari et al. 2024, JMLR)
               < 0.5: q covers the exact posterior well; 0.5-0.7: acceptable; > 0.7: q misses posterior mass

log p(m) does not depend on q: on the SAME observations, the difference between the mean -ELBO of two models is the
difference between their mean KL divergences to the exact posterior (nats per observation). Model selection (e.g. the
number K of Kumaraswamy components) uses the VALIDATION set only (R3.2); the test sets are evaluated for reporting.

Outputs in <run>/Posterior_quality/:
    ELBO_<split>.csv                per observation: neg_elbo, iw_elbo, kl_lower, k_hat, E_q[NLL], entropy
    ELBO_summary.json               means, standard errors and k_hat fractions per split
    Validation_set/                 calibration metrics of the validation set (same code as the test sets)

    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder <run> [--samples 500]
"""

import os
import re
import sys
import json
import argparse

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
from MODULES.review29oct_fixed.posterior_quality import evaluate_split, summarize, psis_khat  # noqa: F401
from MODULES.review29oct_fixed.posterior_families import family_from_preds
from MODULES.review29oct_fixed.reference_posterior import load_problem_config

SPLIT_NAMES = {"val": "Validation set (model selection)", "test": "Test set A: same FE model as training (10 elements)",
               "test_fine": "Test set B: refined FE model (50 elements)"}
PRED_FILES = {"val": "Val_predicted_props.npy", "test": "Test_predicted_props.npy"}


# =============================================================================
# DATA, PREDICTIONS, EVALUATION (core: review29oct_fixed/posterior_quality.py)
# =============================================================================
def resolve_run(name):
    """Absolute path of a run given by path or by folder name (searched in Output/*/)."""
    if os.path.isdir(name):
        return os.path.abspath(name)
    out = os.path.join(C.PROJECT_ROOT, "Output")
    for group in sorted(os.listdir(out)):
        cand = os.path.join(out, group, name)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"run folder not found: {name}")


def load_info(run):
    """Problem_info.npy with the data set path made local (as postprocessing.load_run_info)."""
    info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
    if not os.path.isdir(info["data_dir"]):
        local = os.path.join(C.PROJECT_ROOT, "Data", re.split(r"[\\/]", str(info["data_dir"]).rstrip("\\/"))[-1])
        if not os.path.isdir(local):
            raise FileNotFoundError(f"data set of the run not found: {info['data_dir']} (nor {local})")
        info["data_dir"] = local
    info.setdefault("family", "GC-KS")
    info.setdefault("n_components", info.get("num_KS", 1))
    return info


def load_predictions(run, info, split):
    """Encoder outputs for a split. Runs trained before Val_predicted_props.npy existed (KS model): the trained model is
    rebuilt, applied to the validation set and the file is written."""
    path = os.path.join(run, PRED_FILES.get(split, f"Test_predicted_props_{split}.npy"))
    if not os.path.exists(path):
        if split != "val" or info["family"] != "GC-KS":
            raise FileNotFoundError(path)
        from MODULES.review29oct_fixed.localized_damage_study import load_trained_vae
        model = load_trained_vae(run, info["data_dir"], info)
        f, r, v, a = (np.load(os.path.join(info["data_dir"], f"{k}_val.npy")).astype(np.float32)
                      for k in ("freqs", "rotmodes", "vertmodes", "alphas"))
        f_s = ((np.log(f) - info["mean_f"]) / info["std_f"]).astype(np.float32)
        outs = model.Encoder_model.predict((f_s, r, v, a), batch_size=256, verbose=0)
        keys = ["test_a", "test_b", "test_weight_vals", "test_offdiag_elems", "test_diag_elems"]
        np.save(path, dict(zip(keys, outs)), allow_pickle=True)
        print(f"  wrote {os.path.basename(path)} with the trained model")
    return np.load(path, allow_pickle=True).item()


def load_observations(data_dir, split):
    f, v, r, z = (np.load(os.path.join(data_dir, f"{k}_{split}.npy")).astype(np.float64)
                  for k in ("freqs", "vertmodes", "rotmodes", "alphas"))
    labels = np.load(os.path.join(data_dir, f"labels_{split}.npy"))
    return f, v, r, z, labels


def evaluate_run(run, splits=("val", "test", "test_fine"), n_samples=500, seed=0, val_metrics=True):
    run = resolve_run(run)
    info = load_info(run)
    pcfg = load_problem_config(info["data_dir"])
    lb, gamma2 = float(info["lbound"]), float(info["gamma"]) ** 2
    out_dir = os.path.join(run, "Posterior_quality")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Posterior quality: {os.path.basename(run)} ({info['family']}, {info['n_components']} components), "
          f"S = {n_samples} ===")
    summary = {"run": os.path.basename(run), "family": info["family"], "n_components": int(info["n_components"]),
               "n_samples": n_samples, "definitions": __doc__.split("Outputs")[0].strip()}
    for split in splits:
        if split != "val" and split not in info.get("test_sets", ["test"]):
            continue
        fam = family_from_preds(load_predictions(run, info, split), lb)
        obs = load_observations(info["data_dir"], split)
        df = evaluate_split(fam, obs, pcfg, info["obs_types"], gamma2, n_samples, seed)
        df.to_csv(os.path.join(out_dir, f"ELBO_{split}.csv"), index=False, float_format="%.6g")
        summary[split] = s = summarize(df)
        print(f"  {SPLIT_NAMES.get(split, split):52s} -ELBO = {s['neg_elbo_mean']:.4f} +- {s['neg_elbo_se']:.4f}  "
              f"KL >= {s['kl_lower_mean']:.4f}  k_hat < 0.7: {100 * s['frac_k_hat_below_0.7']:.1f} %  "
              f"(median k_hat {s['k_hat_median']:.2f})")
        if split == "val" and val_metrics:
            from MODULES.review29oct_fixed.postprocessing import manuscript_metrics
            vdir = os.path.join(out_dir, "Validation_set")
            os.makedirs(vdir, exist_ok=True)
            res, _ = manuscript_metrics(fam, obs[3], lb, vdir, label=SPLIT_NAMES["val"], labels=obs[4])
            summary["val_calibration"] = {g: {k: res[g][k] for k in ("MSE", "MAE", "ALL", "MACE", "coverage95")}
                                          for g in ("all_7", "damaged_params", "healthy_params", "prior_scenarios")
                                          if g in res}
    with open(os.path.join(out_dir, "ELBO_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ELBO, importance-weighted ELBO and Pareto-k of a trained run.")
    parser.add_argument("--run-folder", nargs="+", required=True, help="run folder(s): path or name in Output/*/")
    parser.add_argument("--samples", type=int, default=500, help="samples of q per observation")
    parser.add_argument("--splits", nargs="+", default=["val", "test", "test_fine"])
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    for run in args.run_folder:
        evaluate_run(run, args.splits, args.samples, args.seed)


if __name__ == "__main__":
    main()
