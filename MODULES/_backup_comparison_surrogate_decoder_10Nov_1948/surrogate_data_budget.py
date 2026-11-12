#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/surrogate_data_budget.py

R1.4, data budget: how accurate is the surrogate decoder of Ref. 41 when fewer FE simulations are available to train it?
The eigenvalue solver needs no training data, so its accuracy does not depend on this budget.

For every size N, a copy of the final data set is made with a random subset of N training samples (same validation and
test sets, same calibrated likelihood: only the training split changes), the surrogate is trained on it exactly as in the
comparison (architecture of Ref. 41, noise-free nominal targets, same convergence rule), and its error with respect to the
eigenvalue solver is collected (20 000 states drawn uniformly over the prior box, and the damage states of the test set).
The copies are also the data sets of the VAE-level study (train both approaches on the same reduced training split).

    python MODULES/comparison_surrogate_decoder/surrogate_data_budget.py --sizes 500 1000 2000 5000
    python MODULES/comparison_surrogate_decoder/surrogate_data_budget.py --sizes 1000 2000 --make-data-only   # server

Outputs: <out-dir>/data_ntrain<N>/ (reduced data sets), <out-dir>/surrogate_ntrain<N>/ (surrogates) and
<out-dir>/data_budget_summary.csv.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
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

OUT = os.path.join(PROJECT_ROOT, "Output", "comparison_surrogate_decoder")
FULL_SURROGATE = os.path.join(OUT, "surrogate_decoder_06Nov-FINAL5modes_targets-nominal_H10-30-50-70-80")


def make_subset(data_dir, n, dst, seed=2026):
    """Copy of the data set (top-level files) with a random subset of n training samples."""
    if os.path.exists(os.path.join(dst, "problem_config.json")):
        return dst
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(data_dir, "*")):
        if os.path.isfile(f):
            shutil.copy2(f, dst)
    n_full = len(np.load(os.path.join(data_dir, "alphas_train.npy")))
    idx = np.sort(np.random.default_rng(seed).permutation(n_full)[:n])
    for f in glob.glob(os.path.join(data_dir, "*_train.npy")):
        np.save(os.path.join(dst, os.path.basename(f)), np.load(f)[idx])
    with open(os.path.join(dst, "data_budget.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": os.path.abspath(data_dir), "n_train": int(n), "seed": seed,
                   "note": "random subset of the training split; validation and test splits unchanged"}, fh, indent=1)
    return dst


def box_errors(folder):
    acc = pd.read_csv(os.path.join(folder, "surrogate_accuracy.csv"))
    out = {}
    for name, key in (("uniform prior box", "box"), ("test_fine", "test")):
        a = acc[acc["set"] == name]
        if len(a):
            out[f"{key}_mean_err_pct"] = a["freq_err_mean_abs_pct"].mean()
            out[f"{key}_p95_err_pct"] = a["freq_err_p95_abs_pct"].max()
            out[f"{key}_max_err_pct"] = a["freq_err_max_abs_pct"].max()
    cfg = json.load(open(os.path.join(folder, "surrogate_config.json")))
    out["epochs"] = cfg.get("epochs_run", cfg.get("epochs"))
    return out


def main():
    from MODULES.review29oct_fixed import config as C
    p = argparse.ArgumentParser(description="R1.4: accuracy of the surrogate decoder vs the number of training simulations.")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    p.add_argument("--out-dir", default=os.path.join(OUT, "data_budget"))
    p.add_argument("--threads", type=int, default=None, help="CPU threads per surrogate (default: CPUs / number of sizes)")
    p.add_argument("--gpu", default=None)
    p.add_argument("--make-data-only", action="store_true", help="only write the reduced data sets")
    a = p.parse_args()
    data_dir = a.data_dir if os.path.isabs(a.data_dir) else os.path.join(PROJECT_ROOT, a.data_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    dirs = {n: make_subset(data_dir, n, os.path.join(a.out_dir, f"data_ntrain{n}")) for n in a.sizes}
    for n, d in dirs.items():
        print(f"data set with {n} training samples: {d}", flush=True)
    if a.make_data_only:
        return

    threads = a.threads or max(1, (os.cpu_count() or 4) // len(a.sizes))
    procs = {}
    for n, d in dirs.items():
        sdir = os.path.join(a.out_dir, f"surrogate_ntrain{n}")
        if os.path.exists(os.path.join(sdir, "surrogate_accuracy.csv")):
            continue
        cmd = [sys.executable, os.path.join(THIS_DIR, "train_surrogate_decoder.py"), "--data-dir", d, "--targets", "nominal",
               "--hidden", "10", "30", "50", "70", "80", "--seed", "1234", "--out-dir", sdir, "--threads", str(threads)]
        if a.gpu is not None:
            cmd += ["--gpu", str(a.gpu)]
        log = open(os.path.join(a.out_dir, f"surrogate_ntrain{n}.log"), "w")
        procs[n] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT), log)
        print(f"training the surrogate with {n} samples ({threads} threads)", flush=True)
    for n, (pr, log) in procs.items():
        pr.wait()
        log.close()
        print(f"  N = {n}: exit code {pr.returncode}", flush=True)

    rows = []
    for n in sorted(a.sizes) + [None]:
        folder = FULL_SURROGATE if n is None else os.path.join(a.out_dir, f"surrogate_ntrain{n}")
        if os.path.exists(os.path.join(folder, "surrogate_accuracy.csv")):
            n_tr = n or len(np.load(os.path.join(data_dir, "alphas_train.npy")))
            rows.append({"n_train": n_tr, **box_errors(folder)})
    df = pd.DataFrame(rows).sort_values("n_train")
    df.to_csv(os.path.join(a.out_dir, "data_budget_summary.csv"), index=False, float_format="%.4g")
    print("\nError of the surrogate with respect to the eigenvalue solver (frequencies, %; worst mode for p95/max):")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
