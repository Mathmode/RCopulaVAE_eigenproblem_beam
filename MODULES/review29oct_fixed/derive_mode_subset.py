#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/derive_mode_subset.py

Data set with the first M identified modes of the FINAL data set (R3.4: fewer modes). Every observation keeps its
first M frequencies and mode shapes, so the structures, the as-built deviations and the noise realizations of the
remaining modes are exactly those of the M-mode version of the same measurements (the identification noise of a mode
does not depend on the other modes). The likelihood is the marginal of the calibrated Gaussian: bias, covariance and
standard deviations of the first M modes, precision = inverse of the covariance sub-block. Training, reference
posterior, post-processing and the studies read n_modes and the likelihood from problem_config.json, so they need no
other change. The subset is written next to the source, '_5modes' replaced by '_{M}modes', with the run tag
'<tag>{M}modes' (e.g. KSVAE_fixed_03Nov-FINAL3modes_7D_3KS_...).

    python MODULES/review29oct_fixed/derive_mode_subset.py --modes 3      (prints the folder of the subset)
"""

import os
import re
import sys
import copy
import json
import shutil
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from MODULES.review29oct_fixed import config as C


def subset_name(src_dir, n_modes):
    """Folder of the M-mode subset: '..._<N>modes' -> '..._<M>modes' (or '_<M>modes' appended)."""
    src_dir = src_dir.rstrip("\\/")
    base = os.path.basename(src_dir)
    new = re.sub(r"_\d+modes$", f"_{n_modes}modes", base) if re.search(r"_\d+modes$", base) else f"{base}_{n_modes}modes"
    return os.path.join(os.path.dirname(src_dir), new)


def restrict_problem_config(pcfg, n_modes, src_name):
    p = copy.deepcopy(pcfg)
    n = int(n_modes)
    lik = p["likelihood"]
    cov = np.asarray(lik["cov_logf"])[:n, :n]
    lik["cov_logf"] = cov.tolist()
    lik["prec_logf"] = np.linalg.inv(cov).tolist()            # marginal of a Gaussian: invert the covariance block
    lik["corr_logf"] = np.asarray(lik["corr_logf"])[:n, :n].tolist()
    for key in ("bias_logf", "sig_logf", "sig_v", "sig_r"):
        lik[key] = list(lik[key])[:n]
    lik["parts"] = {k: list(v)[:n] for k, v in lik.get("parts", {}).items()}
    p["n_modes"] = n
    p["ref_vert"], p["ref_rot"] = p["ref_vert"][:n], p["ref_rot"][:n]
    p["model_error_budget"] = {src: {k: list(v)[:n] for k, v in b.items()}
                               for src, b in p.get("model_error_budget", {}).items()}
    for split, cc in p.get("calibration_check", {}).items():
        cc["ratio_v"], cc["ratio_r"] = cc["ratio_v"][:n], cc["ratio_r"][:n]
        cc["ratio_f_total"] = None                             # defined for all modes together: not restricted
    tokens = str(p.get("dataset_version", "")).split("_")
    tokens[0] = re.sub(r"\d+modes$", "", tokens[0]) + f"{n}modes"
    p["dataset_version"] = re.sub(r"_\d+modes", f"_{n}modes", "_".join(tokens))
    p["derived_from"] = {"data_dir": src_name, "n_modes_source": int(pcfg["n_modes"]),
                         "how": "first n_modes of every observation; likelihood = marginal of the calibrated Gaussian"}
    return p


def derive_mode_subset(src_dir, n_modes, dst_dir=None, verbose=True):
    """Creates (once) the M-mode subset of src_dir and returns its folder; src_dir itself if M = its n_modes."""
    with open(os.path.join(src_dir, "problem_config.json"), "r", encoding="utf-8") as fh:
        pcfg = json.load(fh)
    n_src = int(pcfg["n_modes"])
    if int(n_modes) == n_src:
        return src_dir
    if not 1 <= int(n_modes) < n_src:
        raise ValueError(f"{n_modes} modes requested; the source data set has {n_src}")
    dst_dir = dst_dir or subset_name(src_dir, n_modes)
    cfg_path = os.path.join(dst_dir, "problem_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            done = json.load(fh)
        if done.get("derived_from", {}).get("data_dir") == os.path.basename(src_dir.rstrip("\\/")) \
                and int(done["n_modes"]) == int(n_modes):
            return dst_dir
        raise FileExistsError(f"{dst_dir} exists and is not the {n_modes}-mode subset of {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    n = int(n_modes)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if name.startswith("freqs_") and name.endswith(".npy"):
            np.save(dst, np.load(src)[:, :n])
        elif name.startswith(("vertmodes_", "rotmodes_")) and name.endswith(".npy"):
            np.save(dst, np.load(src)[:, :n, :])
        elif os.path.isdir(src):
            if name == "asbuilt_fields":                      # the same structures: copied
                shutil.copytree(src, dst, dirs_exist_ok=True)
            # reference_posteriors (other likelihood) and figures are not copied
        elif name not in ("problem_config.json", "dataset_version.json"):
            shutil.copy(src, dst)                             # matrices, alphas, labels, ood_sets.json ...
    new = restrict_problem_config(pcfg, n, os.path.basename(src_dir.rstrip("\\/")))
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(new, fh, indent=2)
    with open(os.path.join(dst_dir, "dataset_version.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": new["dataset_version"]}, fh)
    if verbose:
        print(f"  {n}-mode subset of {os.path.basename(src_dir)} written to {dst_dir}", file=sys.stderr)
    return dst_dir


def resolve_data_dir(data_dir, n_modes=None):
    """Absolute data folder for a run: data_dir itself, or its n_modes subset (created if needed)."""
    data_dir = data_dir if os.path.isabs(data_dir) else os.path.join(PROJECT_ROOT, data_dir)
    return data_dir if n_modes is None else derive_mode_subset(data_dir, int(n_modes))


def main():
    parser = argparse.ArgumentParser(description="M-mode subset of the final data set (prints its folder).")
    parser.add_argument("--data-dir", default=C.DATA_DIR)
    parser.add_argument("--modes", type=int, required=True)
    args, _ = parser.parse_known_args()
    print(resolve_data_dir(args.data_dir, args.modes))


if __name__ == "__main__":
    main()
