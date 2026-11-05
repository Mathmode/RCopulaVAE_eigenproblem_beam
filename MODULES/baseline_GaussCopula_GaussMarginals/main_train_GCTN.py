#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py

Trains BASELINE 1 (Gaussian copula + one truncated Gaussian per marginal; Ref. 41 / submitted version) with the
training procedure of the proposed method (review29oct_fixed/training_common.py: same data, likelihood, gamma = 1,
warm-up, optimizer, schedule, early stopping on the validation -ELBO).

Run from the project root, e.g. on GPU 2 of the server:
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py --gpu 2 \
        --data-dir Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes --postprocess
Outputs: Output/baseline_GaussCopula_GaussMarginals/GCTN_<data tag>_7D_obs-fv_Gamma1.0_Ep1500/
"""

import os
import sys
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.training_common import add_training_arguments, select_gpu

OUTPUT_DIR = os.path.join(C.PROJECT_ROOT, "Output", "baseline_GaussCopula_GaussMarginals")

parser = add_training_arguments(argparse.ArgumentParser(
    description="Baseline 1: Gaussian copula + truncated-Gaussian marginals (same ELBO and training as the KS model)."))
parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128], help="hidden layers of the encoder")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)   # before TensorFlow is imported


def main():
    from MODULES.review29oct_fixed.training_common import train_and_save
    from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import GaussCopulaTNVAE
    GaussCopulaTNVAE.HIDDEN = tuple(args.hidden)
    return train_and_save(args, GaussCopulaTNVAE, family="GC-TN", n_components=1, output_dir=OUTPUT_DIR,
                          run_prefix="GCTN", pred_keys=GaussCopulaTNVAE.PRED_KEYS,
                          extra_info={"variational_family": "Gaussian copula + truncated-Gaussian marginals "
                                                            "(Ref. 41 / submitted version)", "hidden": args.hidden})


if __name__ == "__main__":
    main()
