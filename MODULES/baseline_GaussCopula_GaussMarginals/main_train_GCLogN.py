#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py

Trains GC-LogN: Gaussian copula + one Gaussian in ln z per marginal (GCLogN_model.py), with the training procedure of the
other models (review29oct_fixed/training_common.py). Run it through model_comparison/run_convergence_gpu.sh --family gclogn.
Outputs: Output/baseline_GaussCopula_GaussMarginals/GCLogN_<data tag>_7D_obs-fv_Gamma1.0_Conv6000/
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
    description="Gaussian copula + one Gaussian in ln z per marginal (same ELBO and training as the other models)."))
parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128], help="hidden layers of the encoder")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)   # before TensorFlow is imported


def main():
    from MODULES.review29oct_fixed.training_common import train_and_save
    from MODULES.baseline_GaussCopula_GaussMarginals.GCLogN_model import GaussCopulaLogNVAE
    GaussCopulaLogNVAE.HIDDEN = tuple(args.hidden)
    return train_and_save(args, GaussCopulaLogNVAE, family="GC-LogN", n_components=1, output_dir=OUTPUT_DIR,
                          run_prefix="GCLogN", pred_keys=GaussCopulaLogNVAE.PRED_KEYS,
                          extra_info={"variational_family": "Gaussian copula + truncated log-normal marginals",
                                      "hidden": args.hidden})


if __name__ == "__main__":
    main()
