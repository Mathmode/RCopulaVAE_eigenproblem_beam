#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/baseline_GaussMixture_K5/main_train_GMM.py

Trains BASELINE 2 (Gaussian mixture, K = 5 components with full covariance, no copula; Ref. 39) with the training
procedure of the proposed method (review29oct_fixed/training_common.py: same data, likelihood, gamma = 1, warm-up,
optimizer, schedule, early stopping on the validation -ELBO). --samples is the number of samples PER COMPONENT
(default 2: 10 eigen-solutions per observation and step, against 6 for the KS model).

Run from the project root, e.g. on GPU 2 of the server:
    python MODULES/baseline_GaussMixture_K5/main_train_GMM.py --gpu 2 \
        --data-dir Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes --postprocess
Outputs: Output/baseline_GaussMixture_K5/GMM5full_<data tag>_7D_obs-fv_Gamma1.0_Ep1500/
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

OUTPUT_DIR = os.path.join(C.PROJECT_ROOT, "Output", "baseline_GaussMixture_K5")

parser = add_training_arguments(argparse.ArgumentParser(
    description="Baseline 2: Gaussian mixture posterior, no copula (same ELBO and training as the KS model)."),
    default_samples=2)
parser.add_argument("--components", type=int, default=5, help="number of Gaussian components K")
parser.add_argument("--covariance", choices=["full", "diag"], default="full")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)   # before TensorFlow is imported


def main():
    from MODULES.review29oct_fixed.training_common import train_and_save
    from MODULES.baseline_GaussMixture_K5.GMM_model import GaussMixtureVAE, GaussMixtureDiagVAE
    cls = GaussMixtureVAE if args.covariance == "full" else GaussMixtureDiagVAE
    return train_and_save(args, cls, family="GMM", n_components=args.components, output_dir=OUTPUT_DIR,
                          run_prefix=f"GMM{args.components}{args.covariance}", pred_keys=cls.PRED_KEYS,
                          extra_info={"covariance": args.covariance, "samples_per_component": args.samples,
                                      "variational_family": f"Gaussian mixture, K = {args.components}, "
                                                            f"{args.covariance} covariance, logit space (Ref. 39)"})


if __name__ == "__main__":
    main()
